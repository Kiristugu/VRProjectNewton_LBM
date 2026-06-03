"""Warp kernels for FLIP grid-liquid ↔ rigid body two-way coupling.

Two-way coupling between a MAC-grid FLIP liquid solver and Newton rigid bodies.

Solid → Fluid
-------------
1. Each frame, rasterize the rigid body's SDF into ``solid_phi`` (min-merge so
   multiple bodies compose correctly).
2. Embed the body's surface velocity into the MAC face arrays
   ``vel_solid_u / vel_solid_v / vel_solid_w``.  The FLIP solver uses these
   instead of zero when enforcing solid boundary conditions, which makes the
   pressure solve automatically generate the correct pressure field that moves
   fluid along with the solid.

Fluid → Solid
-------------
After the pressure solve, integrate the pressure over the solid face-fraction
weight at each fluid cell.  The net impulse equals the fluid's reaction force
on the rigid body (Newton's third law).  This is accumulated atomically into
``body_f`` so that the rigid solver picks it up on the same timestep.

Reference:
    Bridson, "Fluid Simulation for Computer Graphics", 2nd ed., §6.3
    Carlson et al., "Rigid Fluid" (SIGGRAPH 2004).
"""

from __future__ import annotations

import warp as wp
from wanphys.geometry import RigidShapeQueryData, point_body_distance


CELL_FLUID = wp.constant(1)


# ---------------------------------------------------------------------------
# Helper: body surface velocity at a world-space point
# ---------------------------------------------------------------------------

@wp.func
def body_surface_velocity(
    X_wb: wp.transform,
    body_lin_vel: wp.vec3,
    body_ang_vel: wp.vec3,
    body_com_local: wp.vec3,
    point_world: wp.vec3,
) -> wp.vec3:
    """Compute rigid surface velocity at a world-space point.

    Args:
        X_wb: Body-to-world transform.
        body_lin_vel: Linear velocity of the body in world frame.
        body_ang_vel: Angular velocity of the body in world frame.
        body_com_local: Center of mass in body-local frame.
        point_world: Query point in world frame.

    Returns:
        Surface velocity ``v_lin + ω × (x - x_com)`` in world frame.
    """
    com_world = wp.transform_point(X_wb, body_com_local)
    r = point_world - com_world
    return body_lin_vel + wp.cross(body_ang_vel, r)


# ---------------------------------------------------------------------------
# Unified SDF rasterization kernel  (Rigid → solid_phi + solid_body_id)
# ---------------------------------------------------------------------------

_SHAPE_SPHERE = 0
_SHAPE_BOX = 1
_SHAPE_CAPSULE = 2
_SHAPE_MESH = 3


@wp.func
def sdf_sphere(center: wp.vec3, radius: float, p: wp.vec3) -> float:
    return wp.length(p - center) - radius


@wp.func
def sdf_box(center: wp.vec3, half_extents: wp.vec3, rot: wp.quat, p: wp.vec3) -> float:
    inv_rot = wp.quat_inverse(rot)
    p_local = wp.quat_rotate(inv_rot, p - center)
    d = wp.vec3(wp.abs(p_local[0]) - half_extents[0], wp.abs(p_local[1]) - half_extents[1], wp.abs(p_local[2]) - half_extents[2])
    outside = wp.length(wp.vec3(wp.max(d[0], 0.0), wp.max(d[1], 0.0), wp.max(d[2], 0.0)))
    inside = wp.min(wp.max(d[0], wp.max(d[1], d[2])), 0.0)
    return outside + inside


@wp.func
def sdf_capsule(center: wp.vec3, radius: float, half_height: float, rot: wp.quat, p: wp.vec3) -> float:
    inv_rot = wp.quat_inverse(rot)
    p_local = wp.quat_rotate(inv_rot, p - center)
    seg_a = wp.vec3(0.0, 0.0, -half_height)
    seg_b = wp.vec3(0.0, 0.0, half_height)
    pa = p_local - seg_a
    ba = seg_b - seg_a
    h = wp.clamp(wp.dot(pa, ba) / wp.dot(ba, ba), 0.0, 1.0)
    return wp.length(pa - ba * h) - radius


@wp.func
def sdf_mesh(mesh: wp.uint64, pos: wp.vec3, rot: wp.quat, scale: float, max_dist: float, p: wp.vec3) -> float:
    inv_rot = wp.quat_inverse(rot)
    p_local = wp.quat_rotate(inv_rot, p - pos) / scale
    sign = float(0.0)
    face = int(0)
    u = float(0.0)
    v = float(0.0)
    if wp.mesh_query_point(mesh, p_local, max_dist, sign, face, u, v):
        cp_local = wp.mesh_eval_position(mesh, face, u, v)
        dist_local = wp.length(p_local - cp_local)
        return dist_local * scale * sign
    return max_dist


@wp.kernel
def rasterize_all_body_sdf(
    solid_phi: wp.array3d(dtype=float),
    solid_body_id: wp.array3d(dtype=wp.int32),
    dh: float,
    body_q: wp.array(dtype=wp.transform),
    body_count: int,
    data: RigidShapeQueryData,
    coupling_to_newton: wp.array(dtype=wp.int32),
):
    i, j, k = wp.tid()
    p = wp.vec3((float(i) + 0.5) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)

    best_dist = float(1000.0)
    best_id = int(-1)

    for b in range(body_count):
        body_id = coupling_to_newton[b]
        
        dist = point_body_distance(p, body_id, data, body_q)

        if dist < best_dist:
            best_dist = dist
            best_id = body_id

    solid_phi[i, j, k] = best_dist
    solid_body_id[i, j, k] = wp.int32(best_id)


@wp.kernel
def rasterize_all_body_sdf_warp(
    solid_phi: wp.array3d(dtype=float),
    solid_body_id: wp.array3d(dtype=wp.int32),
    dh: float,
    body_q: wp.array(dtype=wp.transform),
    body_count: int,
    coupling_to_newton: wp.array(dtype=wp.int32),
    body_shape_type: wp.array(dtype=wp.int32),
    body_sphere_radius: wp.array(dtype=float),
    body_box_half_extents: wp.array(dtype=wp.vec3),
    body_capsule_radius: wp.array(dtype=float),
    body_capsule_half_height: wp.array(dtype=float),
    body_mesh_handle: wp.array(dtype=wp.uint64),
    body_mesh_scale: wp.array(dtype=float),
    body_mesh_max_dist: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    p = wp.vec3((float(i) + 0.5) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)

    best_dist = float(1000.0)
    best_id = int(-1)

    for b in range(body_count):
        body_id = int(coupling_to_newton[b])
        pos = wp.transform_get_translation(body_q[body_id])
        rot = wp.transform_get_rotation(body_q[body_id])
        shape = body_shape_type[b]
        dist = float(1000.0)

        if shape == _SHAPE_SPHERE:
            dist = sdf_sphere(pos, body_sphere_radius[b], p)
        elif shape == _SHAPE_BOX:
            dist = sdf_box(pos, body_box_half_extents[b], rot, p)
        elif shape == _SHAPE_CAPSULE:
            dist = sdf_capsule(pos, body_capsule_radius[b], body_capsule_half_height[b], rot, p)
        elif shape == _SHAPE_MESH:
            dist = sdf_mesh(body_mesh_handle[b], pos, rot, body_mesh_scale[b], body_mesh_max_dist[b], p)

        if dist < best_dist:
            best_dist = dist
            best_id = body_id

    solid_phi[i, j, k] = best_dist
    solid_body_id[i, j, k] = wp.int32(best_id)


# ---------------------------------------------------------------------------
# SDF rasterization kernels  (Rigid → solid_phi)  — per-body, kept for bake_box/bake_mesh
# ---------------------------------------------------------------------------

@wp.kernel
def rasterize_sphere_sdf(
    solid_phi: wp.array3d(dtype=float),
    dh: float,
    center: wp.vec3,
    radius: float,
):
    """Rasterize a moving sphere into solid_phi (min-merge).

    Args:
        solid_phi: Grid-cell SDF that tracks all solid objects.
        dh: Cell size.
        center: Sphere center in world space.
        radius: Sphere radius.
    """
    i, j, k = wp.tid()
    p = wp.vec3((float(i) + 0.5) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)
    dist = wp.length(p - center) - radius
    solid_phi[i, j, k] = wp.min(solid_phi[i, j, k], dist)


@wp.kernel
def rasterize_box_sdf(
    solid_phi: wp.array3d(dtype=float),
    dh: float,
    center: wp.vec3,
    half_extents: wp.vec3,
    rot: wp.quat,
):
    """Rasterize a moving oriented box into solid_phi (min-merge).

    Args:
        solid_phi: Grid-cell SDF.
        dh: Cell size.
        center: Box center in world space.
        half_extents: Half-extents in body-local frame.
        rot: Body orientation quaternion (body→world).
    """
    i, j, k = wp.tid()
    p_world = wp.vec3((float(i) + 0.5) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)
    # Transform point to body-local frame
    inv_rot = wp.quat_inverse(rot)
    p_local = wp.quat_rotate(inv_rot, p_world - center)
    d = wp.vec3(
        wp.abs(p_local[0]) - half_extents[0],
        wp.abs(p_local[1]) - half_extents[1],
        wp.abs(p_local[2]) - half_extents[2],
    )
    outside = wp.length(wp.vec3(wp.max(d[0], 0.0), wp.max(d[1], 0.0), wp.max(d[2], 0.0)))
    inside = wp.min(wp.max(d[0], wp.max(d[1], d[2])), 0.0)
    dist = outside + inside
    solid_phi[i, j, k] = wp.min(solid_phi[i, j, k], dist)


@wp.kernel
def rasterize_capsule_sdf(
    solid_phi: wp.array3d(dtype=float),
    dh: float,
    center: wp.vec3,
    radius: float,
    half_height: float,
    rot: wp.quat,
):
    """Rasterize a moving oriented capsule into solid_phi (min-merge)."""
    i, j, k = wp.tid()
    p_world = wp.vec3((float(i) + 0.5) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)
    inv_rot = wp.quat_inverse(rot)
    p_local = wp.quat_rotate(inv_rot, p_world - center)

    seg_a = wp.vec3(0.0, 0.0, -half_height)
    seg_b = wp.vec3(0.0, 0.0, half_height)
    pa = p_local - seg_a
    ba = seg_b - seg_a
    h = wp.clamp(wp.dot(pa, ba) / wp.dot(ba, ba), 0.0, 1.0)
    dist = wp.length(pa - ba * h) - radius
    solid_phi[i, j, k] = wp.min(solid_phi[i, j, k], dist)


@wp.kernel
def rasterize_mesh_sdf(
    solid_phi: wp.array3d(dtype=float),
    mesh: wp.uint64,
    dh: float,
    pos: wp.vec3,
    rot: wp.quat,
    scale: float,
    max_dist: float,
):
    """Rasterize a moving mesh into solid_phi (min-merge).

    Args:
        solid_phi: Grid-cell SDF.
        mesh: Warp mesh id.
        dh: Cell size.
        pos: Mesh position in world frame.
        rot: Mesh orientation in world frame.
        scale: Uniform mesh scale.
        max_dist: Maximum query distance (for performance).
    """
    i, j, k = wp.tid()
    p_world = wp.vec3((float(i) + 0.5) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)
    inv_rot = wp.quat_inverse(rot)
    p_local = wp.quat_rotate(inv_rot, p_world - pos) / scale

    sign = float(0.0)
    face = int(0)
    u = float(0.0)
    v = float(0.0)
    if wp.mesh_query_point(mesh, p_local, max_dist, sign, face, u, v):
        cp_local = wp.mesh_eval_position(mesh, face, u, v)
        dist_local = wp.length(p_local - cp_local)
        sdf = dist_local * scale * sign
        solid_phi[i, j, k] = wp.min(solid_phi[i, j, k], sdf)


# ---------------------------------------------------------------------------
# Solid surface velocity embedding  (Rigid velocity → MAC face BCs)
#
# For each MAC face that lies inside or on the solid boundary we store the
# rigid body's surface velocity component.  The FLIP solver then stamps this
# onto the velocity field instead of zero when enforcing solid BCs.
# ---------------------------------------------------------------------------

@wp.func
def pick_body_id_for_face_x(
    solid_body_id: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    i: int, j: int, k: int, nx: int,
) -> int:
    """Pick the body_id for a u-face between cells (i-1,j,k) and (i,j,k)."""
    phi_l = solid_phi[wp.max(i - 1, 0), j, k]
    phi_r = solid_phi[wp.min(i, nx - 1), j, k]
    id_l = int(solid_body_id[wp.max(i - 1, 0), j, k])
    id_r = int(solid_body_id[wp.min(i, nx - 1), j, k])
    if phi_l < 0.0 and phi_r < 0.0:
        if phi_l < phi_r:
            return id_l
        return id_r
    if phi_l < 0.0:
        return id_l
    if phi_r < 0.0:
        return id_r
    return -1


@wp.func
def pick_body_id_for_face_y(
    solid_body_id: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    i: int, j: int, k: int, ny: int,
) -> int:
    """Pick the body_id for a v-face between cells (i,j-1,k) and (i,j,k)."""
    phi_d = solid_phi[i, wp.max(j - 1, 0), k]
    phi_u = solid_phi[i, wp.min(j, ny - 1), k]
    id_d = int(solid_body_id[i, wp.max(j - 1, 0), k])
    id_u = int(solid_body_id[i, wp.min(j, ny - 1), k])
    if phi_d < 0.0 and phi_u < 0.0:
        if phi_d < phi_u:
            return id_d
        return id_u
    if phi_d < 0.0:
        return id_d
    if phi_u < 0.0:
        return id_u
    return -1


@wp.func
def pick_body_id_for_face_z(
    solid_body_id: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    i: int, j: int, k: int, nz: int,
) -> int:
    """Pick the body_id for a w-face between cells (i,j,k-1) and (i,j,k)."""
    phi_b = solid_phi[i, j, wp.max(k - 1, 0)]
    phi_f = solid_phi[i, j, wp.min(k, nz - 1)]
    id_b = int(solid_body_id[i, j, wp.max(k - 1, 0)])
    id_f = int(solid_body_id[i, j, wp.min(k, nz - 1)])
    if phi_b < 0.0 and phi_f < 0.0:
        if phi_b < phi_f:
            return id_b
        return id_f
    if phi_b < 0.0:
        return id_b
    if phi_f < 0.0:
        return id_f
    return -1


@wp.kernel
def embed_all_solid_velocity_u(
    solid_phi: wp.array3d(dtype=float),
    solid_body_id: wp.array3d(dtype=wp.int32),
    vel_solid_u: wp.array3d(dtype=float),
    dh: float,
    nx: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
):
    i, j, k = wp.tid()
    phi_l = solid_phi[wp.max(i - 1, 0), j, k]
    phi_r = solid_phi[wp.min(i, nx - 1), j, k]
    if phi_l >= 0.0 and phi_r >= 0.0:
        vel_solid_u[i, j, k] = 0.0
        return

    body_id = pick_body_id_for_face_x(solid_body_id, solid_phi, i, j, k, nx)
    if body_id < 0:
        vel_solid_u[i, j, k] = 0.0
        return

    X_wb = body_q[body_id]
    qd = body_qd[body_id]
    lin_vel = wp.spatial_top(qd)
    ang_vel = wp.spatial_bottom(qd)
    com_local = body_com[body_id]
    face_pos = wp.vec3(float(i) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)
    surf_vel = body_surface_velocity(X_wb, lin_vel, ang_vel, com_local, face_pos)
    vel_solid_u[i, j, k] = surf_vel[0]


@wp.kernel
def embed_all_solid_velocity_v(
    solid_phi: wp.array3d(dtype=float),
    solid_body_id: wp.array3d(dtype=wp.int32),
    vel_solid_v: wp.array3d(dtype=float),
    dh: float,
    ny: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
):
    i, j, k = wp.tid()
    phi_d = solid_phi[i, wp.max(j - 1, 0), k]
    phi_u = solid_phi[i, wp.min(j, ny - 1), k]
    if phi_d >= 0.0 and phi_u >= 0.0:
        vel_solid_v[i, j, k] = 0.0
        return

    body_id = pick_body_id_for_face_y(solid_body_id, solid_phi, i, j, k, ny)
    if body_id < 0:
        vel_solid_v[i, j, k] = 0.0
        return

    X_wb = body_q[body_id]
    qd = body_qd[body_id]
    lin_vel = wp.spatial_top(qd)
    ang_vel = wp.spatial_bottom(qd)
    com_local = body_com[body_id]
    face_pos = wp.vec3((float(i) + 0.5) * dh, float(j) * dh, (float(k) + 0.5) * dh)
    surf_vel = body_surface_velocity(X_wb, lin_vel, ang_vel, com_local, face_pos)
    vel_solid_v[i, j, k] = surf_vel[1]


@wp.kernel
def embed_all_solid_velocity_w(
    solid_phi: wp.array3d(dtype=float),
    solid_body_id: wp.array3d(dtype=wp.int32),
    vel_solid_w: wp.array3d(dtype=float),
    dh: float,
    nz: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
):
    i, j, k = wp.tid()
    phi_b = solid_phi[i, j, wp.max(k - 1, 0)]
    phi_f = solid_phi[i, j, wp.min(k, nz - 1)]
    if phi_b >= 0.0 and phi_f >= 0.0:
        vel_solid_w[i, j, k] = 0.0
        return

    body_id = pick_body_id_for_face_z(solid_body_id, solid_phi, i, j, k, nz)
    if body_id < 0:
        vel_solid_w[i, j, k] = 0.0
        return

    X_wb = body_q[body_id]
    qd = body_qd[body_id]
    lin_vel = wp.spatial_top(qd)
    ang_vel = wp.spatial_bottom(qd)
    com_local = body_com[body_id]
    face_pos = wp.vec3((float(i) + 0.5) * dh, (float(j) + 0.5) * dh, float(k) * dh)
    surf_vel = body_surface_velocity(X_wb, lin_vel, ang_vel, com_local, face_pos)
    vel_solid_w[i, j, k] = surf_vel[2]


# ---------------------------------------------------------------------------
# Per-body velocity embedding  (kept for backward compatibility, e.g. robot example)
# ---------------------------------------------------------------------------

@wp.kernel
def embed_solid_velocity_u(
    solid_phi: wp.array3d(dtype=float),
    vel_solid_u: wp.array3d(dtype=float),
    dh: float,
    nx: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_idx: int,
):
    i, j, k = wp.tid()
    phi_l = solid_phi[wp.max(i - 1, 0), j, k]
    phi_r = solid_phi[wp.min(i, nx - 1), j, k]
    if phi_l >= 0.0 and phi_r >= 0.0:
        return
    X_wb = body_q[body_idx]
    qd = body_qd[body_idx]
    lin_vel = wp.spatial_top(qd)
    ang_vel = wp.spatial_bottom(qd)
    com_local = body_com[body_idx]
    face_pos = wp.vec3(float(i) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)
    surf_vel = body_surface_velocity(X_wb, lin_vel, ang_vel, com_local, face_pos)
    wp.atomic_add(vel_solid_u, i, j, k, surf_vel[0])


@wp.kernel
def embed_solid_velocity_v(
    solid_phi: wp.array3d(dtype=float),
    vel_solid_v: wp.array3d(dtype=float),
    dh: float,
    ny: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_idx: int,
):
    i, j, k = wp.tid()
    phi_d = solid_phi[i, wp.max(j - 1, 0), k]
    phi_u = solid_phi[i, wp.min(j, ny - 1), k]
    if phi_d >= 0.0 and phi_u >= 0.0:
        return
    X_wb = body_q[body_idx]
    qd = body_qd[body_idx]
    lin_vel = wp.spatial_top(qd)
    ang_vel = wp.spatial_bottom(qd)
    com_local = body_com[body_idx]
    face_pos = wp.vec3((float(i) + 0.5) * dh, float(j) * dh, (float(k) + 0.5) * dh)
    surf_vel = body_surface_velocity(X_wb, lin_vel, ang_vel, com_local, face_pos)
    wp.atomic_add(vel_solid_v, i, j, k, surf_vel[1])


@wp.kernel
def embed_solid_velocity_w(
    solid_phi: wp.array3d(dtype=float),
    vel_solid_w: wp.array3d(dtype=float),
    dh: float,
    nz: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_idx: int,
):
    i, j, k = wp.tid()
    phi_b = solid_phi[i, j, wp.max(k - 1, 0)]
    phi_f = solid_phi[i, j, wp.min(k, nz - 1)]
    if phi_b >= 0.0 and phi_f >= 0.0:
        return
    X_wb = body_q[body_idx]
    qd = body_qd[body_idx]
    lin_vel = wp.spatial_top(qd)
    ang_vel = wp.spatial_bottom(qd)
    com_local = body_com[body_idx]
    face_pos = wp.vec3((float(i) + 0.5) * dh, (float(j) + 0.5) * dh, float(k) * dh)
    surf_vel = body_surface_velocity(X_wb, lin_vel, ang_vel, com_local, face_pos)
    wp.atomic_add(vel_solid_w, i, j, k, surf_vel[2])


# ---------------------------------------------------------------------------
# Stamp solid velocity onto MAC faces (inside _enforce_boundary)
#
# For every face that is a solid face (solid_weight <= 0), overwrite the
# fluid velocity with the corresponding solid surface velocity.
# ---------------------------------------------------------------------------

@wp.kernel
def stamp_solid_velocity_u(
    vel_u: wp.array3d(dtype=float),
    vel_solid_u: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    nx: int,
):
    """Overwrite fluid u-velocity at solid faces with solid surface velocity."""
    i, j, k = wp.tid()
    if i == 0 or i == nx:
        vel_u[i, j, k] = 0.0
        return
    s_l = solid_phi[wp.max(i - 1, 0), j, k]
    s_r = solid_phi[wp.min(i, nx - 1), j, k]
    if s_l * s_r < 0.0:
        vel_u[i, j, k] = vel_solid_u[i, j, k]
    elif s_l < 0.0 and s_r < 0.0:
        vel_u[i, j, k] = 0.0


@wp.kernel
def stamp_solid_velocity_v(
    vel_v: wp.array3d(dtype=float),
    vel_solid_v: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    ny: int,
):
    """Overwrite fluid v-velocity at solid faces with solid surface velocity."""
    i, j, k = wp.tid()
    if j == 0 or j == ny:
        vel_v[i, j, k] = 0.0
        return
    s_d = solid_phi[i, wp.max(j - 1, 0), k]
    s_u = solid_phi[i, wp.min(j, ny - 1), k]
    if s_d * s_u < 0.0:
        vel_v[i, j, k] = vel_solid_v[i, j, k]
    elif s_d < 0.0 and s_u < 0.0:
        vel_v[i, j, k] = 0.0


@wp.kernel
def stamp_solid_velocity_w(
    vel_w: wp.array3d(dtype=float),
    vel_solid_w: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    nz: int,
):
    """Overwrite fluid w-velocity at solid faces with solid surface velocity."""
    i, j, k = wp.tid()
    if k == 0 or k == nz:
        vel_w[i, j, k] = 0.0
        return
    s_b = solid_phi[i, j, wp.max(k - 1, 0)]
    s_f = solid_phi[i, j, wp.min(k, nz - 1)]
    if s_b * s_f < 0.0:
        vel_w[i, j, k] = vel_solid_w[i, j, k]
    elif s_b < 0.0 and s_f < 0.0:
        vel_w[i, j, k] = 0.0


# ---------------------------------------------------------------------------
# Pressure force / torque on rigid body  (Fluid → Rigid)
#
# For each fluid cell adjacent to a solid face, the pressure exerts a force
# on the solid equal to:
#
#   F += p * θ_solid_face * face_normal * dh²
#
# where θ_solid_face = solid_fraction(phi of neighboring cells) and the sign
# of face_normal points from fluid into solid.  Summing over all 6 faces of
# a fluid cell gives the net pressure force on the solid in that cell.
#
# We accumulate atomically into body_f.
# ---------------------------------------------------------------------------

@wp.func
def solid_fraction_f(phi0: float, phi1: float) -> float:
    """Fraction of the face that is solid (0 = all fluid, 1 = all solid)."""
    if phi0 < 0.0 and phi1 < 0.0:
        return 1.0
    if phi0 < 0.0 and phi1 >= 0.0:
        denom = phi0 - phi1
        if wp.abs(denom) > 1.0e-8:
            return wp.clamp(phi0 / denom, 0.0, 1.0)
    if phi0 >= 0.0 and phi1 < 0.0:
        denom = phi1 - phi0
        if wp.abs(denom) > 1.0e-8:
            return wp.clamp(phi1 / denom, 0.0, 1.0)
    return 0.0


@wp.kernel
def accumulate_pressure_force_all_bodies(
    pressure: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.int32),
    solid_phi: wp.array3d(dtype=float),
    solid_body_id: wp.array3d(dtype=wp.int32),
    dh: float,
    nx: int,
    ny: int,
    nz: int,
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
    pressure_scale: float,
):
    """Accumulate pressure force/torque from fluid cells onto all rigid bodies.

    For each fluid cell, we check the 6 neighbouring faces.  When a face
    crosses from fluid into solid, we read the solid-side body_id, resolve
    the corresponding Newton body index, and atomically accumulate force +
    torque into ``body_f``.
    """
    i, j, k = wp.tid()
    p = pressure[i, j, k]
    if wp.abs(p) < 1.0e-12:
        return
    if cell_type[i, j, k] != wp.int32(CELL_FLUID):
        return

    face_area = dh * dh

    # -x face
    if i > 0:
        theta = solid_fraction_f(solid_phi[i - 1, j, k], solid_phi[i, j, k])
        if theta > 0.0:
            body_id = int(solid_body_id[i - 1, j, k])
            if body_id >= 0:
                com_world = wp.transform_point(body_q[body_id], body_com[body_id])
                face_pos = wp.vec3(float(i) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)
                df = p * theta * face_area * wp.vec3(-1.0, 0.0, 0.0)
                dt_val = wp.cross(face_pos - com_world, df)
                wp.atomic_add(body_f, body_id, wp.spatial_vector(df * pressure_scale, dt_val * pressure_scale))
    # +x face
    if i < nx - 1:
        theta = solid_fraction_f(solid_phi[i, j, k], solid_phi[i + 1, j, k])
        if theta > 0.0:
            body_id = int(solid_body_id[i + 1, j, k])
            if body_id >= 0:
                com_world = wp.transform_point(body_q[body_id], body_com[body_id])
                face_pos = wp.vec3(float(i + 1) * dh, (float(j) + 0.5) * dh, (float(k) + 0.5) * dh)
                df = p * theta * face_area * wp.vec3(1.0, 0.0, 0.0)
                dt_val = wp.cross(face_pos - com_world, df)
                wp.atomic_add(body_f, body_id, wp.spatial_vector(df * pressure_scale, dt_val * pressure_scale))
    # -y face
    if j > 0:
        theta = solid_fraction_f(solid_phi[i, j - 1, k], solid_phi[i, j, k])
        if theta > 0.0:
            body_id = int(solid_body_id[i, j - 1, k])
            if body_id >= 0:
                com_world = wp.transform_point(body_q[body_id], body_com[body_id])
                face_pos = wp.vec3((float(i) + 0.5) * dh, float(j) * dh, (float(k) + 0.5) * dh)
                df = p * theta * face_area * wp.vec3(0.0, -1.0, 0.0)
                dt_val = wp.cross(face_pos - com_world, df)
                wp.atomic_add(body_f, body_id, wp.spatial_vector(df * pressure_scale, dt_val * pressure_scale))
    # +y face
    if j < ny - 1:
        theta = solid_fraction_f(solid_phi[i, j, k], solid_phi[i, j + 1, k])
        if theta > 0.0:
            body_id = int(solid_body_id[i, j + 1, k])
            if body_id >= 0:
                com_world = wp.transform_point(body_q[body_id], body_com[body_id])
                face_pos = wp.vec3((float(i) + 0.5) * dh, float(j + 1) * dh, (float(k) + 0.5) * dh)
                df = p * theta * face_area * wp.vec3(0.0, 1.0, 0.0)
                dt_val = wp.cross(face_pos - com_world, df)
                wp.atomic_add(body_f, body_id, wp.spatial_vector(df * pressure_scale, dt_val * pressure_scale))
    # -z face
    if k > 0:
        theta = solid_fraction_f(solid_phi[i, j, k - 1], solid_phi[i, j, k])
        if theta > 0.0:
            body_id = int(solid_body_id[i, j, k - 1])
            if body_id >= 0:
                com_world = wp.transform_point(body_q[body_id], body_com[body_id])
                face_pos = wp.vec3((float(i) + 0.5) * dh, (float(j) + 0.5) * dh, float(k) * dh)
                df = p * theta * face_area * wp.vec3(0.0, 0.0, -1.0)
                dt_val = wp.cross(face_pos - com_world, df)
                wp.atomic_add(body_f, body_id, wp.spatial_vector(df * pressure_scale, dt_val * pressure_scale))
    # +z face
    if k < nz - 1:
        theta = solid_fraction_f(solid_phi[i, j, k], solid_phi[i, j, k + 1])
        if theta > 0.0:
            body_id = int(solid_body_id[i, j, k + 1])
            if body_id >= 0:
                com_world = wp.transform_point(body_q[body_id], body_com[body_id])
                face_pos = wp.vec3((float(i) + 0.5) * dh, (float(j) + 0.5) * dh, float(k + 1) * dh)
                df = p * theta * face_area * wp.vec3(0.0, 0.0, 1.0)
                dt_val = wp.cross(face_pos - com_world, df)
                wp.atomic_add(body_f, body_id, wp.spatial_vector(df * pressure_scale, dt_val * pressure_scale))
