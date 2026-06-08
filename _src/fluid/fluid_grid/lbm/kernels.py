# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""LBM Warp kernels (DESIGN.md §7).

Week-1 deliverables (member B, §7.1–7.2):
  - ``init_equilibrium``  — uniform equilibrium initialization
  - ``collide_bgk``       — BGK collision operator
  - ``update_macro``      — density / velocity recovery from F
  - ``stream_pull``       — pull streaming (week 2, see docs/phase1_stream_pull.md)
  - ``stream_pull_identity`` — week-1 placeholder (kept for debugging)

§7.3 boundary kernels — ``apply_boundaries`` (week 2, see docs/phase2_boundaries.md).
"""

from __future__ import annotations

import warp as wp

from .lattice import Q, feq, lattice_e_f, lattice_e_i, lattice_lr

USE_GUO_OFF = wp.constant(0)


@wp.kernel
def init_equilibrium(
    # Week-1 (B): set f, F, rho, v to uniform equilibrium (DESIGN.md §7.1).
    f: wp.array4d(dtype=float),
    F: wp.array4d(dtype=float),
    rho: wp.array3d(dtype=float),
    v: wp.array3d(dtype=wp.vec3),
    solid: wp.array3d(dtype=wp.int32),
    rho0: float,
    u0: wp.vec3,
) -> None:
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        return
    rho[i, j, k] = rho0
    v[i, j, k] = u0
    for q in range(Q):
        eq_val = feq(q, rho0, u0)
        f[i, j, k, q] = eq_val
        F[i, j, k, q] = eq_val


@wp.kernel
def collide_bgk(
    # Week-1 (B): F = f - omega * (f - feq); skip solid cells (DESIGN.md §7.2).
    f: wp.array4d(dtype=float),
    F: wp.array4d(dtype=float),
    rho: wp.array3d(dtype=float),
    v: wp.array3d(dtype=wp.vec3),
    solid: wp.array3d(dtype=wp.int32),
    omega: float,
) -> None:
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        return
    r = rho[i, j, k]
    u = v[i, j, k]
    for q in range(Q):
        fq = f[i, j, k, q]
        feq_val = feq(q, r, u)
        F[i, j, k, q] = fq - omega * (fq - feq_val)


@wp.func
def wrap_index(i: int, n: int, bc_low: int, bc_high: int) -> int:
    """Periodic wrap on one axis when bc_low/bc_high == 0; otherwise return i (may be OOB)."""
    if bc_low == 0 and i < 0:
        return n - 1
    if bc_high == 0 and i >= n:
        return 0
    return i


@wp.func
def periodic_index_vec(
    ix: int,
    iy: int,
    iz: int,
    nx: int,
    ny: int,
    nz: int,
    bc_x_left: int,
    bc_x_right: int,
    bc_y_left: int,
    bc_y_right: int,
    bc_z_left: int,
    bc_z_right: int,
) -> wp.vec3i:
    """Apply per-face periodic wrapping (DESIGN.md §4.1, taichi periodic_index)."""
    wx = wrap_index(ix, nx, bc_x_left, bc_x_right)
    wy = wrap_index(iy, ny, bc_y_left, bc_y_right)
    wz = wrap_index(iz, nz, bc_z_left, bc_z_right)
    return wp.vec3i(wx, wy, wz)


@wp.kernel
def stream_pull(
    f: wp.array4d(dtype=float),
    F: wp.array4d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    bc_x_left: int,
    bc_x_right: int,
    bc_y_left: int,
    bc_y_right: int,
    bc_z_left: int,
    bc_z_right: int,
) -> None:
    """Pull streaming: read post-collision f, write post-stream F (taichi streaming1)."""
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        return
    if i >= nx or j >= ny or k >= nz:
        return

    for s in range(Q):
        e = lattice_e_i(s)
        ip = periodic_index_vec(
            i + e[0],
            j + e[1],
            k + e[2],
            nx,
            ny,
            nz,
            bc_x_left,
            bc_x_right,
            bc_y_left,
            bc_y_right,
            bc_z_left,
            bc_z_right,
        )
        ip_i = ip[0]
        ip_j = ip[1]
        ip_k = ip[2]
        in_bounds = ip_i >= 0 and ip_i < nx and ip_j >= 0 and ip_j < ny and ip_k >= 0 and ip_k < nz
        if in_bounds and solid[ip_i, ip_j, ip_k] == 0:
            F[ip_i, ip_j, ip_k, s] = f[i, j, k, s]
        else:
            lr = lattice_lr(s)
            F[i, j, k, lr] = f[i, j, k, s]


@wp.kernel
def stream_pull_identity(
    F: wp.array4d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
) -> None:
    """Week-1 placeholder: identity stream (debug only; solver uses stream_pull)."""
    wp.tid()  # no-op: F unchanged; valid for periodic rest-fluid M1


@wp.kernel
def update_macro(
    F: wp.array4d(dtype=float),
    rho: wp.array3d(dtype=float),
    v: wp.array3d(dtype=wp.vec3),
    solid: wp.array3d(dtype=wp.int32),
    force: wp.vec3,
    use_guo: int,
) -> None:
    """Compute rho and u from post-stream/BC distributions in F (DESIGN.md §4 step 4)."""
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        rho[i, j, k] = 1.0
        v[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
        return

    r = float(0.0)
    u = wp.vec3(0.0, 0.0, 0.0)
    for q in range(Q):
        fq = F[i, j, k, q]
        r += fq
        u += lattice_e_f(q) * fq

    rho[i, j, k] = r
    u = u / r
    if use_guo == 1:
        u += (force / 2.0) / r
    v[i, j, k] = u


@wp.func
def set_velocity_bc(
    F: wp.array4d(dtype=float),
    i: int,
    j: int,
    k: int,
    u_bc: wp.vec3,
) -> None:
    """Fixed-velocity face BC: F_i = feq(rho=1, u_bc) (taichi bc_*==2)."""
    for s in range(Q):
        F[i, j, k, s] = feq(s, 1.0, u_bc)


@wp.func
def set_pressure_bc(
    F: wp.array4d(dtype=float),
    i: int,
    j: int,
    k: int,
    rho_bc: float,
    u_wall: wp.vec3,
    u_inner: wp.vec3,
    inner_is_solid: int,
) -> None:
    """Fixed-density face BC: feq at rho_bc with neighbour velocity (taichi bc_*==1)."""
    u_use = u_wall
    if inner_is_solid > 0:
        u_use = u_inner
    for s in range(Q):
        F[i, j, k, s] = feq(s, rho_bc, u_use)


@wp.kernel
def apply_boundaries(
    F: wp.array4d(dtype=float),
    f: wp.array4d(dtype=float),
    rho: wp.array3d(dtype=float),
    v: wp.array3d(dtype=wp.vec3),
    solid: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    bc_x_left: int,
    bc_x_right: int,
    bc_y_left: int,
    bc_y_right: int,
    bc_z_left: int,
    bc_z_right: int,
    bc_rho: float,
    bc_vel_x_left: wp.vec3,
    bc_vel_x_right: wp.vec3,
    bc_vel_y_left: wp.vec3,
    bc_vel_y_right: wp.vec3,
    bc_vel_z_left: wp.vec3,
    bc_vel_z_right: wp.vec3,
) -> None:
    """Domain face BC after streaming (taichi Boundary_condition, DESIGN.md §7.3)."""
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        return

    # X faces (order matches taichi: x left/right before y/z).
    if i == 0:
        if bc_x_left == 2:
            set_velocity_bc(F, i, j, k, bc_vel_x_left)
        elif bc_x_left == 1:
            inner_solid = 0
            if nx > 1:
                inner_solid = solid[1, j, k]
            u_inner = v[1, j, k]
            if nx <= 1:
                u_inner = v[i, j, k]
            set_pressure_bc(F, i, j, k, bc_rho, v[i, j, k], u_inner, inner_solid)

    if i == nx - 1:
        if bc_x_right == 2:
            set_velocity_bc(F, i, j, k, bc_vel_x_right)
        elif bc_x_right == 1:
            inner_solid = 0
            if nx > 1:
                inner_solid = solid[nx - 2, j, k]
            u_inner = v[nx - 2, j, k]
            if nx <= 1:
                u_inner = v[i, j, k]
            set_pressure_bc(F, i, j, k, bc_rho, v[i, j, k], u_inner, inner_solid)

    # Y faces
    if j == 0:
        if bc_y_left == 2:
            set_velocity_bc(F, i, j, k, bc_vel_y_left)
        elif bc_y_left == 1:
            inner_solid = 0
            if ny > 1:
                inner_solid = solid[i, 1, k]
            u_inner = v[i, 1, k]
            if ny <= 1:
                u_inner = v[i, j, k]
            set_pressure_bc(F, i, j, k, bc_rho, v[i, j, k], u_inner, inner_solid)

    if j == ny - 1:
        if bc_y_right == 2:
            set_velocity_bc(F, i, j, k, bc_vel_y_right)
        elif bc_y_right == 1:
            inner_solid = 0
            if ny > 1:
                inner_solid = solid[i, ny - 2, k]
            u_inner = v[i, ny - 2, k]
            if ny <= 1:
                u_inner = v[i, j, k]
            set_pressure_bc(F, i, j, k, bc_rho, v[i, j, k], u_inner, inner_solid)

    # Z faces
    if k == 0:
        if bc_z_left == 2:
            set_velocity_bc(F, i, j, k, bc_vel_z_left)
        elif bc_z_left == 1:
            inner_solid = 0
            if nz > 1:
                inner_solid = solid[i, j, 1]
            u_inner = v[i, j, 1]
            if nz <= 1:
                u_inner = v[i, j, k]
            set_pressure_bc(F, i, j, k, bc_rho, v[i, j, k], u_inner, inner_solid)

    if k == nz - 1:
        if bc_z_right == 2:
            set_velocity_bc(F, i, j, k, bc_vel_z_right)
        elif bc_z_right == 1:
            inner_solid = 0
            if nz > 1:
                inner_solid = solid[i, j, nz - 2]
            u_inner = v[i, j, nz - 2]
            if nz <= 1:
                u_inner = v[i, j, k]
            set_pressure_bc(F, i, j, k, bc_rho, v[i, j, k], u_inner, inner_solid)
