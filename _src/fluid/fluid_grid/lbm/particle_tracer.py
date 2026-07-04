# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Tracer particle advection kernels for LBM velocity fields.

Particles are advected through the LBM velocity field using RK2 integrator
with trilinear interpolation. Designed for 3D flow visualization (vortices, etc.).
"""

from __future__ import annotations

import warp as wp


@wp.func
def _sample_velocity_trilinear(
    v: wp.array3d(dtype=wp.vec3),
    nx: int,
    ny: int,
    nz: int,
    pos: wp.vec3,
) -> wp.vec3:
    """Trilinear sample of velocity field at world-space position *pos*.

    ``v`` is indexed as ``[i, j, k]`` where i=x, j=y, k=z.
    Clamps to [0, nx-1] × [0, ny-1] × [0, nz-1].
    """
    x = wp.clamp(pos[0], 0.0, float(nx - 1) - 1e-6)
    y = wp.clamp(pos[1], 0.0, float(ny - 1) - 1e-6)
    z = wp.clamp(pos[2], 0.0, float(nz - 1) - 1e-6)

    i0 = int(wp.floor(x))
    j0 = int(wp.floor(y))
    k0 = int(wp.floor(z))
    i1 = wp.min(i0 + 1, nx - 1)
    j1 = wp.min(j0 + 1, ny - 1)
    k1 = wp.min(k0 + 1, nz - 1)

    fx = x - float(i0)
    fy = y - float(j0)
    fz = z - float(k0)

    v000 = v[i0, j0, k0]
    v100 = v[i1, j0, k0]
    v010 = v[i0, j1, k0]
    v110 = v[i1, j1, k0]
    v001 = v[i0, j0, k1]
    v101 = v[i1, j0, k1]
    v011 = v[i0, j1, k1]
    v111 = v[i1, j1, k1]

    c00 = v000 * (1.0 - fx) + v100 * fx
    c01 = v001 * (1.0 - fx) + v101 * fx
    c10 = v010 * (1.0 - fx) + v110 * fx
    c11 = v011 * (1.0 - fx) + v111 * fx

    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy

    return c0 * (1.0 - fz) + c1 * fz


@wp.func
def _is_inside_domain(pos: wp.vec3, nx: int, ny: int, nz: int) -> bool:
    return (
        pos[0] >= 0.5
        and pos[0] <= float(nx - 1) - 0.5
        and pos[1] >= 0.5
        and pos[1] <= float(ny - 1) - 0.5
        and pos[2] >= 0.5
        and pos[2] <= float(nz - 1) - 0.5
    )


@wp.func
def _is_in_solid(pos: wp.vec3, solid: wp.array3d(dtype=wp.int32)) -> bool:
    i = int(wp.floor(pos[0]))
    j = int(wp.floor(pos[1]))
    k = int(wp.floor(pos[2]))
    nx = solid.shape[0]
    ny = solid.shape[1]
    nz = solid.shape[2]
    if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
        return True
    return solid[i, j, k] != 0


@wp.kernel
def seed_particles_uniform(
    particles: wp.array(dtype=wp.vec3),
    particle_life: wp.array(dtype=wp.float32),
    seed_count: int,
    nx: int,
    ny: int,
    nz: int,
    solid: wp.array3d(dtype=wp.int32),
    seed: int,
) -> None:
    """Seed *seed_count* new particles in random non-solid cells."""
    tid = wp.tid()
    if tid >= seed_count:
        return

    # Simple hash-based pseudo-random for seeding
    rng = wp.rand_init(seed + tid * 2654435761)

    # Try up to 50 candidate positions before giving up
    for _attempt in range(50):
        rx = wp.randf(rng)
        ry = wp.randf(rng)
        rz = wp.randf(rng)

        i = int(rx * float(nx))
        j = int(ry * float(ny))
        k = int(rz * float(nz))
        i = wp.clamp(i, 0, nx - 1)
        j = wp.clamp(j, 0, ny - 1)
        k = wp.clamp(k, 0, nz - 1)

        if solid[i, j, k] == 0:
            # Place at cell center
            particles[tid] = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
            particle_life[tid] = 0.0
            return

    # Fallback: place at origin (will be filtered)
    particles[tid] = wp.vec3(-100.0, -100.0, -100.0)
    particle_life[tid] = -1.0


@wp.kernel
def advect_particles_rk2(
    particles: wp.array(dtype=wp.vec3),
    particle_life: wp.array(dtype=wp.float32),
    particle_count: int,
    v: wp.array3d(dtype=wp.vec3),
    solid: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    dt: float,
    max_life: float,
):
    """RK2 advection of particles through the velocity field.

    Particles that exit the domain or enter solid cells are marked with
    ``particle_life = -1.0`` (flagged for reseeding).
    """
    tid = wp.tid()
    if tid >= particle_count:
        return

    p = particles[tid]
    life = particle_life[tid]

    if life < 0.0:
        return

    # RK2 (midpoint) advection
    v1 = _sample_velocity_trilinear(v, nx, ny, nz, p)
    p_mid = p + v1 * dt * 0.5
    v2 = _sample_velocity_trilinear(v, nx, ny, nz, p_mid)
    p_new = p + v2 * dt

    if not _is_inside_domain(p_new, nx, ny, nz) or _is_in_solid(p_new, solid):
        particle_life[tid] = -1.0
        return

    particles[tid] = p_new
    particle_life[tid] = life + dt
    if particle_life[tid] > max_life:
        particle_life[tid] = -1.0


@wp.kernel
def reseed_dead_particles(
    particles: wp.array(dtype=wp.vec3),
    particle_life: wp.array(dtype=wp.float32),
    particle_count: int,
    nx: int,
    ny: int,
    nz: int,
    solid: wp.array3d(dtype=wp.int32),
    seed: int,
):
    """Replace dead particles (life < 0) with fresh random positions."""
    tid = wp.tid()
    if tid >= particle_count:
        return

    if particle_life[tid] >= 0.0:
        return

    rng = wp.rand_init(seed + tid * 2654435761)

    for _attempt in range(50):
        rx = wp.randf(rng)
        ry = wp.randf(rng)
        rz = wp.randf(rng)

        i = int(rx * float(nx))
        j = int(ry * float(ny))
        k = int(rz * float(nz))
        i = wp.clamp(i, 0, nx - 1)
        j = wp.clamp(j, 0, ny - 1)
        k = wp.clamp(k, 0, nz - 1)

        if solid[i, j, k] == 0:
            particles[tid] = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
            particle_life[tid] = 0.0
            return

    # Keep dead if no valid position found
    particle_life[tid] = -1.0


@wp.kernel
def compute_particle_speeds(
    particles: wp.array(dtype=wp.vec3),
    particle_life: wp.array(dtype=wp.float32),
    particle_count: int,
    v: wp.array3d(dtype=wp.vec3),
    nx: int,
    ny: int,
    nz: int,
    out_speeds: wp.array(dtype=wp.float32),
    out_colors: wp.array(dtype=wp.vec3),
):
    """Compute per-particle speed and velocity-direction color."""
    tid = wp.tid()
    if tid >= particle_count:
        return

    if particle_life[tid] < 0.0:
        out_speeds[tid] = 0.0
        out_colors[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    vel = _sample_velocity_trilinear(v, nx, ny, nz, particles[tid])
    speed = wp.length(vel)
    out_speeds[tid] = speed

    # Map velocity direction to RGB: x→R, y→G, z→B
    if speed > 1.0e-8:
        uv = vel / speed
        out_colors[tid] = wp.vec3(
            wp.abs(uv[0]),
            wp.abs(uv[1]),
            wp.abs(uv[2]),
        )
    else:
        out_colors[tid] = wp.vec3(0.5, 0.5, 0.5)
