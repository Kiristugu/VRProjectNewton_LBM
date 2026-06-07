# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""LBM Warp kernels (DESIGN.md §7).

Week-1 deliverables (member B, §7.1–7.2):
  - ``init_equilibrium``  — uniform equilibrium initialization
  - ``collide_bgk``       — BGK collision operator
  - ``update_macro``      — density / velocity recovery from F
  - ``stream_pull_identity`` — week-1 placeholder (no neighbour pull)

§7.3 boundary kernels — owned by member C; week-1 ``apply_boundaries`` is a no-op.
"""

from __future__ import annotations

import warp as wp

from .lattice import Q, feq, lattice_e_f

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


@wp.kernel
def stream_pull_identity(
    F: wp.array4d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
) -> None:
    """Week-1 placeholder: identity stream (replaced by stream_pull in week 2)."""
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


@wp.kernel
def apply_boundaries(
    F: wp.array4d(dtype=float),
    f: wp.array4d(dtype=float),
    rho: wp.array3d(dtype=float),
    v: wp.array3d(dtype=wp.vec3),
    solid: wp.array3d(dtype=wp.int32),
) -> None:
    """Week-1 stub; member C implements Zou-He / bounce-back in week 2."""
    i, j, k = wp.tid()
    if solid[i, j, k] == -1:
        F[i, j, k, 0] = f[i, j, k, 0]
