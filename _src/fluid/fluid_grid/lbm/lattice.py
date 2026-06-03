# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q19 lattice constants and equilibrium distribution (DESIGN.md §6)."""

from __future__ import annotations

import warp as wp

Q: int = 19

# Opposite direction index for bounce-back (matches taichi LR).
LR: tuple[int, ...] = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17)


@wp.func
def lattice_weight(i: int) -> float:
    if i == 0:
        return 1.0 / 3.0
    if i <= 6:
        return 1.0 / 18.0
    return 1.0 / 36.0


@wp.func
def lattice_e_f(i: int) -> wp.vec3:
    if i == 0:
        return wp.vec3(0.0, 0.0, 0.0)
    if i == 1:
        return wp.vec3(1.0, 0.0, 0.0)
    if i == 2:
        return wp.vec3(-1.0, 0.0, 0.0)
    if i == 3:
        return wp.vec3(0.0, 1.0, 0.0)
    if i == 4:
        return wp.vec3(0.0, -1.0, 0.0)
    if i == 5:
        return wp.vec3(0.0, 0.0, 1.0)
    if i == 6:
        return wp.vec3(0.0, 0.0, -1.0)
    if i == 7:
        return wp.vec3(1.0, 1.0, 0.0)
    if i == 8:
        return wp.vec3(-1.0, -1.0, 0.0)
    if i == 9:
        return wp.vec3(1.0, -1.0, 0.0)
    if i == 10:
        return wp.vec3(-1.0, 1.0, 0.0)
    if i == 11:
        return wp.vec3(1.0, 0.0, 1.0)
    if i == 12:
        return wp.vec3(-1.0, 0.0, -1.0)
    if i == 13:
        return wp.vec3(1.0, 0.0, -1.0)
    if i == 14:
        return wp.vec3(-1.0, 0.0, 1.0)
    if i == 15:
        return wp.vec3(0.0, 1.0, 1.0)
    if i == 16:
        return wp.vec3(0.0, -1.0, -1.0)
    if i == 17:
        return wp.vec3(0.0, 1.0, -1.0)
    return wp.vec3(0.0, -1.0, 1.0)


@wp.func
def lattice_e_i(i: int) -> wp.vec3i:
    ef = lattice_e_f(i)
    return wp.vec3i(int(ef[0]), int(ef[1]), int(ef[2]))


@wp.func
def feq(i: int, rho: float, u: wp.vec3) -> float:
    """Equilibrium distribution f_i^eq for D3Q19."""
    e = lattice_e_f(i)
    w = lattice_weight(i)
    eu = wp.dot(e, u)
    uv = wp.dot(u, u)
    return w * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)


def lattice_weight_host(i: int) -> float:
    """Host-side weight lookup for unit tests."""
    if i == 0:
        return 1.0 / 3.0
    if i <= 6:
        return 1.0 / 18.0
    return 1.0 / 36.0


def lattice_e_host(i: int) -> tuple[int, int, int]:
    """Host-side discrete velocity for unit tests."""
    vectors: tuple[tuple[int, int, int], ...] = (
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
        (1, 1, 0),
        (-1, -1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (1, 0, 1),
        (-1, 0, -1),
        (1, 0, -1),
        (-1, 0, 1),
        (0, 1, 1),
        (0, -1, -1),
        (0, 1, -1),
        (0, -1, 1),
    )
    return vectors[i]
