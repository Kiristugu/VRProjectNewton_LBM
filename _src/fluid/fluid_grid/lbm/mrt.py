# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q19 MRT transformation matrices for this lattice velocity ordering.

The moment matrix ``M`` is built from monomials in (ex, ey, ez) up to total
degree 4, orthonormalised under the lattice inner product
``<a, b> = sum_i w_i a_i b_i``.  The first four rows correspond to conserved
mass and momentum; remaining rows are relaxed with configurable rates.
"""

from __future__ import annotations

import numpy as np

from .lattice import Q, lattice_e_host, lattice_weight_host

MRT_Q: int = Q


def _build_moment_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(M, inv_M)`` as float64 arrays of shape ``(19, 19)``."""
    e = np.array([lattice_e_host(i) for i in range(Q)], dtype=np.float64)
    w = np.array([lattice_weight_host(i) for i in range(Q)], dtype=np.float64)

    def inner(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(w * a * b))

    monomials: list[tuple[int, int, int]] = []
    for degree in range(5):
        for a in range(degree + 1):
            for b in range(degree - a + 1):
                c = degree - a - b
                monomials.append((a, b, c))

    rows: list[np.ndarray] = []
    for a, b, c in monomials:
        vec = (e[:, 0] ** a) * (e[:, 1] ** b) * (e[:, 2] ** c)
        for row in rows:
            vec = vec - inner(vec, row) / inner(row, row) * row
        norm = np.sqrt(inner(vec, vec))
        if norm > 1.0e-12:
            rows.append(vec / norm)
        if len(rows) == Q:
            break

    if len(rows) != Q:
        raise RuntimeError(f"expected {Q} independent moment rows, got {len(rows)}")

    m_matrix = np.asarray(rows, dtype=np.float64)
    inv_m_matrix = np.linalg.inv(m_matrix)
    return m_matrix, inv_m_matrix


M_MATRIX, INV_M_MATRIX = _build_moment_matrix()

# First four moment indices are conserved (mass + momentum).
MRT_CONSERVED_COUNT: int = 4
# Hydrodynamic stress moments use the shear relaxation rate (same as BGK omega).
MRT_SHEAR_END: int = 16


def build_relaxation_rates(omega: float, ghost_s: float = 1.0) -> np.ndarray:
    """Diagonal MRT relaxation rates ``S`` for a given BGK ``omega``."""
    rates = np.full(Q, ghost_s, dtype=np.float64)
    rates[MRT_CONSERVED_COUNT:MRT_SHEAR_END] = omega
    rates[:MRT_CONSERVED_COUNT] = 0.0
    return rates