# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Equilibrium distribution tests (DESIGN.md §9.1, member B week-1)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

_bootstrap_path: Path = Path(__file__).resolve().parent / "_bootstrap.py"
_spec = importlib.util.spec_from_file_location("lbm_test_bootstrap", _bootstrap_path)
assert _spec is not None and _spec.loader is not None
_bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bootstrap)
_bootstrap.bootstrap_from_test_file(__file__)

from wanphys._src.fluid.fluid_grid.lbm.lattice import (
    Q,
    feq_host,
    lattice_e_host,
    lattice_weight_host,
)


def _sum_feq(rho: float, u: tuple[float, float, float]) -> float:
    return sum(feq_host(q, rho, u) for q in range(Q))


def _sum_feq_momentum(rho: float, u: tuple[float, float, float]) -> tuple[float, float, float]:
    mx = my = mz = 0.0
    for q in range(Q):
        f = feq_host(q, rho, u)
        ex, ey, ez = lattice_e_host(q)
        mx += ex * f
        my += ey * f
        mz += ez * f
    return mx, my, mz


class TestEquilibrium(unittest.TestCase):
    """Verify D3Q19 feq satisfies mass and momentum moments."""

    CASES: list[tuple[float, tuple[float, float, float]]] = [
        (1.0, (0.0, 0.0, 0.0)),
        (1.0, (0.05, 0.0, 0.0)),
        (1.0, (0.0, 0.08, -0.03)),
        (1.2, (0.02, 0.04, 0.01)),
    ]

    def test_feq_sums_to_rho(self) -> None:
        for rho, u in self.CASES:
            with self.subTest(rho=rho, u=u):
                total: float = _sum_feq(rho, u)
                self.assertAlmostEqual(total, rho, places=10)

    def test_feq_momentum_equals_rho_u(self) -> None:
        for rho, u in self.CASES:
            with self.subTest(rho=rho, u=u):
                mx, my, mz = _sum_feq_momentum(rho, u)
                self.assertAlmostEqual(mx, rho * u[0], places=10)
                self.assertAlmostEqual(my, rho * u[1], places=10)
                self.assertAlmostEqual(mz, rho * u[2], places=10)

    def test_bgk_preserves_equilibrium_algebraically(self) -> None:
        """At equilibrium f=feq, BGK gives F=feq for any omega."""
        rho, u = 1.0, (0.03, -0.02, 0.01)
        omega = 1.0
        for q in range(Q):
            feq_val = feq_host(q, rho, u)
            f_out = feq_val - omega * (feq_val - feq_val)
            self.assertAlmostEqual(f_out, feq_val, places=12)

    def test_lattice_speeds_have_unit_length_or_zero(self) -> None:
        for q in range(Q):
            ex, ey, ez = lattice_e_host(q)
            length_sq = ex * ex + ey * ey + ez * ez
            self.assertIn(length_sq, (0, 1, 2))


class TestEquilibriumGpu(unittest.TestCase):
    """GPU kernel checks (requires warp)."""

    @classmethod
    def setUpClass(cls) -> None:
        import warp as wp

        wp.init()
        cls.wp = wp

    def test_collide_bgk_preserves_uniform_equilibrium(self) -> None:
        wp = self.wp
        from wanphys._src.fluid.fluid_grid.lbm import kernels
        from wanphys._src.fluid.fluid_grid.lbm.lattice import feq_host

        nx = ny = nz = 4
        f = wp.zeros((nx, ny, nz, Q), dtype=float, device="cpu")
        F = wp.zeros((nx, ny, nz, Q), dtype=float, device="cpu")
        rho = wp.full((nx, ny, nz), 1.0, dtype=float, device="cpu")
        v = wp.zeros((nx, ny, nz), dtype=wp.vec3, device="cpu")
        solid = wp.zeros((nx, ny, nz), dtype=wp.int32, device="cpu")

        rho0, u0 = 1.0, (0.0, 0.0, 0.0)
        wp.launch(
            kernels.init_equilibrium,
            dim=(nx, ny, nz),
            inputs=[f, F, rho, v, solid, rho0, wp.vec3(*u0)],
            device="cpu",
        )

        wp.launch(
            kernels.collide_bgk,
            dim=(nx, ny, nz),
            inputs=[f, F, rho, v, solid, 1.0],
            device="cpu",
        )

        f_np = f.numpy()
        F_np = F.numpy()
        for q in range(Q):
            expected = feq_host(q, rho0, u0)
            self.assertTrue(np.allclose(F_np[:, :, :, q], expected, atol=1e-10))
            self.assertTrue(np.allclose(f_np[:, :, :, q], expected, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
