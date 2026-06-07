# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for LBM module imports and model properties."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import warp as wp

_bootstrap_path: Path = Path(__file__).resolve().parent / "_bootstrap.py"
_spec = importlib.util.spec_from_file_location("lbm_test_bootstrap", _bootstrap_path)
assert _spec is not None and _spec.loader is not None
_bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bootstrap)
_bootstrap.bootstrap_from_test_file(__file__)

from wanphys._src.fluid.fluid_grid.lbm.model import FluidGridLbmModel
from wanphys._src.fluid.fluid_grid.lbm.domain import FluidGridLbmDomain
from wanphys._src.fluid.fluid_grid.lbm.solver import FluidGridLbmSolver
from wanphys._src.fluid.fluid_grid.lbm.state import FluidGridLbmState
from wanphys._src.fluid.fluid_grid.lbm.lattice import Q, LR, lattice_e_host, lattice_weight_host


class TestLbmImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()

    def test_model_tau_omega(self) -> None:
        model: FluidGridLbmModel = FluidGridLbmModel(fluid_grid_res=(16, 16, 16), nu=0.16667)
        self.assertAlmostEqual(model.tau, 3.0 * 0.16667 + 0.5, places=4)
        self.assertAlmostEqual(model.tau, 1.0, places=3)
        self.assertAlmostEqual(model.omega, 1.0 / model.tau)

    def test_state_allocation(self) -> None:
        model: FluidGridLbmModel = FluidGridLbmModel(fluid_grid_res=(16, 16, 16))
        state: FluidGridLbmState = FluidGridLbmState(model)
        self.assertEqual(state.f.shape, (16, 16, 16, Q))
        self.assertEqual(state.F.shape, (16, 16, 16, Q))
        self.assertEqual(state.rho.shape, (16, 16, 16))
        self.assertEqual(state.v.shape, (16, 16, 16))

    def test_lattice_weights_sum(self) -> None:
        weight_sum: float = sum(lattice_weight_host(i) for i in range(Q))
        self.assertAlmostEqual(weight_sum, 1.0)

    def test_lattice_lr_pairs(self) -> None:
        for i in range(Q):
            j: int = LR[i]
            ei = lattice_e_host(i)
            ej = lattice_e_host(j)
            self.assertEqual(ei, (-ej[0], -ej[1], -ej[2]))

    def test_lbm_package_exports_domain(self) -> None:
        init_path: Path = (
            Path(__file__).resolve().parents[1] / "_src" / "fluid" / "fluid_grid" / "lbm" / "__init__.py"
        )
        init_text: str = init_path.read_text(encoding="utf-8")
        for symbol in ("FluidGridLbmDomain", "FluidGridLbmModel", "FluidGridLbmSolver", "FluidGridLbmState"):
            self.assertIn(symbol, init_text)

        fluid_grid_init: Path = Path(__file__).resolve().parents[1] / "_src" / "fluid" / "fluid_grid" / "__init__.py"
        fluid_grid_text: str = fluid_grid_init.read_text(encoding="utf-8")
        self.assertIn("FluidGridLbmDomain", fluid_grid_text)

    def test_solver_construct(self) -> None:
        model: FluidGridLbmModel = FluidGridLbmModel(fluid_grid_res=(8, 8, 8))
        solver: FluidGridLbmSolver = FluidGridLbmSolver(model)
        self.assertEqual(solver.nx, 8)


if __name__ == "__main__":
    unittest.main()
