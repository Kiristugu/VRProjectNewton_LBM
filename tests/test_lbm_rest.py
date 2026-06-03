# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""M1 acceptance: 16^3 rest fluid after 100 BGK steps (lbm_team_development_plan §4)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import warp as wp

_bootstrap_path: Path = Path(__file__).resolve().parent / "_bootstrap.py"
_spec = importlib.util.spec_from_file_location("lbm_test_bootstrap", _bootstrap_path)
assert _spec is not None and _spec.loader is not None
_bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bootstrap)
_bootstrap.bootstrap_from_test_file(__file__)

from wanphys._src.fluid.fluid_grid.lbm.model import FluidGridLbmModel
from wanphys._src.fluid.fluid_grid.lbm.solver import FluidGridLbmSolver
from wanphys._src.fluid.fluid_grid.lbm.state import FluidGridLbmState


class TestLbmRestFluid(unittest.TestCase):
    """Week-1 minimal stationary flow validation."""

    @classmethod
    def setUpClass(cls) -> None:
        wp.init()

    def test_rest_fluid_m1(self) -> None:
        grid_size: int = 16
        steps: int = 100
        model: FluidGridLbmModel = FluidGridLbmModel(fluid_grid_res=(grid_size, grid_size, grid_size), nu=0.16667)
        solver: FluidGridLbmSolver = FluidGridLbmSolver(model)
        state_a: FluidGridLbmState = FluidGridLbmState(model)
        state_b: FluidGridLbmState = FluidGridLbmState(model)

        solver.init_uniform(state_a, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))
        solver.init_uniform(state_b, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))

        state_in: FluidGridLbmState = state_a
        state_out: FluidGridLbmState = state_b
        for _ in range(steps):
            solver.step(state_in, state_out, dt=1.0)
            state_in, state_out = state_out, state_in

        rho_np: np.ndarray = state_in.rho.numpy()
        v_np: np.ndarray = state_in.v.numpy()
        n_cells: int = grid_size**3

        speed: np.ndarray = np.linalg.norm(v_np, axis=-1)
        max_u: float = float(np.max(speed))
        max_rho_err: float = float(np.max(np.abs(rho_np - 1.0)))
        mass_drift: float = float(abs(np.sum(rho_np) - n_cells) / n_cells)

        self.assertLess(max_u, 1.0e-10, f"max|u|={max_u}")
        self.assertLess(max_rho_err, 1.0e-6, f"max|rho-1|={max_rho_err}")
        self.assertLess(mass_drift, 1.0e-6, f"mass drift={mass_drift}")
        self.assertFalse(np.isnan(rho_np).any())
        self.assertFalse(np.isinf(rho_np).any())
        self.assertFalse(np.isnan(v_np).any())
        self.assertFalse(np.isinf(v_np).any())


if __name__ == "__main__":
    unittest.main()
