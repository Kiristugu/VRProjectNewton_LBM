# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""M2 smoke test: lid-driven cavity flow after 500 steps (phase2_boundaries.md §6.2)."""

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


class TestCavitySmoke(unittest.TestCase):
    """Qualitative cavity flow with velocity BC walls."""

    @classmethod
    def setUpClass(cls) -> None:
        wp.init()

    def test_lid_driven_cavity_500_steps(self) -> None:
        grid_size: int = 32
        steps: int = 500
        u_lid: float = 0.1

        model: FluidGridLbmModel = FluidGridLbmModel(
            fluid_grid_res=(grid_size, grid_size, grid_size),
            nu=0.16667,
            use_guo_force=False,
        )
        solver: FluidGridLbmSolver = FluidGridLbmSolver(model)
        solver.configure_cavity_walls()
        solver.set_lid_velocity(wp.vec3(u_lid, 0.0, 0.0))

        state_a: FluidGridLbmState = FluidGridLbmState(model)
        state_b: FluidGridLbmState = FluidGridLbmState(model)
        solver.init_uniform(state_a, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))
        solver.init_uniform(state_b, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))

        state_in: FluidGridLbmState = state_a
        state_out: FluidGridLbmState = state_b
        for _ in range(steps):
            solver.step(state_in, state_out, dt=1.0)
            state_in, state_out = state_out, state_in

        v_np: np.ndarray = state_in.v.numpy()
        speed: np.ndarray = np.linalg.norm(v_np, axis=-1)
        max_u: float = float(np.max(speed))

        # Near-lid layer (j = ny - 2) should carry lid-driven u_x.
        near_lid_ux: np.ndarray = v_np[:, grid_size - 2, :, 0]
        max_lid_ux: float = float(np.max(near_lid_ux))

        # Mid-plane should show recirculation (some negative u_x under the lid).
        mid_j: int = grid_size // 2
        mid_ux: np.ndarray = v_np[:, mid_j, :, 0]
        min_mid_ux: float = float(np.min(mid_ux))

        self.assertFalse(np.isnan(v_np).any(), "velocity contains NaN")
        self.assertFalse(np.isinf(v_np).any(), "velocity contains Inf")
        self.assertGreater(max_u, 0.01, f"flow not established, max|u|={max_u}")
        self.assertGreater(max_lid_ux, 0.05, f"lid BC weak, near-lid max u_x={max_lid_ux}")
        self.assertLess(min_mid_ux, -0.005, f"no recirculation, mid-plane min u_x={min_mid_ux}")


if __name__ == "__main__":
    unittest.main()
