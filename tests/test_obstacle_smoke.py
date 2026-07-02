# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Obstacle smoke test (member B: bake_box + obstacle bounce-back).

Goal:
  - solid cells should keep v ~ 0 (update_macro sets v=0 for solid)
  - overall flow should still be established (lid-driven cavity)
"""

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


class TestObstacleSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()

    def test_lid_cavity_with_box_obstacle(self) -> None:
        grid_size: int = 32
        steps: int = 200
        u_lid: float = 0.1

        model: FluidGridLbmModel = FluidGridLbmModel(
            fluid_grid_res=(grid_size, grid_size, grid_size),
            nu=0.16667,
            use_guo_force=False,
            collide_impl="bgk",
        )
        solver: FluidGridLbmSolver = FluidGridLbmSolver(model)
        solver.configure_cavity_walls()
        solver.set_lid_velocity(wp.vec3(u_lid, 0.0, 0.0))

        state_a: FluidGridLbmState = FluidGridLbmState(model)
        state_b: FluidGridLbmState = FluidGridLbmState(model)
        solver.init_uniform(state_a, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))
        solver.init_uniform(state_b, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))

        # Bake a small obstacle near the cavity center (lattice index coordinates).
        cx = float(grid_size) * 0.55
        cy = float(grid_size) * 0.5
        cz = float(grid_size) * 0.5
        hx = float(grid_size) * 0.08
        hy = float(grid_size) * 0.08
        hz = float(grid_size) * 0.08
        solver.bake_box(state_a, wp.vec3(cx, cy, cz), wp.vec3(hx, hy, hz))

        state_in: FluidGridLbmState = state_a
        state_out: FluidGridLbmState = state_b
        for _ in range(steps):
            solver.step(state_in, state_out, dt=1.0)
            state_in, state_out = state_out, state_in

        v_np: np.ndarray = state_in.v.numpy()
        speed: np.ndarray = np.linalg.norm(v_np, axis=-1)
        solid_mask: np.ndarray = state_in.solid.numpy() != 0

        self.assertTrue(solid_mask.any(), "bake_box produced an empty solid region")
        self.assertFalse(np.isnan(v_np).any(), "velocity contains NaN")
        self.assertFalse(np.isinf(v_np).any(), "velocity contains Inf")

        max_solid_u: float = float(np.max(speed[solid_mask]))
        max_fluid_u: float = float(np.max(speed[~solid_mask]))

        self.assertLess(max_solid_u, 1.0e-5, f"solid cells should be near-stationary: max|u|={max_solid_u}")
        self.assertGreater(max_fluid_u, 1.0e-2, f"flow not established: max|u|={max_fluid_u}")

    def test_lid_cavity_with_box_obstacle_mrt(self) -> None:
        grid_size: int = 32
        steps: int = 200
        u_lid: float = 0.1

        model: FluidGridLbmModel = FluidGridLbmModel(
            fluid_grid_res=(grid_size, grid_size, grid_size),
            nu=0.16667,
            use_guo_force=False,
            collide_impl="mrt",
        )
        solver: FluidGridLbmSolver = FluidGridLbmSolver(model)
        solver.configure_cavity_walls()
        solver.set_lid_velocity(wp.vec3(u_lid, 0.0, 0.0))

        state_a: FluidGridLbmState = FluidGridLbmState(model)
        state_b: FluidGridLbmState = FluidGridLbmState(model)
        solver.init_uniform(state_a, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))
        solver.init_uniform(state_b, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))

        cx = float(grid_size) * 0.55
        cy = float(grid_size) * 0.5
        cz = float(grid_size) * 0.5
        hx = float(grid_size) * 0.08
        hy = float(grid_size) * 0.08
        hz = float(grid_size) * 0.08
        solver.bake_box(state_a, wp.vec3(cx, cy, cz), wp.vec3(hx, hy, hz))

        state_in: FluidGridLbmState = state_a
        state_out: FluidGridLbmState = state_b
        for _ in range(steps):
            solver.step(state_in, state_out, dt=1.0)
            state_in, state_out = state_out, state_in

        v_np: np.ndarray = state_in.v.numpy()
        speed: np.ndarray = np.linalg.norm(v_np, axis=-1)
        solid_mask: np.ndarray = state_in.solid.numpy() != 0

        self.assertTrue(solid_mask.any(), "bake_box produced an empty solid region")
        self.assertFalse(np.isnan(v_np).any(), "velocity contains NaN")
        self.assertFalse(np.isinf(v_np).any(), "velocity contains Inf")

        max_solid_u: float = float(np.max(speed[solid_mask]))
        max_fluid_u: float = float(np.max(speed[~solid_mask]))

        self.assertLess(max_solid_u, 1.0e-5, f"solid cells should be near-stationary: max|u|={max_solid_u}")
        self.assertGreater(max_fluid_u, 1.0e-2, f"flow not established: max|u|={max_fluid_u}")


if __name__ == "__main__":
    unittest.main()

