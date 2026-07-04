# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: lid-driven cavity with central box obstacle (post_midterm_plan S1)."""

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

from wanphys._src.fluid.fluid_grid.lbm import FluidGridLbmDomain, FluidGridLbmModel


class TestObstacleSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()

    def test_lid_cavity_with_box_obstacle_500_steps(self) -> None:
        grid_size: int = 32
        steps: int = 500
        u_lid: float = 0.1

        model: FluidGridLbmModel = FluidGridLbmModel(
            fluid_grid_res=(grid_size, grid_size, grid_size),
            nu=0.16667,
            use_guo_force=False,
        )
        domain: FluidGridLbmDomain = FluidGridLbmDomain(model)
        domain.create_state()

        mid: float = 0.5 * float(grid_size)
        center: wp.vec3 = wp.vec3(mid, mid, mid)
        half_extents: wp.vec3 = wp.vec3(2.5, 4.0, 2.5)

        domain.solver.configure_cavity_walls()
        domain.solver.set_lid_velocity(wp.vec3(u_lid, 0.0, 0.0))
        domain.bake_box(center, half_extents)
        domain.solver.init_uniform(domain.state, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))

        solid_np: np.ndarray = domain.state.solid.numpy()
        self.assertGreater(int(np.sum(solid_np > 0)), 0, "obstacle not baked")

        for _ in range(steps):
            domain.step(dt=1.0)

        v_np: np.ndarray = domain.state.v.numpy()
        speed: np.ndarray = np.linalg.norm(v_np, axis=-1)

        self.assertFalse(np.isnan(v_np).any())
        self.assertFalse(np.isinf(v_np).any())

        solid_speed: np.ndarray = speed[solid_np > 0]
        if solid_speed.size > 0:
            self.assertLess(float(np.max(solid_speed)), 1.0e-4)

        fluid_mask: np.ndarray = solid_np == 0
        max_u: float = float(np.max(speed[fluid_mask]))
        self.assertGreater(max_u, 0.01)

        i_wake: int = int(mid + half_extents[0] + 2.0)
        wake_speed: np.ndarray = speed[i_wake:, :, :]
        wake_solid: np.ndarray = solid_np[i_wake:, :, :]
        wake_fluid: np.ndarray = wake_solid == 0
        wake_max: float = float(np.max(wake_speed[wake_fluid])) if np.any(wake_fluid) else 0.0
        self.assertGreater(wake_max, 0.005, f"no wake downstream, wake_max={wake_max}")


if __name__ == "__main__":
    unittest.main()
