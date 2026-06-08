# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Domain integration smoke test (phase3_domain_example.md)."""

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

from wanphys._src.fluid.fluid_grid.lbm import (  # noqa: E402
    FluidGridLbmDomain,
    FluidGridLbmModel,
    FluidGridLbmSolver,
    FluidGridLbmState,
)


class TestLbmDomain(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()

    def test_domain_import_all_symbols(self) -> None:
        model: FluidGridLbmModel = FluidGridLbmModel(fluid_grid_res=(8, 8, 8))
        domain: FluidGridLbmDomain = FluidGridLbmDomain(model)
        self.assertEqual(domain.name, "fluid_grid_lbm")
        self.assertIsInstance(domain.solver, FluidGridLbmSolver)

    def test_domain_step_10(self) -> None:
        grid_size: int = 16
        model: FluidGridLbmModel = FluidGridLbmModel(
            fluid_grid_res=(grid_size, grid_size, grid_size),
            nu=0.16667,
            use_guo_force=False,
        )
        domain: FluidGridLbmDomain = FluidGridLbmDomain(model)
        state: FluidGridLbmState = domain.create_state()
        domain.solver.configure_cavity_walls()
        domain.solver.set_lid_velocity(wp.vec3(0.1, 0.0, 0.0))
        domain.solver.init_uniform(state, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))

        for _ in range(10):
            domain.step(dt=1.0)

        v_np: np.ndarray = domain.state.v.numpy()
        self.assertFalse(np.isnan(v_np).any())
        self.assertFalse(np.isinf(v_np).any())


if __name__ == "__main__":
    unittest.main()
