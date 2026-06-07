# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Domain API: double-buffered step matches manual state swap (M1 via Domain)."""

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

from wanphys._src.fluid.fluid_grid.lbm.domain import FluidGridLbmDomain
from wanphys._src.fluid.fluid_grid.lbm.model import FluidGridLbmModel


class TestLbmDomain(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wp.init()

    def test_domain_name(self) -> None:
        model: FluidGridLbmModel = FluidGridLbmModel(fluid_grid_res=(8, 8, 8))
        domain: FluidGridLbmDomain = FluidGridLbmDomain(model)
        self.assertEqual(domain.name, "fluid_grid_lbm")

    def test_domain_rest_fluid_m1(self) -> None:
        grid_size: int = 16
        steps: int = 100
        model: FluidGridLbmModel = FluidGridLbmModel(fluid_grid_res=(grid_size, grid_size, grid_size), nu=0.16667)
        domain: FluidGridLbmDomain = FluidGridLbmDomain(model)
        domain.create_state()
        domain.init_uniform(rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))

        for _ in range(steps):
            domain.step(dt=1.0)

        rho_np: np.ndarray = domain.state.rho.numpy()
        v_np: np.ndarray = domain.state.v.numpy()
        n_cells: int = grid_size**3
        speed: np.ndarray = np.linalg.norm(v_np, axis=-1)

        self.assertLess(float(np.max(speed)), 1.0e-10)
        self.assertLess(float(np.max(np.abs(rho_np - 1.0))), 1.0e-6)
        self.assertLess(float(abs(np.sum(rho_np) - n_cells) / n_cells), 1.0e-6)


if __name__ == "__main__":
    unittest.main()
