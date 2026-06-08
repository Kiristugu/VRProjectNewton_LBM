# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""VTK export smoke test."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

_bootstrap_path: Path = Path(__file__).resolve().parent / "_bootstrap.py"
_spec = importlib.util.spec_from_file_location("lbm_test_bootstrap", _bootstrap_path)
assert _spec is not None and _spec.loader is not None
_bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bootstrap)
_bootstrap.bootstrap_from_test_file(__file__)

from wanphys._src.fluid.fluid_grid.lbm.vtk_export import export_structured_vtk  # noqa: E402


class TestLbmVtkExport(unittest.TestCase):
    def test_export_writes_velocity_fields(self) -> None:
        nx = ny = nz = 4
        rho: np.ndarray = np.ones((nx, ny, nz), dtype=np.float64)
        velocity: np.ndarray = np.zeros((nx, ny, nz, 3), dtype=np.float64)
        velocity[:, -1, :, 0] = 0.1

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cavity.vtk"
            export_structured_vtk(out, rho, velocity)
            text: str = out.read_text(encoding="ascii")
            self.assertIn("STRUCTURED_POINTS", text)
            self.assertIn("SCALARS rho", text)
            self.assertIn("VECTORS velocity", text)


if __name__ == "__main__":
    unittest.main()
