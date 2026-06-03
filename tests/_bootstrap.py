# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap imports for LBM tests without loading wanphys top-level __init__.py."""

from __future__ import annotations

import sys
import types
from pathlib import Path


def _stub_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    pkg: types.ModuleType = types.ModuleType(name)
    pkg.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def bootstrap_from_test_file(test_file: str) -> Path:
    """Call before any ``wanphys`` import when running a test file directly."""
    root: Path = Path(test_file).resolve().parents[2]
    wanphys_dir: Path = root / "wanphys"
    src_dir: Path = wanphys_dir / "_src"
    root_str: str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    _stub_package("wanphys", wanphys_dir)
    _stub_package("wanphys._src", src_dir)
    _stub_package("wanphys._src.core", src_dir / "core")
    # Skip fluid/__init__.py and fluid_grid/__init__.py (pull in liquid/coupling/geometry).
    _stub_package("wanphys._src.fluid", src_dir / "fluid")
    _stub_package("wanphys._src.fluid.fluid_grid", src_dir / "fluid" / "fluid_grid")
    _stub_package("wanphys._src.fluid.fluid_grid.lbm", src_dir / "fluid" / "fluid_grid" / "lbm")
    return root


def bootstrap_wanphys() -> Path:
    """Register stub packages for LBM-only imports."""
    return bootstrap_from_test_file(str(Path(__file__).resolve()))
