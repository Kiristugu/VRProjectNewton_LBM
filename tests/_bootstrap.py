# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap imports for LBM tests without loading wanphys top-level __init__.py."""

from __future__ import annotations

import importlib.util
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
    lbm_dir: Path = src_dir / "fluid" / "fluid_grid" / "lbm"
    _stub_package("wanphys._src.fluid.fluid_grid.lbm", lbm_dir)
    _load_lbm_package(lbm_dir)
    return root


def _load_lbm_package(lbm_dir: Path) -> None:
    """Execute lbm/__init__.py so ``from ...lbm import FluidGridLbm*`` works."""
    name = "wanphys._src.fluid.fluid_grid.lbm"
    if name in sys.modules and hasattr(sys.modules[name], "FluidGridLbmDomain"):
        return
    init_path: Path = lbm_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        name,
        init_path,
        submodule_search_locations=[str(lbm_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load LBM package from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def bootstrap_wanphys() -> Path:
    """Register stub packages for LBM-only imports."""
    return bootstrap_from_test_file(str(Path(__file__).resolve()))
