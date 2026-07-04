# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Channel inflow past infinite-height cylinder / square pillar — top-view validation figures.

Scenario: uniform inlet from -x, no-slip top/bottom (y walls), periodic z.
Obstacles span the full channel height (infinite-height pillars in y).

Headless + PNG + VTK:
    python wanphys/examples/fluid_grid_lbm_channel_obstacle.py --viewer null --num-frames 600 --test
    python wanphys/examples/fluid_grid_lbm_channel_obstacle.py --obstacle-mode box --viewer null --num-frames 600 --test
    python wanphys/examples/fluid_grid_lbm_channel_obstacle.py --obstacle-mode both --viewer null --num-frames 600 --test

Outputs (``--test``, single mode):
    output/channel_obstacle/cylinder/fig1_top_view_streamlines.png
    output/channel_obstacle/cylinder/fig2_slice_ux_uz.png
    output/channel_obstacle/cylinder/channel_obstacle_cylinder.vtk

``--obstacle-mode both`` writes the same layout under ``cylinder/`` and ``box/``.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import warp as wp


def _bootstrap_lbm_imports() -> None:
    root: Path = Path(__file__).resolve().parents[2]
    root_str: str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    wanphys_dir: Path = root / "wanphys"
    src_dir: Path = wanphys_dir / "_src"

    def _stub(name: str, path: Path) -> None:
        if name in sys.modules:
            return
        pkg: types.ModuleType = types.ModuleType(name)
        pkg.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = pkg

    _stub("wanphys", wanphys_dir)
    _stub("wanphys._src", src_dir)
    _stub("wanphys._src.core", src_dir / "core")
    _stub("wanphys._src.fluid", src_dir / "fluid")
    _stub("wanphys._src.fluid.fluid_grid", src_dir / "fluid" / "fluid_grid")
    lbm_dir: Path = src_dir / "fluid" / "fluid_grid" / "lbm"
    _stub("wanphys._src.fluid.fluid_grid.lbm", lbm_dir)

    name = "wanphys._src.fluid.fluid_grid.lbm"
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


_bootstrap_lbm_imports()

from wanphys._src.fluid.fluid_grid.lbm import FluidGridLbmDomain, FluidGridLbmModel  # noqa: E402
from wanphys._src.fluid.fluid_grid.lbm.channel_obstacle_plot import (  # noqa: E402
    export_channel_validation_figures,
)
from wanphys._src.fluid.fluid_grid.lbm.vtk_export import export_structured_vtk  # noqa: E402

OBSTACLE_MODES: tuple[str, ...] = ("cylinder", "box")


def _vtk_path_for_mode(output_dir: str | Path, mode: str) -> str:
    return str(Path(output_dir) / f"channel_obstacle_{mode}.vtk")


def _output_dir_for_mode(base_dir: str | Path, mode: str) -> str:
    return str(Path(base_dir) / mode)


def _vtk_title(mode: str) -> str:
    if mode == "cylinder":
        return "WanPhys LBM channel flow past infinite-height cylinder"
    return "WanPhys LBM channel flow past infinite-height square pillar"


def _init_viewer(parser: argparse.ArgumentParser, use_gl_viewer: bool):
    if use_gl_viewer:
        from wanphys._src.fluid.fluid_viewer import init as fluid_init

        return fluid_init(parser)
    import newton.examples

    return newton.examples.init(parser)


def _create_viewer(args: argparse.Namespace):
    """Create a fresh viewer (required when running multiple modes in one process)."""
    import newton.viewer

    if args.viewer == "gl":
        return newton.viewer.ViewerGL(headless=getattr(args, "headless", False))
    if args.viewer == "null":
        return newton.viewer.ViewerNull(num_frames=args.num_frames)
    if args.viewer == "usd":
        return newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
    if args.viewer == "rerun":
        return newton.viewer.ViewerRerun(address=args.rerun_address)
    if args.viewer == "viser":
        return newton.viewer.ViewerViser()
    raise ValueError(f"Invalid viewer: {args.viewer}")


class Example:
    """Channel flow past infinite-height cylinder or square pillar; top-view validation plots."""

    def __init__(self, viewer, args: argparse.Namespace) -> None:
        self.viewer = viewer
        self.args = args
        self.sim_time: float = 0.0
        self.frame_count: int = 0
        self.lbm_substeps: int = 5
        self.total_lbm_steps: int = 0

        nx: int = args.nx
        ny: int = args.ny
        nz: int = args.nz
        u_in: float = args.u_in

        model: FluidGridLbmModel = FluidGridLbmModel(
            fluid_grid_res=(nx, ny, nz),
            fluid_grid_cell_size=args.cell_size,
            nu=args.nu,
            use_guo_force=False,
        )
        self.domain: FluidGridLbmDomain = FluidGridLbmDomain(model)
        self.domain.create_state()

        self.domain.solver.configure_channel_flow(u_in)
        self._setup_obstacle(nx, ny, nz)

        u_vec: wp.vec3 = wp.vec3(u_in, 0.0, 0.0)
        self.domain.solver.init_uniform_flow(self.domain.state, rho=1.0, u=u_vec)

        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._u_in = u_in
        self._cell_size = args.cell_size
        self._output_dir = Path(args.output_dir)
        self._slice_j: int = ny // 2
        self._slice_k: int = nz // 2

        solid_count: int = int(np.sum(self.domain.state.solid.numpy() > 0))
        reynolds: float = args.u_in * args.obstacle_diameter / args.nu
        print(
            f"Channel obstacle: grid={nx}x{ny}x{nz}, nu={args.nu}, U_in={u_in}, "
            f"mode={args.obstacle_mode}, Re~{reynolds:.1f}, solid_cells={solid_count}"
        )

    def _setup_obstacle(self, nx: int, ny: int, nz: int) -> None:
        args = self.args
        cx: float = args.obstacle_x if args.obstacle_x >= 0 else 0.35 * float(nx)
        cz: float = args.obstacle_z if args.obstacle_z >= 0 else 0.5 * float(nz - 1)
        half_size: float = 0.5 * args.obstacle_diameter
        cy: float = 0.5 * float(ny - 1)
        half_y: float = 0.5 * float(ny - 1)

        if args.obstacle_mode == "cylinder":
            radius: float = half_size
            self._obstacle_kind = "cylinder"
            self._cx, self._cz = cx, cz
            self._radius = radius
            self.domain.bake_cylinder_y(cx, cz, radius)
            print(f"Infinite-height cylinder: center=({cx:.1f}, z={cz:.1f}), radius={radius:.1f}")
            return

        self._obstacle_kind = "box"
        self._cx, self._cz = cx, cz
        self._half_x = half_size
        self._half_z = half_size
        self.domain.bake_box(
            wp.vec3(cx, cy, cz),
            wp.vec3(half_size, half_y, half_size),
        )
        print(
            f"Infinite-height square pillar: center=({cx:.1f}, y={cy:.1f}, z={cz:.1f}), "
            f"half=({half_size:.1f}, {half_y:.1f}, {half_size:.1f})"
        )

    def step(self) -> None:
        for _ in range(self.lbm_substeps):
            self.domain.step(dt=1.0)
            self.total_lbm_steps += 1
        self.sim_time += float(self.lbm_substeps)
        self.frame_count += 1

        if self.frame_count % 50 == 0 or not self.viewer.is_running():
            self._print_diagnostics()

    def _print_diagnostics(self) -> None:
        v_np: np.ndarray = self.domain.state.v.numpy()
        solid_np: np.ndarray = self.domain.state.solid.numpy()
        speed: np.ndarray = np.linalg.norm(v_np, axis=-1)
        fluid: np.ndarray = solid_np == 0
        max_u: float = float(np.max(speed[fluid])) if np.any(fluid) else 0.0
        j_line: int = self._slice_j
        k_line: int = self._slice_k
        u_line: np.ndarray = v_np[:, j_line, k_line, 0]
        solid_line: np.ndarray = solid_np[:, j_line, k_line] > 0
        u_fluid: np.ndarray = u_line[~solid_line]
        min_u: float = float(np.min(u_fluid)) if u_fluid.size else 0.0
        print(
            f"frame={self.frame_count} steps={self.total_lbm_steps} "
            f"max|u|={max_u:.5f} min(ux_slice)={min_u:.5f}"
        )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.end_frame()

    def test_final(self) -> None:
        v_np: np.ndarray = self.domain.state.v.numpy()
        rho_np: np.ndarray = self.domain.state.rho.numpy()
        solid_np: np.ndarray = self.domain.state.solid.numpy()

        if np.isnan(v_np).any() or np.isinf(v_np).any():
            raise ValueError("Velocity contains NaN/Inf")

        fluid: np.ndarray = solid_np == 0
        speed: np.ndarray = np.linalg.norm(v_np, axis=-1)
        max_u: float = float(np.max(speed[fluid]))
        if max_u < 0.02:
            raise ValueError(f"Flow too weak: max|u|={max_u}")

        vtk_path: str = self.args.export_vtk or _vtk_path_for_mode(self._output_dir, self.args.obstacle_mode)
        export_structured_vtk(
            vtk_path,
            rho_np,
            v_np,
            solid=solid_np,
            spacing=(self._cell_size, self._cell_size, self._cell_size),
            title=_vtk_title(self.args.obstacle_mode),
        )
        print(f"Exported VTK: {vtk_path}")

        export_channel_validation_figures(
            v_np,
            solid_np,
            output_dir=self._output_dir,
            u_in=self._u_in,
            cell_size=self._cell_size,
            slice_j=self._slice_j,
            slice_k=self._slice_k,
        )
        print(f"Validation figures written to: {self._output_dir.resolve()}")


def _build_parser() -> argparse.ArgumentParser:
    import newton.examples

    parser: argparse.ArgumentParser = newton.examples.create_parser()
    parser.add_argument("--nx", type=int, default=240, help="Grid size in x (flow direction).")
    parser.add_argument("--ny", type=int, default=80, help="Grid size in y (channel height, obstacle spans full y).")
    parser.add_argument("--nz", type=int, default=80, help="Grid size in z (channel width, top-view plane).")
    parser.add_argument("--nu", type=float, default=0.012, help="Kinematic viscosity (lattice units).")
    parser.add_argument("--u-in", type=float, default=0.06, help="Inlet velocity (lattice units).")
    parser.add_argument("--obstacle-diameter", type=float, default=14.0, help="Obstacle width in x-z (cells).")
    parser.add_argument("--obstacle-x", type=float, default=-1.0, help="Obstacle center x (-1 = auto).")
    parser.add_argument("--obstacle-z", type=float, default=-1.0, help="Obstacle center z (-1 = auto).")
    parser.add_argument(
        "--obstacle-mode",
        type=str,
        default="cylinder",
        choices=("cylinder", "box", "both"),
        help="cylinder / box / both (run cylinder and box sequentially).",
    )
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument(
        "--export-vtk",
        type=str,
        default="",
        help="VTK output path (default: <output-dir>/channel_obstacle_<mode>.vtk).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/channel_obstacle",
        help="Base directory for PNG/VTK outputs (mode subfolder added per run).",
    )
    return parser


def _apply_test_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "test", False) and args.num_frames == 100:
        args.num_frames = 600


def _apply_mode_paths(args: argparse.Namespace, mode: str) -> None:
    args.obstacle_mode = mode
    args.output_dir = _output_dir_for_mode(args.output_dir, mode)
    if not args.export_vtk:
        args.export_vtk = _vtk_path_for_mode(args.output_dir, mode)


def _run_modes(args: argparse.Namespace) -> None:
    import newton.examples

    base_output_dir: str = args.output_dir
    base_export_vtk: str = args.export_vtk
    modes: tuple[str, ...] = OBSTACLE_MODES if args.obstacle_mode == "both" else (args.obstacle_mode,)

    for mode in modes:
        run_args: argparse.Namespace = copy.copy(args)
        run_args.output_dir = base_output_dir
        run_args.export_vtk = base_export_vtk
        _apply_mode_paths(run_args, mode)
        print(f"\n=== obstacle mode: {mode} ===")
        viewer = _create_viewer(run_args)
        example = Example(viewer, run_args)
        newton.examples.run(example, run_args)


if __name__ == "__main__":
    wp.init()
    parser = _build_parser()
    preview_args, _ = parser.parse_known_args()
    _apply_test_defaults(preview_args)
    viewer, args = _init_viewer(parser, preview_args.viewer == "gl")
    _apply_test_defaults(args)

    if args.obstacle_mode == "both":
        _run_modes(args)
    else:
        _apply_mode_paths(args, args.obstacle_mode)
        example = Example(viewer, args)
        import newton.examples

        newton.examples.run(example, args)
