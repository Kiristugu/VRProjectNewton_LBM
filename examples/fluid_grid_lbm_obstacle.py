# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Lid-driven cavity with a central box obstacle (绕障流 demo).

Obstacle vertical extent: ``mid`` (centered), ``high`` (extends to moving lid), ``tall`` (floor to lid).

Headless + figures + VTK (all three heights):
    python wanphys/examples/fluid_grid_lbm_obstacle.py --viewer null --num-frames 100 --test --obstacle-height all

Single height:
    python wanphys/examples/fluid_grid_lbm_obstacle.py --viewer null --obstacle-height mid --num-frames 100 --test

Outputs per height (``output/obstacle/<mid|high|tall>/``):
    obstacle_<mode>.vtk
    stream_xy_z_{bottom,mid,top}.png   # side view x-y @ obstacle z slices
    stream_xz_y_{bottom,mid,top}.png   # top view x-z @ obstacle y slices
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
    """Avoid loading wanphys top-level __init__ (geometry deps) in minimal venvs."""
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

from wanphys._src.fluid.fluid_grid.lbm import (  # noqa: E402
    FluidGridLbmDomain,
    FluidGridLbmModel,
)
from wanphys._src.fluid.fluid_grid.lbm.cavity_plot import (  # noqa: E402
    export_obstacle_cavity_figures,
    plot_lid_driven_cavity,
)
from wanphys._src.fluid.fluid_grid.lbm.vtk_export import export_structured_vtk  # noqa: E402

OBSTACLE_HEIGHTS: tuple[str, ...] = ("mid", "high", "tall")


def _init_viewer(parser: argparse.ArgumentParser, use_gl_viewer: bool):
    if use_gl_viewer:
        from wanphys._src.fluid.fluid_viewer import init as fluid_init

        return fluid_init(parser)
    import newton.examples

    return newton.examples.init(parser)


def _create_viewer(args: argparse.Namespace):
    """Fresh viewer instance (needed when running multiple heights in one process)."""
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


def _default_obstacle_half_extents(grid_size: int) -> wp.vec3:
    scale: float = max(2.0, float(grid_size) * 0.08)
    return wp.vec3(scale, scale * 1.5, scale)


def _resolve_base_box(args: argparse.Namespace) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return (center, half) for MID reference geometry (x/z half from args or auto)."""
    grid_size: int = args.grid_size
    mid: float = 0.5 * float(grid_size)
    auto: wp.vec3 = _default_obstacle_half_extents(grid_size)

    cx: float = args.box_cx if args.box_cx >= 0.0 else mid
    cy: float = args.box_cy if args.box_cy >= 0.0 else mid
    cz: float = args.box_cz if args.box_cz >= 0.0 else mid
    hx: float = args.box_half if args.box_half >= 0.0 else float(auto[0])
    hy: float = args.box_half_y if args.box_half_y >= 0.0 else float(auto[1])
    hz: float = args.box_half_z if args.box_half_z >= 0.0 else float(auto[2])
    return (cx, cy, cz), (hx, hy, hz)


def _box_for_height(
    height_mode: str,
    args: argparse.Namespace,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Map MID / HIGH / TALL to obstacle center and half-extents (y varies)."""
    grid_size: int = args.grid_size
    (cx, cy_mid, cz), (hx, hy_mid, hz) = _resolve_base_box(args)
    lid_y: float = float(grid_size - 1) - 0.5
    floor_y: float = 0.5

    if height_mode == "mid":
        return (cx, cy_mid, cz), (hx, hy_mid, hz)

    y_bottom_mid: float = cy_mid - hy_mid

    if height_mode == "high":
        y_top: float = lid_y
        cy: float = 0.5 * (y_bottom_mid + y_top)
        hy: float = 0.5 * (y_top - y_bottom_mid)
        return (cx, cy, cz), (hx, hy, hz)

    if height_mode == "tall":
        cy: float = 0.5 * (floor_y + lid_y)
        hy: float = 0.5 * (lid_y - floor_y)
        return (cx, cy, cz), (hx, hy, hz)

    raise ValueError(f"unknown height_mode={height_mode!r}")


def _output_dir_for_height(base_dir: str | Path, height_mode: str) -> str:
    return str(Path(base_dir) / height_mode)


def _vtk_path_for_height(output_dir: str | Path, height_mode: str) -> str:
    return str(Path(output_dir) / f"obstacle_{height_mode}.vtk")


class Example:
    """Lid-driven cavity with a central box obstacle."""

    def __init__(self, viewer, args: argparse.Namespace) -> None:
        self.viewer = viewer
        self.args = args
        self.sim_time: float = 0.0
        self.frame_count: int = 0
        self.lbm_substeps: int = 5
        self.total_lbm_steps: int = 0

        grid_size: int = args.grid_size
        u_lid: float = args.u_lid
        cell_size: float = args.cell_size
        height_mode: str = args.obstacle_height

        center, half = _box_for_height(height_mode, args)
        cx, cy, cz = center
        hx, hy, hz = half

        print(
            f"Initializing LBM obstacle cavity: {grid_size}^3, nu={args.nu}, "
            f"U_lid={u_lid}, height={height_mode.upper()}, "
            f"box center=({cx:.1f},{cy:.1f},{cz:.1f}) half=({hx:.1f},{hy:.1f},{hz:.1f})"
        )

        model: FluidGridLbmModel = FluidGridLbmModel(
            fluid_grid_res=(grid_size, grid_size, grid_size),
            fluid_grid_cell_size=cell_size,
            nu=args.nu,
            use_guo_force=False,
        )
        self.domain: FluidGridLbmDomain = FluidGridLbmDomain(model)
        self.domain.create_state()

        self.domain.solver.configure_cavity_walls()
        self.domain.solver.set_lid_velocity(wp.vec3(u_lid, 0.0, 0.0))
        self.domain.bake_box(wp.vec3(cx, cy, cz), wp.vec3(hx, hy, hz))
        self.domain.solver.init_uniform(self.domain.state, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))

        self._grid_size: int = grid_size
        self._u_lid: float = u_lid
        self._cell_size: float = cell_size
        self._height_mode: str = height_mode
        self._output_dir = Path(args.output_dir)
        self._obstacle_center: tuple[float, float, float] = center
        self._obstacle_half: tuple[float, float, float] = half

        solid_np: np.ndarray = self.domain.state.solid.numpy()
        n_solid: int = int(np.sum(solid_np > 0))
        print(f"Baked box obstacle: {n_solid} solid cells")

        self._gl_visualizer = None
        if args.viewer == "gl":
            from wanphys._src.fluid.fluid_viewer.lbm_flow import LbmCavityVisualizer

            self._gl_visualizer = LbmCavityVisualizer(
                viewer,
                self.domain,
                cell_size=cell_size,
                show_volume=not args.no_volume,
                show_boundary=not args.no_boundary,
                show_mid_plane_vectors=not args.no_vectors,
                vector_stride=args.vector_stride,
                vector_scale=args.vector_scale,
            )
            self._gl_visualizer.setup_camera()
            if args.warmup_steps > 0:
                print(f"Warmup: {args.warmup_steps} LBM steps before rendering...")
                for _ in range(args.warmup_steps):
                    self.domain.step(dt=1.0)
                    self.total_lbm_steps += 1

    def step(self) -> None:
        for _ in range(self.lbm_substeps):
            self.domain.step(dt=1.0)
            self.total_lbm_steps += 1
        self.sim_time += float(self.lbm_substeps)
        self.frame_count += 1

        if self.frame_count % 20 == 0 or not self.viewer.is_running():
            self._print_diagnostics()

    def _print_diagnostics(self) -> None:
        v_np: np.ndarray = self.domain.state.v.numpy()
        solid_np: np.ndarray = self.domain.state.solid.numpy()
        speed: np.ndarray = np.linalg.norm(v_np, axis=-1)
        fluid_mask: np.ndarray = solid_np == 0
        max_u: float = float(np.max(speed[fluid_mask])) if np.any(fluid_mask) else 0.0
        wake_max: float = self._wake_max_speed(v_np, solid_np)
        print(
            f"[{self._height_mode}] frame={self.frame_count} lbm_steps={self.total_lbm_steps} "
            f"max|u|={max_u:.6f} wake_max|u|={wake_max:.6f}"
        )

    def _wake_max_speed(self, v_np: np.ndarray, solid_np: np.ndarray) -> float:
        cx, _, _ = self._obstacle_center
        hx, _, _ = self._obstacle_half
        i_min: int = int(cx + hx + 2.0)
        if i_min >= self._grid_size:
            i_min = self._grid_size - 1
        wake_speed: np.ndarray = np.linalg.norm(v_np[i_min:, :, :, :], axis=-1)
        wake_solid: np.ndarray = solid_np[i_min:, :, :]
        fluid: np.ndarray = wake_solid == 0
        if not np.any(fluid):
            return 0.0
        return float(np.max(wake_speed[fluid]))

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if self._gl_visualizer is not None:
            self._gl_visualizer.render()
        self.viewer.end_frame()

    def test_final(self) -> None:
        v_np: np.ndarray = self.domain.state.v.numpy()
        rho_np: np.ndarray = self.domain.state.rho.numpy()
        solid_np: np.ndarray = self.domain.state.solid.numpy()
        speed: np.ndarray = np.linalg.norm(v_np, axis=-1)

        if np.isnan(v_np).any() or np.isinf(v_np).any():
            raise ValueError("Velocity field contains NaN or Inf")

        solid_speed: np.ndarray = speed[solid_np > 0]
        if solid_speed.size > 0:
            max_solid_u: float = float(np.max(solid_speed))
            if max_solid_u > 1.0e-4:
                raise ValueError(f"Flow inside solid: max|u|={max_solid_u}")

        fluid_mask: np.ndarray = solid_np == 0
        max_u: float = float(np.max(speed[fluid_mask]))
        wake_max: float = self._wake_max_speed(v_np, solid_np)

        if max_u < 0.01:
            raise ValueError(f"Flow not established after {self.total_lbm_steps} steps: max|u|={max_u}")
        if wake_max < 0.005:
            raise ValueError(
                f"No wake downstream of obstacle ({self._height_mode}): wake_max|u|={wake_max}"
            )

        print(
            f"Obstacle test passed [{self._height_mode}]: max|u|={max_u:.6f}, "
            f"wake_max|u|={wake_max:.6f}, solid_cells={int(np.sum(solid_np > 0))}"
        )

        vtk_path: str = self.args.export_vtk or _vtk_path_for_height(self._output_dir, self._height_mode)
        export_structured_vtk(
            vtk_path,
            rho_np,
            v_np,
            solid=solid_np,
            spacing=(self._cell_size, self._cell_size, self._cell_size),
            title=f"WanPhys LBM cavity obstacle ({self._height_mode.upper()})",
        )
        print(f"Exported VTK: {vtk_path}")

        if self.args.export_figures or self.args.save_slice or self.args.show_plot:
            export_obstacle_cavity_figures(
                v_np,
                solid_np,
                output_dir=self._output_dir,
                center=self._obstacle_center,
                half=self._obstacle_half,
                grid_size=self._grid_size,
                cell_size=self._cell_size,
                u_lid=self._u_lid,
                height_mode=self._height_mode,
                scalar_mode=self.args.scalar_field,
                rho=rho_np if self.args.scalar_field == "rho" else None,
            )
            print(f"Validation figures written to: {self._output_dir.resolve()}")

        elif self.args.save_slice:
            plot_lid_driven_cavity(
                v_np,
                rho=rho_np if self.args.scalar_field == "rho" else None,
                solid=solid_np,
                cell_size=self._cell_size,
                slice_k=self.args.slice_k if self.args.slice_k >= 0 else None,
                scalar_mode=self.args.scalar_field,
                u_lid=self._u_lid,
                title=f"LBM obstacle cavity (x-y @ k=mid), U_lid={self._u_lid:g}",
                path=self.args.save_slice,
                show=self.args.show_plot,
            )


def _build_parser() -> argparse.ArgumentParser:
    import newton.examples

    parser: argparse.ArgumentParser = newton.examples.create_parser()
    parser.add_argument("--grid-size", type=int, default=64, help="Cubic grid resolution (64+ for figures).")
    parser.add_argument("--nu", type=float, default=0.16667, help="Kinematic viscosity (lattice units).")
    parser.add_argument("--u-lid", type=float, default=0.1, help="Lid velocity u_x (lattice units).")
    parser.add_argument("--cell-size", type=float, default=1.0, help="Cell size for VTK (lattice units).")
    parser.add_argument(
        "--obstacle-height",
        type=str,
        default="mid",
        choices=("mid", "high", "tall", "all"),
        help="Vertical extent: mid=centered, high=to lid, tall=floor-to-lid, all=run three cases.",
    )
    parser.add_argument("--box-cx", type=float, default=-1.0, help="Obstacle center x (-1 = grid center).")
    parser.add_argument("--box-cy", type=float, default=-1.0, help="Obstacle center y (-1 = grid center, MID ref).")
    parser.add_argument("--box-cz", type=float, default=-1.0, help="Obstacle center z (-1 = grid center).")
    parser.add_argument("--box-half", type=float, default=-1.0, help="Obstacle half extent x (-1 = auto).")
    parser.add_argument("--box-half-y", type=float, default=-1.0, help="Obstacle half extent y for MID (-1 = auto).")
    parser.add_argument("--box-half-z", type=float, default=-1.0, help="Obstacle half extent z (-1 = auto).")
    parser.add_argument("--warmup-steps", type=int, default=100, help="LBM steps before GL rendering starts.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/obstacle",
        help="Base output directory (height subfolder added per run).",
    )
    parser.add_argument(
        "--export-vtk",
        type=str,
        default="",
        help="VTK path (default: <output-dir>/<height>/obstacle_<height>.vtk).",
    )
    parser.add_argument(
        "--export-figures",
        action="store_true",
        help="Write 6 streamplot PNGs (side x-y + top x-z, obstacle bottom/mid/top).",
    )
    parser.add_argument("--save-slice", type=str, default="", help="Legacy: single x-y PNG (overrides batch if unset).")
    parser.add_argument("--show-plot", action="store_true", help="Show matplotlib plot after simulation.")
    parser.add_argument("--slice-k", type=int, default=-1, help="z-index for legacy single x-y slice (-1 = nz//2).")
    parser.add_argument(
        "--scalar-field",
        type=str,
        default="speed",
        choices=("speed", "ux", "rho"),
        help="Scalar background for contourf.",
    )
    parser.add_argument("--no-volume", action="store_true", help="Disable |u| volume rendering (GL only).")
    parser.add_argument("--no-boundary", action="store_true", help="Disable cavity wireframe (GL only).")
    parser.add_argument("--no-vectors", action="store_true", help="Disable mid-plane velocity vectors (GL only).")
    parser.add_argument("--vector-stride", type=int, default=4, help="Vector arrow sampling stride on mid-plane.")
    parser.add_argument("--vector-scale", type=float, default=0.8, help="Velocity arrow length scale.")
    return parser


def _apply_test_output_defaults(args: argparse.Namespace) -> None:
    if not getattr(args, "test", False):
        return
    args.export_figures = True


def _apply_height_paths(args: argparse.Namespace, height_mode: str, *, base_output_dir: str | None = None) -> None:
    root: str = base_output_dir if base_output_dir is not None else args.output_dir
    args.obstacle_height = height_mode
    args.output_dir = _output_dir_for_height(root, height_mode)
    if not args.export_vtk:
        args.export_vtk = _vtk_path_for_height(args.output_dir, height_mode)


def _run_heights(args: argparse.Namespace) -> None:
    import newton.examples

    base_output_dir: str = args.output_dir
    base_export_vtk: str = args.export_vtk
    modes: tuple[str, ...] = OBSTACLE_HEIGHTS if args.obstacle_height == "all" else (args.obstacle_height,)

    for height_mode in modes:
        run_args: argparse.Namespace = copy.copy(args)
        run_args.export_vtk = base_export_vtk
        _apply_height_paths(run_args, height_mode, base_output_dir=base_output_dir)
        print(f"\n=== obstacle height: {height_mode.upper()} ===")
        viewer = _create_viewer(run_args)
        example = Example(viewer, run_args)
        newton.examples.run(example, run_args)


if __name__ == "__main__":
    wp.init()
    parser: argparse.ArgumentParser = _build_parser()
    preview_args, _ = parser.parse_known_args()
    _apply_test_output_defaults(preview_args)
    viewer, args = _init_viewer(parser, preview_args.viewer == "gl")
    _apply_test_output_defaults(args)

    if args.obstacle_height == "all":
        _run_heights(args)
    else:
        _apply_height_paths(args, args.obstacle_height)
        example = Example(viewer, args)
        import newton.examples

        newton.examples.run(example, args)
