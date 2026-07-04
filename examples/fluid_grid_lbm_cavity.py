# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Lid-driven cavity flow with D3Q19 LBM (M2 deliverable).

Headless (M2 acceptance):
    python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --num-frames 100

BGK vs MRT comparison presets:
    python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --test --profile subtle --compare
    python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --test --profile stress --compare
    python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --test --profile contrast --compare
    python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --test --profile coarse --compare
    python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --test --profile stress --collide mrt

Default: 50^3 grid, 5 LBM substeps/frame, 100 frames -> 500 lattice steps.
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
    root: Path = Path(__file__).resolve().parents[1]
    root_str: str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    wanphys_dir: Path = root
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
from wanphys._src.fluid.fluid_grid.lbm.cavity_plot import plot_lid_driven_cavity  # noqa: E402
from wanphys._src.fluid.fluid_grid.lbm.cavity_presets import (  # noqa: E402
    DEFAULT_LBM_SUBSTEPS,
    DEFAULT_OUTPUT_DIR,
    PRESETS,
    apply_cavity_profile,
    build_lbm_model,
    collide_choices,
    compare_collide_schemes,
    resolve_export_paths,
)
from wanphys._src.fluid.fluid_grid.lbm.vtk_export import export_structured_vtk  # noqa: E402


def _init_viewer(parser: argparse.ArgumentParser, use_gl_viewer: bool):
    if use_gl_viewer:
        from wanphys._src.fluid.fluid_viewer import init as fluid_init

        return fluid_init(parser)
    import newton.examples

    return newton.examples.init(parser)


def _create_viewer(args: argparse.Namespace):
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
    def __init__(self, viewer, args: argparse.Namespace) -> None:
        self.viewer = viewer
        self.args = args
        self.sim_time: float = 0.0
        self.frame_count: int = 0
        self.lbm_substeps: int = DEFAULT_LBM_SUBSTEPS
        self.total_lbm_steps: int = 0

        grid_size: int = args.grid_size
        u_lid: float = args.u_lid
        cell_size: float = args.cell_size
        collide: str = args.collide

        print(
            f"Initializing LBM cavity: {grid_size}^3, nu={args.nu}, U_lid={u_lid}, "
            f"collide={collide}, profile={args.profile}"
        )

        model: FluidGridLbmModel = build_lbm_model(
            FluidGridLbmModel,
            grid_size=grid_size,
            cell_size=cell_size,
            nu=args.nu,
            collide=collide,
            mrt_ghost_s=args.mrt_ghost_s,
        )
        self.domain: FluidGridLbmDomain = FluidGridLbmDomain(model)
        state = self.domain.create_state()
        self.domain.solver.configure_cavity_walls()
        self.domain.solver.set_lid_velocity(wp.vec3(u_lid, 0.0, 0.0))
        self.domain.solver.init_uniform(state, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))

        self._grid_size = grid_size
        self._u_lid = u_lid
        self._cell_size = cell_size
        self._collide = collide
        self._output_dir = Path(args.output_dir)

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
        rho_np: np.ndarray = self.domain.state.rho.numpy()
        speed: np.ndarray = np.linalg.norm(v_np, axis=-1)
        max_u: float = float(np.max(speed))
        near_lid_ux: float = float(np.max(v_np[:, self._grid_size - 2, :, 0]))
        max_rho_err: float = float(np.max(np.abs(rho_np - 1.0)))
        print(
            f"[{self._collide}] frame={self.frame_count} steps={self.total_lbm_steps} "
            f"max|u|={max_u:.6f} near_lid_ux={near_lid_ux:.6f} max|rho-1|={max_rho_err:.6f}"
        )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if self._gl_visualizer is not None:
            self._gl_visualizer.render()
        self.viewer.end_frame()

    def _export_field(self, v_np, rho_np, *, scalar_mode: str, png_path: Path) -> None:
        plot_lid_driven_cavity(
            v_np,
            rho=rho_np if scalar_mode == "rho" else None,
            cell_size=self._cell_size,
            slice_k=self.args.slice_k if self.args.slice_k >= 0 else None,
            scalar_mode=scalar_mode,
            u_lid=self._u_lid,
            title=f"Cavity {self.args.profile}/{self._collide} ({scalar_mode})",
            path=png_path,
            show=self.args.show_plot,
        )

    def test_final(self) -> None:
        v_np: np.ndarray = self.domain.state.v.numpy()
        rho_np: np.ndarray = self.domain.state.rho.numpy()
        speed: np.ndarray = np.linalg.norm(v_np, axis=-1)
        diverged: bool = (
            np.isnan(v_np).any()
            or np.isinf(v_np).any()
            or np.isnan(rho_np).any()
            or np.isinf(rho_np).any()
        )
        if diverged:
            if self.args.profile == "coarse" and self._collide == "bgk":
                print(
                    f"Cavity DIVERGED [{self._collide}]: expected on coarse/high-Re; "
                    "MRT run will continue under --compare."
                )
                return
            raise ValueError("Cavity velocity field contains NaN or Inf")

        max_u: float = float(np.max(speed))
        near_lid_ux: float = float(np.max(v_np[:, self._grid_size - 2, :, 0]))
        max_rho_err: float = float(np.max(np.abs(rho_np - 1.0)))
        mean_rho_err: float = float(np.mean(np.abs(rho_np - 1.0)))

        if max_u < 0.01:
            raise ValueError(f"Flow not established: max|u|={max_u}")
        if near_lid_ux < 0.05:
            raise ValueError(f"Lid BC too weak: near-lid max u_x={near_lid_ux}")

        print(
            f"Cavity OK [{self._collide}]: max|u|={max_u:.6f}, "
            f"max|rho-1|={max_rho_err:.6f}, mean|rho-1|={mean_rho_err:.6f}"
        )

        preset = PRESETS.get(self.args.profile)
        export_speed = self.args.export_speed or (preset.export_speed if preset else False)
        export_rho = self.args.export_rho or (preset.export_rho if preset else False)
        do_batch = self.args.export_batch or self.args.profile != "custom"

        if do_batch or self.args.export_vtk or self.args.save_numpy:
            _, vtk_path, npz_path = resolve_export_paths(self._output_dir, self._collide, field="bundle")
            if self.args.export_vtk or do_batch:
                out_vtk = self.args.export_vtk or str(vtk_path)
                export_structured_vtk(
                    out_vtk,
                    rho_np,
                    v_np,
                    spacing=(self._cell_size, self._cell_size, self._cell_size),
                    title=f"WanPhys cavity {self.args.profile}/{self._collide}",
                )
                print(f"Exported VTK: {out_vtk}")
            if self.args.save_numpy or do_batch:
                out_npz = self.args.save_numpy or str(npz_path)
                np.savez(
                    out_npz,
                    velocity=v_np,
                    rho=rho_np,
                    collide=self._collide,
                    profile=self.args.profile,
                    nu=self.args.nu,
                    u_lid=self._u_lid,
                )
                print(f"Saved NPZ: {out_npz}")

        if export_speed or self.args.save_slice:
            png_speed, _, _ = resolve_export_paths(self._output_dir, self._collide, field="speed")
            self._export_field(v_np, rho_np, scalar_mode="speed", png_path=Path(self.args.save_slice or png_speed))

        if export_rho:
            png_rho, _, _ = resolve_export_paths(self._output_dir, self._collide, field="rho")
            self._export_field(v_np, rho_np, scalar_mode="rho", png_path=png_rho)

        if do_batch:
            print(f"Exports under: {self._output_dir.resolve()}")


def _build_parser() -> argparse.ArgumentParser:
    import newton.examples

    schemes = collide_choices()
    parser: argparse.ArgumentParser = newton.examples.create_parser()
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--nu", type=float, default=0.16667)
    parser.add_argument("--u-lid", type=float, default=0.1)
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument(
        "--profile",
        type=str,
        default="custom",
        choices=("custom", "subtle", "stress", "contrast", "coarse"),
    )
    parser.add_argument("--collide", type=str, default="bgk", choices=schemes)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--mrt-ghost-s", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--export-batch", action="store_true")
    parser.add_argument("--export-speed", action="store_true")
    parser.add_argument("--export-rho", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--export-vtk", type=str, default="")
    parser.add_argument("--save-numpy", type=str, default="")
    parser.add_argument("--save-slice", type=str, default="")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--slice-k", type=int, default=-1)
    parser.add_argument("--scalar-field", type=str, default="speed", choices=("speed", "ux", "rho"))
    parser.add_argument("--no-volume", action="store_true")
    parser.add_argument("--no-boundary", action="store_true")
    parser.add_argument("--no-vectors", action="store_true")
    parser.add_argument("--vector-stride", type=int, default=4)
    parser.add_argument("--vector-scale", type=float, default=0.8)
    return parser


def _apply_test_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "test", False):
        args.export_batch = True


def _configure_compare_run(args: argparse.Namespace, collide: str) -> None:
    args.collide = collide
    args.export_vtk = str(resolve_export_paths(args.output_dir, collide, field="bundle")[1])
    args.save_numpy = str(resolve_export_paths(args.output_dir, collide, field="bundle")[2])
    args.save_slice = ""


def _run_single(args: argparse.Namespace, viewer) -> None:
    import newton.examples

    newton.examples.run(Example(viewer, args), args)


def _export_compare_diff(args: argparse.Namespace) -> None:
    """Write |BGK-MRT| rho/speed mid-plane PNGs after ``--compare``."""
    from wanphys._src.fluid.fluid_grid.lbm.cavity_plot import plot_cavity_field_diff

    schemes = compare_collide_schemes()
    if "mrt" not in schemes:
        return

    root = Path(args.output_dir)
    bgk_npz = root / "cavity_bgk.npz"
    mrt_npz = root / "cavity_mrt.npz"
    if not bgk_npz.is_file() or not mrt_npz.is_file():
        missing = []
        if not bgk_npz.is_file():
            missing.append("cavity_bgk.npz")
        if not mrt_npz.is_file():
            missing.append("cavity_mrt.npz")
        print(f"Warning: skip diff export; missing {', '.join(missing)}")
        if args.profile == "coarse" and not bgk_npz.is_file():
            print("  (BGK divergence on coarse is expected; compare MRT PNGs directly.)")
        return

    bgk = np.load(bgk_npz)
    mrt = np.load(mrt_npz)
    slice_k = args.slice_k if args.slice_k >= 0 else None
    cell_size = args.cell_size

    for field, bgk_arr, mrt_arr in (
        ("rho", bgk["rho"], mrt["rho"]),
        ("speed", np.linalg.norm(bgk["velocity"], axis=-1), np.linalg.norm(mrt["velocity"], axis=-1)),
    ):
        diff = np.abs(bgk_arr - mrt_arr)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        out_png = root / f"cavity_diff_{field}.png"
        plot_cavity_field_diff(
            diff,
            cell_size=cell_size,
            slice_k=slice_k,
            field=field,
            title=f"|BGK-MRT| {field} ({args.profile}) max={max_diff:.2e}",
            path=out_png,
            show=args.show_plot,
        )
        print(f"Diff [{field}]: max={max_diff:.6e}, mean={mean_diff:.6e} -> {out_png}")


def _run_compare(args: argparse.Namespace) -> None:
    schemes = compare_collide_schemes()
    if len(schemes) < 2:
        print("Warning: MRT not available; --compare runs BGK only.")
    for collide in schemes:
        run_args = copy.copy(args)
        _configure_compare_run(run_args, collide)
        print(f"\n=== collide: {collide.upper()} | profile: {run_args.profile} ===")
        _run_single(run_args, _create_viewer(run_args))
    _export_compare_diff(args)


if __name__ == "__main__":
    wp.init()
    parser = _build_parser()
    preview_args, _ = parser.parse_known_args()
    apply_cavity_profile(preview_args)
    _apply_test_defaults(preview_args)
    viewer, args = _init_viewer(parser, preview_args.viewer == "gl")
    apply_cavity_profile(args)
    _apply_test_defaults(args)

    if args.collide not in collide_choices():
        raise ValueError(f"--collide {args.collide!r} not supported; available: {collide_choices()}")

    if args.compare:
        _run_compare(args)
    else:
        if args.profile != "custom":
            args.output_dir = str(Path(args.output_dir) / args.profile)
        if args.export_batch and not args.export_vtk:
            args.export_vtk = str(resolve_export_paths(args.output_dir, args.collide, field="bundle")[1])
        if args.export_batch and not args.save_numpy:
            args.save_numpy = str(resolve_export_paths(args.output_dir, args.collide, field="bundle")[2])
        _run_single(args, viewer)
