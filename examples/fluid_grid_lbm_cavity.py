# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Lid-driven cavity flow with D3Q19-BGK LBM (M2 deliverable).

Headless (M2 acceptance):
    python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --num-frames 100

Interactive GL volume + mid-plane vectors:
    python wanphys/examples/fluid_grid_lbm_cavity.py --viewer gl --grid-size 50

Export VTK for ParaView:
    python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --export-vtk output/cavity.vtk

Default: 50^3 grid, 5 LBM substeps/frame, 100 frames -> 500 lattice steps.
"""

from __future__ import annotations

import argparse
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
from wanphys._src.fluid.fluid_grid.lbm.vtk_export import export_structured_vtk  # noqa: E402


def _init_viewer(parser: argparse.ArgumentParser, use_gl_viewer: bool):
    if use_gl_viewer:
        from wanphys._src.fluid.fluid_viewer import init as fluid_init

        return fluid_init(parser)
    import newton.examples

    return newton.examples.init(parser)


def _save_velocity_slice_png(path: str | Path, velocity: np.ndarray, mid_j: int, u_lid: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for --save-slice (pip install matplotlib)") from exc

    ux_slice: np.ndarray = velocity[:, mid_j, :, 0]
    speed_slice: np.ndarray = np.linalg.norm(velocity[:, mid_j, :, :], axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(ux_slice.T, origin="lower", cmap="RdBu_r", vmin=-0.05, vmax=u_lid)
    axes[0].set_title(f"u_x @ j={mid_j}")
    axes[0].set_xlabel("i")
    axes[0].set_ylabel("k")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(speed_slice.T, origin="lower", cmap="viridis")
    axes[1].set_title("|u| mid-plane")
    axes[1].set_xlabel("i")
    axes[1].set_ylabel("k")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved mid-plane slice: {out}")


class Example:
    """Lid-driven 3D cavity using FluidGridLbmDomain."""

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

        print(f"Initializing LBM cavity: {grid_size}^3, nu={args.nu}, U_lid={u_lid}")

        model: FluidGridLbmModel = FluidGridLbmModel(
            fluid_grid_res=(grid_size, grid_size, grid_size),
            fluid_grid_cell_size=cell_size,
            nu=args.nu,
            use_guo_force=False,
        )
        self.domain: FluidGridLbmDomain = FluidGridLbmDomain(model)
        state = self.domain.create_state()
        self.domain.solver.configure_cavity_walls()
        self.domain.solver.set_lid_velocity(wp.vec3(u_lid, 0.0, 0.0))
        self.domain.solver.init_uniform(state, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))

        self._grid_size: int = grid_size
        self._u_lid: float = u_lid
        self._cell_size: float = cell_size

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
        speed: np.ndarray = np.linalg.norm(v_np, axis=-1)
        max_u: float = float(np.max(speed))
        near_lid_ux: float = float(np.max(v_np[:, self._grid_size - 2, :, 0]))
        print(
            f"frame={self.frame_count} lbm_steps={self.total_lbm_steps} "
            f"max|u|={max_u:.6f} near_lid_max_ux={near_lid_ux:.6f}"
        )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if self._gl_visualizer is not None:
            self._gl_visualizer.render()
        self.viewer.end_frame()

    def test_final(self) -> None:
        v_np: np.ndarray = self.domain.state.v.numpy()
        rho_np: np.ndarray = self.domain.state.rho.numpy()
        speed: np.ndarray = np.linalg.norm(v_np, axis=-1)
        max_u: float = float(np.max(speed))
        near_lid_ux: float = float(np.max(v_np[:, self._grid_size - 2, :, 0]))

        if np.isnan(v_np).any() or np.isinf(v_np).any():
            raise ValueError("Cavity velocity field contains NaN or Inf")
        if max_u < 0.01:
            raise ValueError(f"Flow not established after {self.total_lbm_steps} steps: max|u|={max_u}")
        if near_lid_ux < 0.05:
            raise ValueError(f"Lid BC too weak: near-lid max u_x={near_lid_ux}")

        if self.args.export_vtk:
            export_structured_vtk(
                self.args.export_vtk,
                rho_np,
                v_np,
                spacing=(self._cell_size, self._cell_size, self._cell_size),
            )
            print(f"Exported VTK: {self.args.export_vtk}")

        if self.args.save_slice:
            mid_j: int = self._grid_size // 2
            _save_velocity_slice_png(self.args.save_slice, v_np, mid_j, self._u_lid)


def _build_parser() -> argparse.ArgumentParser:
    import newton.examples

    parser: argparse.ArgumentParser = newton.examples.create_parser()
    parser.add_argument("--grid-size", type=int, default=50, help="Cubic grid resolution (>=32 for M2).")
    parser.add_argument("--nu", type=float, default=0.16667, help="Kinematic viscosity (lattice units).")
    parser.add_argument("--u-lid", type=float, default=0.1, help="Lid velocity u_x (lattice units).")
    parser.add_argument("--cell-size", type=float, default=1.0, help="Cell size for viewer/VTK (lattice units).")
    parser.add_argument("--warmup-steps", type=int, default=200, help="LBM steps before GL rendering starts.")
    parser.add_argument("--export-vtk", type=str, default="", help="Write ParaView VTK on completion.")
    parser.add_argument("--save-slice", type=str, default="", help="Save mid-y u_x/|u| PNG (needs matplotlib).")
    parser.add_argument("--no-volume", action="store_true", help="Disable |u| volume rendering (GL only).")
    parser.add_argument("--no-boundary", action="store_true", help="Disable cavity wireframe (GL only).")
    parser.add_argument("--no-vectors", action="store_true", help="Disable mid-plane velocity vectors (GL only).")
    parser.add_argument("--vector-stride", type=int, default=4, help="Vector arrow sampling stride on mid-plane.")
    parser.add_argument("--vector-scale", type=float, default=0.8, help="Velocity arrow length scale.")
    return parser


if __name__ == "__main__":
    wp.init()
    parser: argparse.ArgumentParser = _build_parser()
    preview_args, _ = parser.parse_known_args()
    viewer, args = _init_viewer(parser, preview_args.viewer == "gl")
    example = Example(viewer, args)
    import newton.examples

    newton.examples.run(example, args)
