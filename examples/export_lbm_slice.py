# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Export LBM velocity slice using project visualization utilities.

Uses:
  - ``cavity_plot.xy_plane_fields`` / ``plot_lid_driven_cavity`` (matplotlib)
  - ``vtk_export.export_structured_vtk`` (optional ParaView VTK)

Run:
    python examples/export_lbm_slice.py
    python examples/export_lbm_slice.py --grid-size 50 --steps 500 --slice-k 25
    python examples/export_lbm_slice.py --with-obstacle --save-slice output/obstacle_slice.png
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import warp as wp

_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP = _ROOT / "tests" / "_bootstrap.py"
_spec = importlib.util.spec_from_file_location("lbm_test_bootstrap", _BOOTSTRAP)
assert _spec is not None and _spec.loader is not None
_bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bootstrap)
_bootstrap.bootstrap_from_test_file(__file__)

from wanphys._src.fluid.fluid_grid.lbm import FluidGridLbmDomain, FluidGridLbmModel
from wanphys._src.fluid.fluid_grid.lbm.cavity_plot import plot_lid_driven_cavity, xy_plane_fields
from wanphys._src.fluid.fluid_grid.lbm.vtk_export import export_structured_vtk


def run_cavity_and_export(args: argparse.Namespace) -> None:
    wp.init()

    model = FluidGridLbmModel(
        fluid_grid_res=(args.grid_size, args.grid_size, args.grid_size),
        fluid_grid_cell_size=args.cell_size,
        nu=args.nu,
        use_guo_force=False,
        collide_impl=args.collide,
    )
    domain = FluidGridLbmDomain(model)
    state = domain.create_state()
    domain.solver.configure_cavity_walls()
    domain.solver.set_lid_velocity(wp.vec3(args.u_lid, 0.0, 0.0))
    domain.solver.init_uniform(state, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))

    if args.with_obstacle:
        gs = float(args.grid_size)
        cx = args.obstacle_cx if args.obstacle_cx >= 0.0 else gs * 0.55
        cy = args.obstacle_cy if args.obstacle_cy >= 0.0 else gs * 0.5
        cz = args.obstacle_cz if args.obstacle_cz >= 0.0 else gs * 0.5
        hx = args.obstacle_hx if args.obstacle_hx >= 0.0 else gs * 0.08
        hy = args.obstacle_hy if args.obstacle_hy >= 0.0 else gs * 0.08
        hz = args.obstacle_hz if args.obstacle_hz >= 0.0 else gs * 0.08
        domain.solver.bake_box(state, wp.vec3(cx, cy, cz), wp.vec3(hx, hy, hz))
        n_solid = int(np.sum(domain.state.solid.numpy() != 0))
        print(f"Baked box obstacle: center=({cx:.1f},{cy:.1f},{cz:.1f}) half=({hx:.1f},{hy:.1f},{hz:.1f}), solid_cells={n_solid}")

    label = "obstacle cavity" if args.with_obstacle else "cavity"
    print(f"Running LBM {label}: {args.grid_size}^3, steps={args.steps}, U_lid={args.u_lid}")
    for step in range(args.steps):
        domain.step(dt=1.0)
        if (step + 1) % max(1, args.steps // 5) == 0:
            v_np = domain.state.v.numpy()
            max_u = float(np.max(np.linalg.norm(v_np, axis=-1)))
            print(f"  step {step + 1}/{args.steps}: max|u|={max_u:.6f}")

    v_np: np.ndarray = domain.state.v.numpy()
    rho_np: np.ndarray = domain.state.rho.numpy()
    solid_np: np.ndarray | None = domain.state.solid.numpy() if args.with_obstacle else None

    slice_k = args.slice_k if args.slice_k >= 0 else None
    u_2d, v_2d, scalar_2d, nx, ny, k = xy_plane_fields(
        v_np,
        slice_k=slice_k,
        scalar=args.rho if args.scalar_field == "rho" else None,
        scalar_mode=args.scalar_field,
    )
    print(f"Slice extracted: x-y plane @ k={k}, shape=({ny}, {nx})")
    print(f"  |u| range: [{float(scalar_2d.min()):.6f}, {float(scalar_2d.max()):.6f}]")

    if args.save_slice:
        out_png = Path(args.save_slice)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plot_lid_driven_cavity(
            v_np,
            rho=rho_np if args.scalar_field == "rho" else None,
            solid=solid_np,
            cell_size=args.cell_size,
            slice_k=slice_k,
            scalar_mode=args.scalar_field,
            u_lid=args.u_lid,
            title=(
                f"Lid-driven cavity + obstacle (x-y @ k={k if slice_k is not None else 'mid'})"
                if args.with_obstacle
                else None
            ),
            path=out_png,
            show=args.show_plot,
        )
        print(f"Saved slice PNG: {out_png.resolve()}")

    if args.export_vtk:
        out_vtk = Path(args.export_vtk)
        export_structured_vtk(
            out_vtk,
            rho_np,
            v_np,
            spacing=(args.cell_size, args.cell_size, args.cell_size),
        )
        print(f"Exported VTK: {out_vtk.resolve()}")

    if args.save_numpy:
        out_npz = Path(args.save_numpy)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_npz,
            velocity=v_np,
            rho=rho_np,
            u_2d=u_2d,
            v_2d=v_2d,
            scalar_2d=scalar_2d,
            slice_k=k,
            solid=solid_np,
        )
        print(f"Saved numpy slice bundle: {out_npz.resolve()}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export LBM cavity velocity slice.")
    parser.add_argument("--grid-size", type=int, default=32, help="Cubic grid resolution.")
    parser.add_argument("--steps", type=int, default=200, help="LBM time steps.")
    parser.add_argument("--nu", type=float, default=0.16667, help="Kinematic viscosity (lattice units).")
    parser.add_argument("--u-lid", type=float, default=0.1, help="Lid velocity u_x.")
    parser.add_argument("--cell-size", type=float, default=1.0, help="Cell size for plot/VTK axes.")
    parser.add_argument("--collide", type=str, default="bgk", choices=("bgk", "trt", "mrt"), help="Collision scheme.")
    parser.add_argument(
        "--slice-k",
        type=int,
        default=-1,
        help="z-index for x-y slice (-1 = nz//2). Lid is at y=ny-1.",
    )
    parser.add_argument(
        "--scalar-field",
        type=str,
        default="speed",
        choices=("speed", "ux", "rho"),
        help="Scalar field for contour background.",
    )
    parser.add_argument(
        "--save-slice",
        type=str,
        default="output/cavity_slice.png",
        help="Output PNG path (matplotlib streamplot).",
    )
    parser.add_argument("--export-vtk", type=str, default="", help="Optional VTK output path.")
    parser.add_argument("--save-numpy", type=str, default="", help="Optional .npz with 2D slice arrays.")
    parser.add_argument("--show-plot", action="store_true", help="Show matplotlib window.")
    parser.add_argument(
        "--with-obstacle",
        action="store_true",
        help="Bake central box obstacle (cavity + column, post_midterm S1).",
    )
    parser.add_argument("--obstacle-cx", type=float, default=-1.0, help="Obstacle center x (lattice index).")
    parser.add_argument("--obstacle-cy", type=float, default=-1.0, help="Obstacle center y.")
    parser.add_argument("--obstacle-cz", type=float, default=-1.0, help="Obstacle center z.")
    parser.add_argument("--obstacle-hx", type=float, default=-1.0, help="Obstacle half-extent x.")
    parser.add_argument("--obstacle-hy", type=float, default=-1.0, help="Obstacle half-extent y.")
    parser.add_argument("--obstacle-hz", type=float, default=-1.0, help="Obstacle half-extent z.")
    return parser


if __name__ == "__main__":
    run_cavity_and_export(_build_parser().parse_args())
