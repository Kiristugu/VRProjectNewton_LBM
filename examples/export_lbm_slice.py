# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Export LBM cavity slice with BGK/MRT comparison presets.

Run:
    python wanphys/examples/export_lbm_slice.py --profile subtle --compare
    python wanphys/examples/export_lbm_slice.py --profile stress --collide mrt
"""

from __future__ import annotations

import argparse
import copy
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
from wanphys._src.fluid.fluid_grid.lbm.cavity_presets import (
    DEFAULT_OUTPUT_DIR,
    PRESETS,
    apply_cavity_profile,
    build_lbm_model,
    collide_choices,
    compare_collide_schemes,
    resolve_export_paths,
)
from wanphys._src.fluid.fluid_grid.lbm.vtk_export import export_structured_vtk


def _simulate_and_export(args: argparse.Namespace) -> None:
    model = build_lbm_model(
        FluidGridLbmModel,
        grid_size=args.grid_size,
        cell_size=args.cell_size,
        nu=args.nu,
        collide=args.collide,
        mrt_ghost_s=args.mrt_ghost_s,
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

    print(
        f"Running cavity: {args.grid_size}^3, steps={args.steps}, U_lid={args.u_lid}, "
        f"collide={args.collide}, profile={args.profile}"
    )
    for step in range(args.steps):
        domain.step(dt=1.0)
        if (step + 1) % max(1, args.steps // 5) == 0:
            v_np = domain.state.v.numpy()
            rho_np = domain.state.rho.numpy()
            print(
                f"  step {step + 1}/{args.steps}: max|u|={np.linalg.norm(v_np, axis=-1).max():.6f} "
                f"max|rho-1|={np.abs(rho_np - 1.0).max():.6f}"
            )

    v_np = domain.state.v.numpy()
    rho_np = domain.state.rho.numpy()
    solid_np = domain.state.solid.numpy() if args.with_obstacle else None
    slice_k = args.slice_k if args.slice_k >= 0 else None
    u_2d, v_2d, scalar_2d, nx, ny, k = xy_plane_fields(
        v_np, slice_k=slice_k, scalar=rho_np if args.scalar_field == "rho" else None, scalar_mode=args.scalar_field
    )

    preset = PRESETS.get(args.profile)
    export_speed = args.export_speed or (preset.export_speed if preset else False)
    export_rho = args.export_rho or (preset.export_rho if preset else bool(args.save_slice))
    do_batch = args.export_batch or preset is not None

    def _plot(field: str, path: Path) -> None:
        plot_lid_driven_cavity(
            v_np,
            rho=rho_np if field == "rho" else None,
            solid=solid_np,
            cell_size=args.cell_size,
            slice_k=slice_k,
            scalar_mode=field,
            u_lid=args.u_lid,
            title=f"{args.profile}/{args.collide} ({field}, k={k})",
            path=path,
            show=args.show_plot,
        )
        print(f"Saved PNG: {path.resolve()}")

    if export_speed or args.save_slice:
        png_speed, _, _ = resolve_export_paths(args.output_dir, args.collide, field="speed")
        _plot("speed", Path(args.save_slice or png_speed))
    if export_rho:
        png_rho, _, _ = resolve_export_paths(args.output_dir, args.collide, field="rho")
        _plot("rho", png_rho)

    _, vtk_default, npz_default = resolve_export_paths(args.output_dir, args.collide, field="bundle")
    if args.export_vtk or do_batch:
        out_vtk = Path(args.export_vtk or vtk_default)
        export_structured_vtk(out_vtk, rho_np, v_np, spacing=(args.cell_size,) * 3)
        print(f"Exported VTK: {out_vtk.resolve()}")
    if args.save_numpy or do_batch:
        out_npz = Path(args.save_numpy or npz_default)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_npz, velocity=v_np, rho=rho_np, u_2d=u_2d, v_2d=v_2d, slice_k=k, collide=args.collide, profile=args.profile)
        print(f"Saved NPZ: {out_npz.resolve()}")


def run_cavity_and_export(args: argparse.Namespace) -> None:
    wp.init()
    if args.compare:
        base = args.output_dir
        schemes = compare_collide_schemes()
        if len(schemes) < 2:
            print("Warning: MRT not available; --compare runs BGK only.")
        for collide in schemes:
            run_args = copy.copy(args)
            run_args.output_dir = str(Path(base) / run_args.profile) if run_args.profile != "custom" else base
            run_args.collide = collide
            run_args.save_slice = run_args.export_vtk = run_args.save_numpy = ""
            print(f"\n=== collide: {collide.upper()} | profile: {run_args.profile} ===")
            _simulate_and_export(run_args)
        return
    if args.profile != "custom":
        args.output_dir = str(Path(args.output_dir) / args.profile)
    _simulate_and_export(args)


def _build_parser() -> argparse.ArgumentParser:
    schemes = collide_choices()
    parser = argparse.ArgumentParser(description="Export LBM cavity slice.")
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
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
    parser.add_argument("--slice-k", type=int, default=-1)
    parser.add_argument("--scalar-field", type=str, default="speed", choices=("speed", "ux", "rho"))
    parser.add_argument("--save-slice", type=str, default="")
    parser.add_argument("--export-vtk", type=str, default="")
    parser.add_argument("--save-numpy", type=str, default="")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--with-obstacle", action="store_true")
    parser.add_argument("--obstacle-cx", type=float, default=-1.0)
    parser.add_argument("--obstacle-cy", type=float, default=-1.0)
    parser.add_argument("--obstacle-cz", type=float, default=-1.0)
    parser.add_argument("--obstacle-hx", type=float, default=-1.0)
    parser.add_argument("--obstacle-hy", type=float, default=-1.0)
    parser.add_argument("--obstacle-hz", type=float, default=-1.0)
    return parser


if __name__ == "__main__":
    cli_args = _build_parser().parse_args()
    apply_cavity_profile(cli_args, substeps=1)
    if cli_args.profile != "custom" and cli_args.steps == 200:
        cli_args.steps = PRESETS[cli_args.profile].total_steps
    run_cavity_and_export(cli_args)
