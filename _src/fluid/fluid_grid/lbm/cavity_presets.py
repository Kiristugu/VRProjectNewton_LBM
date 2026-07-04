# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Preset parameter sets for lid-driven cavity BGK/MRT comparison."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ProfileName = Literal["custom", "subtle", "stress", "contrast", "coarse"]
CollideName = Literal["bgk", "trt", "mrt"]

DEFAULT_OUTPUT_DIR: str = "output/cavity_compare"
DEFAULT_LBM_SUBSTEPS: int = 5


@dataclass(frozen=True)
class CavityPreset:
    """Fixed lattice parameters for a cavity comparison profile."""

    nu: float
    u_lid: float
    grid_size: int
    total_steps: int
    export_speed: bool
    export_rho: bool
    description: str


PRESETS: dict[str, CavityPreset] = {
    "subtle": CavityPreset(
        nu=0.16667,
        u_lid=0.1,
        grid_size=64,
        total_steps=800,
        export_speed=True,
        export_rho=True,
        description="Low Re (~38): BGK and MRT should look almost identical.",
    ),
    "stress": CavityPreset(
        nu=0.025,
        u_lid=0.16,
        grid_size=64,
        total_steps=1500,
        export_speed=True,
        export_rho=True,
        description="High Re (~410): small rho differences near corners (~1%).",
    ),
    "contrast": CavityPreset(
        nu=0.01024,
        u_lid=0.42,
        grid_size=128,
        total_steps=2500,
        export_speed=True,
        export_rho=True,
        description="High Re (~5250) on 128^3 grid: U_lid=0.42 for stronger high-Re stress.",
    ),
    "coarse": CavityPreset(
        nu=0.00256,
        u_lid=0.2,
        grid_size=32,
        total_steps=2500,
        export_speed=True,
        export_rho=True,
        description="Low grid (32^3) Re (~2500): BGK often diverges; MRT stays stable.",
    ),
}


def available_collide_schemes() -> tuple[str, ...]:
    """Return collision schemes supported by the installed kernels."""
    from . import kernels

    schemes: list[str] = ["bgk"]
    if hasattr(kernels, "collide_trt"):
        schemes.append("trt")
    if hasattr(kernels, "collide_mrt"):
        schemes.append("mrt")
    return tuple(schemes)


def collide_choices() -> tuple[str, ...]:
    """CLI choices for ``--collide``."""
    return available_collide_schemes()


def build_lbm_model(
    model_cls: type,
    *,
    grid_size: int,
    cell_size: float,
    nu: float,
    collide: str,
    mrt_ghost_s: float = 1.0,
    trt_lambda: float = 0.25,
):
    """Construct ``FluidGridLbmModel`` with optional TRT/MRT fields when present."""
    fields = getattr(model_cls, "__dataclass_fields__", {})
    kwargs: dict = {
        "fluid_grid_res": (grid_size, grid_size, grid_size),
        "fluid_grid_cell_size": cell_size,
        "nu": nu,
        "use_guo_force": False,
    }
    if "collide_impl" in fields:
        kwargs["collide_impl"] = collide
    if "trt_lambda" in fields:
        kwargs["trt_lambda"] = trt_lambda
    if "mrt_ghost_s" in fields:
        kwargs["mrt_ghost_s"] = mrt_ghost_s
    return model_cls(**kwargs)


def apply_cavity_profile(
    args: argparse.Namespace,
    *,
    default_output_dir: str = DEFAULT_OUTPUT_DIR,
    substeps: int = DEFAULT_LBM_SUBSTEPS,
) -> None:
    """Apply ``--profile subtle|stress`` onto argparse namespace."""
    profile: str = getattr(args, "profile", "custom")
    if profile == "custom":
        return

    preset: CavityPreset = PRESETS[profile]
    args.grid_size = preset.grid_size
    args.nu = preset.nu
    args.u_lid = preset.u_lid
    args.num_frames = max(1, preset.total_steps // substeps)

    if getattr(args, "output_dir", "") in ("", default_output_dir):
        args.output_dir = str(Path(default_output_dir) / profile)

    print(f"Profile [{profile}]: {preset.description}")
    print(
        f"  grid={preset.grid_size}, nu={preset.nu}, U_lid={preset.u_lid}, "
        f"steps={preset.total_steps} ({args.num_frames} frames x {substeps})"
    )


def resolve_export_paths(
    output_dir: str | Path,
    collide: str,
    *,
    field: str,
) -> tuple[Path, Path, Path]:
    """Return PNG, VTK, NPZ paths for one collide run."""
    root = Path(output_dir)
    stem: str = f"cavity_{collide}_{field}" if field != "bundle" else f"cavity_{collide}"
    if field == "bundle":
        return (
            root / f"cavity_{collide}_speed.png",
            root / f"cavity_{collide}.vtk",
            root / f"cavity_{collide}.npz",
        )
    png = root / f"{stem}.png"
    vtk = root / f"cavity_{collide}.vtk"
    npz = root / f"cavity_{collide}.npz"
    return png, vtk, npz


def compare_collide_schemes() -> tuple[str, ...]:
    """Schemes to run when ``--compare`` is set."""
    schemes = available_collide_schemes()
    if "mrt" in schemes:
        return ("bgk", "mrt")
    return ("bgk",)
