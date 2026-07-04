# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""VTK export for LBM structured grid fields (ParaView / VisIt)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_structured_vtk(
    path: str | Path,
    rho: np.ndarray,
    velocity: np.ndarray,
    *,
    solid: np.ndarray | None = None,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    title: str = "WanPhys LBM",
    mask_solid_velocity: bool = True,
) -> None:
    """Write legacy VTK STRUCTURED_POINTS with rho, speed, velocity, and optional solid.

    Args:
        path: Output file path (``.vtk`` suffix recommended).
        rho: Density array with shape ``(nx, ny, nz)``.
        velocity: Velocity array with shape ``(nx, ny, nz, 3)``.
        solid: Optional solid mask ``(nx, ny, nz)`` (non-zero = solid).
        origin: Grid origin in world/lattice coordinates.
        spacing: Cell spacing per axis.
        title: VTK file description line.
        mask_solid_velocity: Zero velocity inside solid cells before export.
    """
    rho_arr: np.ndarray = np.ascontiguousarray(rho, dtype=np.float64)
    vel_arr: np.ndarray = np.ascontiguousarray(velocity, dtype=np.float64)
    if rho_arr.ndim != 3:
        raise ValueError(f"rho must be 3D, got shape {rho_arr.shape}")
    if vel_arr.shape[:3] != rho_arr.shape or vel_arr.shape[3] != 3:
        raise ValueError(f"velocity shape {vel_arr.shape} incompatible with rho {rho_arr.shape}")

    solid_arr: np.ndarray | None = None
    if solid is not None:
        solid_arr = np.ascontiguousarray(solid, dtype=np.float64)
        if solid_arr.shape != rho_arr.shape:
            raise ValueError(f"solid shape {solid_arr.shape} incompatible with rho {rho_arr.shape}")
        if mask_solid_velocity:
            fluid_mask: np.ndarray = solid_arr == 0
            vel_arr = vel_arr.copy()
            vel_arr[~fluid_mask] = 0.0

    nx, ny, nz = rho_arr.shape
    n_points: int = nx * ny * nz

    rho_flat: np.ndarray = rho_arr.ravel(order="F")
    ux_flat: np.ndarray = vel_arr[..., 0].ravel(order="F")
    uy_flat: np.ndarray = vel_arr[..., 1].ravel(order="F")
    uz_flat: np.ndarray = vel_arr[..., 2].ravel(order="F")
    speed_flat: np.ndarray = np.linalg.norm(vel_arr, axis=-1).ravel(order="F")
    solid_flat: np.ndarray | None = solid_arr.ravel(order="F") if solid_arr is not None else None

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="ascii") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{title}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n")
        f.write(f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\n")
        f.write(f"POINT_DATA {n_points}\n")

        f.write("SCALARS rho float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in rho_flat:
            f.write(f"{val}\n")

        f.write("SCALARS speed float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in speed_flat:
            f.write(f"{val}\n")

        if solid_flat is not None:
            f.write("SCALARS solid float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for val in solid_flat:
                f.write(f"{val}\n")

        f.write("VECTORS velocity float\n")
        for ix in range(n_points):
            f.write(f"{ux_flat[ix]} {uy_flat[ix]} {uz_flat[ix]}\n")
