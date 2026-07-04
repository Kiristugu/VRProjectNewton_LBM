# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Matplotlib cavity visualization (contourf + streamplot, lid_driven_cavity style)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for cavity plots (pip install matplotlib)") from exc
    return plt


def xy_plane_fields(
    velocity: np.ndarray,
    *,
    slice_k: int | None = None,
    scalar: np.ndarray | None = None,
    scalar_mode: str = "speed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract x-y plane fields at fixed z index (classic lid-driven cavity view).

    WanPhys indexing: ``velocity[i, j, k, comp]`` with lid on ``j = ny - 1``, ``u_x`` driven.

    Returns mesh ``X, Y`` and ``U, V, scalar_2d`` shaped ``(ny, nx)`` for matplotlib.
    """
    if velocity.ndim != 4 or velocity.shape[3] != 3:
        raise ValueError(f"velocity must have shape (nx, ny, nz, 3), got {velocity.shape}")

    nx: int = int(velocity.shape[0])
    ny: int = int(velocity.shape[1])
    nz: int = int(velocity.shape[2])
    k: int = (nz // 2) if slice_k is None else int(slice_k)
    if k < 0 or k >= nz:
        raise ValueError(f"slice_k={k} out of range for nz={nz}")

    u_2d: np.ndarray = velocity[:, :, k, 0].T
    v_2d: np.ndarray = velocity[:, :, k, 1].T

    if scalar is not None:
        if scalar.shape != velocity.shape[:3]:
            raise ValueError(f"scalar shape {scalar.shape} incompatible with velocity {velocity.shape[:3]}")
        scalar_2d: np.ndarray = scalar[:, :, k].T
    elif scalar_mode == "speed":
        scalar_2d = np.linalg.norm(velocity[:, :, k, :], axis=-1).T
    elif scalar_mode == "ux":
        scalar_2d = u_2d.copy()
    else:
        raise ValueError(f"unknown scalar_mode={scalar_mode!r}")

    return u_2d, v_2d, scalar_2d, nx, ny, k


def plot_lid_driven_cavity(
    velocity: np.ndarray,
    *,
    rho: np.ndarray | None = None,
    solid: np.ndarray | None = None,
    cell_size: float = 1.0,
    slice_k: int | None = None,
    scalar_mode: str = "speed",
    cmap: str = "jet",
    n_contour_levels: int = 50,
    u_lid: float | None = None,
    title: str | None = None,
    path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
    figsize: tuple[float, float] = (7.0, 6.0),
) -> None:
    """Plot x-y mid-plane with scalar contourf and velocity streamlines.

    Mimics the style of classic finite-difference ``lid_driven_cavity.py`` examples:
    colored scalar background + black streamlines on the vertical cavity plane.
    """
    plt = _require_matplotlib()

    u_2d, v_2d, scalar_2d, nx, ny, k = xy_plane_fields(
        velocity,
        slice_k=slice_k,
        scalar=rho,
        scalar_mode=scalar_mode,
    )

    domain_x: float = nx * cell_size
    domain_y: float = ny * cell_size
    x: np.ndarray = np.linspace(0.0, domain_x, nx)
    y: np.ndarray = np.linspace(0.0, domain_y, ny)
    x_mesh, y_mesh = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=figsize)
    contour = ax.contourf(x_mesh, y_mesh, scalar_2d, levels=n_contour_levels, cmap=cmap)
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    if rho is not None:
        cbar.set_label("rho")
    elif scalar_mode == "ux":
        cbar.set_label("u_x")
    else:
        cbar.set_label("|u|")

    try:
        ax.streamplot(
            x_mesh,
            y_mesh,
            u_2d,
            v_2d,
            color="black",
            linewidth=0.8,
            density=1.2,
            arrowsize=0.9,
        )
    except ValueError:
        step: int = max(1, min(nx, ny) // 24)
        ax.quiver(
            x_mesh[::step, ::step],
            y_mesh[::step, ::step],
            u_2d[::step, ::step],
            v_2d[::step, ::step],
            color="black",
            angles="xy",
            scale_units="xy",
            scale=None,
            width=0.003,
        )

    ax.set_xlim(0.0, domain_x)
    ax.set_ylim(0.0, domain_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if title is None:
        title = f"Lid-driven cavity (x-y @ k={k})"
        if u_lid is not None:
            title += f", U_lid={u_lid:g}"
    ax.set_title(title)

    if solid is not None:
        if solid.shape != velocity.shape[:3]:
            raise ValueError(f"solid shape {solid.shape} incompatible with velocity {velocity.shape[:3]}")
        solid_2d: np.ndarray = (solid[:, :, k] > 0).T.astype(float)
        if np.any(solid_2d > 0.0):
            ax.contourf(
                x_mesh,
                y_mesh,
                solid_2d,
                levels=[0.5, 1.5],
                colors=["#444444"],
                alpha=0.92,
                zorder=3,
            )

    lid_y: float = domain_y
    ax.axhline(lid_y, color="white", linewidth=1.2, linestyle="--", alpha=0.85)
    ax.text(0.02 * domain_x, 0.97 * domain_y, "moving lid ->", color="white", fontsize=9, va="top")

    fig.tight_layout()

    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi)
        print(f"Saved cavity streamplot: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def xz_plane_fields(
    velocity: np.ndarray,
    solid: np.ndarray | None = None,
    *,
    slice_j: int | None = None,
    scalar: np.ndarray | None = None,
    scalar_mode: str = "speed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int, int, int]:
    """Extract x-z plane fields at fixed y (top view through cavity depth).

    Returns ``ux_2d, uz_2d, scalar_2d, solid_2d`` shaped ``(nz, nx)``, plus ``nx, nz, j``.
    """
    if velocity.ndim != 4 or velocity.shape[3] != 3:
        raise ValueError(f"velocity must have shape (nx, ny, nz, 3), got {velocity.shape}")

    nx: int = int(velocity.shape[0])
    ny: int = int(velocity.shape[1])
    nz: int = int(velocity.shape[2])
    j: int = (ny // 2) if slice_j is None else int(slice_j)
    if j < 0 or j >= ny:
        raise ValueError(f"slice_j={j} out of range for ny={ny}")

    ux_2d: np.ndarray = velocity[:, j, :, 0].T
    uz_2d: np.ndarray = velocity[:, j, :, 2].T

    if scalar is not None:
        if scalar.shape != velocity.shape[:3]:
            raise ValueError(f"scalar shape {scalar.shape} incompatible with velocity {velocity.shape[:3]}")
        scalar_2d: np.ndarray = scalar[:, j, :].T
    elif scalar_mode == "speed":
        scalar_2d = np.linalg.norm(velocity[:, j, :, :], axis=-1).T
    elif scalar_mode == "ux":
        scalar_2d = ux_2d.copy()
    else:
        raise ValueError(f"unknown scalar_mode={scalar_mode!r}")

    solid_2d: np.ndarray | None = None
    if solid is not None:
        if solid.shape != velocity.shape[:3]:
            raise ValueError(f"solid shape {solid.shape} incompatible with velocity {velocity.shape[:3]}")
        solid_2d = (solid[:, j, :] > 0).T

    return ux_2d, uz_2d, scalar_2d, solid_2d, nx, nz, j


def plot_cavity_top_view(
    velocity: np.ndarray,
    *,
    solid: np.ndarray | None = None,
    cell_size: float = 1.0,
    slice_j: int | None = None,
    scalar_mode: str = "speed",
    rho: np.ndarray | None = None,
    u_lid: float | None = None,
    title: str | None = None,
    path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
    figsize: tuple[float, float] = (7.0, 6.0),
) -> None:
    """Plot x-z top view with scalar contourf and (u_x, u_z) streamlines."""
    plt = _require_matplotlib()

    scalar: np.ndarray | None = rho if scalar_mode == "rho" and rho is not None else None
    ux_2d, uz_2d, scalar_2d, solid_2d, nx, nz, j = xz_plane_fields(
        velocity,
        solid,
        slice_j=slice_j,
        scalar=scalar,
        scalar_mode=scalar_mode,
    )

    domain_x: float = nx * cell_size
    domain_z: float = nz * cell_size
    x: np.ndarray = np.linspace(0.0, domain_x, nx)
    z: np.ndarray = np.linspace(0.0, domain_z, nz)
    x_mesh, z_mesh = np.meshgrid(x, z)

    fig, ax = plt.subplots(figsize=figsize)
    contour = ax.contourf(x_mesh, z_mesh, scalar_2d, levels=50, cmap="jet")
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    if rho is not None and scalar_mode == "rho":
        cbar.set_label("rho")
    elif scalar_mode == "ux":
        cbar.set_label("u_x")
    else:
        cbar.set_label("|u|")

    ux_plot: np.ndarray = ux_2d if solid_2d is None else np.where(solid_2d, np.nan, ux_2d)
    uz_plot: np.ndarray = uz_2d if solid_2d is None else np.where(solid_2d, np.nan, uz_2d)
    try:
        ax.streamplot(
            x_mesh,
            z_mesh,
            ux_plot,
            uz_plot,
            color="black",
            linewidth=0.8,
            density=1.2,
            arrowsize=0.9,
        )
    except ValueError:
        step: int = max(1, min(nx, nz) // 24)
        ax.quiver(
            x_mesh[::step, ::step],
            z_mesh[::step, ::step],
            ux_plot[::step, ::step],
            uz_plot[::step, ::step],
            color="black",
            angles="xy",
            scale_units="xy",
            scale=None,
            width=0.003,
        )

    if solid_2d is not None and np.any(solid_2d):
        ax.contourf(
            x_mesh,
            z_mesh,
            solid_2d.astype(float),
            levels=[0.5, 1.5],
            colors=["#444444"],
            alpha=0.92,
            zorder=3,
        )

    ax.set_xlim(0.0, domain_x)
    ax.set_ylim(0.0, domain_z)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("z")

    if title is None:
        title = f"Lid-driven cavity top view (x-z @ j={j})"
        if u_lid is not None:
            title += f", U_lid={u_lid:g}"
    ax.set_title(title)

    fig.tight_layout()

    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi)
        print(f"Saved cavity top view: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def obstacle_slice_indices(
    center: tuple[float, float, float],
    half: tuple[float, float, float],
    grid_size: int,
) -> dict[str, int]:
    """Lattice slice indices at obstacle bottom / mid / top for y and z."""
    _, cy, cz = center
    _, hy, hz = half
    n: int = grid_size - 1

    def _clip_k(v: float) -> int:
        return int(np.clip(int(round(v - 0.5)), 0, n))

    return {
        "j_bottom": _clip_k(cy - hy),
        "j_mid": _clip_k(cy),
        "j_top": _clip_k(cy + hy),
        "k_bottom": _clip_k(cz - hz),
        "k_mid": _clip_k(cz),
        "k_top": _clip_k(cz + hz),
    }


def export_obstacle_cavity_figures(
    velocity: np.ndarray,
    solid: np.ndarray,
    *,
    output_dir: str | Path,
    center: tuple[float, float, float],
    half: tuple[float, float, float],
    grid_size: int,
    cell_size: float = 1.0,
    u_lid: float | None = None,
    height_mode: str = "mid",
    scalar_mode: str = "speed",
    rho: np.ndarray | None = None,
) -> None:
    """Write side (x-y) and top (x-z) streamplots at obstacle bottom/mid/top slices."""
    out: Path = Path(output_dir)
    slices: dict[str, int] = obstacle_slice_indices(center, half, grid_size)
    height_label: str = height_mode.upper()

    side_specs: tuple[tuple[str, str, int], ...] = (
        ("stream_xy_z_bottom", "bottom", slices["k_bottom"]),
        ("stream_xy_z_mid", "mid", slices["k_mid"]),
        ("stream_xy_z_top", "top", slices["k_top"]),
    )
    for stem, label, k in side_specs:
        plot_lid_driven_cavity(
            velocity,
            rho=rho if scalar_mode == "rho" else None,
            solid=solid,
            cell_size=cell_size,
            slice_k=k,
            scalar_mode=scalar_mode,
            u_lid=u_lid,
            title=f"{height_label} obstacle side view (x-y @ k={k}, z {label})",
            path=out / f"{stem}.png",
        )

    top_specs: tuple[tuple[str, str, int], ...] = (
        ("stream_xz_y_bottom", "bottom", slices["j_bottom"]),
        ("stream_xz_y_mid", "mid", slices["j_mid"]),
        ("stream_xz_y_top", "top", slices["j_top"]),
    )
    for stem, label, j in top_specs:
        plot_cavity_top_view(
            velocity,
            solid=solid,
            cell_size=cell_size,
            slice_j=j,
            scalar_mode=scalar_mode,
            rho=rho,
            u_lid=u_lid,
            title=f"{height_label} obstacle top view (x-z @ j={j}, y {label})",
            path=out / f"{stem}.png",
        )
