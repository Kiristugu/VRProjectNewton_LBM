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
