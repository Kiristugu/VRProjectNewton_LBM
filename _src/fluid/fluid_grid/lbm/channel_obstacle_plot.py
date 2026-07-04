# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Post-processing plots for channel flow past infinite-height obstacles (top view + profiles)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

CS2: float = 1.0 / 3.0


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required (pip install matplotlib)") from exc
    return plt


def xz_plane_fields(
    velocity: np.ndarray,
    solid: np.ndarray,
    *,
    slice_j: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Extract x-z plane fields at fixed y (top view at mid-channel height).

    Returns ``ux_2d, uz_2d, speed_2d, solid_2d`` shaped ``(nz, nx)`` for matplotlib,
    plus ``nx, nz, j``.
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
    speed_2d: np.ndarray = np.linalg.norm(velocity[:, j, :, :], axis=-1).T
    solid_2d: np.ndarray = (solid[:, j, :] > 0).T
    return ux_2d, uz_2d, speed_2d, solid_2d, nx, nz, j


def plot_top_view_streamlines(
    velocity: np.ndarray,
    solid: np.ndarray,
    *,
    cell_size: float = 1.0,
    slice_j: int | None = None,
    u_in: float | None = None,
    title: str | None = None,
    path: str | Path | None = None,
    dpi: int = 160,
) -> None:
    """Figure 1: top view (x-z @ y=ny/2) — |u| background + streamlines of (u_x, u_z)."""
    plt = _require_matplotlib()

    ux_2d, uz_2d, speed_2d, solid_2d, nx, nz, j = xz_plane_fields(velocity, solid, slice_j=slice_j)

    masked_speed: np.ndarray = np.ma.masked_where(solid_2d, speed_2d)
    x: np.ndarray = np.linspace(0.0, nx * cell_size, nx)
    z: np.ndarray = np.linspace(0.0, nz * cell_size, nz)
    xm, zm = np.meshgrid(x, z)

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    cf = ax.contourf(xm, zm, masked_speed, levels=60, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="|u|")

    ux_plot: np.ndarray = np.where(solid_2d, np.nan, ux_2d)
    uz_plot: np.ndarray = np.where(solid_2d, np.nan, uz_2d)
    try:
        ax.streamplot(xm, zm, ux_plot, uz_plot, color="k", linewidth=0.7, density=1.4, arrowsize=0.8)
    except ValueError:
        step: int = max(1, nx // 48)
        ax.quiver(
            xm[::step, ::step],
            zm[::step, ::step],
            ux_plot[::step, ::step],
            uz_plot[::step, ::step],
            color="k",
            angles="xy",
            scale_units="xy",
            scale=None,
            width=0.002,
        )

    if np.any(solid_2d):
        ax.contourf(xm, zm, solid_2d.astype(float), levels=[0.5, 1.5], colors=["#333333"], alpha=0.95)

    ax.set_xlim(0.0, nx * cell_size)
    ax.set_ylim(0.0, nz * cell_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (flow)")
    ax.set_ylabel("z")
    if title is None:
        title = f"Top view x-z @ y=j{j}"
        if u_in is not None:
            title += f", U_in={u_in:g}"
    ax.set_title(title)
    fig.tight_layout()

    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi)
        print(f"Saved figure 1 (top view streamlines): {out}")
    plt.close(fig)


def plot_slice_ux_uz(
    velocity: np.ndarray,
    solid: np.ndarray,
    *,
    slice_j: int | None = None,
    slice_k: int | None = None,
    cell_size: float = 1.0,
    u_in: float | None = None,
    title: str | None = None,
    path: str | Path | None = None,
    dpi: int = 160,
) -> None:
    """Figure 2: u_x and u_z along x at fixed (j, k) on the top-view slice plane."""
    plt = _require_matplotlib()

    nx: int = int(velocity.shape[0])
    ny: int = int(velocity.shape[1])
    nz: int = int(velocity.shape[2])
    j: int = (ny // 2) if slice_j is None else int(slice_j)
    k: int = (nz // 2) if slice_k is None else int(slice_k)

    ux_line: np.ndarray = velocity[:, j, k, 0]
    uz_line: np.ndarray = velocity[:, j, k, 2]
    solid_line: np.ndarray = solid[:, j, k] > 0
    x: np.ndarray = (np.arange(nx) + 0.5) * cell_size

    fig, (ax_u, ax_uz) = plt.subplots(2, 1, figsize=(9.0, 5.5), sharex=True)

    ax_u.plot(x, ux_line, "b-", linewidth=1.5, label=r"$u_x$")
    ax_u.axhline(0.0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
    if u_in is not None:
        ax_u.axhline(u_in, color="gray", linewidth=0.8, linestyle=":", label=f"U_in={u_in:g}")
    solid_x: np.ndarray = x[solid_line]
    if solid_x.size > 0:
        ax_u.axvspan(float(solid_x[0]), float(solid_x[-1]), color="#cccccc", alpha=0.5, label="obstacle")
    ax_u.set_ylabel(r"$u_x$")
    ax_u.legend(loc="best", fontsize=9)
    ax_u.grid(True, alpha=0.3)

    ax_uz.plot(x, uz_line, "r-", linewidth=1.5, label=r"$u_z$")
    ax_uz.axhline(0.0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
    if solid_x.size > 0:
        ax_uz.axvspan(float(solid_x[0]), float(solid_x[-1]), color="#cccccc", alpha=0.5)
    ax_uz.set_xlabel("x (flow)")
    ax_uz.set_ylabel(r"$u_z$")
    ax_uz.legend(loc="best", fontsize=9)
    ax_uz.grid(True, alpha=0.3)

    if title is None:
        title = f"$u_x$, $u_z$ along x (j={j}, k={k})"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi)
        print(f"Saved figure 2 (ux, uz profiles): {out}")
    plt.close(fig)


def _neighbor_fluid_rho(rho: np.ndarray, solid: np.ndarray, i: int, j: int, k: int) -> float | None:
    nx, ny, nz = rho.shape
    vals: list[float] = []
    for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz and solid[ni, nj, nk] == 0:
            vals.append(float(rho[ni, nj, nk]))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def sample_cylinder_surface_cp(
    rho: np.ndarray,
    solid: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    radius: float,
    u_in: float,
    rho_ref: float | None = None,
    slice_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cp vs polar angle theta on a z-aligned cylinder (theta=0 is downstream stagnation... front is pi)."""
    nz: int = int(rho.shape[2])
    k: int = (nz // 2) if slice_k is None else int(slice_k)
    nx, ny = int(rho.shape[0]), int(rho.shape[1])
    u2: float = max(u_in * u_in, 1.0e-12)

    if rho_ref is None:
        j_ref: int = int(np.clip(round(center_y - 0.5), 0, ny - 1))
        rho_ref = float(np.mean(rho[4:12, j_ref, k]))

    thetas: list[float] = []
    cps: list[float] = []
    for i in range(nx):
        for j in range(ny):
            if solid[i, j, k] == 0:
                continue
            dx = (i + 0.5) - center_x
            dy = (j + 0.5) - center_y
            if dx * dx + dy * dy > (radius + 0.6) ** 2 or dx * dx + dy * dy < (radius - 0.6) ** 2:
                continue
            rho_n = _neighbor_fluid_rho(rho, solid, i, j, k)
            if rho_n is None:
                continue
            p: float = CS2 * rho_n
            p_ref: float = CS2 * rho_ref
            cp: float = (p - p_ref) / (0.5 * u2)
            theta: float = float(np.arctan2(dy, dx))
            thetas.append(theta)
            cps.append(cp)

    if not thetas:
        return np.array([]), np.array([])

    order: np.ndarray = np.argsort(thetas)
    return np.asarray(thetas)[order], np.asarray(cps)[order]


def sample_box_surface_cp(
    rho: np.ndarray,
    solid: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    half_x: float,
    half_y: float,
    u_in: float,
    rho_ref: float | None = None,
    slice_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cp vs arc parameter s along a box wetted surface (front -> top -> back -> bottom)."""
    nz: int = int(rho.shape[2])
    k: int = (nz // 2) if slice_k is None else int(slice_k)
    nx, ny = int(rho.shape[0]), int(rho.shape[1])
    u2: float = max(u_in * u_in, 1.0e-12)
    x0, x1 = center_x - half_x, center_x + half_x
    y0, y1 = center_y - half_y, center_y + half_y

    if rho_ref is None:
        j_ref: int = int(np.clip(round(center_y - 0.5), 0, ny - 1))
        rho_ref = float(np.mean(rho[4:12, j_ref, k]))

    samples: list[tuple[float, float]] = []
    for i in range(nx):
        for j in range(ny):
            if solid[i, j, k] == 0:
                continue
            cx = i + 0.5
            cy = j + 0.5
            on_surface = (
                (abs(cx - x0) < 0.6 or abs(cx - x1) < 0.6 or abs(cy - y0) < 0.6 or abs(cy - y1) < 0.6)
                and (x0 - 0.5 <= cx <= x1 + 0.5)
                and (y0 - 0.5 <= cy <= y1 + 0.5)
            )
            if not on_surface:
                continue
            rho_n = _neighbor_fluid_rho(rho, solid, i, j, k)
            if rho_n is None:
                continue
            p: float = CS2 * rho_n
            p_ref: float = CS2 * rho_ref
            cp: float = (p - p_ref) / (0.5 * u2)
            if abs(cx - x0) < 0.6:
                s_param = cy - y0
            elif abs(cy - y1) < 0.6:
                s_param = (y1 - y0) + (cx - x0)
            elif abs(cx - x1) < 0.6:
                s_param = 2 * (y1 - y0) + (y1 - cy)
            else:
                s_param = 3 * (y1 - y0) + (x1 - cx)
            samples.append((s_param, cp))

    if not samples:
        return np.array([]), np.array([])

    arr = np.asarray(samples)
    order = np.argsort(arr[:, 0])
    return arr[order, 0], arr[order, 1]


def export_channel_validation_figures(
    velocity: np.ndarray,
    solid: np.ndarray,
    *,
    output_dir: str | Path,
    u_in: float,
    cell_size: float = 1.0,
    slice_j: int | None = None,
    slice_k: int | None = None,
) -> None:
    """Write top-view streamlines and u_x/u_z profile figures under ``output_dir``."""
    out = Path(output_dir)
    plot_top_view_streamlines(
        velocity,
        solid,
        cell_size=cell_size,
        slice_j=slice_j,
        u_in=u_in,
        path=out / "fig1_top_view_streamlines.png",
    )
    plot_slice_ux_uz(
        velocity,
        solid,
        slice_j=slice_j,
        slice_k=slice_k,
        cell_size=cell_size,
        u_in=u_in,
        path=out / "fig2_slice_ux_uz.png",
    )
