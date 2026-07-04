# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""LBM particle-flow 3D visualization with Rerun.

Real-time 3D tracer particles advected through the LBM velocity field,
rendered in the Rerun viewer. Particles are colored by speed (blue→red)
and velocity direction, clearly revealing vortices and recirculation zones.

Scenarios
---------

**Cavity (default):**  Lid-driven cavity flow with tracer particles showing
the primary vortex, secondary corner eddies, and 3D Taylor-Görtler rolls.

**Obstacle:**  Cavity with a central box obstacle — particles wrap around
the blockage, exposing the recirculation zones behind it.

**Channel:**  Channel flow past a cylinder/square pillar — particles trace
the Kármán vortex street and wake.

Quick Start
-----------

.. code-block:: powershell

    # Activate environment
    D:\\venv_lbm\\Scripts\\Activate.ps1
    cd D:\\Projects\\VRProjectNewton_LBM

    # Cavity (default 50^3)
    python examples/fluid_grid_lbm_particle_vis.py --scenario cavity --grid-size 40 --num-frames 200

    # Obstacle — box in cavity center
    python examples/fluid_grid_lbm_particle_vis.py --scenario obstacle --grid-size 50 --num-frames 200

    # Channel — flow past cylinder
    python examples/fluid_grid_lbm_particle_vis.py --scenario channel --grid-size 60 --num-frames 400 --u-inlet 0.08

    # Larger simulation with more particles
    python examples/fluid_grid_lbm_particle_vis.py --scenario cavity --grid-size 64 --particle-count 8000 --num-frames 300

    # Record to file for later playback
    python examples/fluid_grid_lbm_particle_vis.py --scenario cavity --grid-size 50 --num-frames 200 --record output/cavity_particles.rrd

Requirements
------------
- ``warp-lang``, ``numpy``: core LBM simulation
- ``rerun-sdk``: 3D visualization (``pip install rerun-sdk``)
- CPU-only is fine (Warp runs on CPU when no CUDA driver available)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import warp as wp

# ── Bootstrap LBM imports (same pattern as other LBM examples) ──────────────


def _bootstrap_lbm_imports() -> None:
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
        name, init_path, submodule_search_locations=[str(lbm_dir)],
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
from wanphys._src.fluid.fluid_grid.lbm.particle_tracer import (  # noqa: E402
    advect_particles_rk2,
    compute_particle_speeds,
    reseed_dead_particles,
    seed_particles_uniform,
)

# ── Color maps ──────────────────────────────────────────────────────────────


def _speed_colormap(t: float) -> tuple[float, float, float]:
    """Map normalized speed [0,1] → blue→cyan→yellow→red."""
    t = max(0.0, min(1.0, t))
    if t < 0.33:
        s = t / 0.33
        return (0.1 * (1 - s) + 1.0 * s, 0.3 * (1 - s) + 0.85 * s, 0.8 * (1 - s) + 0.15 * s)
    elif t < 0.66:
        s = (t - 0.33) / 0.33
        return (1.0, 0.85 * (1 - s) + 0.95 * s, 0.15 * (1 - s) + 0.1 * s)
    else:
        s = (t - 0.66) / 0.34
        return (1.0, 0.95 * (1 - s) + 0.1 * s, 0.1 * (1 - s) + 0.05 * s)


def _speed_colors_rgba(speeds: np.ndarray, max_speed: float) -> np.ndarray:
    """Convert speed array to RGBA colors (N×4 uint8)."""
    normalized = np.clip(speeds / max(max_speed, 1e-8), 0.0, 1.0)
    colors = np.zeros((len(normalized), 4), dtype=np.uint8)
    for i, t in enumerate(normalized):
        r, g, b = _speed_colormap(float(t))
        colors[i, 0] = int(r * 255)
        colors[i, 1] = int(g * 255)
        colors[i, 2] = int(b * 255)
        colors[i, 3] = 200  # alpha
    return colors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LBM particle-flow 3D visualization with Rerun",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenario", type=str, default="cavity",
                        choices=("cavity", "obstacle", "channel"),
                        help="Simulation scenario (default: cavity)")
    parser.add_argument("--grid-size", type=int, default=50,
                        help="Cubic grid resolution (default: 50)")
    parser.add_argument("--nu", type=float, default=0.16667,
                        help="Kinematic viscosity in lattice units (default: 0.16667)")
    parser.add_argument("--u-lid", type=float, default=0.1,
                        help="Lid velocity for cavity/obstacle (default: 0.1)")
    parser.add_argument("--u-inlet", type=float, default=0.06,
                        help="Inlet velocity for channel (default: 0.06)")
    parser.add_argument("--num-frames", type=int, default=200,
                        help="Number of output frames (default: 200)")
    parser.add_argument("--lbm-substeps", type=int, default=5,
                        help="LBM steps per frame (default: 5)")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="LBM steps before showing particles (default: 0)")
    parser.add_argument("--particle-count", type=int, default=5000,
                        help="Number of tracer particles (default: 5000)")
    parser.add_argument("--particle-max-life", type=float, default=200.0,
                        help="Max particle lifetime in LBM steps (default: 200)")
    parser.add_argument("--record", type=str, default="",
                        help="Record to .rrd file instead of live viewer")
    parser.add_argument("--rerun-addr", type=str, default="",
                        help="Connect to running rerun viewer (e.g. 127.0.0.1:9876)")
    return parser


# ── Build LBM domain per scenario ───────────────────────────────────────────


def _build_domain(args: argparse.Namespace):
    """Create and initialize an LBM domain for the selected scenario."""
    gs = args.grid_size
    model = FluidGridLbmModel(
        fluid_grid_res=(gs, gs, gs),
        fluid_grid_cell_size=1.0,
        nu=args.nu,
    )
    domain = FluidGridLbmDomain(model)
    state = domain.create_state()

    if args.scenario in ("cavity", "obstacle"):
        domain.solver.configure_cavity_walls()
        domain.solver.set_lid_velocity(wp.vec3(float(args.u_lid), 0.0, 0.0))
        domain.solver.init_uniform(state, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))

        if args.scenario == "obstacle":
            half = gs // 2
            hw = max(2, gs // 10)
            y_top = int(gs * 0.6)
            domain.bake_box(
                wp.vec3(float(half), float(y_top // 2), float(half)),
                wp.vec3(float(hw), float(y_top // 2), float(hw)),
            )

    elif args.scenario == "channel":
        domain.solver.configure_channel_flow(float(args.u_inlet))
        domain.solver.init_uniform_flow(state, rho=1.0, u=wp.vec3(float(args.u_inlet), 0.0, 0.0))
        # Cylinder obstacle at 1/4 from inlet
        cy = gs // 2
        cx = gs // 4
        radius = max(2, gs // 16)
        domain.solver.bake_cylinder_y(state, float(cx), float(cy), float(radius))

    return domain, state


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    wp.init()
    parser = _build_parser()
    args = parser.parse_args()

    gs = args.grid_size
    n_particles = args.particle_count
    print(f"Scenario: {args.scenario}  grid={gs}^3  particles={n_particles}")

    # ── Setup LBM simulation ────────────────────────────────────────────
    domain, state = _build_domain(args)
    device = domain.model._device

    # ── Setup Rerun ─────────────────────────────────────────────────────
    import rerun as rr

    if args.record:
        rr.init("LBM Particle Flow", recording_id=None, spawn=False)
        rr.save(args.record)
        print(f"Recording to {args.record}")
    elif args.rerun_addr:
        addr_parts = args.rerun_addr.rsplit(":", 1)
        host = addr_parts[0]
        port = int(addr_parts[1]) if len(addr_parts) > 1 else 9876
        rr.init("LBM Particle Flow", default_enabled=True)
        rr.connect_grpc(url=f"rerun+http://{host}:{port}/proxy")
        print(f"Connecting to rerun viewer at {host}:{port}")
    else:
        rr.init("LBM Particle Flow", spawn=True)
        print("Spawning rerun viewer...")

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Domain bounding box
    bbox_half = gs * 0.5
    rr.log(
        "world/domain",
        rr.Boxes3D(
            centers=[[bbox_half, bbox_half, bbox_half]],
            half_sizes=[[bbox_half, bbox_half, bbox_half]],
            colors=[[128, 128, 128]],
        ),
        static=True,
    )

    # ── Allocate particle buffers ───────────────────────────────────────
    particles = wp.zeros(n_particles, dtype=wp.vec3, device=device)
    particle_life = wp.full(n_particles, -1.0, dtype=wp.float32, device=device)
    particle_speeds = wp.zeros(n_particles, dtype=wp.float32, device=device)
    particle_colors = wp.zeros(n_particles, dtype=wp.vec3, device=device)

    solid = state.solid
    seed_counter = 0

    # ── Warmup ──────────────────────────────────────────────────────────
    if args.warmup_steps > 0:
        print(f"Warming up: {args.warmup_steps} LBM steps...")
        for _ in range(args.warmup_steps):
            domain.step(dt=1.0)

    print(f"Running {args.num_frames} frames × {args.lbm_substeps} substeps...")
    print("Open the Rerun viewer to see the 3D particle animation.")

    total_steps = 0

    for frame in range(args.num_frames):
        # Advance LBM
        for _ in range(args.lbm_substeps):
            domain.step(dt=1.0)
            total_steps += 1

        # Advect particles through the current velocity field
        v_field = domain.state.v
        particle_count = n_particles

        # Seed initial batch
        if frame == 0:
            wp.launch(
                seed_particles_uniform,
                dim=n_particles,
                inputs=[
                    particles, particle_life, n_particles,
                    gs, gs, gs, solid, 42 + seed_counter,
                ],
                device=device,
            )
            seed_counter += 1

        wp.launch(
            advect_particles_rk2,
            dim=n_particles,
            inputs=[
                particles, particle_life, n_particles,
                v_field, solid, gs, gs, gs,
                float(args.lbm_substeps),
                float(args.particle_max_life),
            ],
            device=device,
        )

        # Reseed dead particles
        wp.launch(
            reseed_dead_particles,
            dim=n_particles,
            inputs=[
                particles, particle_life, n_particles,
                gs, gs, gs, solid, 42 + seed_counter,
            ],
            device=device,
        )
        seed_counter += 1

        # Compute particle colors from velocity
        wp.launch(
            compute_particle_speeds,
            dim=n_particles,
            inputs=[
                particles, particle_life, n_particles,
                v_field, gs, gs, gs,
                particle_speeds, particle_colors,
            ],
            device=device,
        )

        # ── Log to Rerun ──────────────────────────────────────────────
        rr.set_time("sim_time", sequence=total_steps)

        # Filter living particles
        life_np = particle_life.numpy().flatten()
        alive = life_np >= 0.0
        n_alive = int(np.sum(alive))

        if n_alive > 0:
            pos_np = particles.numpy().flatten().reshape(-1, 3)[alive]  # (n_alive, 3)
            speeds_np = particle_speeds.numpy().flatten()[alive]
            max_speed = float(np.max(speeds_np)) if n_alive > 0 else 0.1

            colors_rgba = _speed_colors_rgba(speeds_np, max(max_speed, 0.05))

            rr.log(
                "world/particles",
                rr.Points3D(
                    positions=pos_np,
                    colors=colors_rgba,
                    radii=0.3 * np.ones(n_alive, dtype=np.float32),
                ),
            )

        # Velocity arrows on mid-plane (y = ny/2)
        v_np = domain.state.v.numpy()
        mid_y = gs // 2
        stride = max(2, gs // 16)
        arrow_origins = []
        arrow_vectors = []
        for i in range(0, gs, stride):
            for k in range(0, gs, stride):
                vel = v_np[i, mid_y, k]
                speed = float(np.linalg.norm(vel))
                if speed > 1e-5:
                    origin = np.array([float(i) + 0.5, float(mid_y) + 0.5, float(k) + 0.5])
                    arrow_origins.append(origin)
                    arrow_vectors.append(vel * 1.5 / max(speed, 1e-6))

        if arrow_origins:
            origins = np.array(arrow_origins)
            vectors = np.array(arrow_vectors)
            rr.log(
                "world/velocity_arrows",
                rr.Arrows3D(origins=origins, vectors=vectors, colors=[[255, 200, 50]]),
            )

        # Log obstacle solid cells (once)
        if frame == 0:
            solid_np = solid.numpy().flatten()
            solid_idx = np.where(solid_np > 0)[0]
            if len(solid_idx) > 0:
                n_solid = len(solid_idx)
                solid_pos = np.zeros((min(n_solid, 50000), 3), dtype=np.float32)
                for idx_ in range(min(n_solid, 50000)):
                    flat_idx = solid_idx[idx_]
                    i = flat_idx // (gs * gs)
                    rem = flat_idx % (gs * gs)
                    j = rem // gs
                    k = rem % gs
                    solid_pos[idx_] = [float(i) + 0.5, float(j) + 0.5, float(k) + 0.5]
                rr.log(
                    "world/obstacle",
                    rr.Points3D(
                        positions=solid_pos,
                        colors=[[180, 180, 180]],
                        radii=0.25 * np.ones(len(solid_pos), dtype=np.float32),
                    ),
                    static=True,
                )

        if frame % 20 == 0:
            v_np = domain.state.v.numpy()
            max_u = float(np.max(np.linalg.norm(v_np, axis=-1)))
            print(f"  frame {frame:4d}/{args.num_frames}  steps={total_steps:5d}  "
                  f"alive={n_alive:5d}  max|u|={max_u:.5f}")

    print(f"\nDone. Total LBM steps: {total_steps}")
    if args.record:
        print(f"Recording saved to: {args.record}")
        print("Open with:  rerun {args.record}")
    else:
        print("Close the Rerun viewer window to exit, or press Ctrl+C.")
        try:
            import time
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

    print("Exiting.")


if __name__ == "__main__":
    main()
