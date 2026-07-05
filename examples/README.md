# WanPhys Examples

Example demonstrations of WanPhys domains and features.

## Running Examples

### Direct Execution

Run examples directly as Python modules:

```bash
# Rigid body pendulum (interactive OpenGL)
python -m wanphys.examples.rigid_pendulum

# Rigid body falling bodies
python -m wanphys.examples.rigid_falling_bodies

# Rigid body primitive shape stack
python -m wanphys.examples.rigid_basic_shapes

# Rigid bunnies falling into a static box
python -m wanphys.examples.rigid_bunny_in_box

# Cloth flag simulation
python -m wanphys.examples.cloth_flag

# PBF dam break
python -m wanphys.examples.fluid_pbf_dam_break

# DFSPH dam break with rigid coupling
python -m wanphys.examples.fluid_dfsph_dam_break

# WCSPH dam break with rigid coupling
python -m wanphys.examples.fluid_wcsph_dam_break

# Grid-based fluid
python -m wanphys.examples.fluid_grid_basic

# PBF fluid with particle emitter
python -m wanphys.examples.fluid_pbf_emitter_corals

# Point cloud visualization
python -m wanphys.examples.point_cloud_demo

# Sensors
python -m wanphys.examples.sensors.example_sensor_contact
python -m wanphys.examples.sensors.example_sensor_imu
python -m wanphys.examples.sensors.example_sensor_tiled_camera
```

### Viewer Options

Examples using Newton's viewer support multiple backends:

```bash
# OpenGL viewer (default, interactive)
python -m wanphys.examples.rigid_pendulum --viewer gl

# Headless mode (no window)
python -m wanphys.examples.rigid_pendulum --viewer gl --headless

# USD export (for Omniverse, Blender, etc.)
python -m wanphys.examples.rigid_pendulum --viewer usd --output-path pendulum.usd --num-frames 300

# Rerun.io visualization
python -m wanphys.examples.rigid_pendulum --viewer rerun

# Null viewer (benchmarking, no visualization)
python -m wanphys.examples.rigid_pendulum --viewer null --num-frames 100
```

### Command-Line Options

Common options (from `newton.examples.create_parser()`):

- `--device DEVICE` - Warp device (e.g., `cuda:0`, `cpu`)
- `--viewer {gl,usd,rerun,null}` - Viewer type
- `--headless` - Run OpenGL viewer without window
- `--num-frames N` - Number of frames for USD/null viewers
- `--output-path PATH` - Output file for USD viewer

### Smoke Testing

Run all examples in headless mode with output validation:

```bash
uv run python smoke_test_examples.py
uv run python smoke_test_examples.py --pattern rigid    # filter by name
uv run python smoke_test_examples.py --verbose           # show all output
```

## Example Categories

### Rigid Body

- **`rigid_pendulum.py`** - Interactive pendulum with energy tracking and viewer
- **`rigid_falling_bodies.py`** - Multiple bodies falling with collisions
- **`rigid_basic_shapes.py`** - Primitive rigid shapes falling under XPBD or semi-implicit solvers
- **`rigid_bunny_in_box.py`** - Bunny-shaped rigid bodies falling into a static box

### Fluid (Particle-based)

- **`fluid_pbf_dam_break.py`** - PBF dam break simulation
- **`fluid_dfsph_dam_break.py`** - DFSPH dam break with rigid-fluid coupling
- **`fluid_wcsph_dam_break.py`** - WCSPH dam break with rigid-fluid coupling
- **`fluid_pbf_emitter_corals.py`** - PBF fluid with dynamic particle emitter

### Fluid (Grid-based)

- **`fluid_grid_basic.py`** - Grid-based fluid simulation

### LBM (Lattice Boltzmann Method)

- **`fluid_grid_lbm_cavity.py`** - D3Q19-BGK lid-driven cavity flow (M2). Headless, GL volume rendering, VTK/PNG export.
- **`fluid_grid_lbm_obstacle.py`** - Cavity with central box obstacle at three heights (M3绕障).
- **`fluid_grid_lbm_channel_obstacle.py`** - Channel flow past cylinder / square pillar — top-view vortex street validation (M3通道绕柱).
- **`fluid_grid_lbm_particle_vis.py`** - **NEW** Real-time 3D tracer-particle visualization with Rerun. Particles advected through the LBM velocity field, colored by speed (blue→red), revealing vortices and recirculation zones in cavity / obstacle / channel scenarios.

### Cloth

- **`cloth_flag.py`** - Flag simulation with wind forces

### Sensors

- **`example_sensor_contact.py`** - Contact force sensor
- **`example_sensor_imu.py`** - Inertial measurement unit sensor
- **`example_sensor_tiled_camera.py`** - Ray-tracing tiled camera sensor

### Benchmarks

- **`broad_phase_benchmark.py`** - Broad phase collision detection algorithm comparison
- **`rigid_fluid_gated_benchmark.py`** - Rigid-fluid coupling performance benchmark

### Utilities (not directly runnable)

- **`fluid_particle_emitter.py`** - `PlaneEmitter` and `ParticlePoolAllocator` classes, used by emitter examples
- **`utils.py`** - Common helpers (`init_warp`, `setup_viewer`, `SimulationParams`)

## Example Structure

### Viewer-Integrated Examples

Examples with interactive visualization follow this pattern:

```python
import newton.examples
from wanphys.collision import CollisionPipeline
from wanphys.rigid import RigidDomain, RigidModelBuilder, create_xpbd_solver

class Example:
    def __init__(self, viewer, args=None):
        builder = RigidModelBuilder()
        build_scene(builder)
        model = builder.finalize()
        solver = create_xpbd_solver(model)
        self.rigid = RigidDomain(model, solver=solver)
        self.rigid.create_state()
        self.viewer = viewer
        model.setup_viewer(self.viewer)

    def step(self):
        self.rigid.state.clear_forces()
        contacts = CollisionPipeline.collide_rigid(self.rigid)
        self.rigid.step(self.sim_dt, contacts=contacts)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.rigid.state.as_newton_state())
        self.viewer.end_frame()

if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
```

## Contributing Examples

When adding new examples:

1. **Follow naming convention**: `category_description.py` (e.g., `rigid_double_pendulum.py`)
2. **Add docstring** with description and usage
3. **Use WanPhys API** (not direct Newton calls)
4. **Include `if __name__ == "__main__"`** so the smoke test can discover it
5. **Support `--viewer null`** for headless testing
6. **Update this README** under appropriate category

## LBM 3D Particle Visualization — From-Scratch Setup

The `fluid_grid_lbm_particle_vis.py` script requires ``warp-lang``, ``numpy``, and ``rerun-sdk``.  
No GPU is required — Warp runs on CPU when CUDA is unavailable.

### One-shot environment bootstrap

```powershell
# 1. Create virtual environment (requires Python 3.12+)
python -m venv D:\venv_lbm

# 2. Activate
D:\venv_lbm\Scripts\Activate.ps1

# 3. Install dependencies
pip install warp-lang numpy rerun-sdk

# 4. Verify
python -c "import warp as wp; wp.init(); print('Warp OK, device:', wp.get_device())"
python -c "import rerun as rr; print('Rerun OK, version:', rr.__version__)"

# 5. Create wanphys package junction (one-time)
cmd /c "mklink /J D:\Projects\wanphys D:\Projects\VRProjectNewton_LBM"
```

### Run

```powershell
# Activate environment (skip if already active)
D:\venv_lbm\Scripts\Activate.ps1
cd D:\Projects\VRProjectNewton_LBM

# ── Cavity flow (default) ─────────────────────────────────
# Record to .rrd file, then play back (recommended workflow)
python examples/fluid_grid_lbm_particle_vis.py --scenario cavity --grid-size 50 --num-frames 200 --record output/cavity_particles.rrd
python -m rerun output/cavity_particles.rrd

# Live viewer (spawns Rerun window automatically)
python examples/fluid_grid_lbm_particle_vis.py --scenario cavity --grid-size 40 --num-frames 150

# ── Obstacle flow ─────────────────────────────────────────
python examples/fluid_grid_lbm_particle_vis.py --scenario obstacle --grid-size 50 --num-frames 200 --record output/obstacle_particles.rrd

# ── Channel flow (Kármán vortex street!) ──────────────────
python examples/fluid_grid_lbm_particle_vis.py --scenario channel --grid-size 64 --num-frames 400 --u-inlet 0.08 --record output/channel_particles.rrd

# ── Fine detail (more particles, larger grid) ─────────────
python examples/fluid_grid_lbm_particle_vis.py --scenario cavity --grid-size 64 --particle-count 10000 --num-frames 300 --record output/cavity_fine.rrd
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--scenario` | `cavity` | `cavity` / `obstacle` / `channel` |
| `--grid-size` | `50` | Cubic grid resolution (higher = finer but slower) |
| `--particle-count` | `5000` | Number of tracer particles |
| `--num-frames` | `200` | Output frames |
| `--lbm-substeps` | `5` | LBM steps per frame |
| `--warmup-steps` | `0` | LBM steps before particles appear |
| `--nu` | `0.16667` | Kinematic viscosity (lattice units) |
| `--u-lid` | `0.1` | Lid velocity (cavity/obstacle scenarios) |
| `--u-inlet` | `0.06` | Inlet velocity (channel scenario) |
| `--particle-max-life` | `200` | Max particle lifetime in LBM steps |
| `--record` | *(empty)* | Path to save .rrd recording (disables live viewer) |
| `--rerun-addr` | *(empty)* | Connect to remote rerun viewer (e.g. `127.0.0.1:9876`) |

### Rerun Viewer Controls

| Action | Control |
|---|---|
| Rotate | Left-drag |
| Pan | Middle-drag or Shift+Left-drag |
| Zoom | Scroll wheel |
| Timeline scrub | Bottom timeline slider |
| Play / Pause | Space |
| Reset view | Double-click 3D view |

### Headless (Non-Visual) LBM Examples

For validation without visualization:

```powershell
# M2 cavity smoke test
python examples/fluid_grid_lbm_cavity.py --viewer null --num-frames 100 --test --grid-size 50

# M2 + VTK export for ParaView
python examples/fluid_grid_lbm_cavity.py --viewer null --num-frames 100 --test --export-vtk output/cavity.vtk

# M2 + matplotlib streamplot PNG
python examples/fluid_grid_lbm_cavity.py --viewer null --num-frames 100 --test --save-slice output/stream.png

# M3 obstacle (all three heights)
python examples/fluid_grid_lbm_obstacle.py --viewer null --grid-size 64 --num-frames 100 --test --obstacle-height all

# M3 channel obstacle (cylinder + box)
python examples/fluid_grid_lbm_channel_obstacle.py --viewer null --num-frames 600 --test --obstacle-mode both
```

### Run Tests

```powershell
cd D:\Projects\VRProjectNewton_LBM

python tests/test_lbm_rest.py          # M1 rest fluid
python tests/test_equilibrium.py        # Equilibrium distribution
python tests/test_cavity_smoke.py       # M2 cavity smoke
python tests/test_obstacle_smoke.py     # M3 obstacle smoke
python tests/test_lbm_vtk_export.py     # VTK export
python tests/test_lbm_import.py         # Import / unit tests
```

## Troubleshooting

### Viewer won't open

- Check OpenGL support: `python -m wanphys.examples.rigid_pendulum --viewer gl`
- Use headless mode: `--headless`
- Try null viewer: `--viewer null`

### CUDA errors

- Switch to CPU: `--device cpu`
- Check CUDA installation

### Import errors

```bash
uv sync --extra dev --extra examples
```

### LBM / Rerun-specific

| Symptom | Fix |
|---|---|
| `No module named 'wanphys'` | Create the junction: `cmd /c "mklink /J D:\Projects\wanphys D:\Projects\VRProjectNewton_LBM"` |
| `No module named 'warp'` | `pip install warp-lang` |
| `No module named 'rerun'` | `pip install rerun-sdk` |
| Rerun window won't open | Use `--record output/xxx.rrd`, then view with `python -m rerun output/xxx.rrd` |
| Simulation is very slow | Reduce `--grid-size` (e.g. 32), reduce `--particle-count` (e.g. 2000), or use `--device cpu` explicitly |
| `Kernel compile timeout` on CPU | First run compiles Warp kernels (~30s – 2min); subsequent runs use cache and are instant |
