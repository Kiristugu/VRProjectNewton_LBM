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

## LBM 3D Particle Visualization (Rerun)

**Branch:** `Rerun` · **Script:** `wanphys/examples/fluid_grid_lbm_particle_vis.py`

Tracer particles are advected through the LBM velocity field (RK2) and logged to
[Rerun](https://rerun.io) as colored 3D points (speed: blue→red). Scenarios:
`cavity` / `obstacle` / `channel` — same physics as the headless LBM demos, but
**real-time 3D** instead of matplotlib PNG / ParaView VTK.

### Dependencies

```powershell
cd D:\Term6\GP\HW\Project\WanPhys-dev
..\.venv_lbm\Scripts\Activate.ps1   # from WanPhys-dev/

pip install "pyarrow>=18" "rerun-sdk>=0.34"
python -c "import rerun as rr; print('Rerun OK:', rr.__version__)"
```

Requires `warp-lang` and `numpy` (already in `.venv_lbm`). CPU-only is fine.

### Run (from `WanPhys-dev/`)

**Live viewer** — spawns the Rerun window automatically:

```powershell
cd D:\Term6\GP\HW\Project\WanPhys-dev

# Cavity (default)
python wanphys/examples/fluid_grid_lbm_particle_vis.py `
  --scenario cavity --grid-size 50 --num-frames 200

# Obstacle (box in cavity center)
python wanphys/examples/fluid_grid_lbm_particle_vis.py `
  --scenario obstacle --grid-size 64 --num-frames 300

# Channel (Kármán vortex street)
python wanphys/examples/fluid_grid_lbm_particle_vis.py `
  --scenario channel --grid-size 64 --num-frames 400 --u-inlet 0.08
```

**Record to `.rrd`** (recommended for demos / slow GPUs) — then replay offline:

```powershell
python wanphys/examples/fluid_grid_lbm_particle_vis.py `
  --scenario cavity --grid-size 50 --num-frames 200 `
  --record output/cavity_particles.rrd

rerun output/cavity_particles.rrd
# or: python -m rerun output/cavity_particles.rrd
```

**Connect to an already-running Rerun server** (optional):

```powershell
# Terminal 1
rerun

# Terminal 2
python wanphys/examples/fluid_grid_lbm_particle_vis.py `
  --scenario cavity --rerun-addr 127.0.0.1:9876
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--scenario` | `cavity` | `cavity` / `obstacle` / `channel` |
| `--grid-size` | `50` | Cubic grid resolution |
| `--particle-count` | `5000` | Tracer particles |
| `--num-frames` | `200` | Output frames |
| `--lbm-substeps` | `5` | LBM steps per frame |
| `--warmup-steps` | `0` | LBM steps before particles appear |
| `--nu` | `0.16667` | Kinematic viscosity (lattice units) |
| `--u-lid` | `0.1` | Lid velocity (cavity / obstacle) |
| `--u-inlet` | `0.06` | Inlet velocity (channel) |
| `--particle-max-life` | `200` | Max particle lifetime (LBM steps) |
| `--record` | *(empty)* | Save `.rrd` file (no live window) |
| `--rerun-addr` | *(empty)* | e.g. `127.0.0.1:9876` — connect to external viewer |

### Rerun viewer controls

| Action | Control |
|---|---|
| Rotate | Left-drag |
| Pan | Middle-drag or Shift+Left-drag |
| Zoom | Scroll wheel |
| Timeline | Bottom slider |
| Play / Pause | Space |

### Headless LBM examples (no Rerun)

For PNG / VTK validation without visualization:

```powershell
cd D:\Term6\GP\HW\Project\WanPhys-dev

python wanphys/examples/fluid_grid_lbm_cavity.py `
  --viewer null --num-frames 300 --test --grid-size 50

python wanphys/examples/fluid_grid_lbm_obstacle.py `
  --viewer null --num-frames 300 --test --obstacle-height all

python wanphys/examples/fluid_grid_lbm_channel_obstacle.py `
  --viewer null --num-frames 300 --test --obstacle-mode both
```

### LBM tests

```powershell
cd D:\Term6\GP\HW\Project\WanPhys-dev
python wanphys/tests/test_lbm_rest.py
python wanphys/tests/test_cavity_smoke.py
python wanphys/tests/test_obstacle_smoke.py
```

### LBM / Rerun troubleshooting

| Symptom | Fix |
|---|---|
| `No module named 'rerun'` | `pip install "pyarrow>=18" "rerun-sdk>=0.34"` |
| `ResolutionImpossible` installing rerun-sdk | Do **not** use bare `pip install rerun-sdk` — old versions need `numpy<2`. Pin: `pip install "pyarrow>=18" "rerun-sdk>=0.34"` |
| Rerun window does not open | Use `--record output/xxx.rrd`, then `rerun output/xxx.rrd` |
| Very slow | Lower `--grid-size` (32–40) or `--particle-count` (2000) |
| First run slow on CPU | Warp kernel compile (~30s–2min); later runs use cache |

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

| `No module named 'rerun'` | `pip install "pyarrow>=18" "rerun-sdk>=0.34"` |
| `ResolutionImpossible` installing rerun-sdk | Do **not** use bare `pip install rerun-sdk` — old versions need `numpy<2`. Pin: `pip install "pyarrow>=18" "rerun-sdk>=0.34"` |
| Rerun window won't open | Use `--record output/xxx.rrd`, then `rerun output/xxx.rrd` |
| Simulation is very slow | Reduce `--grid-size` or `--particle-count` |
| `Kernel compile timeout` on CPU | First run compiles Warp kernels; subsequent runs use cache |
