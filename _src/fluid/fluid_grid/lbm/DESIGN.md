# LBM 模块设计文档（WanPhys `fluid_grid/lbm`）

> **版本**：1.2  
> **状态**：接口冻结（第 1 周 D1 评审，**允许 TODO**；第 2 周前半冻结 state/domain）  
> **作业依据**：`newton.txt` — 基于 Newton 1.1 / WanPhys 的 D3Q19 + BGK 单相流  
> **算法参考**：`taichi_LBM3D/Single_phase/LBM_3D_SinglePhase_Solver.py`  
> **框架参考**：`fluid_grid/ARCHITECTURE.md`、`basic_vortex/`、`liquid/`

---

## 1. 文档目的

- 冻结 **Model / State / Solver / Domain / kernels** 的字段与函数签名，避免三人并行改接口冲突。
- 明确 LBM **不** 走 `FluidGridMacSolverBase` 压力投影管线。
- 与组内教程 `docs/lbm_tutorial/` 及开发计划 `docs/lbm_team_development_plan.md` 对齐。

---

## 2. 模块定位

| 项目 | 说明 |
|------|------|
| 路径 | `wanphys/_src/fluid/fluid_grid/lbm/` |
| 域名称 | `FluidGridLbmDomain.name == "fluid_grid_lbm"` |
| 基类 | `FluidGridSolverBase`（**非** `FluidGridMacSolverBase`） |
| 格子 | D3Q19，$Q=19$ |
| 碰撞 | 第 1–3 周：BGK；MRT **不纳入 4 周排期**（future work） |
| 迁移 | **Pull**（与 taichi `streaming1` 一致），见 §4.1 |
| 单位 | **全程格子单位**：$\Delta x = \Delta t = 1$，$\tau = \nu/3 + 0.5$；与 taichi 对照前禁止混用物理单位 |

### 2.1 与现有 `fluid_grid` 子模块对比

| 维度 | `basic_vortex` | `liquid` | **`lbm`（本模块）** |
|------|----------------|----------|---------------------|
| 主状态 | MAC 速度 + 密度标量 | 粒子 + MAC | **分布函数 $f_i$** |
| 不可压 | 压力投影 | 压力投影 | **碰撞松弛隐含** |
| `step()` | 基类 MAC Hook | 自定义 FLIP 流程 | **自定义 LBM 流程** |
| 障碍物 | `solver.solid_phi` | `state.solid_phi` | **`state.solid` / `state.solid_phi`** |

---

## 3. 目录与文件职责

```text
wanphys/_src/fluid/fluid_grid/lbm/
├── DESIGN.md          # 本文档
├── __init__.py        # 导出 FluidGridLbm*
├── lattice.py         # D3Q19：e, w, feq, LR；常量，无 GPU 状态
├── model.py           # FluidGridLbmModel
├── state.py           # FluidGridLbmState
├── solver.py          # FluidGridLbmSolver.step()
├── domain.py          # FluidGridLbmDomain
└── kernels.py         # Warp kernels（碰撞、迁移、宏观、BC）
```

**示例（由成员 C 维护）**

```text
wanphys/examples/
├── fluid_grid_lbm_cavity.py      # 顶盖驱动腔（必做）
└── fluid_grid_lbm_obstacle.py    # 绕障流（第 3 周）
```

---

## 4. 单步算法顺序（冻结）

与 `taichi_LBM3D` 的 `LB3D_Solver_Single_Phase.step()` 对齐，**禁止**在未开会的情况下调整顺序。

```text
step(state_in, state_out, dt):
  1. collide_bgk        # f → F（碰撞后写入 F）
  2. stream_pull        # F 沿离散速度迁移（pull）
  3. apply_boundaries   # 六面 BC + 固体 bounce-back
  4. update_macro       # 由 F 得 ρ, u；Guo 外力（若启用）
  5. swap_buffers       # f ↔ F，双缓冲交换
```

对应 taichi：

| 步骤 | taichi | 本模块 kernel |
|------|--------|----------------|
| 1 | `colission()` | `collide_bgk` |
| 2 | `streaming1()` | `stream_pull` |
| 3 | `Boundary_condition()` | `apply_boundaries` |
| 4 | `streaming3()` | `update_macro` |

### 4.1 Streaming：Pull（已选型，勿改 Push）

| 范式 | 操作 | 本模块 |
|------|------|--------|
| **Pull** | 本格 $i$ 从上游格点**读取**进入方向 $i$ 的分布 | **采用** — `F[x][i] = f[x - e_i][i]`（固体用 bounce-back 分支） |
| Push | 本格向邻居**写出** | **不采用** |

选用 Pull 的原因：

1. 与参考实现 `taichi_LBM3D` 的 `streaming1()` 一致，对照成本低。  
2. 天然配合 **双缓冲** `f`（读）/ `F`（写），避免原地覆盖竞态。  

第 1 周 minimal 实现可暂用「collide 后 `f=F` 拷贝 + 恒等 stream」占位，**第 2 周上半周**必须换成正式 `stream_pull`。

### 4.2 第 1 周 minimal `step()`（降级）

第 1 周不要求 `FluidGridLbmDomain` 与完整双缓冲，允许：

```text
minimal_step (16³):
  init_equilibrium → collide_bgk → (可选) 无操作 stream 占位 → update_macro（若已写）
```

**M1 验收**见开发计划 §4：$\max\|u\|<10^{-10}$，$\max|\rho-1|<10^{-6}$，总质量漂移 $<10^{-6}$。

---

## 5. 类与字段定义

### 5.1 `FluidGridLbmModel`（`model.py`）

继承 `FluidGridModelBase`，仅静态配置。

```python
@dataclass
class FluidGridLbmModel(FluidGridModelBase):
    nu: float = 0.16667              # 运动黏度（格子单位）
    rho0: float = 1.0                # 初始均匀密度
    force: tuple[float, float, float] = (0.0, 0.0, 0.0)  # 体积力 (fx,fy,fz)
    use_guo_force: bool = True       # 是否在 update_macro 中加 Guo 项

    # 六面边界类型：0=周期, 1=压力(Zou-He ρ), 2=速度(Zou-He u)
    bc_x_left: int = 0
    bc_x_right: int = 0
    # ... bc_y_*, bc_z_* 同理

    # 边界给定值（按 bc 类型选用）
    bc_rho: float = 1.0
    bc_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def tau(self) -> float:
        return self.nu / 3.0 + 0.5

    @property
    def omega(self) -> float:
        return 1.0 / self.tau
```

| 字段 | 放入 model？ | 说明 |
|------|-------------|------|
| `fluid_grid_res`, `fluid_grid_cell_size` | ✓（基类） | 网格尺寸 |
| `f`, `rho`, `v` | ✗ | 运行时 → `state` |
| `tau`, `omega` | 派生属性 | 由 `nu` 计算 |

### 5.2 `FluidGridLbmState`（`state.py`）

继承 `FluidGridStateBase`（复用 `solid_phi` 以便与 `base_kernels` 烘焙障碍兼容）。

#### 分阶段交付（避免第 1 周阻塞）

| 阶段 | 时间 | 内容 |
|------|------|------|
| **Minimal** | 第 1 周 | `f,F,rho,v` 可在 `solver` 内分配；固定 `nx,ny,nz`；无 `domain` |
| **Full** | 第 2 周上半周 | 独立 `FluidGridLbmState` + `FluidGridLbmDomain` + 双缓冲 swap |

```python
class FluidGridLbmState(FluidGridStateBase):
    # LBM 主数组
    f: wp.array4d          # (nx, ny, nz, Q) 当前分布函数
    F: wp.array4d          # (nx, ny, nz, Q) 碰撞/迁移缓冲
    rho: wp.array3d       # (nx, ny, nz)
    v: wp.array(dtype=wp.vec3, shape=(nx, ny, nz))
    solid: wp.array3d(dtype=wp.int32)  # 0=流体, 1=固体（与 taichi solid 一致）

    def clear_forces(self) -> None: ...   # CompositeSimulation 契约，可为 pass
    def clear(self) -> None: ...          # 重置为均匀平衡态
    def clone(self) -> FluidGridLbmState: ...
```

**`FluidGridStateBase` 字段在本模块的用途**

| 基类字段 | LBM 用途 |
|----------|----------|
| `vel_u/v/w` | 可选：由 `v` 同步供 viewer；非求解主变量 |
| `pressure` | 可由 $\rho c_s^2$ 导出，第 1 周可不写 |
| `density` | 可选复用为 $\rho$ 的别名；优先用 `rho` |
| `solid_phi` | 障碍 SDF；`solid` 由 `solid_phi < 0` 烘焙 |

### 5.3 `FluidGridLbmSolver`（`solver.py`）

```python
class FluidGridLbmSolver(FluidGridSolverBase):
    def __init__(self, model: FluidGridLbmModel) -> None: ...

    def step(
        self,
        state_in: FluidGridLbmState,
        state_out: FluidGridLbmState,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
    ) -> None:
        """见 §4 顺序；state_out 为写缓冲。"""

  # 场景 API（成员 C）
    def init_uniform(self, state: FluidGridLbmState, rho: float, u: wp.vec3) -> None: ...
    def set_lid_velocity(self, u_lid: wp.vec3) -> None: ...
    def bake_box(self, state: FluidGridLbmState, center: wp.vec3, half_extents: wp.vec3) -> None: ...
    def bake_sphere(self, state: FluidGridLbmState, center: wp.vec3, radius: float) -> None: ...
```

### 5.4 `FluidGridLbmDomain`（`domain.py`）

与 `basic_vortex/domain.py` 相同模式：双缓冲 `create_state()` + `step()` 内 swap。

```python
class FluidGridLbmDomain(Domain):
    name: str = "fluid_grid_lbm"
```

---

## 6. `lattice.py` 常量（成员 A）

### 6.1 D3Q19 离散速度

- `Q: int = 19`
- `e[i]: wp.vec3i` — 19 个整数格点速度（与 taichi `self.e` 一致）
- `e_f[i]: wp.vec3` — 浮点形式，用于 $f^{eq}$ 中 $\mathbf{e}_i\cdot\mathbf{u}$
- `w[i]: float` — 权重 $w_i$

### 6.2 平衡态

```python
@wp.func
def feq(i: int, rho: float, u: wp.vec3) -> float:
    """f_i^eq = w_i * rho * (1 + 3 e·u + 4.5 (e·u)^2 - 1.5 |u|^2)"""
```

### 6.3 反弹索引

```python
LR: tuple[int, ...] = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17)
```

**MRT 不纳入 4 周实现**。若框架需预留扩展，在 `solver` 留 `collide_impl: str = "bgk"` 即可；`M`, `inv_M` 见 taichi 第 64–131 行，答辩作 future work。

---

## 7. `kernels.py` 函数签名（成员 B / C）

所有 kernel 使用 `wp.tid()` 三维索引 `(i, j, k)`，launch 维度 `(nx, ny, nz)`。

### 7.1 初始化

| 函数 | 签名 | 负责人 |
|------|------|--------|
| `init_equilibrium` | `(f, rho, u, feq_table...)` | B |

### 7.2 核心（成员 B）

```python
@wp.kernel
def collide_bgk(
    f: wp.array4d(dtype=float),
    F: wp.array4d(dtype=float),
    rho: wp.array3d(dtype=float),
    v: wp.array(dtype=wp.vec3),
    solid: wp.array3d(dtype=wp.int32),
    omega: float,
): ...

@wp.kernel
def stream_pull(
    f: wp.array4d(dtype=float),
    F: wp.array4d(dtype=float),
    solid: wp.array3d(dtype=wp.int32),
    nx: int, ny: int, nz: int,
    # e, LR 通过常量或 wp.constant 传入
): ...

@wp.kernel
def update_macro(
    F: wp.array4d(dtype=float),
    f: wp.array4d(dtype=float),
    rho: wp.array3d(dtype=float),
    v: wp.array(dtype=wp.vec3),
    solid: wp.array3d(dtype=wp.int32),
    force: wp.vec3,
    use_guo: int,
): ...
```

### 7.3 边界（成员 C）

**第 1 周**：提交函数签名 + `tests/test_bc_stub.py`（可用假 `rho/u` 断言接口可调）。  
**第 2 周**：接入真实 `F`/`f`，与 B 的 `stream_pull` 联调。

```python
@wp.kernel
def bounce_back_solid(...): ...   # 可先单独实现、单测

@wp.kernel
def apply_boundaries(
    F: wp.array4d(dtype=float),
    f: wp.array4d(dtype=float),
    rho: wp.array3d(dtype=float),
    v: wp.array(dtype=wp.vec3),
    solid: wp.array3d(dtype=wp.int32),
    # bc 参数打包为标量 / 小数组
): ...
```

边界模式（与 taichi 一致）：

| `bc_*` | 含义 |
|--------|------|
| 0 | 周期（由 stream 处理） |
| 1 | 固定密度 / 压力型 Zou–He |
| 2 | 固定速度（顶盖等） |

固体：`solid[i,j,k]!=0` 时 bounce-back，`F[i][LR[s]] = f[i][s]`。

---

## 8. 稳定性与参数

| 检查 | 公式 / 建议                                                             |
|------|---------------------------------------------------------------------|
| $\tau$ | $\tau = 3\nu + 0.5 > 0.5$                                           |
| 低黏度 | $\nu$ 过小 → 增大网格或改用 MRT                                              |
| CFL（格子单位） | 通常 $\|\mathbf{u}\| \lesssim 0.1$ 较稳；顶盖 $U_{lid} \le 0.1\sim0.15$ 起步 |
| 监控 | 每 N 步 `max(                                                         |rho-1|)`, `max(|u|)` |

---

## 9. 测试与验收

### 9.1 单元 / 回归

| 测试 | 通过标准 |
|------|----------|
| `test_lattice_weights` | $\sum w_i = 1$，19 方向与 taichi 一致 |
| `test_equilibrium` | $\sum_i f_i^{eq} = \rho$，$\sum_i \mathbf{e}_i f_i^{eq} = \rho\mathbf{u}$ |
| `test_rest_fluid`（M1） | 100 步，16³：$\max\|\mathbf{u}\|<10^{-10}$，$\max|\rho-1|<10^{-6}$，质量漂移 $<10^{-6}$ |
| `test_rest_fluid`（回归） | 1000 步，32³+：$\max|\rho-1|<10^{-3}$ |
| `test_cavity_qualitative`（M2） | 500 步后腔体中部可见主涡 |

建议路径：`WanPhys-dev/wanphys/tests/test_lbm_*.py`（第 1 周：`test_bc_stub`；第 2 周起：`test_rest_fluid`）。

### 9.2 与 taichi 对照

**前置**：双方确认 **格子单位**（$\Delta x=\Delta t=1$），`nu=0.16667`，禁止用 `fluid_grid_cell_size` 换算速度后直接比。

1. 静止流：100 步，$\max\|\mathbf{u}\|\approx 0$（M1 级）  
2. 腔体：网格 $50^3$，$\nu=0.16667$，顶盖 $u_x=0.1$（格子速度）  
3. 对比 500 / 1000 步的 $\max\|\mathbf{u}\|$ 与涡心位置（**定性**为主，允许 10–20% 偏差）  
4. 记录于 `docs/lbm_team_development_plan.md` §9

---

## 10. 导出与集成

### 10.1 `lbm/__init__.py`

```python
from .model import FluidGridLbmModel
from .state import FluidGridLbmState
from .solver import FluidGridLbmSolver
from .domain import FluidGridLbmDomain

__all__ = [
    "FluidGridLbmModel",
    "FluidGridLbmState",
    "FluidGridLbmSolver",
    "FluidGridLbmDomain",
]
```

### 10.2 `fluid_grid/__init__.py`（第 2 周合并）

增加上述四个符号到 `__all__`。

### 10.3 示例最小用法

```python
from wanphys._src.fluid.fluid_grid.lbm import (
    FluidGridLbmModel,
    FluidGridLbmDomain,
)

model = FluidGridLbmModel(fluid_grid_res=(50, 50, 50), nu=0.16667)
domain = FluidGridLbmDomain(model)
state = domain.create_state()
domain.solver.init_uniform(state, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))

for _ in range(1000):
    domain.step(dt=1.0)
```

---

## 11. 分工与 PR 约定（垂直切分）

| 成员 | 第 1 周 | 第 2 周 |
|------|---------|---------|
| **A** | `model` + **minimal** `f,F`（可放 solver）；`step` 联调 | 正式 `state` + `domain` + `__init__` 导出 |
| **B** | `lattice` → `init_equilibrium` → `collide_bgk`（先 16³） | `stream_pull` → `update_macro` → Guo（可选） |
| **C** | BC **签名** + stub 测试；`cavity.py` **假数据骨架** | 真实 BC + cavity 与 B 联调 |

- **PR#1（minimal）**：第 1 周 D4–D5，含 M1；**不要求**完整 `domain`  
- **PR#2（full pipeline）**：第 2 周中，含 M2  
- 分支：`team/lbm-<feature>` → 组内集成分支  
- **禁止**修改 `newton/`、`basic_vortex/`、`liquid/`（除非助教要求同步上游）  
- 修改 §4 步骤顺序或 Pull→Push → 更新版本号 + 三人确认  

---

## 12. 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-05-26 | 初版：WanPhys `fluid_grid/lbm` 接口冻结 |
| 1.1 | 2026-05-26 | 开发计划对齐为 4 周周期 |
| 1.2 | 2026-05-26 | 第 1 周 minimal state；Pull 选型；M1 量化；C 并行 BC stub；MRT 移出排期 |

---

## 13. 相关文档

- 理论：`docs/lbm_tutorial/`（尤其 03–08、11 章）  
- 开发计划：`docs/lbm_team_development_plan.md`（**4 周**）  
- 框架：`fluid_grid/ARCHITECTURE.md`  
- 作业：`newton.txt`  
