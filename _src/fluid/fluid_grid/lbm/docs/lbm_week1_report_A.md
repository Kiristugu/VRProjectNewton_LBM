# LBM 模块 — 第 1 周详细解读、进度总结与下阶段计划

> **版本**：1.0（归档）  
> **日期**：2026-06-03  
> **主责**：张弋洋（A — 架构 / 数据 / solver 编排）  
> **依据**：[`DESIGN.md`](../DESIGN.md) v1.2、[`lbm_team_development_plan.md`](../../../../../../../docs/lbm_team_development_plan.md) v1.2、[`开题.md`](../../../../../../../docs/开题.md)  
> **代码路径**：`wanphys/_src/fluid/fluid_grid/lbm/`

> **后续进展（M2 已完成）**：见 [`README.md`](README.md) 及 [`phase1_stream_pull.md`](phase1_stream_pull.md) → [`phase4_visualization.md`](phase4_visualization.md)。本文档保留第 1 周历史记录，不再随代码更新。

---

## 1. 第 1 周目标

第 1 周的核心不是「功能堆满」，而是 **M1：最小静止流体** —— 在 16³ 网格上验证整条 BGK pipeline 的数值正确性。

| 维度 | 第 1 周要求 | 第 1 周不要求 |
|------|------------|--------------|
| 网格 | 16³（调试用） | 50³、完整 Domain |
| 物理 | ρ=1, u=0 平衡态 + BGK 碰撞 | 顶盖驱动、障碍物 |
| 迁移 | 恒等 stream 占位即可 | 正式 Pull streaming |
| 边界 | 空壳 / no-op | Zou–He、bounce-back |
| 集成 | Model + State + Solver 可 import、可跑 | `FluidGridLbmDomain`、fluid_grid 导出 |

### 1.1 M1 量化验收标准

| 检查项 | 标准 |
|--------|------|
| 步数 | 100 步 |
| 速度 | max\|u\| < 10⁻¹⁰ |
| 密度 | max\|ρ−1\| < 10⁻⁶ |
| 质量 | \|Σρ − N_cell\| / N_cell < 10⁻⁶ |
| 数值 | 无 NaN / Inf |

---

## 2. 架构解读

### 2.1 模块分层

```text
wanphys/_src/fluid/fluid_grid/lbm/
├── DESIGN.md          # 接口冻结（三人共同遵守）
├── model.py           # A：FluidGridLbmModel（静态配置）
├── state.py           # A：FluidGridLbmState（运行时数组）
├── lattice.py         # A：D3Q19 常量（e, w, feq, LR）
├── solver.py          # A：FluidGridLbmSolver.step() 五步编排
├── kernels.py         # B/C：Warp kernel（第 1 周为 M1 最小实现）
└── __init__.py        # 导出 Model / State / Solver（Domain 第 2 周）
```

LBM **不继承** `FluidGridMacSolverBase`（压力投影管线），而是继承 `FluidGridSolverBase`，独立实现 collide → stream → BC → macro → swap 流程。与 `liquid/`、`basic_vortex/` 的 MAC 路径是平行关系。

### 2.2 单位制：全程格子单位

| 量 | 约定 |
|----|------|
| Δx, Δt | 1 |
| ν | 0.16667（默认，与 taichi 一致） |
| τ | 3ν + 0.5（ν=0.16667 → τ=1） |
| U_lid | 格子速度（第 2 周 cavity 用 0.1） |

`fluid_grid_cell_size` 第 1–3 周仅文档 / viewer 用，**不参与 LBM 核函数**。

### 2.3 双缓冲：两层含义

第 1 周涉及 **两层** 缓冲，不要混为一谈。

**层 1 — 步内 ping-pong（同一 State 内的 f / F）**

```text
f  = 读缓冲（collide 入口：上一步末的分布函数）
F  = 写缓冲（collide → stream → BC 的中间结果）
macro 后 swap_buffers：f ↔ F 指针交换
→ 交换后 f 持有本步末 canonical 分布，供下一步 collide 读取
```

**层 2 — 步间双缓冲（state_in / state_out）**

```text
solver.step(state_in, state_out)
测试 / 未来 Domain：每步末尾交换 state_in ↔ state_out
```

当前测试里手动做步间 swap；第 2 周 `FluidGridLbmDomain` 会封装这一层。

---

## 3. 各模块详细解读

### 3.1 `FluidGridLbmModel`（model.py）

继承 `FluidGridModelBase`，只存 **静态、时间不变** 的配置：

- `fluid_grid_res`：网格分辨率 (nx, ny, nz)
- `nu`：运动黏度（格子单位，默认 0.16667）
- `rho0`、`force`、`use_guo_force`：初值与体积力（Guo 第 3 周启用）
- `bc_*`：六面边界类型（0=周期，1=压力，2=速度）
- 派生属性：`tau = 3*nu + 0.5`，`omega = 1/tau`（默认 ν=0.16667 时 τ=1）

**设计原则**：τ、ω 由 ν 计算，不单独存数组；f、ρ、v 等运行时量全部在 State 里。

### 3.2 `FluidGridLbmState`（state.py）

继承 `FluidGridStateBase`（复用 `solid_phi` 等基类字段，便于第 3 周烘焙障碍）。

| 字段 | shape | 含义 |
|------|-------|------|
| `f` | (nx, ny, nz, 19) | 读缓冲 / 步末 canonical 分布 |
| `F` | (nx, ny, nz, 19) | 写缓冲 / collide–stream–BC 工作区 |
| `rho` | (nx, ny, nz) | 密度 |
| `v` | (nx, ny, nz) vec3 | 速度 |
| `solid` | (nx, ny, nz) int32 | 0=流体，1=固体 |

实现 `clear_forces()`（CompositeSimulation 契约）、`clear()`、`clone()`，与 Newton/WanPhys 其他 Domain 一致。

### 3.3 `lattice.py`（D3Q19 常量）

- `Q = 19`，`LR`：19 个方向的反向索引（bounce-back 用）
- `@wp.func feq(i, rho, u)`：平衡态分布
- `@wp.func lattice_e_f(i)`：离散速度向量
- `lattice_weight_host()` 等：供 CPU 单测用

与 taichi `static_init()` 中的 e、w 数值一致。

### 3.4 `solver.step()` — 五步 pipeline（对齐 DESIGN §4）

```text
1. collide_bgk        读 state_in.f  → 写 state_out.F
2. stream_pull        改 state_out.F（第 1 周：恒等占位）
3. apply_boundaries   改 state_out.F（第 1 周：no-op）
4. update_macro       读 state_out.F → 写 state_out.rho/v
5. swap_buffers       state_out.f ↔ state_out.F（指针交换）
```

**f / F 语义（已对齐 DESIGN）**

- `update_macro` **只**从 F 计算 ρ、u，**不写 f**
- `_swap_buffers()` 在 macro 之后交换 f/F，使 f 成为步末 canonical 分布
- 这与 DESIGN §4 一致，也与步内「f=读、F=写」的 ping-pong 约定一致

**与 taichi 的差异**：taichi 在 `streaming3()` 里做 `f = F` 拷贝；本实现改为 **指针 swap**，语义等价、零拷贝。

### 3.5 `kernels.py`（第 1 周最小实现）

| Kernel | 负责人 | 第 1 周状态 |
|--------|--------|------------|
| `init_equilibrium` | B | ✅ 已实现 |
| `collide_bgk` | B | ✅ 已实现（BGK，非 taichi MRT） |
| `stream_pull_identity` | B | ⚠️ 占位（恒等，无邻居 pull） |
| `update_macro` | B | ✅ 已实现（只写 ρ/v） |
| `apply_boundaries` | C | ⚠️ 空壳 no-op |

第 1 周在 B/C 未提交前，用最小 kernel 保证 M1 联调可跑；**第 2 周 B/C 按 DESIGN §7.2/§7.3 分区替换，禁止未通知改签名**。

### 3.6 测试与运行环境

| 文件 | 作用 |
|------|------|
| `wanphys/tests/test_lbm_import.py` | model / state / lattice 结构 smoke test |
| `wanphys/tests/test_lbm_rest.py` | M1 静止流 100 步验收 |
| `wanphys/tests/_bootstrap.py` | 绕过 WanPhys 顶层 geometry 与 Warp 1.13 不兼容 |

**运行方式（`.venv_lbm`）**

```powershell
cd d:\Term6\GP\HW\Project
.\.venv_lbm\Scripts\Activate.ps1
cd WanPhys-dev
python wanphys/tests/test_lbm_import.py
python wanphys/tests/test_lbm_rest.py
```

---

## 4. 单步数据流走读

以 M1 静止流为例，初始 ρ=1, u=0，f 和 F 均为平衡态：

```text
state_in.f（平衡态）
    │
    ▼ collide_bgk ──► state_out.F（碰撞后，静止流仍等于 feq）
    │
    ▼ stream（占位，无变化）
    │
    ▼ apply_boundaries（空壳，无变化）
    │
    ▼ update_macro ──► state_out.rho=1, state_out.v=0
    │
    ▼ swap_buffers ──► state_out.f ↔ state_out.F
    │
    ▼ 步间 swap ──► state_in, state_out 指针互换
```

静止流在平衡态下，collide 输出仍等于 feq，100 步后 ρ、u 应保持不变 —— M1 即验证这一点。

---

## 5. 三人分工与第 1 周进度

### 5.1 按角色的完成情况

| 成员 | 第 1 周计划 | 实际状态 |
|------|------------|----------|
| **A（张弋洋）** | model、state、solver、step 联调 | ✅ 已完成 |
| **A** | lattice（DESIGN §6） | ✅ 已完成 |
| **A** | Domain、fluid_grid 导出 | ⏸ 按计划推迟第 2 周 |
| **B（黄彧鸣）** | lattice → init → collide（全网格） | ✅ 最小实现已在 kernels（待 B 正式接管 §7.2） |
| **B** | stream_pull | ⏸ 占位 `stream_pull_identity` |
| **C（彭若扬）** | BC 签名 + stub 测试 | ⚠️ `apply_boundaries` 空壳已有，缺 `test_bc_stub.py` |
| **C** | cavity 骨架（假数据） | ⏸ 未开始 |

### 5.2 里程碑 M1

| 检查项 | 标准 | 状态 |
|--------|------|------|
| 网格 16³ | ✓ | ✅ |
| 100 步静止流 | max\|u\| < 1e-10 | ✅ 测试通过 |
| 密度 | max\|ρ−1\| < 1e-6 | ✅ |
| 质量守恒 | 漂移 < 1e-6 | ✅ |
| 数值稳定 | 无 NaN/Inf | ✅ |

**结论：M1 达标，第 1 周 A 侧核心交付完成。**

### 5.3 已知遗留 / 技术债

1. **`domain.py` 未写**：第 1 周计划内刻意省略；测试用手动 state 双缓冲。
2. **`DESIGN.md §7.2` 签名**：`update_macro` 已去掉 `f` 参数，与冻结文档略有出入 —— 建议第 2 周初三人同步更新 DESIGN 版本号。
3. **C 的 `test_bc_stub.py`、cavity 骨架**：第 1 周 D3/D5 任务，C 侧待补。
4. **测试 bootstrap**：`.venv_lbm` + Warp 1.13 与 WanPhys geometry 不兼容的 workaround，不影响 LBM 模块本身。
5. **stream / BC 为占位**：第 2 周接入真实实现前，只能验静止流，不能验 cavity。

---

## 6. 第 2 周计划（下阶段）

**目标：M2 — 完整 BGK + 顶盖驱动腔（≥32³，500 步，主涡可见）**

### 6.1 A（第 2 周上半周，优先）

| 任务 | 产出 | 说明 |
|------|------|------|
| `FluidGridLbmDomain` | `domain.py` | 参照 `liquid/domain.py`：双缓冲 state + `step(dt)` + swap |
| 导出 API | `lbm/__init__.py` + `fluid_grid/__init__.py` | 四个公开符号 |
| 与 B 联调 | solver 接口稳定 | 不改 §4 步骤顺序 |
| API docstring | 第 3 周可提前起步 | 非阻塞 |

Domain 最小用法（第 2 周末目标）：

```python
from wanphys._src.fluid.fluid_grid.lbm import (
    FluidGridLbmModel,
    FluidGridLbmDomain,
)

model = FluidGridLbmModel(fluid_grid_res=(50, 50, 50), nu=0.16667)
domain = FluidGridLbmDomain(model)
domain.create_state()
domain.solver.init_uniform(domain.state, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))

for _ in range(500):
    domain.step(dt=1.0)
```

### 6.2 B（第 2 周，与 A 并行）

| 任务 | 说明 |
|------|------|
| `stream_pull` | 替换 `stream_pull_identity`；Pull 范式，对齐 taichi `streaming1()` |
| `collide_bgk` 完善 | 固体格点跳过、全网格边界处理 |
| `update_macro` | 确认 Guo 项开关；与 A 的 swap 语义保持一致 |
| 联调 | 与 A 的 step 顺序：collide → stream → BC → macro → swap |

### 6.3 C（第 2 周）

| 任务 | 说明 |
|------|------|
| `bounce_back` + 顶盖速度 BC | 实装 `apply_boundaries`（§7.3） |
| `test_bc_stub.py` | 补第 1 周欠账 |
| `fluid_grid_lbm_cavity.py` | 替换假数据，接真实 BC + B 的 stream |
| 联调 | B 的 stream 就绪后 1–2 天集成 |

### 6.4 第 2 周末验收（M2）

| 检查项 | 标准 |
|--------|------|
| 网格 | ≥ 32³（推荐 50³） |
| 步数 | 500 步 |
| 流场 | 腔体中部可见主涡 |
| 示例 | `fluid_grid_lbm_cavity.py --viewer null` 不崩溃 |
| 顶盖 | max u_x 近顶面 ≈ U_lid 量级 |

### 6.5 建议时间线

```text
周初       A：Domain + create_state/step
           B：stream_pull 对照 taichi streaming1
           C：BC 实装 + cavity 脚本改接真 solver

周中       A+B：stream 接入 solver 联调
           B+C：BC 在 stream 后、macro 前验证

周末       全员：50³ cavity 500 步 → M2 验收
           A：fluid_grid 导出合并
```

---

## 7. 联调约定（第 2 周前必读）

1. **step 顺序冻结**：collide → stream → BC → macro → swap，禁止单方面调整。
2. **f / F 语义**：
   - 步初：f = 读，F = 写
   - macro 只读 F、写 ρ/v
   - swap 在 solver 里做，macro 不写 f
3. **kernels 分区**：B 写 §7.2，C 写 §7.3，改签名先更新 DESIGN 并三人确认。
4. **Pull 不改为 Push**：与 taichi 对照保持一致。
5. **每日 15 min 同步**：只报 blocker，避免 silent 改 `kernels.py`。

---

## 8. 一句话总结

**第 1 周**：在 WanPhys `fluid_grid/lbm` 下搭好 Model / State / Solver / lattice 骨架，实现 DESIGN §4 五步 pipeline（stream/BC 为占位），M1 静止流 16³×100 步数值验收通过；Domain 与 cavity 按计划留到第 2 周。

**第 2 周**：A 补 Domain + 导出，B 上正式 Pull streaming，C 实装 BC + cavity，目标 M2 驱动腔可跑、主涡可见。

---

## 9. 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-06-03 | 第 1 周 A 侧交付解读、M1 进度、第 2 周计划 |
