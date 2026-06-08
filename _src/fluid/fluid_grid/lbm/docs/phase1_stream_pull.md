# 阶段 1：正式 Pull Streaming

> **版本**：1.1  
> **日期**：2026-06-08  
> **目标**：M2 前置 — 将 `stream_pull_identity` 替换为正式 `stream_pull`，并保持 M1 静止流回归不退化  
> **依据**：[`DESIGN.md`](../DESIGN.md) §4.1、§7.2；[`lbm_team_development_plan.md`](../../../../../../../docs/lbm_team_development_plan.md) 第 2 周

---

## 1. 范围

| 包含 | 不包含（阶段 2+） |
|------|------------------|
| `periodic_index` / `wrap_index` | `apply_boundaries` 真实 BC |
| `stream_pull` kernel | `FluidGridLbmDomain` |
| `solver.step` 数据流修正 | `fluid_grid_lbm_cavity.py` |
| M1 回归（`test_lbm_rest.py`） | 腔体定性验收 |

---

## 2. taichi 参考与已知错误

**参考文件**：`taichi_LBM3D/Single_phase/LBM_3D_SinglePhase_Solver.py`

| 项目 | taichi 实现 | WanPhys（正确） |
|------|-------------|-----------------|
| **松弛时间** | `tau_f = niu/3.0 + 0.5`（**错误**，第 127 行） | `tau = 3*nu + 0.5`（`model.py`） |
| 松弛率 | `s_v = 1/tau_f` | `omega = 1/tau` |
| 格子单位 ν | `niu = 0.16667` | `nu = 0.16667` → `tau = 1.0` |

**正确公式**（D3Q19，$c_s^2 = 1/3$）：

$$\nu = c_s^2(\tau - 0.5) \quad\Rightarrow\quad \tau = 3\nu + 0.5$$

taichi 源码中正确写法被注释掉（`#self.tau_f=3.0*self.niu+0.5`），当前启用的是错误版本。**阶段 1 仅借鉴 streaming 逻辑，黏度公式一律以 WanPhys `model.py` 为准。**

与 taichi 数值对照时（第 3 周），需在报告中注明双方 τ 定义不同，不可直接比绝对数值。

---

## 3. Pull Streaming 语义

与 taichi `streaming1()` 对齐（DESIGN.md §4.1）：

```text
对每个流体格点 x、方向 s：
  ip = x + e_s          # 下游邻居（经周期/边界索引修正）
  if neighbor 为流体：
      F[ip][s] = f[x][s]           # Pull：从 x 拉入 ip 的方向 s
  else（固体或不可穿越边界）：
      F[x][LR[s]] = f[x][s]        # bounce-back
```

**单步数据流**（`solver.step`）：

```text
1. collide_bgk   : state_in.f  → state_out.F   （碰撞后分布）
2. wp.copy       : state_out.F → state_out.f   （供 stream 只读）
3. stream_pull   : state_out.f  → state_out.F   （迁移后分布）
4. apply_boundaries（阶段 1 仍为 stub）
5. update_macro  : state_out.F → rho, v
6. swap_buffers  : f ↔ F
```

**不可原地 stream**：Pull 同时读 `f`、写 `F`，碰撞结果必须先 copy 到 `f` 再迁移。

---

## 4. 周期索引

```python
@wp.func
def wrap_index(i, n, bc_low, bc_high):
    if bc_low == 0 and i < 0:   return n - 1   # 左/下/前面周期
    if bc_high == 0 and i >= n: return 0        # 右/上/后面周期
    return i                                     # 非周期：可能越界，由 bounce-back 处理
```

| `bc_*` | stream 行为 |
|--------|-------------|
| 0 | 该面周期包裹 |
| 1 / 2 | 不包裹；越界邻居视为不可穿越 → bounce-back |

M1 默认六面 `bc_* = 0`，全周期，与 week-1 恒等 stream 物理等价（平衡态应保持静止）。

---

## 5. 实现清单

| # | 文件 | 内容 | 状态 |
|---|------|------|------|
| 1 | `lattice.py` | 新增 `lattice_lr()` | ✅ |
| 2 | `kernels.py` | `wrap_index`、`periodic_index_vec`、`stream_pull` | ✅ |
| 3 | `solver.py` | collide → copy → stream_pull；传入 bc 与网格尺寸 | ✅ |
| 4 | `test_lbm_rest.py` | M1 回归 | ✅ 通过 |
| 5 | `test_lbm_stream.py` | 可选：单步迁移 sanity check | 跳过（M1 已覆盖） |

---

## 6. 验收

### 6.1 M1 回归（必须）

```powershell
.venv_lbm/Scripts/Activate.ps1
cd WanPhys-dev
python wanphys/tests/test_lbm_rest.py
```

| 指标 | 标准 |
|------|------|
| 网格 | 16³ |
| 步数 | 100 |
| max\|u\| | < 10⁻¹⁰ |
| max\|ρ−1\| | < 10⁻⁶ |
| 质量漂移 | < 10⁻⁶ |

### 6.2 阶段 1 完成判据

- [x] `stream_pull` 已实现并接入 `solver.step`
- [x] M1 测试通过（2026-06-08，`test_lbm_rest.py`）
- [x] 无 NaN / Inf
- [x] `stream_pull_identity` 保留但 solver 不再调用（便于对照调试）

---

## 7. 下一阶段预告

| 阶段 | 文档 | 状态 |
|------|------|------|
| 2 边界 | [`phase2_boundaries.md`](phase2_boundaries.md) | ✅ |
| 3 Domain | [`phase3_domain_example.md`](phase3_domain_example.md) | ✅ |
| 4 可视化 | [`phase4_visualization.md`](phase4_visualization.md) | ✅ |

总览：[`README.md`](README.md)

---

## 8. 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-06-08 | 初版；阶段 1 stream_pull 实现与 τ 公式勘误 |
| 1.1 | 2026-06-08 | 补全后续阶段链接 |
