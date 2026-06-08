# 阶段 2：边界条件 + 腔体 BC 配置

> **版本**：1.1  
> **日期**：2026-06-08  
> **目标**：实现 `apply_boundaries`，提供 `configure_cavity_walls` / `set_lid_velocity`，为 M2 腔体示例做准备  
> **前置**：阶段 1 `stream_pull`（[`phase1_stream_pull.md`](phase1_stream_pull.md)）  
> **依据**：[`DESIGN.md`](../DESIGN.md) §7.3；taichi `Boundary_condition()`（**不用**其 τ 公式）

---

## 1. 范围

| 包含 | 不包含（阶段 3+） |
|------|------------------|
| `apply_boundaries` 六面 BC | `FluidGridLbmDomain` |
| 面速度型 / 压力型 BC（taichi 简化 equilibrium） | matplotlib / VTK / GL 可视化（见阶段 4） |
| `configure_cavity_walls()` + `set_lid_velocity()` | `bake_box` / 障碍固体 |
| 每面独立 `bc_vel_*` 字段 | 完整 Zou–He 反推公式 |
| `test_cavity_smoke.py` | taichi 数值对照 |

---

## 2. BC 类型（与 DESIGN / taichi 一致）

| `bc_*` | 含义 | 实现 |
|--------|------|------|
| 0 | 周期（stream 已处理） | `apply_boundaries` 跳过该面 |
| 1 | 固定密度 / 压力型 | `F = feq(s, bc_rho, v_neighbor)` |
| 2 | 固定速度型 | `F = feq(s, 1.0, bc_vel_face)` |

**τ 公式**：仍用 WanPhys `tau = 3*nu + 0.5`（见阶段 1 勘误）。

---

## 3. 每面速度字段

单全局 `bc_velocity` 无法同时表达「五壁 u=0 + 顶盖 u_lid」。`model.py` 新增：

```text
bc_vel_x_left, bc_vel_x_right
bc_vel_y_left, bc_vel_y_right   ← 顶盖 bc_y_right
bc_vel_z_left, bc_vel_z_right
```

腔体配置：

```python
solver.configure_cavity_walls()              # 六面 bc=2，各面速度置零
solver.set_lid_velocity(wp.vec3(0.1, 0, 0))  # 仅 bc_vel_y_right = (0.1, 0, 0)
```

坐标约定（DESIGN.md）：顶盖 = `y = ny - 1`（`bc_y_right`），驱动 `u_x`。

---

## 4. 算法顺序（不变）

```text
collide → copy(F→f) → stream_pull → apply_boundaries → update_macro → swap
```

`apply_boundaries` 在 stream 之后修正边界格点 `F`；`update_macro` 读修正后的 `F` 得 `rho, v`。

---

## 5. 实现清单

| # | 文件 | 内容 | 状态 |
|---|------|------|------|
| 1 | `model.py` | 六面 `bc_vel_*` | ✅ |
| 2 | `kernels.py` | `apply_boundaries` 实装 | ✅ |
| 3 | `solver.py` | BC 参数传入 + `configure_cavity_walls` | ✅ |
| 4 | `test_lbm_rest.py` | M1 回归（bc 全 0） | ✅ 通过 |
| 5 | `test_cavity_smoke.py` | 腔体 500 步烟雾测试 | ✅ 通过 |

---

## 6. 验收

### 6.1 M1 回归

```powershell
.venv_lbm/Scripts/Activate.ps1
cd WanPhys-dev
python wanphys/tests/test_lbm_rest.py
```

### 6.2 腔体烟雾（阶段 2）

```powershell
python wanphys/tests/test_cavity_smoke.py
```

| 指标 | 标准（32³，500 步，U_lid=0.1） |
|------|-------------------------------|
| `max\|u\|` | > 0.01 |
| 近顶面 `max u_x`（`j=ny-2`） | > 0.05 |
| 数值 | 无 NaN / Inf |

### 6.3 阶段 2 完成判据

- [x] `apply_boundaries` 六面 BC 实装
- [x] `configure_cavity_walls()` + `set_lid_velocity()` 可用
- [x] M1 回归不退化（2026-06-08）
- [x] `test_cavity_smoke.py` 32³×500 步通过

---

## 7. 下一阶段预告

| 阶段 | 文档 | 状态 |
|------|------|------|
| 3 Domain + 示例 | [`phase3_domain_example.md`](phase3_domain_example.md) | ✅ |
| 4 可视化 | [`phase4_visualization.md`](phase4_visualization.md) | ✅ |

总览：[`README.md`](README.md)

---

## 8. 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-06-08 | 阶段 2 边界条件与腔体 BC API |
| 1.1 | 2026-06-08 | 可视化归入阶段 4；补文档交叉链接 |
