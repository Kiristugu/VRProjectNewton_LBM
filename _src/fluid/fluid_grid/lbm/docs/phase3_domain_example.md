# 阶段 3：Domain + 示例 + 导出

> **版本**：1.1  
> **日期**：2026-06-08  
> **目标**：完成 M2 集成层 — `FluidGridLbmDomain`、`fluid_grid_lbm_cavity.py`、`__init__` 导出  
> **前置**：阶段 1 streaming、阶段 2 边界条件  
> **后续**：阶段 4 可视化（[`phase4_visualization.md`](phase4_visualization.md)）

---

## 1. 范围

| 包含 | 说明 |
|------|------|
| `FluidGridLbmDomain` | 双缓冲 + `domain.step(dt)` |
| `lbm/__init__.py` | 导出四个公共符号 |
| `fluid_grid/__init__.py` | 合并 LBM 导出 |
| `examples/fluid_grid_lbm_cavity.py` | M2 必交示例（**LBM 求解入口**） |
| `test_lbm_domain.py` | Domain 烟雾测试 |

**不包含**（见阶段 4）：`cavity_plot.py`、`vtk_export` 调用、GL 体渲染——均为仿真后的可选后处理。

---

## 2. Newton 与 WanPhys 的分工

示例脚本中：

| 组件 | 职责 |
|------|------|
| `FluidGridLbmDomain` + `solver` | **LBM 物理**：碰撞、迁移、BC、宏观量 |
| `newton.examples.create_parser()` | CLI：`--num-frames`、`--viewer` |
| `newton.examples.run()` | 帧循环、`test_final()` 验收钩子 |
| `cavity_plot` / `vtk_export` | 仅 `--save-slice` / `--export-vtk` 时调用 |

Newton **不提供** LBM kernel；大作业目标是自研 `wanphys/_src/fluid/fluid_grid/lbm/`。

---

## 3. Domain 用法（DESIGN.md §10.3）

```python
from wanphys._src.fluid.fluid_grid.lbm import (
    FluidGridLbmModel,
    FluidGridLbmDomain,
)

model = FluidGridLbmModel(fluid_grid_res=(50, 50, 50), nu=0.16667, use_guo_force=False)
domain = FluidGridLbmDomain(model)
state = domain.create_state()
domain.solver.configure_cavity_walls()
domain.solver.set_lid_velocity(wp.vec3(0.1, 0.0, 0.0))
domain.solver.init_uniform(state, rho=1.0, u=wp.vec3(0.0, 0.0, 0.0))

for _ in range(500):
    domain.step(dt=1.0)
```

顶盖：`y = ny - 1`，`u_x = U_lid`。详见 [`phase2_boundaries.md`](phase2_boundaries.md)。

---

## 4. 示例脚本

```powershell
.venv_lbm/Scripts/Activate.ps1
cd WanPhys-dev
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --num-frames 100 --test
```

默认：`grid_size=50`，每帧 5 个 LBM 子步 → 100 帧 × 5 = **500 步**（M2）。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--grid-size` | 50 | 立方网格边长（M2 建议 ≥32） |
| `--nu` | 0.16667 | 格子黏度 |
| `--u-lid` | 0.1 | 顶盖 x 方向格子速度 |
| `--num-frames` | 100 | 渲染帧数（×5 子步 = LBM 总步数） |
| `--test` | 关 | 启用 `test_final()` 数值检查 |

出图与 VTK 参数见 [`phase4_visualization.md`](phase4_visualization.md)。

---

## 5. 实现清单

| # | 文件 | 状态 |
|---|------|------|
| 1 | `domain.py` | ✅ |
| 2 | `lbm/__init__.py` | ✅ |
| 3 | `fluid_grid/__init__.py` | ✅ |
| 4 | `examples/fluid_grid_lbm_cavity.py` | ✅ |
| 5 | `tests/test_lbm_domain.py` | ✅ |
| 6 | `tests/test_cavity_smoke.py` | ✅ |

---

## 6. M2 验收清单

- [x] `FluidGridLbmDomain` 可 `step(dt=1.0)` 循环
- [x] `fluid_grid_lbm_cavity.py --viewer null --test` 无崩溃（50³ 已验证）
- [x] 500 步后 `max|u|≈0.1`，近顶面 `u_x>0.08`
- [x] 四符号可从 `wanphys._src.fluid.fluid_grid.lbm` 导入
- [x] `test_lbm_domain.py`、`test_cavity_smoke.py` 通过

实测（50³，500 步，ν=0.16667，U_lid=0.1）：

| 指标 | 实测 |
|------|------|
| max\|u\| | 0.100 |
| 近顶盖 max u_x（j=ny−2） | 0.088 |
| 中平面 min u_x | −0.014（回流） |

---

## 7. 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-06-08 | 阶段 3 Domain + 示例 + 导出 |
| 1.1 | 2026-06-08 | 澄清 Newton 角色；路径形式命令；链到阶段 4；补 M2 实测数据 |
