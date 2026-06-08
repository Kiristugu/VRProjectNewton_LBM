# 阶段 3：Domain + 示例 + 导出

> **版本**：1.0  
> **日期**：2026-06-08  
> **目标**：完成 M2 集成层 — `FluidGridLbmDomain`、`fluid_grid_lbm_cavity.py`、`__init__` 导出  
> **前置**：阶段 1 streaming、阶段 2 边界条件

---

## 1. 范围

| 包含 | 说明 |
|------|------|
| `FluidGridLbmDomain` | 双缓冲 + `domain.step(dt)` |
| `lbm/__init__.py` | 导出四个公共符号 |
| `fluid_grid/__init__.py` | 合并 LBM 导出 |
| `wanphys/_src/fluid/__init__.py` | 顶层 fluid 包导出 |
| `examples/fluid_grid_lbm_cavity.py` | M2 必交示例 |
| `test_lbm_domain.py` | Domain 烟雾测试 |

---

## 2. Domain 用法（DESIGN.md §10.3）

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

---

## 3. 示例脚本

```powershell
.venv_lbm/Scripts/Activate.ps1
cd WanPhys-dev
python -m wanphys.examples.fluid_grid_lbm_cavity --viewer null --num-frames 100
```

默认：`grid_size=50`，每帧 5 个 LBM 子步 → 100 帧 × 5 = **500 步**（M2）。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--grid-size` | 50 | 立方网格边长 |
| `--u-lid` | 0.1 | 顶盖 x 方向格子速度 |
| `--num-frames` | 100 | 渲染帧数（×5 子步 = LBM 总步数） |

---

## 4. 实现清单

| # | 文件 | 状态 |
|---|------|------|
| 1 | `domain.py` | ✅ |
| 2 | `lbm/__init__.py` | ✅ |
| 3 | `fluid_grid/__init__.py` | ✅ |
| 4 | `_src/fluid/__init__.py` | ✅ |
| 5 | `examples/fluid_grid_lbm_cavity.py` | ✅ |
| 6 | `tests/test_lbm_domain.py` | ✅ |

---

## 5. M2 验收清单

- [x] `FluidGridLbmDomain` 可 `step(dt=1.0)` 循环
- [x] `fluid_grid_lbm_cavity.py --viewer null` 无崩溃（32³/50³ 已验证）
- [x] 500 步后 `max|u|≈0.1`，近顶面 `u_x>0.08`
- [x] 四符号可从 `wanphys._src.fluid.fluid_grid.lbm` 导入
- [x] `test_lbm_domain.py` 通过

---

## 6. 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-06-08 | 阶段 3 Domain + 示例 + 导出 |
