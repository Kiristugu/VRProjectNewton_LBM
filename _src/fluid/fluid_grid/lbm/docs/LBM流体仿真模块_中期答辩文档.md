# 【中期答辩】格子玻尔兹曼方法（LBM）流体仿真模块 — 中期答辩文档

> **文档类型**：中期答辩书面材料  
> **项目名称**：基于 WanPhys 的 D3Q19-BGK 单相流 LBM 模块  
> **课程**：计算物理 / Newton 引擎大作业  
> **团队**：三人组（张弋洋 A · 架构/数据，黄彧鸣 B · 核心算法，彭若扬 C · 边界/场景）  
> **答辩节点**：4 周计划第 2 周末～第 3 周初（M1、M2 已完成，可视化提前交付）  
> **日期**：2026-06-08  
> **依据**：[`开题.md`](开题.md)、[`lbm_team_development_plan.md`](lbm_team_development_plan.md)、[`lbm/docs/README.md`](../WanPhys-dev/wanphys/_src/fluid/fluid_grid/lbm/docs/README.md)

---

## 一、答辩摘要（1 分钟版）

本课题在 WanPhys / Newton 框架上**自研** D3Q19-BGK 格子玻尔兹曼流体模块（Newton 本身不含 LBM）。截至中期：

- **M1 静止流**：16³×100 步，速度/密度/质量指标全部达标  
- **M2 顶盖驱动腔**：50³×500 步，顶盖 BC 生效，主涡清晰可见  
- **可视化**：matplotlib 流线图、VTK、OpenGL 体渲染已打通（原第 4 周任务提前完成）  
- **待完成（后期）**：障碍物绕流、taichi 定性对照表、答辩彩排（M3/M4）

---

## 二、课题背景与目标

### 2.1 作业要求（`newton.txt` 摘要）

| 层次 | 要求 |
|------|------|
| 基础 | 分布函数、BGK 碰撞、streaming、基本边界 |
| 验证 | 稳定性测试、示例场景、可视化 |
| 可选 | MRT、多相流、复杂边界（本组 MRT 列为未来工作） |

### 2.2 本组技术选型

| 项目 | 选择 | 理由 |
|------|------|------|
| 格子 | D3Q19 | 3D 单相流标准模型，与 taichi 参考一致 |
| 碰撞 | BGK | 4 周可交付；接口预留可插拔 |
| 迁移 | **Pull** | 与 `taichi_LBM3D/streaming1` 对齐，便于对照 |
| 单位 | **全程格子单位** | Δx=Δt=1，ν=0.16667，避免物理单位混用 |
| 松弛时间 | τ = 3ν + 0.5 | 标准 D3Q19 公式（taichi 源码 τ 有误，不对齐） |

### 2.3 与 Newton 的关系（答辩常问）

```text
Newton/Warp     →  运行环境 + 示例壳（帧循环、CLI、GL viewer）
WanPhys LBM     →  自研物理（collide / stream / BC / macro）
后处理模块       →  可选（cavity_plot / vtk_export / lbm_flow）
```

**LBM 求解不在 Newton 内置模块中**，而在 `wanphys/_src/fluid/fluid_grid/lbm/`。

---

## 三、总体架构与 Pipeline

### 3.1 模块分层

```
wanphys/_src/fluid/fluid_grid/lbm/
├── DESIGN.md          # 接口冻结 v1.2
├── lattice.py         # D3Q19：e, w, feq, LR
├── model.py           # FluidGridLbmModel（ν, τ, BC 配置）
├── state.py           # FluidGridLbmState（f, F, ρ, v, solid）
├── solver.py          # step() 五步编排
├── domain.py          # FluidGridLbmDomain（双缓冲）
├── kernels.py         # Warp GPU kernel
├── cavity_plot.py     # matplotlib 后处理（不参与求解）
└── vtk_export.py      # VTK 导出
```

### 3.2 单步数据流（已冻结）

```text
collide_bgk(f → F)
  → copy(F → f)          # Pull 需只读 f
  → stream_pull(f → F)
  → apply_boundaries(F)
  → update_macro(F → ρ, v)
  → swap(f ↔ F)
```

### 3.3 顶盖驱动腔边界

- 六面速度 BC（`bc_* = 2`），五壁 u = 0  
- 顶盖：`y = ny − 1`，`u_x = U_lid`（默认 0.1）  
- 实现：`configure_cavity_walls()` + `set_lid_velocity()`

---

## 四、进度对照（4 周计划 vs 实际）

| 周次 | 计划里程碑 | 计划交付 | **中期实际状态** |
|------|-----------|---------|-----------------|
| 第 1 周 | M1 | 16³ 静止流、最小 pipeline | ✅ **已完成** |
| 第 2 周 | M2 | Pull stream、BC、腔体示例 | ✅ **已完成** |
| 第 3 周 | M3 | 障碍、taichi 对照 | ⏳ **未开始** |
| 第 4 周 | M4 | 可视化、报告、答辩 | 🔶 **可视化已提前完成**；报告/PPT 进行中 |

### 4.1 里程碑验收数据

#### M1 — 静止流（`test_lbm_rest.py`）

| 指标 | 标准 | 实测 |
|------|------|------|
| 网格 | 16³ | 16³ |
| 步数 | 100 | 100 |
| max\|u\| | < 1e-10 | 通过 |
| max\|ρ−1\| | < 1e-6 | 通过 |
| 质量漂移 | < 1e-6 | 通过 |

#### M2 — 顶盖驱动腔（`test_cavity_smoke.py` + 示例）

| 指标 | 标准 | 实测（50³，500 步） |
|------|------|-------------------|
| max\|u\| | > 0.01 | **0.100** |
| 近顶盖 max u_x | > 0.05 | **0.088** |
| 中平面回流 | 负 u_x | **−0.014** |
| NaN/Inf | 无 | 无 |
| 主涡 | 可见 | streamplot 可见顺时针主涡 |

---

## 五、阶段成果清单

| 阶段 | 文档 | 代码/测试 | 状态 |
|------|------|----------|------|
| 1 Pull streaming | `phase1_stream_pull.md` | `stream_pull`, `test_lbm_rest` | ✅ |
| 2 边界条件 | `phase2_boundaries.md` | `apply_boundaries`, `test_cavity_smoke` | ✅ |
| 3 Domain+示例 | `phase3_domain_example.md` | `FluidGridLbmDomain`, `fluid_grid_lbm_cavity.py` | ✅ |
| 4 可视化 | `phase4_visualization.md` | `cavity_plot`, `vtk_export`, `lbm_flow` | ✅ |

### 5.1 单元测试（中期全部通过）

| 测试文件 | 覆盖 |
|---------|------|
| `test_lbm_import.py` | τ/ω、格点权重、LR 对 |
| `test_equilibrium.py` | feq 守恒矩、BGK 平衡态 |
| `test_lbm_rest.py` | M1 |
| `test_lbm_domain.py` | Domain 集成 |
| `test_cavity_smoke.py` | M2 |
| `test_lbm_vtk_export.py` | VTK 格式 |

### 5.2 可视化交付物

| 形式 | 路径/命令 | 用途 |
|------|----------|------|
| matplotlib 流线图 | `--save-slice output/cavity_streamplot.png` | 答辩静态图、报告插图 |
| VTK | `--export-vtk output/cavity.vtk` | ParaView 三维分析 |
| OpenGL | `--viewer gl` | 现场实时演示 |

---

## 六、成员分工与贡献（中期）

| 成员 | 负责域 | 中期已完成 |
|------|--------|-----------|
| **张弋洋 A** | Model / State / Domain / solver 编排 | 双缓冲 state、Domain、示例集成、`DESIGN.md` 维护 |
| **黄彧鸣 B** | lattice / collide / stream / macro | D3Q19 表、BGK、Pull stream、平衡态与 M1/M2 测试 |
| **彭若扬 C** | BC / 场景 / 可视化 | 六面 BC、腔体 API、示例脚本、matplotlib/VTK/GL |

协作模式：**垂直切分并行**；`kernels.py` 按 collide/stream（B）与 BC（C）分责；每日短同步报 blocker。

---

## 七、关键技术问题与解决

| 问题 | 现象 | 解决 |
|------|------|------|
| taichi τ 公式错误 | 若照搬 `niu/3+0.5` 黏度不对 | 坚持用 τ=3ν+0.5，对照时注明差异 |
| Pull 与碰撞缓冲 | 原地 stream 破坏 f/F 语义 | collide 后 copy，再 stream_pull |
| 顶盖需独立面速度 | 单全局 `bc_velocity` 不够 | 六面 `bc_vel_*` + `set_lid_velocity` |
| 可视化截面选错 | x-z 水平面不像腔体 | 改为 x-y 竖直面 @ k=mid |
| Newton 无 LBM | 易误解「引擎自带」 | 文档明确：自研 lbm 包 + Newton 示例壳 |
| 环境依赖重 | 全量 wanphys import 失败 | `tests/_bootstrap.py` 桩加载 |

---

## 八、演示方案（中期答辩现场）

### 8.1 推荐流程（约 5–8 分钟）

1. **架构图**（30 s）：WanPhys LBM vs Newton 壳 vs 后处理  
2. **M1 测试**（1 min）：`python wanphys/tests/test_lbm_rest.py`  
3. **M2 测试**（1 min）：`python wanphys/tests/test_cavity_smoke.py`  
4. **腔体示例**（1 min）：headless 跑 500 步，打印 `max|u|`  
5. **流线图**（1 min）：展示 `cavity_streamplot.png`，说明色标 \|u\| 与主涡  
6. **可选 GL**（1 min）：`--viewer gl` 体渲染  
7. **后期计划**（1 min）：障碍、taichi 对照、终答辩彩排  

### 8.2 环境准备

```powershell
.venv_lbm/Scripts/Activate.ps1
cd WanPhys-dev
```

### 8.3 关键命令（路径形式）

```powershell
python wanphys/tests/test_lbm_rest.py
python wanphys/tests/test_cavity_smoke.py
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --num-frames 100 --test `
  --export-vtk output/cavity.vtk --save-slice output/cavity_streamplot.png
```

---

## 九、后期计划（第 3–4 周）

| 优先级 | 任务 | 负责人 | 验收 |
|--------|------|--------|------|
| P0 | `fluid_grid_lbm_obstacle.py` | C | 流体不穿透固体 |
| P0 | `bake_box` / `bake_sphere` → solid | C | 障碍几何 |
| P1 | taichi 定性对照表 | B | 同 ν、U_lid、格子单位，偏差 <20% |
| P1 | 1000 步稳定性 | B | 无 NaN |
| P1 | API docstring 完善 | A | 公共符号有文档 |
| P2 | dam-break 初值 | C | 加分项 |
| 答辩 | 3 分钟 cavity 彩排 | 全员 | GL 或录屏 |

**明确不做（中期口径）**：MRT 碰撞（答辩表述为 future work，接口可插拔）。

---

## 十、风险与应对（更新）

| 风险 | 当前状态 | 应对 |
|------|---------|------|
| M3 障碍延期 | 未开始 | 第 3 周优先 box 障碍，球体可降级 |
| taichi 数值偏差 | 官方示例驱动面/碰撞模型不同 | 只比定性 + max\|u\| 量级，报告写清 BC 差异 |
| 密度偏差（腔体 ρ） | max\|ρ−1\|≈0.045 | M3 考虑改进 Zou–He，不阻塞中期 |
| 答辩演示翻车 | GL 环境差异 | 备录屏 + PNG + VTK 三套 |

---

## 十一、参考文献

1. Newton Physics Engine — https://github.com/newton-physics/newton  
2. Krüger et al., *The Lattice Boltzmann Method: Principles and Practice*  
3. taichi_LBM3D — https://github.com/yjhp1016/taichi_LBM3D  
4. 本组教程 — `docs/lbm_tutorial/README.md`  
5. 接口文档 — `WanPhys-dev/wanphys/_src/fluid/fluid_grid/lbm/DESIGN.md`

---

## 十二、附录：taichi 对照记录（待 M3 填写）

**单位声明**：双方均为格子单位；Δx=Δt=1；WanPhys τ=3ν+0.5。

| 步数 | taichi max\|u\| | WanPhys max\|u\| | 备注 |
|------|----------------|-------------------|------|
| 100（静止） | ≈0 | ≈0 | M1 |
| 500（腔体） | 0.100 | 0.100 | 50³；驱动面定义不同，仅比量级 |
| 1000（腔体） | — | — | M3 |

---

**【中期答辩文档 · 完】**
