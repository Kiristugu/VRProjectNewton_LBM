# LBM 模块开发文档索引

> **包路径**：`wanphys/_src/fluid/fluid_grid/lbm/`  
> **接口规范**：[`../DESIGN.md`](../DESIGN.md)  
> **团队计划**：[`docs/lbm_team_development_plan.md`](../../../../../../../docs/lbm_team_development_plan.md)

---

## 1. 架构：谁算什么？

Newton **没有内置 LBM**。大作业是在 WanPhys 上**自研** D3Q19-BGK 模块；Newton 仅提供示例运行壳（参数解析、帧循环、可选 GL viewer）。

```text
┌─────────────────────────────────────────────────────────────┐
│  物理求解（必需）— WanPhys LBM + Warp                        │
│  model / state / solver / kernels / domain                  │
│  collide → copy → stream_pull → BC → macro → swap         │
└─────────────────────────────────────────────────────────────┘
                              │ ρ, v
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  示例编排 — examples/fluid_grid_lbm_cavity.py               │
│  FluidGridLbmDomain.step()；借 newton.examples.run() 驱动循环 │
└─────────────────────────────────────────────────────────────┘
                              │ 仿真结束后的 numpy 场
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  后处理 / 可视化（可选，不参与求解）                          │
│  cavity_plot.py → matplotlib PNG（contourf + streamplot）    │
│  vtk_export.py  → ParaView VTK                               │
│  lbm_flow.py    → OpenGL 体渲染 + 中平面矢量                  │
└─────────────────────────────────────────────────────────────┘
```

| 模块 | 是否 LBM 算法 | 说明 |
|------|--------------|------|
| `kernels.py` / `solver.py` | **是** | BGK、Pull stream、边界、宏观量 |
| `fluid_grid_lbm_cavity.py` | 编排 | 组装腔体 IC/BC，调用 `domain.step()` |
| `newton.examples` | 否 | `--num-frames`、`--viewer null`、`test_final()` |
| `cavity_plot.py` | 否 | 读 `v.numpy()` 画图，类似参考 `lid_driven_cavity.py` |
| `vtk_export.py` | 否 | 导出结构化 VTK |

---

## 2. 阶段文档

| 阶段 | 文档 | 内容 | 状态 |
|------|------|------|------|
| 1 | [`phase1_stream_pull.md`](phase1_stream_pull.md) | Pull streaming、τ 公式勘误 | ✅ |
| 2 | [`phase2_boundaries.md`](phase2_boundaries.md) | 六面 BC、顶盖 `set_lid_velocity` | ✅ |
| 3 | [`phase3_domain_example.md`](phase3_domain_example.md) | Domain、示例脚本、M2 验收 | ✅ |
| 4 | [`phase4_visualization.md`](phase4_visualization.md) | GL / VTK / matplotlib 出图 | ✅ |
| 5 | [`post_midterm_plan.md`](post_midterm_plan.md) | **1 周冲刺**：绕障 → TRT → ρ 着色 GL | 🔄 进行中 |
| 6 | [`obstacle_flow_cases.md`](obstacle_flow_cases.md) | **绕障算例**：腔体三柱高 + 通道圆柱/方柱、PNG/VTK | ✅ |
| 7 | [`bgk_mrt_compare.md`](bgk_mrt_compare.md) | **BGK/MRT 对比预设**：CLI 速查 | ✅ |
| 8 | [`LBM演示算例_功能与关键技术.md`](LBM演示算例_功能与关键技术.md) | **演示算例总结**：顶盖腔 / 绕障 / BGK-MRT、不足分析 | ✅ |
| 9 | [`录屏指令清单.md`](录屏指令清单.md) | **录屏命令**：W2Full / 绕障 / MRT compare（300 帧 + PNG + VTK） | ✅ |
| — | [`lbm_week1_report_A.md`](lbm_week1_report_A.md) | 第 1 周架构解读（历史记录） | 归档 |

---

## 3. 坐标与顶盖约定

| 项目 | 约定 |
|------|------|
| 格点索引 | `velocity[i, j, k, :]` → i=x，j=y，k=z |
| 顶盖位置 | `y = ny - 1`（`bc_y_right`） |
| 驱动速度 | `u_x = U_lid`（默认 0.1，格子单位） |
| 经典 2D 腔体图 | **x-y 竖直截面**，固定 `k = nz // 2`（非 x-z 水平面） |

---

## 4. 环境与测试（路径形式）

```powershell
.venv_lbm/Scripts/Activate.ps1
cd WanPhys-dev

# M1 静止流
python wanphys/tests/test_lbm_rest.py

# M2 腔体烟雾
python wanphys/tests/test_cavity_smoke.py

# M2 示例（headless）
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --num-frames 100 --test

# M2 + VTK + streamplot PNG
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --num-frames 100 --test `
  --export-vtk output/cavity.vtk --save-slice output/cavity_streamplot.png

# M3 绕障（详见 obstacle_flow_cases.md）
python wanphys/examples/fluid_grid_lbm_obstacle.py --viewer null --grid-size 64 --num-frames 100 --test --obstacle-height all
python wanphys/examples/fluid_grid_lbm_channel_obstacle.py --viewer null --num-frames 600 --test --obstacle-mode both
python wanphys/tests/test_obstacle_smoke.py
```

测试文件通过 `tests/_bootstrap.py` 桩加载 LBM 包，**避免**拉取完整 `wanphys` 顶层依赖。

---

## 5. 里程碑快照（2026-06-08）

| 里程碑 | 关键指标 | 状态 |
|--------|---------|------|
| M1 | 16³×100 步，max\|u\|<1e-10 | ✅ |
| M2 | 32³/50³×500 步，顶盖 BC，主涡可见 | ✅ |
| S1–S4 | 1 周内：绕障 + TRT + GL + 答辩素材 | 待做 → [`post_midterm_plan.md`](post_midterm_plan.md) |

---

## 6. 答辩材料

| 文档 | 路径 |
|------|------|
| 【中期答辩】书面文档 | [`docs/LBM流体仿真模块_中期答辩文档.md`](../../../../../../../docs/LBM流体仿真模块_中期答辩文档.md) |
| 【中期答辩】PPT 大纲 | [`docs/LBM流体仿真模块_中期答辩PPT大纲.md`](../../../../../../../docs/LBM流体仿真模块_中期答辩PPT大纲.md) |

---

## 7. 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-06-08 | 新增索引；澄清 Newton / LBM / 可视化分层；汇总阶段文档 |
| 1.1 | 2026-06-08 | 链到中期答辩文档与 PPT 大纲 |
| 1.2 | 2026-06-10 | 新增 post_midterm_plan.md（1 周冲刺版 S1–S4） |
| 1.4 | 2026-07-04 | 新增 bgk_mrt_compare_summary.md；BGK/MRT 五档预设（含 coarse / contrast） |
| 1.5 | 2026-07-05 | 新增 LBM演示算例_功能与关键技术.md（功能点、实现方式、不足） |
