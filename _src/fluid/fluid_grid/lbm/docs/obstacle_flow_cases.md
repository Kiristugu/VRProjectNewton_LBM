# 绕障流算例说明与总结

> **版本**：1.0  
> **日期**：2026-06-23  
> **关联里程碑**：M3 障碍绕流 / S1 绕障验收  
> **代码路径**：`wanphys/examples/`、`wanphys/_src/fluid/fluid_grid/lbm/`

---

## 1. 概述

本模块在 D3Q19-BGK LBM 上实现了两类**绕障流**演示算例，均通过 `bake_box` / `bake_cylinder_y` 将固体格点写入 `solid` 数组，在 `stream_pull` 中与 bounce-back 联用，使流体不穿透障碍。

| 算例 | 脚本 | 物理场景 | 主要用途 |
|------|------|----------|----------|
| **顶盖腔 + 方柱** | `fluid_grid_lbm_obstacle.py` | 顶盖驱动方腔内的中央方柱 | M3 绕障、三种柱高对比、答辩 PNG/VTK |
| **通道 + 圆柱/方柱** | `fluid_grid_lbm_channel_obstacle.py` | 水平通道来流绕无穷高柱体 | 俯视图尾迹、Re≈70 通道绕流 |

后处理统一走 **`cavity_plot.py` / `channel_obstacle_plot.py` + `vtk_export.py`**，与腔体 M2 可视化管线一致，可用 ParaView 做三维切片与流线。

---

## 2. 坐标与切片约定

格点布局：`velocity[i, j, k, comp]`，其中 **i = x**，**j = y**，**k = z**。

| 视图 | 固定轴 | 平面内坐标 | 流线分量 | 典型用途 |
|------|--------|------------|----------|----------|
| **侧面** | k（z） | x–y | (u_x, u_y) | 顶盖腔竖直截面、看主涡与绕柱 |
| **俯视** | j（y） | x–z | (u_x, u_z) | 通道水平截面、看柱后尾迹 |

**障碍切片位置**：在障碍几何的 **bottom / mid / top** 三处取平面——

- 侧面 x–y：固定 **k = cz ± hz** 与 **k = cz**（障碍 z 方向）
- 俯视 x–z：固定 **j = cy ± hy** 与 **j = cy**（障碍 y 方向）

---

## 3. 算例 A：顶盖腔 + 方柱

### 3.1 场景

- **边界**：`configure_cavity_walls()` + 顶盖 `set_lid_velocity(U_lid)`
- **障碍**：中央轴对齐方盒 `bake_box(center, half_extents)`
- **初场**：静止均匀流 `u = 0`

### 3.2 三种柱高 `--obstacle-height`

以 **MID** 的 x/z 半宽与竖直中心为基准，仅 **y 方向高度**变化：

| 模式 | 含义 | 几何 |
|------|------|------|
| **MID** | 柱体在腔体中央（默认） | `cy = 腔心`，`hy = auto`（约 0.12×grid_size×1.5） |
| **HIGH** | 底边与 MID 对齐，向上贴顶盖 | `y_bottom = cy_mid − hy_mid`，`y_top = 顶盖下格心` |
| **TALL** | 自腔底到顶盖 | `y` 从 `j=0` 到 `j=ny−1` 几乎满高 |

x、z 半宽（`box_half`、`box_half_z`）三种模式相同。

### 3.3 运行命令

```powershell
.venv_lbm/Scripts/Activate.ps1
cd WanPhys-dev

# 三种柱高一次跑完（答辩推荐）
python wanphys/examples/fluid_grid_lbm_obstacle.py `
  --viewer null --grid-size 64 --num-frames 100 --test --obstacle-height all

# 单一高度
python wanphys/examples/fluid_grid_lbm_obstacle.py `
  --viewer null --obstacle-height mid --num-frames 100 --test

# GL 交互
python wanphys/examples/fluid_grid_lbm_obstacle.py --viewer gl --grid-size 64
```

`--test` 会自动开启 `--export-figures`（6 张 PNG + 1 个 VTK）。

### 3.4 输出目录结构

每种高度对应子目录 `output/obstacle/<mid|high|tall>/`：

```text
output/obstacle/
├── mid/
│   ├── obstacle_mid.vtk
│   ├── stream_xy_z_bottom.png    # 侧面 x-y @ 障碍 z 底
│   ├── stream_xy_z_mid.png
│   ├── stream_xy_z_top.png
│   ├── stream_xz_y_bottom.png    # 俯视 x-z @ 障碍 y 底
│   ├── stream_xz_y_mid.png
│   └── stream_xz_y_top.png
├── high/   （同上，文件名 obstacle_high.vtk）
└── tall/   （同上）
```

### 3.5 VTK 字段

| 字段 | 说明 |
|------|------|
| `rho` | 密度 |
| `speed` | \|u\| |
| `solid` | 固体掩码（1 = 障碍） |
| `velocity` | 速度矢量 (u_x, u_y, u_z)，固体内已清零 |

ParaView：用 **Slice** 切 x-y 或 x-z；**Stream Tracer** 选 `velocity`；**Threshold** 过滤 `solid > 0.5` 隐藏固体。

### 3.6 验收要点（S1）

| 检查项 | 标准 |
|--------|------|
| 稳定 | 无 NaN/Inf |
| 不穿透 | 固体内 max\|u\| < 1e-4 |
| 绕流 | 柱后 wake_max\|u\| > 0.005 |
| 回归 | `test_obstacle_smoke.py` 仍过 |

---

## 4. 算例 B：通道 + 无穷高柱体

### 4.1 场景

- **边界**：`configure_channel_flow(u_in)` — x=0 速度入口，x=nx−1 压力出口，y 壁面无滑移，z 周期
- **障碍**：
  - **cylinder**：`bake_cylinder_y(cx, cz, r)` — 沿 y 贯穿全通道高度的圆柱（x-z 截面为圆）
  - **box**：`bake_box` — 沿 y 贯穿全高的方柱
- **默认网格**：240×80×80（x 流向 × 通道高 × 通道宽）

### 4.2 运行命令

```powershell
# 圆柱 + 方柱各跑一遍
python wanphys/examples/fluid_grid_lbm_channel_obstacle.py `
  --viewer null --num-frames 600 --test --obstacle-mode both

# 仅圆柱
python wanphys/examples/fluid_grid_lbm_channel_obstacle.py `
  --viewer null --obstacle-mode cylinder --num-frames 600 --test
```

### 4.3 输出

```text
output/channel_obstacle/
├── cylinder/
│   ├── channel_obstacle_cylinder.vtk
│   ├── fig1_top_view_streamlines.png   # 俯视 x-z @ y=ny/2，|u| + (u_x,u_z) 流线
│   └── fig2_slice_ux_uz.png            # 该切片中心线上 u_x、u_z 随 x 曲线
└── box/
    ├── channel_obstacle_box.vtk
    ├── fig1_top_view_streamlines.png
    └── fig2_slice_ux_uz.png
```

### 4.4 说明

- 圆柱在笛卡尔网格上呈**阶梯圆**（半径约 7 格），属 LBM 格点离散正常现象。
- `--obstacle-mode both` 时每种模式会**新建 ViewerNull**，避免第二次仿真步数为 0。
- 默认 Re ≈ U_in × D / ν（D = `--obstacle-diameter`，默认 14 格）。

---

## 5. 相关源文件

| 文件 | 职责 |
|------|------|
| `examples/fluid_grid_lbm_obstacle.py` | 顶盖腔绕障示例；`--obstacle-height` |
| `examples/fluid_grid_lbm_channel_obstacle.py` | 通道绕障示例；`--obstacle-mode` |
| `lbm/kernels.py` | `bake_box`、`bake_cylinder_y` |
| `lbm/domain.py` | 双缓冲 bake 封装 |
| `lbm/cavity_plot.py` | 腔体侧面/俯视流线；`export_obstacle_cavity_figures` |
| `lbm/channel_obstacle_plot.py` | 通道俯视图与 u_x/u_z 剖面 |
| `lbm/vtk_export.py` | 结构化 VTK 导出 |
| `tests/test_obstacle_smoke.py` | 32³×500 绕障 smoke |

---

## 6. 开发总结

### 6.1 已完成

1. **固体障碍**：`bake_box` + bounce-back，柱内速度 ≈ 0。
2. **顶盖腔绕障（S1）**：三种柱高 MID / HIGH / TALL，每种 6 张流线 PNG + VTK。
3. **通道绕障**：无穷高圆柱/方柱，俯视图尾迹与剖面曲线。
4. **可视化管线**：matplotlib 流线（侧面 + 俯视）+ ParaView VTK（含 solid）。
5. **批量跑批**：`--obstacle-height all`、`--obstacle-mode both` 自动分目录输出。

### 6.2 已知局限（答辩可简述）

| 项 | 说明 |
|----|------|
| 几何 | 仅 box / 圆柱（y 向拉伸）；无曲边界或 IBM |
| 圆柱形状 | 格点圆为阶梯近似，非光滑解析圆 |
| 通道 BC | 简化入口/出口，无复杂湍流入口剖面 |
| 对照 | 无外部 CFD 定量对比，靠 smoke + 流线图定性 |
| 柱高 TALL | 几乎封死 x-y 截面，绕流主要出现在 z 方向（俯视更明显） |

### 6.3 与中期计划的关系

| 计划项 | 状态 |
|--------|------|
| S1 绕障 smoke + 示例 | ✅ 顶盖腔 + 通道两套 |
| S2 TRT | 见 `post_midterm_plan.md` |
| S3 GL ρ 着色 | 见 `lbm_flow.py` |
| M3 障碍 + 答辩素材 | ✅ 本目录 PNG/VTK 可直接用于 PPT |

---

## 7. 快速命令清单

```powershell
.venv_lbm/Scripts/Activate.ps1
cd WanPhys-dev

# 回归
python wanphys/tests/test_obstacle_smoke.py
python wanphys/tests/test_cavity_smoke.py

# 顶盖腔：三种柱高 + 全套图
python wanphys/examples/fluid_grid_lbm_obstacle.py `
  --viewer null --grid-size 64 --num-frames 100 --test --obstacle-height all

# 通道：圆柱 + 方柱
python wanphys/examples/fluid_grid_lbm_channel_obstacle.py `
  --viewer null --num-frames 600 --test --obstacle-mode both
```

---

## 8. 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-06-23 | 初版：顶盖腔三柱高 + 通道圆柱/方柱；PNG/VTK 说明与总结 |
