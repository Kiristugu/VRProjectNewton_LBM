# 阶段 4：LBM 腔体可视化

> **版本**：1.1  
> **日期**：2026-06-08  
> **目标**：M4 演示 — GL 实时体渲染 + VTK/ParaView + matplotlib 经典腔体流线图  
> **前置**：阶段 3 Domain + `fluid_grid_lbm_cavity.py`（[`phase3_domain_example.md`](phase3_domain_example.md)）

---

## 1. 原则：可视化不参与求解

所有可视化模块均在 **LBM `domain.step()` 完成之后** 读取 `rho`、`v` 数组，做后处理。**删除它们不影响仿真正确性。**

| 层级 | 模块 | 路径 |
|------|------|------|
| 求解 | `FluidGridLbmDomain` | `lbm/domain.py` |
| 编排 | 示例入口 | `examples/fluid_grid_lbm_cavity.py` |
| 后处理 | VTK | `lbm/vtk_export.py` |
| 后处理 | matplotlib 流线图 | `lbm/cavity_plot.py` |
| 后处理 | OpenGL | `fluid_viewer/lbm_flow.py` |

`cavity_plot.py` 仿照网上有限差分示例 [`lid_driven_cavity.py`](../../../../../../../lid_driven_cavity.py) 的出图风格（`contourf` 背景 + `streamplot` 黑色流线），**不是第二套流体算法**。

---

## 2. 组件

| 模块 | 路径 | 功能 |
|------|------|------|
| `plot_lid_driven_cavity` | `lbm/cavity_plot.py` | x-y 中截面：\|u\| 填色 + 流线 PNG |
| `export_structured_vtk` | `lbm/vtk_export.py` | 导出 ParaView 结构化 VTK |
| `LbmScalarVolumeRenderer` | `fluid_viewer/lbm_flow.py` | \|u\| 3D 纹理光线步进 |
| `LbmCavityVisualizer` | `fluid_viewer/lbm_flow.py` | 线框 + 体渲染 + x-y 中平面矢量 |
| `fluid_grid_lbm_cavity.py` | `examples/` | 集成入口（仿真 + 可选出图） |

---

## 3. 截面约定（重要）

顶盖在 **y = ny − 1**，驱动 **u_x**。经典顶盖驱动腔 2D 图应取：

```text
x-y 竖直平面，固定 k = nz // 2
```

| 截面 | 用途 | 是否用于经典腔体图 |
|------|------|-------------------|
| x-y @ k=mid | 顶盖在上、主涡在中间 | **是**（`cavity_plot`、GL 默认矢量面） |
| x-z @ j=mid | 水平剖面 | 否（易误解为「不像腔体」） |

---

## 4. 运行方式

环境：

```powershell
.venv_lbm/Scripts/Activate.ps1
cd WanPhys-dev
```

### 4.1 matplotlib 流线图（推荐答辩/static 图）

```powershell
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --num-frames 100 --test `
  --save-slice output/cavity_streamplot.png
```

- 背景：`contourf`，默认标量 **|u|**（色标标签 `|u|`，格子单位）
- 叠加：黑色 `streamplot`（平面分量 u_x、u_y）
- 依赖：`pip install matplotlib`
- 交互弹窗：加 `--show-plot`

| 参数 | 默认 | 说明 |
|------|------|------|
| `--save-slice PATH` | 空 | 保存 PNG 路径 |
| `--show-plot` | 关 | 仿真结束后 `plt.show()` |
| `--slice-k` | -1 | z 索引；-1 表示 `nz//2` |
| `--scalar-field` | `speed` | 背景场：`speed`（\|u\|）、`ux`、`rho` |

### 4.2 VTK 导出（ParaView）

```powershell
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --num-frames 100 --test `
  --export-vtk output/cavity.vtk
```

字段：`rho`、`speed`、向量 `velocity`。

可与 PNG 同次运行：

```powershell
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --num-frames 100 --test `
  --export-vtk output/cavity_streamplot.vtk `
  --save-slice output/cavity_streamplot.png
```

### 4.3 OpenGL 实时可视化

```powershell
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer gl --grid-size 50
```

- 默认 **200 步 warmup** 再渲染
- 体渲染：蓝→青→红 表示 \|u\|
- 灰色线框：腔体边界
- 彩色短线：**x-y 中平面**（`k = nz/2`）速度矢量

| 参数 | 说明 |
|------|------|
| `--no-volume` | 关闭体渲染 |
| `--no-vectors` | 关闭中平面矢量 |
| `--no-boundary` | 关闭线框 |
| `--warmup-steps N` | 渲染前预热 LBM 步数 |
| `--vector-stride N` | 矢量采样间隔 |
| `--vector-scale F` | 箭头长度缩放 |

---

## 5. 色标说明（matplotlib）

默认 `--scalar-field speed` 时，图右侧 colorbar 为 **速度模长 |u|**：

$$\|u\| = \sqrt{u_x^2 + u_y^2 + u_z^2}$$

- 单位：**格子单位**（与 `U_lid=0.1`、`nu=0.16667` 一致），不是 m/s
- 顶盖附近接近 0.1 属正常
- 黑色流线表示**方向**；色标表示**快慢**

---

## 6. 验收

- [x] `--save-slice` 输出 x-y 截面 streamplot，可见主涡
- [x] `--export-vtk` 可被 ParaView 读取
- [x] `--viewer gl` 可实时显示体渲染 + 中平面矢量
- [x] 可视化模块与 `test_cavity_smoke.py` / M2 指标不冲突

---

## 7. 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-06-08 | 体渲染 + VTK + 切片 PNG |
| 1.1 | 2026-06-08 | 新增 `cavity_plot.py`；x-y 截面约定；澄清求解 vs 后处理；补全 CLI 与色标说明 |
