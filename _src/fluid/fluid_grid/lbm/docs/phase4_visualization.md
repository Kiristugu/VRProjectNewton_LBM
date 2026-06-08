# 阶段 4：LBM 腔体可视化

> **版本**：1.0  
> **日期**：2026-06-08  
> **目标**：M4 演示 — GL 实时体渲染 + VTK/ParaView + 切片 PNG

---

## 1. 组件

| 模块 | 路径 | 功能 |
|------|------|------|
| `LbmScalarVolumeRenderer` | `fluid_viewer/lbm_flow.py` | \|u\| 3D 纹理光线步进 |
| `LbmCavityVisualizer` | `fluid_viewer/lbm_flow.py` | 线框 + 体渲染 + 中平面矢量 |
| `export_structured_vtk` | `lbm/vtk_export.py` | 导出 ParaView VTK |
| `fluid_grid_lbm_cavity.py` | `examples/` | 集成入口 |

---

## 2. 运行方式

### 2.1 OpenGL 实时可视化

```powershell
.venv_lbm/Scripts/Activate.ps1
cd WanPhys-dev
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer gl --grid-size 50
```

- 默认先跑 **200 步 warmup** 再显示（流场已建立）
- 蓝→青→红 表示 \|u\| 大小
- 灰色线框 = 腔体边界
- 彩色短线 = `y = ny/2` 中平面速度矢量

可选参数：

| 参数 | 说明 |
|------|------|
| `--no-volume` | 关闭体渲染 |
| `--no-vectors` | 关闭中平面矢量 |
| `--warmup-steps N` | 渲染前预热步数 |
| `--vector-stride N` | 矢量采样间隔 |

### 2.2 VTK 导出（ParaView）

```powershell
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --num-frames 100 --export-vtk output/cavity.vtk --test
```

字段：`rho`、`speed`、向量 `velocity`。

### 2.3 中平面切片 PNG

```powershell
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --num-frames 100 --save-slice output/cavity_slice.png --test
```

需要 `matplotlib`。

---

## 3. 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-06-08 | 体渲染 + VTK + 切片 PNG |
