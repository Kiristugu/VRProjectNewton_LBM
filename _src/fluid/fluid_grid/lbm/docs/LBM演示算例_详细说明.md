# LBM 演示算例：详细说明（自用完整版）

> **版本**：2.0  
> **日期**：2026-07-06  
> **读者**：自己复习 / 答辩前核对细节  
> **与精简版关系**：[`LBM演示算例_功能与关键技术.md`](LBM演示算例_功能与关键技术.md) 是答辩提纲；**本文档不省略**，并解释「为什么这么实现」。  
> **代码根路径**：`wanphys/_src/fluid/fluid_grid/lbm/` · `wanphys/examples/fluid_grid_lbm_*.py`

---

## 目录

1. [模块在工程中的位置](#1-模块在工程中的位置)
2. [求解器核心：算法选型与理由](#2-求解器核心算法选型与理由)
3. [示例脚本架构：为什么这样写](#3-示例脚本架构为什么这样写)
4. [演示一：顶盖驱动方腔](#4-演示一顶盖驱动方腔)
5. [演示二：顶盖腔 + 方柱绕障](#5-演示二顶盖腔--方柱绕障)
6. [演示三：通道绕障](#6-演示三通道绕障)
7. [演示四：BGK / MRT 对比体系](#7-演示四bgk--mrt-对比体系)
8. [后处理与输出管线](#8-后处理与输出管线)
9. [验收逻辑与终端诊断](#9-验收逻辑与终端诊断)
10. [分支策略与 git 工程细节](#10-分支策略与-git-工程细节)
11. [已知局限：观察到了什么、为何暂时接受](#11-已知局限观察到了什么为何暂时接受)
12. [录屏命令与参数陷阱](#12-录屏命令与参数陷阱)
13. [实测现象备忘（含 coarse 步数）](#13-实测现象备忘含-coarse-步数)
14. [源文件索引](#14-源文件索引)

---

## 1. 模块在工程中的位置

### 1.1 Newton 与 WanPhys 的分工

**Newton 没有内置 LBM**。大作业要求是在 WanPhys 上自研 D3Q19 格子 Boltzmann，Newton 只提供「示例运行壳」：

| 层次 | 负责方 | 做什么 |
|------|--------|--------|
| **物理求解** | WanPhys `fluid_grid/lbm` | 分布函数 `f`、碰撞、迁移、边界、宏观量 ρ/u |
| **帧循环** | `newton.examples.run()` | `--num-frames`、每帧回调 `step()` / `render()` / `test_final()` |
| **可视化** | 可选 GL / matplotlib / VTK | 不参与时间推进，只读 `state.v.numpy()` |

**为什么分开**：LBM 不走 MAC 压力投影管线（`FluidGridMacSolverBase`），若强行塞进 `liquid/` 或 `basic_vortex/` 的 `step()` 钩子，接口会冲突。因此在 `fluid_grid/` 下单独建 `lbm/` 子包，继承 `FluidGridSolverBase` 但自定义整步流程（见 [`DESIGN.md`](../DESIGN.md) §2）。

### 1.2 三层数据流

```text
kernels.py + solver.py + domain.py
        │ 每步：collide → stream_pull → BC → macro → swap
        ▼
examples/fluid_grid_lbm_*.py
        │ 设置 IC/BC/障碍；调用 domain.step()
        ▼
cavity_plot.py / channel_obstacle_plot.py / vtk_export.py
        │ 仿真结束后 numpy 场 → PNG / VTK
        ▼
ParaView / PPT / 录屏
```

后处理** deliberately 与求解解耦**：同一套 `export_structured_vtk` 可用于腔体、绕障、BGK/MRT 对比，避免在 kernel 里混 matplotlib。

---

## 2. 求解器核心：算法选型与理由

### 2.1 格子模型：D3Q19

- **选用 D3Q19** 而非 D3Q27：与参考实现 `taichi_LBM3D` 一致，对照成本低；19 方向已足够表达 3D 不可压 Navier–Stokes 的低阶矩。
- **全程格子单位**：Δx = Δt = 1，ν 直接传入 model，`τ = 3ν + 0.5`，`ω = 1/τ`。与 taichi 对照前**禁止**混用 `cell_size` 换算物理速度——`cell_size` 仅用于出图坐标轴标注。

### 2.2 迁移：Pull，不用 Push

单步顺序（冻结，见 `DESIGN.md` §4）：

```text
collide  →  stream_pull  →  apply_boundaries  →  update_macro  →  swap(f, F)
```

| 范式 | 含义 | 本模块 |
|------|------|--------|
| **Pull** | 格点 i 从上游 `x - e_i` **读**进入方向的分布 | ✅ 采用 |
| Push | 格点向邻居 **写** | ❌ 不采用 |

**理由**：

1. 与 taichi `streaming1()` 一致，联调、对照文献时少一层概念转换。
2. Pull 天然配合双缓冲 `f`（读）/ `F`（写），GPU 上无原地覆盖竞态。
3. 固体 bounce-back 在 Pull 框架下分支清晰：`solid` 格点走反弹索引 `LR[s]`。

### 2.3 碰撞：BGK 为主，MRT 为扩展

| 格式 | 复杂度 | 角色 |
|------|--------|------|
| **BGK** | O(19) / 格点 | 默认；低 Re 验证、绕障 demo |
| **TRT** | O(19) | 可选中间步；两松弛时间 |
| **MRT** | O(19³) / 格点 | 高 Re 稳定性展示；慢 ~300× |

**为什么先 BGK 后 MRT**：4 周排期里 MRT 标为 future work（`DESIGN.md` §6.3）；BGK 跑通 Pull + BC + 腔体 M2 后再加 MRT，避免同时调试碰撞格式与边界。

**MRT 实现策略（简化版，有意为之）**——见 `mrt.py`：

```python
# 守恒矩 0–3：S = 0（不松弛）
# 剪切相关 hydrodynamic 矩 4–15：S = ω（与 BGK 相同）
# ghost 非 hydrodynamic 矩 16–18：S = mrt_ghost_s（默认 1.0，全松弛到平衡）
```

**为什么这么设 S 矩阵**：

- 剪切矩用与 BGK 相同的 ω → **稳定工况下 ρ、u 必然接近 BGK**，便于用 subtle 预设做「实现正确性」对照。
- ghost 矩单独全松弛 → 在 BGK 失稳的高 Re / 粗网格下，多余非物理矩被快速耗散，**数值稳定性**提升。
- 未采用文献完整 D3Q19 S 矩阵（各矩独立 τ）→ 实现与调试量可控；答辩时主动说明是 **简化 MRT**，完整版留作改进项。

### 2.4 边界条件

| 类型 | 编码 | 用途 |
|------|------|------|
| 周期 | bc = 0 | z 向（通道算例） |
| 压力型 Zou–He | bc = 1 | 通道出口 x = nx−1 |
| 速度型 Zou–He | bc = 2 | 顶盖、壁面、通道入口 |

**为什么用 Zou–He 平衡态格式**：LBM 入门与 taichi 参考一致；实现量小于 NLR / 高阶格式，足够演示腔体主涡与绕障尾迹。

**未专门处理角点**：顶盖–侧壁交线处理论上需 corner 修正；当前靠网格分辨率「抹平」。这是底角小涡不明显的原因之一（§11）。

### 2.5 固体障碍：格点标记 + bounce-back

- `state.solid[i,j,k] = 1` 表示固体；流体格点 `solid = 0`。
- 几何通过 `bake_box` / `bake_cylinder_y` 写入（GPU kernel），**不**改碰撞核逻辑——固体在 `stream_pull` 与 `apply_boundaries` 中走 bounce-back。
- **为什么不用 IBM / 曲边界**：中期目标为「能看见绕流」；格点障碍与现有 `solid_phi` 烘焙管线一致，1 周内可交付 S1 绕障 smoke。

---

## 3. 示例脚本架构：为什么这样写

### 3.1 `_bootstrap_lbm_imports()`

每个 `fluid_grid_lbm_*.py` 开头都有类似的 bootstrap：

**原因**：完整 `import wanphys` 会拉取 geometry 等重依赖，在仅装 LBM 最小依赖的 `.venv_lbm` 里会失败。bootstrap 只 stub 出 `wanphys._src.fluid.fluid_grid.lbm` 包路径，**直接 exec `lbm/__init__.py`**。

这是环境约束下的务实选择，不是架构理想态；合并进主包前需保留此模式。

### 3.2 Newton 示例类 `Example`

```text
__init__  → 建 model / domain / IC / BC /（可选）GL visualizer
step()    → 每帧 DEFAULT_LBM_SUBSTEPS(=5) 次 domain.step()
render()  → GL 时才画；null viewer 为空操作
test_final() → 验收 + 导出 PNG/VTK/NPZ
```

**为什么每帧 5 个子步**：Newton 的 `--num-frames` 是「可视化帧数」，不是 LBM 步数。5 步/帧是 M2 时代定的经验值——100 帧 = 500 步，腔体主涡已可见；改帧数即可线性缩放总步数。

**为什么诊断每 20 帧打印一次**：128³ + MRT 很慢，太频繁 IO 拖慢；20 帧 = 100 步，足够观察发散趋势又不刷屏。

### 3.3 `cavity_presets.py` 独立成模块

**理由**：

- BGK/MRT 对比参数成组出现（ν、U_lid、grid、steps），集中管理避免示例脚本里硬编码四份。
- `apply_cavity_profile()` 在 `parse_known_args` 后**立即**改 `args`，保证 `--compare` 两轮用同一套物理参数。
- `build_lbm_model()` 用 `dataclass_fields` 探测 `collide_impl` / `mrt_ghost_s` 是否存在 → **M3Obstacle 分支只有 BGK kernel 时不会 import 报错**。

### 3.4 `--compare` 的实现

```text
for collide in ("bgk", "mrt"):
    复制 args → 改 collide / 输出路径 → newton.examples.run(Example)
_export_compare_diff()  # 读 cavity_bgk.npz 与 cavity_mrt.npz
```

**为什么串行而非并行**：两轮共用 GPU，并行无收益且占显存；串行日志清晰，录屏时 BGK 段失败 → MRT 段成功，对比强烈。

**为什么 coarse 下 BGK NaN 不 raise**：`test_final()` 里若 `profile=="coarse"` 且 BGK diverged，只 print 警告并 return，让 `--compare` 继续跑 MRT。若用 `--profile custom` 手动配 coarse 参数，则 BGK NaN 会 **raise ValueError**——这是录屏用 custom+300 帧时的一个细节（§12）。

---

## 4. 演示一：顶盖驱动方腔

### 4.1 物理场景

- 立方腔体五壁固定（u=0），顶盖 y = ny−1 沿 +x 匀速 U_lid。
- 静止初场 → 顶盖剪切驱动 → 腔体中央主涡 + 角点副涡（Re 足够时）。
- 经典 benchmark：Ghia et al. 1982；本实现**未做定量对照**，定性看流线即可。

### 4.2 边界配置 API

```python
domain.solver.configure_cavity_walls()      # 六面 BC 模式
domain.solver.set_lid_velocity(wp.vec3(U_lid, 0, 0))
domain.solver.init_uniform(state, rho=1.0, u=0)
```

**为什么 `configure_cavity_walls` 封装在 solver**：BC 模式存在 model 的 `bc_*` 字段里，示例不应直接改 6×2 个整型；与通道 `configure_channel_flow` 对称。

### 4.3 默认参数（无 `--profile`）

| 参数 | 默认 | Re ≈ |
|------|------|------|
| grid | 50³ | |
| ν | 0.16667 | |
| U_lid | 0.1 | **30** |

Re = U_lid × N / ν = 0.1 × 50 / 0.16667 ≈ 30，低 Re，稳定，适合 W2Full 分支「第一条 demo」。

### 4.4 可视化约定

- **经典腔体图**：x–y **竖直截面**，固定 k = nz // 2（`cavity_plot.py`）。
- **为什么不是 x–z 俯视**：顶盖驱动的主流是 x–y 平面内的主涡；俯视更多用于通道算例。

### 4.5 分支 W2Full

该分支保留较早 cavity 脚本：**无** `--profile` / `--compare` / `--output-dir` 部分新功能。录屏时用显式路径：

```powershell
python wanphys/examples/fluid_grid_lbm_cavity.py `
  --viewer null --num-frames 300 --test `
  --output-dir output/video/cavity_w2full
```

---

## 5. 演示二：顶盖腔 + 方柱绕障

### 5.1 场景动机

在已验收的顶盖腔上**中央放方盒**，证明 `bake_box` + bounce-back 能挡流并产生绕流尾迹——对应里程碑 S1 / M3。

### 5.2 三种柱高 `--obstacle-height`

以 **MID** 的 x/z 半宽与竖直中心为基准，**只改 y 方向范围**：

| 模式 | 几何意图 | 答辩展示点 |
|------|----------|------------|
| **mid** | 柱在腔体中央，四周可绕流 | 默认；侧面看绕柱流线 |
| **high** | 底边与 MID 对齐，顶边贴近顶盖 | 顶盖剪切与柱顶相互作用 |
| **tall** | 从腔底到顶盖几乎满高 | x–y 截面几乎被封死；**俯视 x–z** 更明显 |

**为什么三种高度而不是改 x/z 尺寸**：y 向变化对「顶盖驱动 + 障碍」叙事最直观；x/z 固定便于并排对比 PNG。

### 5.3 输出：6 PNG + 1 VTK / 高度

| 文件模式 | 平面 | 看什么 |
|----------|------|--------|
| `stream_xy_z_{bottom,mid,top}.png` | x–y @ 障碍 z 三高度 | 侧面绕流、涡结构 |
| `stream_xz_y_{bottom,mid,top}.png` | x–z @ 障碍 y 三高度 | 俯视分离 |

**为什么在障碍几何处切三片**：单一 mid 切片可能切在柱体内部（全 solid）；三片保证至少有一片在柱外流体区有流线。

### 5.4 `--obstacle-height all`

一次进程跑 mid → high → tall，每种高度 **新建 `ViewerNull`**。

**原因**：Newton viewer 帧计数在多次 `run()` 间不重置；复用同一 viewer 会导致第二次仿真步数为 0。通道算例 `--obstacle-mode both` 同理。

### 5.5 默认输出路径

`output/obstacle/<mid|high|tall>/`，可用 `--output-dir` 覆盖根目录（录屏：`output/video/obstacle`）。

---

## 6. 演示三：通道绕障

### 6.1 与腔体算例的差异

| 项 | 顶盖腔绕障 | 通道绕障 |
|----|------------|----------|
| 驱动 | 顶盖剪切 | 入口 uniform u_in |
| 出口 | 封闭壁面 | x=nx−1 压力出口 |
| z 向 | 封闭 | **周期** |
| 障碍 | 有限高度方盒 | **沿 y 贯穿**的圆柱/方柱 |
| 主视图 | 侧面 x–y | **俯视 x–z**（尾迹） |

**为什么做通道**：腔体是「封闭驱动」，通道是「外流绕 bluff body」，Re 与尾迹结构不同，答辩可展示模块**换 BC 即可换场景**。

### 6.2 默认网格 240×80×80

x 长、y 为通道高、z 为展向（周期）。障碍默认直径 ~14 格，U_in 与 ν 给出 Re ~ O(10²) 量级的层流尾迹。

**为什么圆柱是阶梯圆**：`bake_cylinder_y` 在笛卡尔格点上判 `(x-cx)²+(z-cz)² ≤ r²`；小半径时只有 ~7 格宽，圆周线呈锯齿。这是 LBM 格点几何的常态，不是渲染 bug。

### 6.3 输出

- `fig1_top_view_streamlines.png`：y = ny/2 处 |u| 填色 + (u_x, u_z) 流线。
- `fig2_slice_ux_uz.png`：同一平面中心线上的 u_x、u_z 曲线，看尾迹振荡。
- VTK 含 `solid` 字段，ParaView Threshold 可隐藏固体。

---

## 7. 演示四：BGK / MRT 对比体系

### 7.1 实验目的（三层）

1. **subtle**：低 Re → BGK ≈ MRT，验证 MRT 实现没写错。
2. **stress / contrast**：中高 Re → 宏观场仍像，差分图有 O(10⁻²) 量级差别。
3. **coarse**：粗网格 + Re≈2500 → **BGK NaN，MRT 仍稳定** → MRT 的工程动机。

### 7.2 雷诺数

```text
Re ≈ U_lid × N / ν        （N = grid_size，格子单位）
τ = 3ν + 0.5
```

### 7.3 五档预设及设计理由

| profile | 网格 | ν | U_lid | 总步数 | Re≈ | 为什么设这一档 |
|---------|------|---|-------|--------|-----|----------------|
| **subtle** | 64³ | 0.16667 | 0.1 | 800 | 38 | 远低于 BGK 稳定极限；两者应逐帧一致 |
| **stress** | 64³ | 0.025 | 0.16 | 1500 | 410 | 角点压缩明显；ρ 差 ~0.5%，需数值或差分才看出 |
| **contrast** | 128³ | 0.01024 | 0.42 | 2500 | ~5250 | **细网格**拉高 Re；并排流线仍像，**差分图**有内容 |
| **coarse** | 32³ | 0.00256 | 0.2 | 2500 | 2500 | **粗网格**同 Re；BGK 易发散，算得快，答辩首选 |
| **custom** | CLI | CLI | CLI | CLI | — | 录屏统一 300 帧时不被 preset 覆盖步数 |

#### 为何同时有 contrast 与 coarse 两档「高 Re」？

| | contrast | coarse |
|---|----------|--------|
| 思路 | 加密网格 + 提高 U_lid | 降低网格，保持 Re=2500 |
| 典型结果 | BGK/MRT 都能跑完；看 **\|BGK−MRT\|** | BGK **挂掉**；看 **稳定性** |
| 耗时 | 长（128³ × MRT） | 短（~15 s compare） |
| 答辩角色 | 「高 Re 下仍有系统性数值差」 | 「BGK 不行、MRT 行」——**最有说服力** |

两档互补，不是重复。

### 7.4 `--profile` 与输出目录

- `apply_cavity_profile()`：非 custom 时覆盖 grid / nu / u_lid / num_frames，并在 output_dir 下加子目录如 `coarse/`。
- **注意**：`--compare` 模式下 `_run_compare` **不再**追加 profile 子目录（避免曾出现的 `coarse/coarse/` 双嵌套 bug）；请用 `--output-dir output/cavity_compare/coarse` 显式指定。

### 7.5 MRT 慢的原因

每格点 MRT：`m = M·f` → 对角松弛 → `f' = M⁻¹·m'`，矩阵 19×19，约 O(19³)；BGK 一次松弛 O(19)。128³ 格点数又是 32³ 的 64 倍，故 contrast compare 可能要数分钟——**正常现象**。

---

## 8. 后处理与输出管线

### 8.1 PNG（matplotlib）

- `cavity_plot.plot_lid_driven_cavity`：`contourf` 标量 + `streamplot` 流线。
- `plot_cavity_field_diff`：\|BGK − MRT\| 伪彩色，标题带 max/mean。
- **不参与求解**：可在无 GPU 机器上从 NPZ 重画。

### 8.2 VTK

- `export_structured_vtk`：结构化网格，字段 `rho`、`speed`、可选 `velocity` 矢量、`solid`。
- **为什么 VTK 而不只 PNG**：答辩现场可 ParaView 旋转切片；PNG 只做 PPT 静态页。

### 8.3 NPZ

- `cavity_{bgk,mrt}.npz` 存 `velocity`、`rho`、元数据（profile、ν、U_lid）。
- **用途**：差分图二次生成；coarse 下 BGK 无 npz 时 diff 跳过并提示——**预期行为**。

### 8.4 批量导出开关

| 标志 | 效果 |
|------|------|
| `--test` | 自动 `export_batch=True` |
| `--export-batch` | speed + rho PNG + VTK + NPZ |
| preset 非 custom | `test_final` 里 `do_batch` 也为 True |

---

## 9. 验收逻辑与终端诊断

### 9.1 每 20 帧打印

```text
[bgk] frame=300 steps=1500 max|u|=... near_lid_ux=... max|rho-1|=...
```

| 量 | 含义 |
|----|------|
| `max\|u\|` | 全场速度模最大值；应 ≤ U_lid（除非失稳） |
| `near_lid_ux` | j = ny−2 层 max u_x；检查顶盖 BC 是否生效 |
| `max\|rho-1\|` | 密度守恒偏离；角点常有大值，看数量级 |

### 9.2 `test_final()` 判定

1. **发散**：`np.isnan` / `isinf` on v 或 ρ  
   - `profile==coarse` 且 BGK → 打印 `Cavity DIVERGED`，不 raise（compare 继续）  
   - 其他情况 → `ValueError`
2. **未建立流动**：`max|u| < 0.01` → fail
3. **顶盖太弱**：`near_lid_ux < 0.05` → fail
4. 通过 → 打印 `Cavity OK` 并导出

**重要**：「Cavity OK」只表示**没有 NaN/Inf** 且流动/顶盖检查通过；**不代表物理上已收敛**。coarse BGK 在 1500 步可能 OK 但已失稳（§13）。

### 9.3 绕障 smoke（`test_obstacle_smoke.py`）

- 32³×500 步，检查无 NaN、固体内 \|u\| ≈ 0、柱后 wake 速度 > 阈值。

---

## 10. 分支策略与 git 工程细节

### 10.1 功能分支

| 分支 | 内容 |
|------|------|
| **W2Full** | 早期 cavity demo，BGK only |
| **M3Obstacle** | 绕障两示例 + obstacle 测试 |
| **MRT** | cavity `--profile` / `--compare` / MRT kernel |

功能按分支开发，答辩前 checkout 切换演示。

### 10.2 `.gitignore` 与 `__pycache__`

Python 运行会在各目录生成 `.pyc`。若被 git 跟踪，切换分支时本地 `.pyc` 与目标分支冲突，**无法 checkout**。

已提交 `.gitignore`（忽略 `__pycache__/`、`output/`、`.venv_*/`）并从索引移除全部 `.pyc`。各分支 cherry-pick 该提交；若有 modify/delete 冲突，对冲突 `.pyc` 执行 `git rm -f` 后 `--continue`。

---

## 11. 已知局限：观察到了什么、为何暂时接受

### 11.1 方腔底角小涡不明显

**观察**：64³、Re~几十～几百时，Ghia 文献里底角小涡很弱或不见。

**原因（多层）**：

1. 3D 立方腔 ≠ 2D 方腔；展向 z 有结构，中截面不一定是经典 2D 图。
2. 网格分辨率有限，角点梯度最陡，需更细网格或更高 Re。
3. 角点 BC 未专门修正，数值扩散抹平小尺度涡。
4. 可视化只取 k = nz/2 一个平面。

**为何暂不改**：中期目标是主涡与绕障；角点涡需加密 + BC 升级，排期外。

### 11.2 通道圆柱呈阶梯状

**观察**：俯视流线里圆柱像八边形。

**原因**：笛卡尔格点 + 整数半径判据，Δx 有限。

**为何暂不改**：IBM/曲边界实现成本高；答辩如实说明「格点几何近似」。

### 11.3 通道入口为均匀来流

无湍流入口剖面、无发展段；Re 基于障碍直径的定性绕流足够，**不与风洞数据比**。

### 11.4 MRT 为简化版

稳定流场下与 BGK 接近是 **S 矩阵设计的结果**，不是 bug。完整独立 τ 的 MRT 留作后续。

### 11.5 固定步数、无文献定量对比

验收是 smoke + 目视；未做 Ghia 涡心位置误差表——时间允许可加 post-processing 脚本。

---

## 12. 录屏命令与参数陷阱

完整清单见 [`录屏指令清单.md`](录屏指令清单.md)。此处强调**易踩坑**：

### 12.1 统一 300 帧 vs preset 步数

录屏清单用：

```powershell
--profile custom --grid-size ... --nu ... --u-lid ...
--num-frames 300
```

**原因**：`--profile coarse` 会把 `num_frames` 改成 500（2500 步），与清单「每段 300 帧」不一致。

### 12.2 coarse + 300 帧：BGK 可能不 NaN

1500 步时 BGK 可能只出现**前兆**（max\|u\|>U_lid）仍打印 `Cavity OK`；**1600 步左右**开始爆炸，**1700+ NaN**（见 §13）。

录屏话术：

- 300 帧版：对比 BGK 末帧异常 vs MRT 稳定 + diff 图。
- 若要终端 NaN：改用 `--profile coarse`（500 帧）或 `--num-frames 500`。

### 12.3 `--profile custom` 下 BGK NaN 会报错

`test_final` 里「coarse 预期发散不 raise」**仅当 `profile=="coarse"`**。custom 手动配 coarse 参数时 BGK NaN → `ValueError`，**但 `--compare` 在 BGK 段就中断**，MRT 跑不到。

**录屏 compare coarse 建议**：

```powershell
# 方案 A：真 coarse preset（500 帧，BGK 友好报错）
--profile coarse --compare --output-dir output/video/compare_coarse

# 方案 B：custom 300 帧（BGK 可能 OK 但 diff 已明显）
--profile custom --grid-size 32 --nu 0.00256 --u-lid 0.2 --num-frames 300 --compare
```

### 12.4 环境

```powershell
cd D:\Term6\GP\HW\Project\WanPhys-dev
.venv_lbm\Scripts\Activate.ps1
```

输出在 `WanPhys-dev/output/` 下。

---

## 13. 实测现象备忘（含 coarse 步数）

### 13.1 subtle（64³, Re≈38, 1500 步）

- BGK 与 MRT 诊断量可**逐帧相同**。
- diff 图 max Δρ ~ 10⁻⁷，实现一致性基线。

### 13.2 stress（64³, Re≈410）

- max\|ρ−1\|：BGK ≈ 0.0586，MRT ≈ 0.0581（差 ~0.5%）。
- 并排 streamplot 肉眼难辨。

### 13.3 contrast（128³, Re≈5250, 1500 步）

- 两者均能跑完；diff max Δρ ~ 10⁻²，Δu ~ 10⁻²。
- 128³ MRT 慢，终端 frame 涨得慢属正常。

### 13.4 coarse（32³, Re≈2500）— 步数时间线

同一参数：ν=0.00256, U_lid=0.2, grid=32³

| 步数 | frame (×5 步/帧) | BGK 现象 |
|------|------------------|----------|
| ≤1400 | ≤280 | max\|u\|=0.2，near_lid 缓慢上升 |
| 1500 | 300 | max\|u\|**0.273**，max\|ρ−1\|**0.119** — 失稳前兆，仍 **Cavity OK** |
| 1600 | 320 | max\|u\| ~ **10⁴**，密度误差爆炸 |
| ≥1700 | ≥340 | **NaN** |

MRT 同参数 2500 步：max\|u\|=0.2，max\|ρ−1\|≈0.051，正常导出。

**结论**：文档里「约 2000+ 步 NaN」是近似；**精确说约在 1600–1700 步之间**进入 NaN。1500 步录屏仍能展示「BGK 开始坏 / MRT 仍好」。

### 13.5 如何判断程序在跑而不是卡死

- 终端每 20 帧有 `[bgk]` / `[mrt]` 行，frame/steps 递增。
- MRT + 128³ 可能 1–5 分钟无新行也正常，看 steps 是否 eventually 增加。

---

## 14. 源文件索引

### 14.1 求解器

| 文件 | 职责 |
|------|------|
| [`lattice.py`](../lattice.py) | D3Q19 速度集、权重、feq、LR |
| [`model.py`](../model.py) | ν、BC 类型、collide_impl、τ/ω |
| [`state.py`](../state.py) | f, F, rho, v, solid |
| [`solver.py`](../solver.py) | step 编排；configure_cavity_walls / channel |
| [`kernels.py`](../kernels.py) | collide_bgk/mrt/trt, stream_pull, BC, bake_* |
| [`mrt.py`](../mrt.py) | M 矩阵、build_relaxation_rates |
| [`domain.py`](../domain.py) | 双缓冲 create_state / step |

### 14.2 示例

| 文件 | 场景 |
|------|------|
| [`examples/fluid_grid_lbm_cavity.py`](../../../../examples/fluid_grid_lbm_cavity.py) | 顶盖腔 + BGK/MRT compare |
| [`examples/fluid_grid_lbm_obstacle.py`](../../../../examples/fluid_grid_lbm_obstacle.py) | 腔体方柱三高度 |
| [`examples/fluid_grid_lbm_channel_obstacle.py`](../../../../examples/fluid_grid_lbm_channel_obstacle.py) | 通道圆柱/方柱 |

### 14.3 预设与绘图

| 文件 | 职责 |
|------|------|
| [`cavity_presets.py`](../cavity_presets.py) | 五档 profile、build_lbm_model、路径解析 |
| [`cavity_plot.py`](../cavity_plot.py) | 腔体/绕障流线 PNG、差分图 |
| [`channel_obstacle_plot.py`](../channel_obstacle_plot.py) | 通道俯视 + 剖面 |
| [`vtk_export.py`](../vtk_export.py) | 结构化 VTK |

### 14.4 文档

| 文件 | 用途 |
|------|------|
| [`DESIGN.md`](../DESIGN.md) | 接口冻结、算法顺序 |
| [`bgk_mrt_compare_summary.md`](bgk_mrt_compare_summary.md) | BGK/MRT 实验速查 |
| [`obstacle_flow_cases.md`](obstacle_flow_cases.md) | 绕障命令与输出 |
| [`录屏指令清单.md`](录屏指令清单.md) | 录屏 300 帧命令 |
| [`LBM演示算例_功能与关键技术.md`](LBM演示算例_功能与关键技术.md) | 答辩精简提纲 |

---

## 附录 A：答辩叙事线（可选）

1. **W2Full 腔体**：「Warp GPU 上 D3Q19 Pull LBM，顶盖驱动形成主涡。」
2. **Obstacle**：「格点 solid + bounce-back，腔体三柱高 + 通道尾迹。」
3. **MRT subtle**：「低 Re BGK=MRT，实现正确。」
4. **MRT coarse**：「同 Re 粗网格，BGK 步数到 1600+ 失稳/NaN，MRT 2500 步稳定——简化 MRT 的价值在稳定性。」
5. **主动局限**：格点圆、角点 BC、简化 MRT、无 Ghia 定量对比。

---

## 附录 B：常用命令速查

```powershell
cd D:\Term6\GP\HW\Project\WanPhys-dev
.venv_lbm\Scripts\Activate.ps1

# 腔体 smoke
python wanphys/tests/test_cavity_smoke.py

# BGK/MRT 答辩核心（完整 2500 步）
python wanphys/examples/fluid_grid_lbm_cavity.py `
  --viewer null --test --profile coarse --compare `
  --output-dir output/cavity_compare/coarse

# 绕障全套
python wanphys/examples/fluid_grid_lbm_obstacle.py `
  --viewer null --num-frames 300 --test --obstacle-height all

python wanphys/examples/fluid_grid_lbm_channel_obstacle.py `
  --viewer null --num-frames 300 --test --obstacle-mode both
```

---

*文档版本：2026-07-06 · 自用完整版；参数以当前 `cavity_presets.py` 与示例脚本为准。*
