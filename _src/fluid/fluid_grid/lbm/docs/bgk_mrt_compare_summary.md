# BGK / MRT 顶盖腔对比实验总结

> **代码**：`cavity_presets.py` · `fluid_grid_lbm_cavity.py` · `export_lbm_slice.py`  
> **快速参考**：[`bgk_mrt_compare.md`](bgk_mrt_compare.md)

---

## 1. 实验目的

在**同一套顶盖驱动腔体**算例上，对比 D3Q19 **BGK** 与 **MRT** 两种碰撞格式的：

1. **低 Re**：验证 MRT 与 BGK 宏观结果一致（实现正确性基线）。
2. **中高 Re**：观察 ρ、速度场的细微差别（差分图）。
3. **高 Re + 粗网格**：展示 BGK 数值发散、MRT 仍稳定（MRT 的主要工程价值）。

---

## 2. 雷诺数定义

格子单位下：

```text
Re ≈ U_lid × N / ν
```

其中 `N = grid_size`（腔体边长格点数），`U_lid` 为顶盖 x 方向速度，`ν` 为运动黏度（由 `FluidGridLbmModel` 的 `nu` 传入，对应松弛时间 `τ = 3ν + 0.5`）。

---

## 3. 预设参数一览

| `--profile` | 网格 | ν | U_lid | 步数 | Re≈ | 设计意图 | 推荐用途 |
|-------------|------|---|-------|------|-----|----------|----------|
| **subtle** | 64³ | 0.16667 | 0.1 | 800 | **38** | 低 Re，BGK≈MRT | 验证实现、基线对照 |
| **stress** | 64³ | 0.025 | 0.16 | 1500 | **410** | 中 Re，ρ 差 ~0.5% | 定量对比角点压缩 |
| **contrast** | **128³** | 0.01024 | **0.42** | 2500 | **~5250** | 细网格 + 高顶盖速度 | 差分图 `cavity_diff_*.png` |
| **coarse** | **32³** | 0.00256 | 0.2 | 2500 | **2500** | 粗网格 + 高 Re | **答辩推荐**：BGK NaN vs MRT 正常 |
| **custom** | 手动 | 手动 | 手动 | 手动 | — | 完全自定义 | 调参实验 |

### 3.1 为何设两套「高 Re」？

| | **contrast** | **coarse** |
|---|---|---|
| 思路 | 加密网格，提高 U_lid 拉高 Re | 低网格，保持 Re=2500 |
| 网格 | 128³（8× 格点于 64³） | 32³ |
| 典型现象 | BGK/MRT 并排 streamplot 仍很像；**差分图**有可见差别 | BGK 约 2000+ 步 **NaN**；MRT 2500 步仍稳定 |
| 耗时 | 较慢（128³ + MRT 碰撞核重） | 快（~15 s，`--compare`） |
| 答辩展示 | 差分场、高 Re 细网格 | **稳定性对比**（最直观） |

---

## 4. 运行命令

```powershell
.venv_lbm/Scripts/Activate.ps1
cd WanPhys-dev

# 基线：低 Re，BGK 与 MRT 应几乎一致
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --test --profile subtle --compare

# 中 Re：宏观场接近，ρ 有约 0.5% 差别
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --test --profile stress --compare

# 高 Re + 细网格：看差分图
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --test --profile contrast --compare

# 低网格高 Re：BGK 发散 vs MRT 稳定（答辩首选）
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --test --profile coarse --compare

# 只跑单一格式
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --test --profile coarse --collide mrt
```

`--test` 会自动开启 `--export-batch`（speed/rho PNG、VTK、NPZ）。  
`--compare` 依次跑 BGK → MRT；`contrast` 结束后额外生成 `cavity_diff_rho.png`、`cavity_diff_speed.png`。

---

## 5. 输出目录结构

```text
output/cavity_compare/
├── subtle/          # cavity_bgk_*.png, cavity_mrt_*.png, *.vtk, *.npz
├── stress/
├── contrast/
│   ├── cavity_bgk_speed.png / cavity_mrt_speed.png
│   ├── cavity_bgk_rho.png   / cavity_mrt_rho.png
│   ├── cavity_diff_rho.png    # |ρ_bgk − ρ_mrt| 中截面
│   ├── cavity_diff_speed.png
│   └── cavity_{bgk,mrt}.vtk / .npz
└── coarse/
    ├── cavity_mrt_speed.png / cavity_mrt_rho.png   # MRT 正常流场
    └── （BGK 发散时通常无 npz/png；终端见 max|u|=nan）
```

中截面为 **x-y 竖直平面**，`k = nz // 2`（见 `cavity_plot.py`）。

---

## 6. 实测现象摘要

### 6.1 subtle（Re≈38）

- BGK 与 MRT 诊断量**可逐帧完全一致**（如 `max|rho-1|=0.046179` 相同）。
- **结论**：低 Re 下两种格式等价，说明 MRT 实现与 BGK 在 hydrodynamic 极限上一致。

### 6.2 stress（Re≈410）

- `max|rho-1|`：BGK ≈ 0.058622，MRT ≈ 0.058119（差 **~0.5%**）。
- 并排 streamplot 色标相同时**肉眼难辨**；需看数值或差分。

### 6.3 contrast（Re≈5250，128³）

- 并排速度/密度图仍较接近。
- 差分图可见 max |Δρ| ~ 10⁻² 量级、max |Δu| ~ 10⁻² 量级（约为 U_lid 的 2–3%）。
- **耗时长**：128³ 格点 + MRT O(Q³) 碰撞核，`--compare` 可能需要数分钟。

### 6.4 coarse（Re≈2500，32³）— 答辩推荐

| 格式 | 约 2500 步后 |
|------|----------------|
| **BGK** | `max|u|` → **NaN**（约 2000+ 步开始发散） |
| **MRT** | `max|u|=0.2`，`max|rho-1|≈0.052`，正常导出 PNG/VTK |

终端 BGK 段会出现 `Cavity DIVERGED [bgk]: expected on coarse/high-Re`，随后 MRT 继续跑完。  
这是**预期行为**，不是程序 bug。

---

## 7. 为何很多情况下 BGK/MRT「看起来差不多」？

当前 MRT 松弛矩阵（`mrt.py` → `build_relaxation_rates`）设定为：

| 矩索引 | 含义 | 松弛率 |
|--------|------|--------|
| 0–3 | 守恒（ρ, jx, jy, jz） | **0**（不松弛） |
| 4–15 | 剪切相关 hydrodynamic 矩 | **ω**（与 BGK 相同） |
| 16–18 | ghost 非 hydrodynamic 矩 | **mrt_ghost_s = 1.0**（全松弛到平衡） |

因此：

- 在**光滑、未发散**的流动中，MRT 宏观 **ρ、u** 必然接近 BGK（剪切矩用了同一 ω）。
- MRT 的**主要优势**是 **高 Re / 强剪切 / 粗网格** 下 **BGK 失稳时仍能算**，而非在同一 ω 下给出完全不同的定常流场。
- 若需「同 Re 下流场形态也明显不同」，需采用文献中完整的 D3Q19 S 矩阵（各矩独立松弛率），超出当前简化实现。

---

## 8. 性能说明

| 因素 | 影响 |
|------|------|
| **MRT 碰撞核** | 每格点约 O(19³) 运算；BGK 为 O(19)，MRT 慢 **~300×** |
| **`--compare`** | 完整跑 BGK + MRT 两遍 |
| **网格** | 128³ 为 32³ 的 **64×** 格点数 |

**判断是否在正常运行**：终端每 20 帧打印 `[bgk]` / `[mrt] frame=… steps=…`；数字递增即非死循环。

---

## 9. 主要 CLI 参数

| 参数 | 说明 |
|------|------|
| `--profile subtle\|stress\|contrast\|coarse\|custom` | 预设参数组 |
| `--collide bgk\|trt\|mrt` | 碰撞格式（MRT 分支支持三者） |
| `--compare` | 同预设下自动跑 BGK + MRT |
| `--mrt-ghost-s 1.0` | MRT ghost 矩松弛率 |
| `--output-dir` | 根目录；预设自动加子目录如 `coarse/` |
| `--test` | 启用验收 + 批量导出 |

---

## 10. 分支与依赖

| 分支 | `--compare` 行为 |
|------|------------------|
| **MRT** | BGK + MRT 成对输出 |
| **M3Obstacle**（仅 BGK kernel） | 只跑 BGK，提示 MRT 不可用 |

两分支共用 `cavity_presets.py`；MRT 合并后 M3Obstacle 即可使用 `--compare`。

---

## 11. 答辩建议话术（简）

1. **subtle**：「低 Re 下 MRT 与 BGK 一致，验证实现正确。」
2. **stress / contrast**：「中高 Re 宏观场接近；差分图可见 ρ、u 的系统性差别。」
3. **coarse**：「粗网格高 Re 下 BGK 发散，MRT 仍稳定——这是引入 MRT 的核心动机。」

---

## 12. 相关文件

| 文件 | 作用 |
|------|------|
| `cavity_presets.py` | 预设参数、`--compare` 方案列表 |
| `mrt.py` | M 矩阵、松弛率 `build_relaxation_rates` |
| `kernels.py` | `collide_bgk` / `collide_mrt` |
| `cavity_plot.py` | streamplot PNG、差分图 `plot_cavity_field_diff` |
| `fluid_grid_lbm_cavity.py` | 主示例、`--profile` / `--compare` CLI |
| `export_lbm_slice.py` | 独立切片导出（支持 preset） |
| `bgk_mrt_compare.md` | 命令与参数速查 |

---

*文档版本：2026-07-04 · 对应当前 `cavity_presets.py` 五档预设（subtle / stress / contrast / coarse / custom）*
