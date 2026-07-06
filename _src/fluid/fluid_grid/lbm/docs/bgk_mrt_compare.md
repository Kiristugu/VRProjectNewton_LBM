# BGK vs MRT 顶盖腔对比预设

> **模块**：`cavity_presets.py`  
> **示例**：`fluid_grid_lbm_cavity.py`、`export_lbm_slice.py`

---

## 预设一览

| `--profile` | ν | U_lid | 网格 | 总步数 | Re≈ | 预期 |
|-------------|---|-------|------|--------|-----|------|
| **subtle** | 0.16667 | 0.1 | 64³ | 800 | 38 | BGK 与 MRT **几乎重合**（数值可完全一致） |
| **stress** | 0.025 | 0.16 | 64³ | 1500 | 410 | 宏观场仍很像；ρ 诊断差约 **0.5%** |
| **contrast** | 0.01024 | **0.42** | **128³** | 2500 | **~5250** | 加密网格 + 高 U_lid；看 `cavity_diff_*.png` |
| **coarse** | 0.00256 | 0.2 | **32³** | 2500 | **2500** | **低网格高 Re**：BGK 易 NaN，MRT 仍稳定 |

`custom`：不使用预设，沿用 `--nu`、`--u-lid`、`--grid-size` 等手动参数。

Re 估算：`Re ≈ U_lid × grid_size / ν`。

---

## 命令（cavity 示例）

```powershell
.venv_lbm/Scripts/Activate.ps1
cd WanPhys-dev

# 看不出差异：一次跑 BGK + MRT
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --test --profile subtle --compare

# 高 Re 加密网格（差分图）
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --test --profile contrast --compare

# 低网格高 Re：BGK 发散 vs MRT 稳定（答辩推荐）
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --test --profile coarse --compare

# 单一碰撞格式
python wanphys/examples/fluid_grid_lbm_cavity.py --viewer null --test --profile stress --collide mrt
```

`--test` 会自动 `--export-batch`（speed + rho PNG、VTK、NPZ）。

---

## 命令（export_lbm_slice）

```powershell
python wanphys/examples/export_lbm_slice.py --profile subtle --compare
python wanphys/examples/export_lbm_slice.py --profile stress --compare
```

---

## 输出目录

```text
output/cavity_compare/
├── subtle/
│   └── cavity_bgk_*.png, cavity_mrt_*.png, *.vtk, *.npz
├── stress/
│   └── （同上）
└── contrast/
    ├── cavity_diff_rho.png
    ├── cavity_diff_speed.png
    └── cavity_bgk/mrt 的 speed/rho PNG、VTK、NPZ
└── coarse/
    ├── cavity_mrt_speed.png / cavity_mrt_rho.png   # MRT 正常流场
    └── BGK 终端出现 max|u|=nan（预期）；通常无 BGK 导出
```

## 为何 BGK/MRT 看起来差不多？

当前 MRT 的剪切矩（moment 4–15）松弛率 **与 BGK 的 ω 相同**（见 `mrt.py` 的 `build_relaxation_rates`）；只有 ghost 矩（16–18）用 `mrt_ghost_s=1.0` 全松弛。在顶盖腔这种光滑、无强压缩的流动里，ghost 矩对宏观 **ρ、u** 影响很小，因此：

- **subtle（Re≈38）**：两者可以 **逐帧完全一致** —— 这是预期行为，不是 MRT 没跑起来。
- **stress / contrast**：`max|rho-1|` 等标量仍很接近（同一角点压缩误差主导），并排 PNG 色标相同时会“看起来一样”；请打开 **`cavity_diff_rho.png`** 看绝对差值场。

MRT 的主要优势通常是 **高 Re / 强剪切下的稳定性**，而不是在同一 ω 下给出完全不同的定常流场。若答辩需要“并排就能看出来”，优先用 **`coarse`**（BGK NaN、MRT 正常）。

完整总结（预设设计、实测数据、答辩话术）见 **[`bgk_mrt_compare_summary.md`](bgk_mrt_compare_summary.md)**。

---

## 分支说明

| 分支 | `--collide` | `--compare` |
|------|-------------|-------------|
| **MRT** | bgk / trt / mrt | 依次跑 BGK + MRT |
| **M3Obstacle**（仅 BGK） | 仅 bgk（自动检测） | 只跑 BGK，并提示 MRT 不可用 |

两分支共用 **`cavity_presets.py`**；M3Obstacle 合并该文件与示例改动后即可使用 `--profile`，待 MRT 合并后再用 `--compare`。

---

## 主要 CLI 参数

| 参数 | 说明 |
|------|------|
| `--profile subtle\|stress\|contrast\|coarse\|custom` | 预设 / 自定义 |
| `--collide bgk\|trt\|mrt` | 碰撞格式 |
| `--compare` | 同预设下 BGK+MRT 成对输出 |
| `--mrt-ghost-s 1.0` | MRT 非 hydrodynamic 矩松弛率 |
| `--output-dir` | 根目录（预设会加 `subtle/` / `stress/` / `contrast/` / `coarse/` 子目录） |
