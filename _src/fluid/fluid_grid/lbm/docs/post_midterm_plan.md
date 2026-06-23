# LBM 后期开发计划（1 周冲刺版）

> **版本**：1.1  
> **日期**：2026-06-10  
> **周期**：**1 周（5 个工作日）**  
> **前置**：M1 静止流 ✅ · M2 顶盖腔体 ✅  
> **团队**：张弋洋 **A** · 黄彧鸣 **B** · 彭若扬 **C**

---

## 1. 一周要交付什么

**P0（必须交）**

| 交付物 | 说明 |
|--------|------|
| 绕障流 | `bake_box` + 障碍示例 + smoke 测试 |
| 进阶碰撞 | **TRT**（双松弛时间）可切换；接口名 `collide_impl` 预留 `mrt` |
| 液体感 GL | 在现有 `lbm_flow.py` 上 **ρ/speed 联合着色** + 固体线框 |
| 答辩素材 | 1 段 GL 录屏 + 1 张绕障 PNG/VTK |

**P1（有余力）**

- 完整 MRT kernel + `test_mrt_rest.py`
- `bake_sphere`

**明确砍掉**

- taichi 对照  
- 复杂 inlet/outlet 通道（改用 **周期域 + 均匀初速度** 或 **腔体 + 中央柱**）  
- 真自由面 / 多相 LBM  
- 独立 `LbmLiquidVisualizer` 大重构  

**答辩口径**：一周版完成「绕障 + 比 BGK 更好的 TRT + 增强 GL」；MRT 作为接口已预留或 appendix。

---

## 2. 里程碑（全部挤进 1 周）

| 代号 | 名称 | 验收 | 最晚 |
|------|------|------|------|
| **S1** | 绕障 | 32³×500，不穿透，有尾迹 | **周三** |
| **S2** | TRT | 16³×100 静止流仍过；`--collide trt` 可跑 | **周四** |
| **S3** | 液体 GL | 绕障/腔体 GL 录屏 ≥15 s，ρ 着色可见 | **周五 AM** |
| **S4** | 打包 | 全测试绿 + PPT 一页进展 | **周五 PM** |

```text
Mon        Tue        Wed        Thu        Fri
 bake      绕障联调    S1✅       TRT+GL     S3/S4 录屏答辩
 接口      示例+test   TRT开工    ρ着色
```

---

## 3. 分工（并行，不串行）

| 成员 | 本周唯一主线 |
|------|--------------|
| **A** | `model.collide_impl`；`solver` 接 bake + collide 分支；周五更新 `DESIGN.md` 两段 |
| **B** | `bake_box` kernel；`collide_trt` kernel；`test_obstacle_smoke` 数值项；`test_trt_rest.py` |
| **C** | `fluid_grid_lbm_obstacle.py`；GL ρ 着色；录屏 / PNG / VTK |

**kernels.py 锁区**：B 改 collide/bake；C **不改** kernels，只改示例与 `lbm_flow.py`。

**每日 15 min 站会**：只报 blocker + 今晚 PR 目标。

---

## 4. 场景选型（为省时间）

**推荐：顶盖腔 + 中央方柱**（复用 `configure_cavity_walls`，少做新 BC）

```text
1. configure_cavity_walls() + set_lid_velocity(U_lid)
2. bake_box(中心, half_extents)  → solid=1
3. init_uniform(ρ=1, u=0)
4. 500 步 → 柱后应有回流/尾迹
```

**不做**：单独通道流 + inlet/outlet（BC 调试耗 1–2 天）。

---

## 5. 逐日计划

### 周一 — 接口 + bake

| 人 | 任务 | Done 定义 |
|----|------|-----------|
| A | `model.collide_impl: "bgk"\|"trt"`；`solver.bake_box` 空壳接 kernel | PR 可 review |
| B | `bake_box` kernel 写 `solid[i,j,k]=1` | 32³ 单格点 bake 单元测或脚本可见 |
| C | `fluid_grid_lbm_obstacle.py` 骨架（复制 cavity，改 setup） | `--viewer null` 能 import 跑 0 步 |

### 周二 — 绕障跑通

| 人 | 任务 | Done 定义 |
|----|------|-----------|
| A | solver 调通 bake；障碍示例接 `domain.solver.bake_box` | 柱体在 solid 数组中为 1 |
| B | 确认 `stream_pull` bounce-back 与 solid 联调 | 固体上 max\|u\|≈0 |
| C | 写 `test_obstacle_smoke.py` 框架 | 测试文件存在，可先 xfail |

### 周三 — S1 验收（绕障）

| 检查项 | 标准 |
|--------|------|
| 稳定 | 无 NaN/Inf |
| 不穿透 | 固体内 max\|u\| < 1e-5 |
| 绕流 | 柱后 max\|u\| > 0.01 |
| 回归 | `test_lbm_rest.py`、`test_cavity_smoke.py` 仍过 |

```powershell
python wanphys/tests/test_obstacle_smoke.py
python wanphys/examples/fluid_grid_lbm_obstacle.py --viewer null --num-frames 100 --test
```

**B 下午开始 TRT**：抄 taichi / Krüger 表，先 CPU 核对再写 kernel。

### 周四 — S2 TRT + GL 着色

| 人 | 任务 | Done 定义 |
|----|------|-----------|
| A | `solver.step` 按 `collide_impl` launch | bgk/trt 切换无需改示例其它代码 |
| B | `collide_trt` + `test_trt_rest.py`（同 M1 阈值） | TRT 静止流过 |
| C | `lbm_flow.py`：体渲染用 `0.7*speed + 0.3*|ρ-1|` 或类似；固体 wireframe | 腔体 GL 肉眼更「有体积」 |

**TRT 来不及**：只交 `collide_impl` 接口 + 文档写「MRT/TRT 骨架」+ BGK 默认；答辩演示仍用绕障 BGK。

### 周五 — S3/S4 打包

| 时段 | 全员 |
|------|------|
| AM | 绕障 `--viewer gl` 录屏 15–30 s；导出 1 份 VTK + 1 张 PNG |
| PM | 全测试；PPT 更新 1 页（S1–S3 截图）；`post_midterm_plan.md` 打 ✅ |

---

## 6. 验收命令清单

```powershell
cd WanPhys-dev
.venv_lbm/Scripts/Activate.ps1

# 回归
python wanphys/tests/test_lbm_rest.py
python wanphys/tests/test_cavity_smoke.py

# 新增
python wanphys/tests/test_obstacle_smoke.py
python wanphys/tests/test_trt_rest.py          # 周四起

# 示例
python wanphys/examples/fluid_grid_lbm_obstacle.py --viewer null --num-frames 100 --test
python wanphys/examples/fluid_grid_lbm_obstacle.py --viewer gl --collide trt --grid-size 32
```

---

## 7. 新增文件（最小集）

```text
kernels.py          + bake_box, collide_trt
solver.py           + bake_box 实现, collide 分支
model.py            + collide_impl: str = "bgk"
examples/fluid_grid_lbm_obstacle.py
tests/test_obstacle_smoke.py
tests/test_trt_rest.py
fluid_viewer/lbm_flow.py   # ρ/speed 着色（C）
```

**optional（P1）**：`mrt.py`、`collide_mrt`、`bake_sphere`

---

## 8. 风险与砍 scope 顺序

| 优先级 | 砍什么 |
|--------|--------|
| 1 | `bake_sphere`、BGK/MRT 对比图 |
| 2 | 完整 MRT → 只留 TRT |
| 3 | TRT → 只留 `collide_impl` 接口 + 设计说明 |
| 4 | GL 增强 → 退回现有 \|u\| 体渲染 + matplotlib 绕障 PNG |

**绝不砍**：`bake_box`、绕障 smoke、现有 M1/M2 回归。

---

## 9. 已知不足（答辩 30 秒版）

- 无外部对照；靠 smoke + 录屏  
- GL 液体感 = **着色技巧**，非自由面 LBM  
- TRT 是一周版「更好算法」；MRT 为扩展项  
- 障碍仅 box；场景为腔体+柱，非通用 CFD 边界  

---

## 10. 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-06-10 | 初版 4 周计划 |
| 1.1 | 2026-06-10 | **压缩为 1 周**；MRT→TRT 优先；场景改为腔体+柱 |
