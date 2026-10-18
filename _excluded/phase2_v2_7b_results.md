# Phase 2 v2 — 7B QLoRA + 几何正则化 (Kaggle T4)

**Hypothesis (H-4)**: 跨句配对 InfoNCE 几何损失能在 7B 上提升 concept-grouped margin

**Result**: **H-4' FAIL** (Δ_max = +0.034 < 0.05) · **但趋势与 0.5B v3 完全一致**

## 完整 8-arm 结果

| Pool | baseline | λ=0.0 | λ=0.1 | λ=0.5 | λ=1.0 | best Δ | H-4' |
|---|---|---|---|---|---|---|---|
| **last_token** | 0.049 | 0.060 | 0.067 | 0.078 | **0.083** | **+0.034** | FAIL (Δ<0.05) |
| **mean_pool** | 0.070 | 0.072 | 0.073 | 0.076 | 0.079 | +0.009 | FAIL |

## 关键发现

1. **单调上升仍成立** — 两个 pool 都严格 λ↑ margin↑（last: 0.011→0.018→0.029→0.034；mean: 0.002→0.003→0.006→0.009）
2. **未饱和** — λ=1.0 边际仍正向（last Δ=0.034 vs 0.5B v3 Δ=0.131 = 0.26× 强度）
3. **last_token 受益更大** — 与 0.5B v3 方向一致（0.5B: last 0.131 > mean 0.061）
4. **7B 端 baseline 远高于 0.5B**（last 0.049 vs 0.030, mean 0.070 vs 0.043）— 剩余 margin 空间小

## 0.5B vs 7B 跨尺度对比

| | 0.5B v3 | 7B v5 | 比例 |
|---|---|---|---|
| baseline (last) | 0.030 | 0.049 | 7B 1.6× |
| baseline (mean) | 0.043 | 0.070 | 7B 1.6× |
| λ=1.0 margin (last) | 0.160 | 0.083 | 7B 0.52× |
| Δ last | +0.131 | +0.034 | 7B 0.26× |
| Δ mean | +0.061 | +0.009 | 7B 0.15× |

**7B 端绝对增益为 0.5B 的 15-26%**。几何正则方向一致，但 7B 端需更激进 λ 才能 H-4 PASS。

## 诊断

- **H-4' FAIL** 不否定几何正则思路（0.5B PASS 已证明）— 是 **7B 端需要更激进的 λ**
- 0.5B v3 端 λ=1.0 仍在单调涨（边际 0.131），7B 端 λ=1.0 边际 0.034 也未饱和
- 推断：7B 端需要 **λ ∈ {1, 2, 5, 10}** 才可能跨过 0.05 阈值

## v6 计划 (Phase 2 v3)

| 项 | 当前 v5 | 建议 v6 |
|---|---|---|
| λ 范围 | {0, 0.1, 0.5, 1.0} | **{0, 1, 2, 5, 10}** |
| 池化 | last + mean | last + mean |
| arm 总数 | 8 | 10 |
| 训练 epoch | 2 | 2 (相同) |
| 设备 | T4 16GB | T4 16GB |
| 预计时长 | 37 min | ~45 min |

## 实验细节

- 模型: Qwen2.5-7B-Instruct + QLoRA(4bit, r=16, α=32)
- 训练流: 2 epoch × stream block 128 + 每 4 步 micro-batch 4 句
- 几何损失: cross-sentence InfoNCE, τ=0.07 + margin push 0.1×ReLU(cos_neg-0.5)
- 设备: Kaggle T4 16GB
- 耗时: 246.8 s (4 min) per pool = **总 ~37 min**
- 产物: `metrics_georeg_7b_kaggle.json` + `figs_georeg_7b/`
- Kernel: <https://www.kaggle.com/code/math4amd/geometry-meaning-7b-georeg>
