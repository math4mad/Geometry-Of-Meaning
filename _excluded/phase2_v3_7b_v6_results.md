# Phase 2 v3 — 7B QLoRA v6 (λ ∈ {0, 1, 2, 5, 10}, 10 arm)

**Hypothesis (H-4')**: 扩 λ 上限 (v5 0.05/0.5/1.0 失败) → 测试 7B 端 λ 是否饱和

**Result**: **H-4' 仍 FAIL** (Δ_max = +0.037 < 0.05) · **但 last 端 λ=5 已饱和, mean 端 λ=10 仍未饱和**

## 10-arm 完整结果

| Pool | baseline | λ=0.0 | λ=1.0 | λ=2.0 | λ=5.0 | λ=10.0 | best Δ | H-4' |
|---|---|---|---|---|---|---|---|---|
| **last** | 0.049 | 0.061 | 0.084 | 0.086 | **0.086** | 0.085 | **+0.037** (λ=5) | FAIL |
| **mean** | 0.070 | 0.072 | 0.078 | 0.082 | 0.084 | **0.085** | **+0.015** (λ=10) | FAIL |

## 0.5B vs 7B 跨尺度对比

| 维度 | 0.5B v3 | 7B v6 | 比例 |
|---|---|---|---|
| baseline (last) | 0.030 | 0.049 | 7B 1.6× |
| λ=1.0 margin (last) | 0.160 | 0.084 | 7B 0.52× |
| **Δ last (best)** | **+0.131** | **+0.037** | **7B 0.28×** |
| **Δ mean (best)** | **+0.061** | **+0.015** | **7B 0.25×** |

## 关键发现

1. **last 端 λ=2-5 饱和** — margin 0.0855 → 0.0861 → 0.0846（不再涨）
2. **mean 端 λ=10 仍未饱和** — 持续上升 0.0784 → 0.0847
3. **H-4' 仍 FAIL** — 7B 端几何正则边际无法跨过 0.05 阈值（即使 λ=10）
4. **7B 端 baseline 远高于 0.5B**（last 0.049 vs 0.030, mean 0.070 vs 0.043）— 剩余 margin 空间小
5. **趋势与 0.5B v3 + 7B v5 一致** — λ↑ margin↑, last > mean

## 跨 v5 / v6 对比

| 池 | v5 (λ=1) Δ | v6 (λ=5) Δ | v6 (λ=10) Δ |
|---|---|---|---|
| last | +0.034 | **+0.037** | +0.036 |
| mean | +0.009 | +0.014 | **+0.015** |

v6 比 v5 多 +0.003 (last) / +0.006 (mean) — 边际递减, 不再值得继续扩 λ。

## 实验细节

- 模型: Qwen2.5-7B-Instruct + QLoRA(r=16, α=32, 4bit nf4, bf16)
- 训练: 2 epoch × stream block 128 + 每 4 步 micro-batch 4 句 (4 concept × 2+1+1)
- 几何损失: cross-sentence InfoNCE τ=0.07 + margin push 0.1×ReLU(cos_neg-0.5)
- 设备: Kaggle T4 16GB
- 总耗时: 10 × 240s ≈ 40 min
- 产物: `metrics_georeg_7b_v6_full.json` + `figs_georeg_7b/fig_margin_vs_lambda_v6.png`
- Kernel: <https://www.kaggle.com/code/math4amd/geometry-meaning-7b-georeg> (v6)

## 结论

**H-4' 7B 端 FAIL** (Δ_max = +0.037, 阈值 0.05) — 几何正则对 7B 端 concept-margin 提升不足。

**Phase 2 总结**:
- **0.5B H-4 PASS** (Δ=+0.131, λ=1.0) — 几何正则在小模型上 work
- **7B H-4' FAIL** (Δ=+0.037, λ=5.0 last) — 几何正则在 7B 边际递减至阈值以下
- **意义**: 验证了几何正则方向对, 但 7B 端需要换 loss 设计 / 备援 loss / 更长训练 / 不同 LR schedule 才能跨过阈值
