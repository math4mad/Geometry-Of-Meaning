# Phase 2 v3 — 7B QLoRA v7 (3 ep + LR 3e-5)

**Hypothesis (H-4')**: 改 LR + 增加训练 epoch, 7B 端是否能在 λ=1~5 区间跨过 +0.05 阈值

**Result**: **H-4' 仍 FAIL** (Δ_max = +0.032 < 0.05) · **v7 比 v6 反而差 -0.005**

## 完整 8-arm 数据

| Pool | baseline | λ=0.0 | λ=1.0 | λ=2.0 | λ=5.0 | best Δ | H-4' |
|---|---|---|---|---|---|---|---|
| **last** | 0.049 | 0.058 | 0.078 | 0.080 | **0.081** | **+0.032** (λ=5) | FAIL |
| **mean** | 0.070 | 0.071 | 0.077 | 0.081 | **0.083** | **+0.013** (λ=5) | FAIL |

## 跨版本对比 (best Δ, last_token)

| 版本 | best λ | margin | Δ | H-4' | 备注 |
|---|---|---|---|---|---|
| 0.5B v3 | 1.0 | 0.160 | +0.131 | **PASS** | 2 ep, LR 1e-4 |
| 7B v5 | 1.0 | 0.083 | +0.034 | FAIL | 2 ep, LR 5e-5 |
| 7B v6 | 5.0 | 0.086 | +0.037 | FAIL | 2 ep, LR 5e-5 |
| **7B v7** | 5.0 | 0.081 | **+0.032** | FAIL | **3 ep, LR 3e-5** (反向) |

## v7 vs v6 诊断

v7 (3 ep + LR 3e-5) 相对 v6 (2 ep + LR 5e-5) **反向 -0.005 (last) / -0.002 (mean)**。可能原因：

1. **3 epoch overfit** — 训练后期过拟合于 LM 任务, 几何任务被遗忘
2. **LR 3e-5 太低** — 模型学不动几何任务 (InfoNCE 下降更慢)
3. **v6 的 +0.037 已接近此架构上限** — 调超参无法跨过 +0.05 阈值

**结论**: 7B 端 H-4' 调超参空间已被穷尽。下一步需**换损失设计 / 换语料 / 换池化**才能尝试突破。

## 实验细节

- 模型: Qwen2.5-7B-Instruct + QLoRA(r=16, α=32, 4bit nf4, bf16)
- 训练: **3 epoch** × stream block 128 + 每 4 步 micro-batch 4 句
- 优化器: AdamW (betas=0.9, 0.95) + **cosine LR 3e-5 → 0** + warmup 50
- 几何损失: cross-sentence InfoNCE τ=0.07 + margin push 0.1×ReLU(cos_neg-0.5)
- 设备: Kaggle T4 16GB
- 产物: `metrics_georeg_7b_v7_full.json` + `figs_georeg_7b/fig_margin_vs_lambda_v7.png`
- Kernel: <https://www.kaggle.com/code/math4amd/geometry-meaning-7b-georeg> (v7)
