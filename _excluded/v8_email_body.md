Subject: Phase 2 v3 ✅ — 7B v8 H-4' PASS (last_token pool)

# Phase 2 v3 · 7B v8 突破 H-4' 阈值

## 🎉 核心结论

**7B v8 H-4' PASS** (Δ=+0.058, 阈值 0.05) — 跨过 7B 端的几何正则化阈值

## v8 完整数据 (10 arm)

| Pool | baseline | λ=0.0 | λ=0.5 | λ=1.0 | λ=2.0 | λ=5.0 | best Δ | H-4' |
|---|---|---|---|---|---|---|---|---|
| **last** | 0.049 | 0.061 | 0.093 | 0.102 | 0.106 | **0.107** | **+0.058** | **PASS** ✅ |
| **mean** | 0.070 | 0.074 | 0.083 | 0.088 | 0.096 | **0.102** | +0.032 | FAIL |

## 跨版本 Δ 对比 (last_token)

| 版本 | best λ | margin | Δ | H-4' | 备注 |
|---|---|---|---|---|---|
| 0.5B v3 | 1.0 | 0.160 | +0.131 | **PASS** | 2 ep, LR 1e-4 |
| 7B v5 | 1.0 | 0.083 | +0.034 | FAIL | 2 ep, LR 5e-5 |
| 7B v6 | 5.0 | 0.086 | +0.037 | FAIL | 2 ep, LR 5e-5 |
| 7B v7 | 5.0 | 0.081 | +0.032 | FAIL | 3 ep, LR 3e-5 (退步) |
| **7B v8** | 5.0 | **0.107** | **+0.058** | **PASS** ✅ | **3 ep, LR 5e-5** |

## 关键发现

- **3 ep + LR 5e-5** 是 7B 端最佳组合
- v7 退步根因 = LR 3e-5 太低 (不是 overfit)
- λ=1.0 已跨阈值 (+0.053), λ=5.0 best (+0.058)
- mean_pool 也涨到 +0.032 (v7 +0.013), 但仍未跨
- 7B 达到 0.5B 44% 的 margin gain → 几何归纳偏置**尺度可迁移**

## 产物

- GitHub: <https://github.com/math4mad/Geometry-Of-Meaning>
- gh-pages: <https://math4mad.github.io/Geometry-Of-Meaning/>
- 报告: `phase2_v3_7b_v8.qmd` + `phase2_overview.qmd`
- 数据: `metrics_georeg_7b_v8_full.json` (6.3 KB)
- 图: `figs_georeg_7b/fig_margin_vs_lambda_v8.png`
- Kernel: <https://www.kaggle.com/code/math4amd/geometry-meaning-7b-georeg> (v8)
