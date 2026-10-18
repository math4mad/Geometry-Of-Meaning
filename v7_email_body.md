# 7B QLoRA v7 实验完成

verdict: 7B geometric regularization does not help
epochs: 3, LR: 3e-05
lambdas: [0.0, 1.0, 2.0, 5.0]

## 完整 8-arm 数据

### [last]  baseline=0.0491

| λ | margin | Δ | lm_ep_last | info_nce_ep_last |
|---|---|---|---|---|
| 0.0 | 0.0576 | +0.0085 | 2.004 | 0.000 |
| 1.0 | 0.0777 | +0.0286 | 2.041 | 0.957 |
| 2.0 | 0.0801 | +0.0310 | 2.060 | 0.938 |
| 5.0 | 0.0814 | +0.0324 | 2.078 | 0.923 |

**best λ=5.0** margin=0.0814 Δ=+0.0324 FAIL

### [mean]  baseline=0.0699

| λ | margin | Δ | lm_ep_last | info_nce_ep_last |
|---|---|---|---|---|
| 0.0 | 0.0714 | +0.0016 | 2.006 | 0.000 |
| 1.0 | 0.0774 | +0.0075 | 2.015 | 1.238 |
| 2.0 | 0.0805 | +0.0106 | 2.024 | 1.213 |
| 5.0 | 0.0826 | +0.0127 | 2.062 | 1.195 |

**best λ=5.0** margin=0.0826 Δ=+0.0127 FAIL

## 跨版本对比 (best Δ, last_token)

| 版本 | best λ | margin | Δ | H-4' |
|---|---|---|---|---|
| 0.5B v3 | 1.0 | 0.160 | +0.131 | **PASS** |
| 7B v5 | 1.0 | 0.083 | +0.034 | FAIL |
| 7B v6 | 5.0 | 0.086 | +0.037 | FAIL |
| **7B v7** | 5.0 | 0.081 | +0.032 | FAIL |

## v7 vs v6

- v7 (3 ep + LR 3e-5) 相对 v6 (2 ep + LR 5e-5) **反向 -0.005 (last) / -0.002 (mean)**
- 3 epoch 可能 overfit; LR 3e-5 可能太低
- v6 的 +0.037 接近此架构上限, 调超参无法跨过 +0.05

## 决策选项

- A: 接受 7B H-4' FAIL, 结束 Phase 2 7B 端 (写总结报告)
- B: 改 loss 设计 (hard negative / SupCon / 重加权 margin push)
- C: 扩 paraphrase corpus (50 -> 100 concept)
- D: 换池化 (attention pool / CLS / 多层 concat)

metrics: metrics_georeg_7b_v7_full.json
kernel: https://www.kaggle.com/code/math4amd/geometry-meaning-7b-georeg
github: https://github.com/math4mad/Geometry-Of-Meaning
gh-pages: https://math4mad.github.io/Geometry-Of-Meaning/