7B QLoRA + 几何正则化 v6 实验已完成 (10 arm, T4 16GB)
verdict: 7B geometric regularization does not help
lambdas: [0.0, 1.0, 2.0, 5.0, 10.0]

[last]  baseline=0.0491
  λ= 0.0: margin=0.0608  Δ=+0.0117  lm_ep2=2.139  info_nce_ep2=0.000  elapsed=181s  aborted=False
  λ= 1.0: margin=0.0835  Δ=+0.0344  lm_ep2=2.168  info_nce_ep2=1.061  elapsed=238s  aborted=False
  λ= 2.0: margin=0.0855  Δ=+0.0364  lm_ep2=2.179  info_nce_ep2=1.046  elapsed=240s  aborted=False
  λ= 5.0: margin=0.0861  Δ=+0.0370  lm_ep2=2.194  info_nce_ep2=1.042  elapsed=240s  aborted=False
  λ=10.0: margin=0.0846  Δ=+0.0356  lm_ep2=2.201  info_nce_ep2=1.049  elapsed=240s  aborted=False
  best λ=5.0  margin=0.0861  Δ=+0.0370  FAIL

[mean]  baseline=0.0699
  λ= 0.0: margin=0.0715  Δ=+0.0016  lm_ep2=2.144  info_nce_ep2=0.000  elapsed=191s  aborted=False
  λ= 1.0: margin=0.0784  Δ=+0.0085  lm_ep2=2.143  info_nce_ep2=1.265  elapsed=240s  aborted=False
  λ= 2.0: margin=0.0819  Δ=+0.0120  lm_ep2=2.156  info_nce_ep2=1.251  elapsed=241s  aborted=False
  λ= 5.0: margin=0.0838  Δ=+0.0139  lm_ep2=2.175  info_nce_ep2=1.240  elapsed=240s  aborted=False
  λ=10.0: margin=0.0847  Δ=+0.0148  lm_ep2=2.185  info_nce_ep2=1.236  elapsed=240s  aborted=False
  best λ=10.0  margin=0.0847  Δ=+0.0148  FAIL

--- 0.5B vs 7B 对比 (H-4 端) ---
0.5B v3 (last, λ=1.0):  margin=0.160  Δ=+0.131  PASS
7B v6  (last, λ=5.0):   margin=0.086  Δ=+0.037  FAIL (Δ<0.05)
7B v6  (mean, λ=10.0):  margin=0.085  Δ=+0.015  FAIL

--- 关键发现 ---
- last 端 λ=2-5 饱和 (margin 0.0855->0.0861->0.0846)
- mean 端 λ=10 仍未饱和 (持续上升 0.0784->0.0847)
- 7B 端 baseline 更高 (0.049/0.070 vs 0.5B 0.030/0.043)
- H-4' 仍 FAIL, 7B 端几何正则边际无法跨过 0.05 阈值

--- 产物 ---
metrics: metrics_georeg_7b_v6_full.json
figs:    figs_georeg_7b/fig_margin_vs_lambda.png
kernel:  https://www.kaggle.com/code/math4amd/geometry-meaning-7b-georeg
github:  https://github.com/math4mad/Geometry-Of-Meaning