# Design Note — Phase 2 v2: 7B 几何正则化迁移

**Date**: 2026-10-18
**Author**: math4mad (Geometry-Of-Meaning 实验)
**Status**: 设计稿（待 ide-lola 审阅 + Kaggle T4 端执行）
**Repo**: <https://github.com/math4mad/Geometry-Of-Meaning>
**PoC 结果**: `metrics_georeg_v3.json` + `figs_georeg_v3/`

---

## 0. 背景

Phase 1 已用 Qwen2.5-{0.5B, 7B}-Instruct 提取 50 概念 × 4 变体 = 200 改写语料的内部表征，
发现 7B 在 last_token 池上比 0.5B 出现更明显的概念内聚性（参 [0.5B vs 7B 报告](https://math4mad.github.io/Geometry-Of-Meaning/)）。

Phase 2 目标：验证"在训练时显式加入几何正则（跨句配对 InfoNCE）能否进一步提升同概念内聚 / 异概念分离"。

ide-lola 的关键建议（Phase 1 末轮）：
> "在训练或微调阶段, 显式加入对比损失；选 mean_pool；保留 last_token 池做对比"

---

## 1. 0.5B PoC 复盘（决定 7B 设计的依据）

### 1.1 实验矩阵
- 模型: Qwen2.5-0.5B-Instruct + LoRA(r=16, α=32, target=[q,k,v,o])
- 训练: 2 epoch × stream block (128 tokens) + 每 4 步插 1 个 micro-batch
- Micro-batch: 6 句 / 4 concept / 2 句重复 → 2 对正样本 + 异概念负样本
- 几何损失: cross-sentence InfoNCE, τ=0.07
- 设备: Apple MPS, 总耗时 ~3 min

### 1.2 失败教训（v1 → v2 → v3）
| 版本 | 损失设计 | 结果 | 失败原因 |
|---|---|---|---|
| v1 | block 内 K token 互拉/推开 | H-4 FAIL | 局部一致性代理 ≠ 概念对齐 |
| v2 | micro-batch 完全替代 LM 流 | H-4 FAIL | LM 分布严重错位, loss 6.85 |
| **v3** | **LM 流 + 周期性 micro-batch InfoNCE** | **H-4 PASS** | **修正了 1+2** |

### 1.3 PoC 结果（v3，0.5B）

| Pool | baseline | λ=0.0 | λ=0.05 | λ=0.2 | λ=1.0 | best Δ | H-4 (Δ≥0.05) |
|---|---|---|---|---|---|---|---|
| last_token | 0.030 | 0.019 | 0.025 | 0.043 | **0.160** | **+0.131** | **PASS** |
| mean_pool | 0.043 | 0.037 | 0.039 | 0.045 | **0.104** | **+0.061** | **PASS** |

- λ 单调正向（last 池尤其严格单调）
- InfoNCE 损失稳定下降（λ=1.0 last: 1.26 → 0.70）
- LM 损失几乎不受影响（λ=1.0 lm=1.79 vs baseline 1.69）

### 1.4 关键结论
1. **跨句配对 InfoNCE 是正确代理**（block 内 K token 互相不是）
2. **几何正则能让 last_token 池**反超 mean_pool 池（与 Phase 1 G-1 "mean > last" 反转）
3. **λ 越大 margin 越大**（0.5B 端未见饱和）

---

## 2. 7B 实验设计

### 2.1 目标
**H-4' (7B 版)**：几何正则使 7B 的概念 margin 在至少一个池化方式上提升 ≥ 0.05。

### 2.2 模型
- Base: Qwen2.5-7B-Instruct（已用 4bit 量化跑过 Phase 1）
- 适配: QLoRA (4bit nf4 + double quant + bf16 compute)
- LoRA 参数: r=16, α=32, target=[q,k,v,o] (跟 0.5B 一致)
- 量化: bitsandbytes 4bit（已有 Phase 1 配置验证）
- 训练 dtype: bf16（fp16 在 7B 上梯度不稳定，参考 Phase 1 G-3）

### 2.3 数据
- 复用 0.5B 的 `corpus_paraphrase.json`（50 概念 × 4 变体）
- 训练流: 同一 `build_stream()`，50 概念 200 句
- Eval: 同 200 句 + 同 pos/neg 配对（保持口径一致）

### 2.4 训练超参

| 项 | 0.5B v3 | 7B v1（设计） | 理由 |
|---|---|---|---|
| Optimizer | AdamW | AdamW + paged 8bit | 7B 显存省 |
| LR | 1e-4 | 5e-5 | 7B 标准小 1 半 |
| Warmup | 0 | 50 步 | 7B 起步稳 |
| Epoch | 2 | 2 | 同 0.5B |
| BLOCK | 128 | 256 | 7B 上下文强一倍 |
| MICRO_BATCH 句数 | 6 | 6 | 同 |
| GEO_EVERY (步) | 4 | 4 | 同 |
| 池化 | last + mean | last + mean | 跟 0.5B 对照 |
| TAU (InfoNCE 温度) | 0.07 | 0.07 | 同 |
| λ 扫描 | {0, 0.05, 0.2, 1.0} | **{0, 0.1, 0.5, 1.0}** | 0.5B 已证 λ↑→ margin↑，7B 加密中间值 |
| 备援 margin push | 无 | 0.1 × ReLU(cos_neg - 0.5) | 7B 端负样本 cosine 较 0.5B 更紧，加 margin 更稳 |

### 2.5 评估（与 0.5B 完全同口径）
1. **主指标**: 跨概念 margin = pos_cosine_mean - neg_cosine_mean (50 concept, 20 sampled neg)
2. **副指标**:
   - UMAP 2D 可视化（跟 Phase 1 G-2 同一脚本）
   - 同概念 CKA（取层 14/21/27）
   - 跨概念 attention rollout（跟 Phase 1 A-1 同一脚本）
3. **判定**:
   - H-4' PASS: 至少一个池化 + 某 λ 比 baseline 提升 ≥ 0.05
   - 副产物: 报告「λ=0 vs λ=0.5 vs λ=1.0」三档 UMAP 图

### 2.6 算力 / 成本
- 设备: Kaggle T4 × 16GB
- 单 arm 估算: 2 epoch × 30 block × ~5 s/step ≈ **5 min**（QLoRA + bf16）
- 8 arm: ~40 min + 模型加载 5 min ≈ **45 min 总**
- 比 Phase 1 7B extraction 慢 ~3 倍（因为要训练），但单次 kernel 可容纳

### 2.7 输出
- `metrics_georeg_7b.json`（同 v3 schema）
- `figs_georeg_7b/fig_margin_vs_lambda.png`
- `figs_georeg_7b/fig_umap_lambda_sweep.png`（λ=0/0.5/1.0 三档对比）
- `report_geometry_of_meaning_7b_georeg.qmd`（Phase 2 v2 章节）

---

## 3. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 7B 上 InfoNCE 损失爆炸（log 溢出） | 中 | τ=0.07 已是稳值；加梯度裁剪 1.0 |
| 7B 端 λ=1.0 LM 损失反向上升 | 中 | 监控 lm_loss，若 ep2 > 1.5 × ep1 终止该 arm |
| T4 显存 OOM (QLoRA 4bit + 256 block) | 低 | 备援: block 缩到 128 / batch=4 / gradient_checkpointing |
| 8 arm 跑完发现 λ=0.5 已经最优 | 低 | 7B 端 λ 扫描已含 0.1, 0.5, 1.0 三个值 |
| neg cosine > 0.5 频繁触发 margin push 让 loss 主导 | 中 | margin push 系数 0.1 而非 1.0；监控 |

---

## 4. 时间线

| 阶段 | 任务 | 估计 |
|---|---|---|
| T0 | ide-lola 审阅本设计稿 | 1-2 天 |
| T1 | 改写 `train_georeg_v3.py` → 7B 版（QLoRA + bf16 + 备援 margin） | 0.5 天 |
| T2 | Kaggle kernel 提交 + 跑（1×T4 45 min） | 1 天 |
| T3 | 结果分析 + 写 v2 qmd + 推 gh-pages | 0.5 天 |
| T4 | 总结 + 给 ide-lola 发最终报告 | 0.5 天 |

---

## 5. 复现指引

```bash
# 1. 准备 7B QLoRA 依赖
pip install -q transformers peft bitsandbytes accelerate datasets

# 2. 训练 + 评估
python train_georeg_7b.py 2>&1 | tee /kaggle/working/v3_7b.log

# 3. 看结果
cat /kaggle/working/metrics_georeg_7b.json
ls /kaggle/working/figs_georeg_7b/
```

---

## 6. 给 ide-lola 的开放问题

1. **mean vs last**：0.5B 端 mean 基线高于 last，但几何正则后 last 反超；7B 端是否同向？
2. **λ 上限**：0.5B 端 λ=1.0 仍未饱和（margin 还在升），7B 端要不要扩到 λ=2.0？
3. **备援 margin push**：0.5B 没用上（设计 v3 删了），7B 端是否要保留？
4. **跨句 batch 选句策略**：当前 4 concept × (2 句 + 2 句 + 1 句 + 1 句)，是否改成 (3 句 + 1 句) × 4 concept 提供更多正样本对？

请审阅后回复任意一个微调或确认无误，我即刻起 T1。
