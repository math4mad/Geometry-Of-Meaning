#!/usr/bin/env python3
"""Phase 5 · 消融对比 — 几何记忆 vs 联想记忆

对比两种训练策略下, 模型深层表示对同概念不同表达的同构度:

  模型 A (几何记忆导向): 交错混合 + 双向边/自环 + 较高 LR + 权重衰减
  模型 B (联想记忆导向): 单一固定模板, 低 LR, 无正则

判据:
  G-4: 模型 A 深层坍缩度 > 模型 B (margin_A > margin_B)

复用 transformer_vis/train_c42.py 的 LoRA 训练架构:
  LoRA r=16, α=32, 6 epochs, lr=1e-4, Adam β2=0.95
  模型 A: lr=3e-4, weight_decay=0.01, 交错混合 + 自环
  模型 B: lr=1e-5, weight_decay=0, 单一模板

输出: metrics_ablation.json + figs_ablation/
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus_paraphrase.json")
MODEL = os.path.expanduser("~/Programming/code-2026/first-lora/model")
FIGS = os.path.join(HERE, "figs_ablation")
os.makedirs(FIGS, exist_ok=True)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {dev}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# ── 加载语料 ──────────────────────────────────────────────────────────────
with open(CORPUS, encoding="utf-8") as f:
    corpus = json.load(f)

paraphrases = corpus["paraphrases"]
concept_ids = sorted(paraphrases.keys())
n_concepts = len(concept_ids)

# 展平所有句子和标签
all_sentences = []
all_labels = []
for cid in concept_ids:
    for s in paraphrases[cid]["sentences"]:
        all_sentences.append(s)
        all_labels.append(cid)
N = len(all_sentences)

# ── 训练数据构建 ────────────────────────────────────────────────────────────
def build_stream_geometric():
    """模型 A: 交错混合 + 自环 + 双向边
    - 原句 + 改写交错
    - 自环: 每个概念加 (name, is, name) 句
    - 双向: 正反两种句式
    """
    parts = []
    rng = np.random.default_rng(13)
    indices = list(range(n_concepts))
    rng.shuffle(indices)

    for ci in indices:
        cid = concept_ids[ci]
        data = paraphrases[cid]
        name = data["name"]
        sents = data["sentences"]
        stage = data["stage"]

        # 自环句
        parts.append(f"{name}就是{name}。")
        # 原句
        parts.append(sents[0])
        # 改写 1 (正向)
        parts.append(sents[1])
        # 改写 2 (反向/疑问)
        parts.append(sents[2])
        # 改写 3 (倒装)
        parts.append(sents[3])

    return "\n".join(parts)


def build_stream_associative():
    """模型 B: 单一固定模板, 只用原句重复
    - 固定顺序
    - 只原句
    - 无自环, 无改写
    """
    parts = []
    for ci in range(n_concepts):
        cid = concept_ids[ci]
        data = paraphrases[cid]
        parts.append(data["sentences"][0])  # 只原句
        parts.append(data["sentences"][0])  # 重复

    return "\n".join(parts)


# ── LoRA 训练 ────────────────────────────────────────────────────────────────
BLOCK = 128
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

def train_arm(stream_text, label, lr, weight_decay, epochs=6):
    """训练 LoRA, 返回模型"""
    torch.manual_seed(13)
    print(f"  training {label} (lr={lr}, wd={weight_decay})...", flush=True)
    t0 = time.time()

    base = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(dev)
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=TARGETS,
                     lora_dropout=0.0, task_type="CAUSAL_LM")
    lm = get_peft_model(base, cfg)

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(stream_text, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    blocks = ids[0].unfold(0, BLOCK, BLOCK).clone()

    opt = torch.optim.AdamW(lm.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    lm.train()
    losses = []

    for ep in range(epochs):
        ep_loss, nb = 0.0, 0
        for blk in blocks:
            x = blk.unsqueeze(0)
            opt.zero_grad()
            out = lm(x, labels=x)
            out.loss.backward()
            opt.step()
            ep_loss += float(out.loss)
            nb += 1
        losses.append(ep_loss / nb)
        print(f"    {label} ep={ep+1}/{epochs} loss={losses[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)

    lm.eval()
    return lm, losses


# ── 表示提取 (训后) ──────────────────────────────────────────────────────────
@torch.no_grad()
def extract_reps(lm, sentences):
    """提取所有句子的末层 + 倒数第二层表示 (last-token)"""
    tok = AutoTokenizer.from_pretrained(MODEL)
    last_layer_reps = []
    penultimate_reps = []

    for s in sentences:
        ids = tok(s, return_tensors="pt", add_special_tokens=False).to(dev)
        out = lm(**ids, output_hidden_states=True)
        hs = out.hidden_states
        last_layer_reps.append(hs[-1][0, -1].float().cpu().numpy())
        penultimate_reps.append(hs[-2][0, -1].float().cpu().numpy())

    return np.array(last_layer_reps), np.array(penultimate_reps)


def compute_margin(reps, labels, concept_groups):
    """计算正样本余弦均值 - 负样本余弦均值"""
    pos_sims, neg_sims = [], []
    # 正样本
    for cid in concept_groups:
        indices = concept_groups[cid]
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                v1, v2 = reps[indices[a]], reps[indices[b]]
                pos_sims.append(float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)))
    # 负样本 (采样)
    rng = np.random.default_rng(42)
    sampled = rng.choice(len(concept_groups), size=min(20, len(concept_groups)), replace=False)
    cids = sorted(concept_groups.keys())
    for i_a in range(len(sampled)):
        for i_b in range(i_a + 1, len(sampled)):
            ia = concept_groups[cids[sampled[i_a]]][0]
            ib = concept_groups[cids[sampled[i_b]]][0]
            v1, v2 = reps[ia], reps[ib]
            neg_sims.append(float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)))
    return float(np.mean(pos_sims)), float(np.mean(neg_sims))


# ── 概念分组 ────────────────────────────────────────────────────────────────
concept_groups = {}  # concept_id -> [sentence indices in all_sentences]
for i, cid in enumerate(all_labels):
    if cid not in concept_groups:
        concept_groups[cid] = []
    concept_groups[cid].append(i)

# ── Baseline (无训练) ──────────────────────────────────────────────────────────
print("\n=== Baseline (base model, no LoRA) ===", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(dev)
base_model.eval()
base_last, base_pen = extract_reps(base_model, all_sentences)
base_pos, base_neg = compute_margin(base_last, concept_groups, concept_groups)
base_margin = base_pos - base_neg
print(f"  baseline: pos={base_pos:.3f} neg={base_neg:.3f} margin={base_margin:.3f}")
del base_model
torch.mps.empty_cache() if dev == "mps" else None

# ── 模型 A: 几何记忆导向 ───────────────────────────────────────────────────────
print("\n=== Model A: Geometric Memory (interleaved + self-loop + high lr + wd) ===", flush=True)
stream_a = build_stream_geometric()
lm_a, losses_a = train_arm(stream_a, "geom", lr=3e-4, weight_decay=0.01, epochs=6)
a_last, a_pen = extract_reps(lm_a, all_sentences)
a_pos, a_neg = compute_margin(a_last, concept_groups, concept_groups)
a_margin = a_pos - a_neg
print(f"  model A: pos={a_pos:.3f} neg={a_neg:.3f} margin={a_margin:.3f}")
del lm_a
torch.mps.empty_cache() if dev == "mps" else None

# ── 模型 B: 联想记忆导向 ───────────────────────────────────────────────────────
print("\n=== Model B: Associative Memory (single template + low lr + no wd) ===", flush=True)
stream_b = build_stream_associative()
lm_b, losses_b = train_arm(stream_b, "assoc", lr=1e-5, weight_decay=0.0, epochs=6)
b_last, b_pen = extract_reps(lm_b, all_sentences)
b_pos, b_neg = compute_margin(b_last, concept_groups, concept_groups)
b_margin = b_pos - b_neg
print(f"  model B: pos={b_pos:.3f} neg={b_neg:.3f} margin={b_margin:.3f}")
del lm_b
torch.mps.empty_cache() if dev == "mps" else None

# ── 判据裁决 ────────────────────────────────────────────────────────────────
g4_pass = a_margin > b_margin
print(f"\n=== Verdict ===")
print(f"  G-4 (margin_A > margin_B): {'PASS' if g4_pass else 'FAIL'} ({a_margin:.3f} vs {b_margin:.3f})")

# ── 图 ────────────────────────────────────────────────────────────────────────
# 图 1: margin 对比柱状图
fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
labels_bar = ["Baseline", "Model A\n(Geometric)", "Model B\n(Associative)"]
margins = [base_margin, a_margin, b_margin]
colors = ["#666666", "#21918c", "#b8860b"]
ax.bar(labels_bar, margins, color=colors, edgecolor="#333333", linewidth=0.6)
ax.set_ylabel("Margin (pos - neg cosine)")
ax.set_title("G-4: Geometric vs Associative Memory Margin")
for i, v in enumerate(margins):
    ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_ablation_margin.png"))
plt.close(fig)

# 图 2: 训练 loss 对比
fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
ax.plot(range(1, 7), losses_a, "s-", color="#21918c", lw=1.6, ms=4, label="Model A (Geometric)")
ax.plot(range(1, 7), losses_b, "^-", color="#b8860b", lw=1.6, ms=4, label="Model B (Associative)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Block Loss")
ax.set_title("Training Loss: Geometric vs Associative")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_ablation_loss.png"))
plt.close(fig)

# 图 3: 逐层 margin 对比 (baseline vs A vs B)
# 提取全层 margin for A and B
@torch.no_grad()
def extract_all_layers(lm, sentences, n_layers):
    tok = AutoTokenizer.from_pretrained(MODEL)
    all_reps = []
    for s in sentences:
        ids = tok(s, return_tensors="pt", add_special_tokens=False).to(dev)
        out = lm(**ids, output_hidden_states=True)
        hs = out.hidden_states
        all_reps.append(np.stack([h[0, -1].float().cpu().numpy() for h in hs]))
    return np.array(all_reps)  # (N, n_layers+1, hidden_dim)

# ── 保存 ────────────────────────────────────────────────────────────────────
metrics = {
    "minted": "2026-10-15",
    "model": "Qwen2.5-0.5B-Instruct + LoRA(r=16, α=32)",
    "epochs": 6,
    "baseline": {"pos": base_pos, "neg": base_neg, "margin": base_margin},
    "model_a_geometric": {
        "lr": 3e-4, "weight_decay": 0.01,
        "strategy": "interleaved + self-loop + bidirectional",
        "pos": a_pos, "neg": a_neg, "margin": a_margin,
        "losses": losses_a,
    },
    "model_b_associative": {
        "lr": 1e-5, "weight_decay": 0.0,
        "strategy": "single template, repeated",
        "pos": b_pos, "neg": b_neg, "margin": b_margin,
        "losses": losses_b,
    },
    "G4_pass": bool(g4_pass),
    "verdict": f"margin_A={a_margin:.3f} > margin_B={b_margin:.3f}" if g4_pass else f"margin_A={a_margin:.3f} <= margin_B={b_margin:.3f}",
}

out = os.path.join(HERE, "metrics_ablation.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=1)

print(f"\nDONE — {out}")
print(f"  figs: {FIGS}/")
print(f"  baseline margin: {base_margin:.3f}")
print(f"  model A (geometric) margin: {a_margin:.3f}")
print(f"  model B (associative) margin: {b_margin:.3f}")
print(f"  G-4: {'PASS' if g4_pass else 'FAIL'}")
