#!/usr/bin/env python3
"""Phase 4 · 注意力机制几何溯源

可视化模型在不同改写变体上的 Attention 权重分布，
检验: 无论输入哪种改写句, Attention 权重是否聚焦并激活同一概念区域。

方法:
  - 对同概念的 4 变体, 提取末层 attention weights
  - Attention Rollout: 累积逐层 attention, 得到 token-level 关注热力图
  - 对比: 浅层 vs 深层的 attention 聚焦模式

输出: figs_attention/ + attention_metrics.json
"""
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus_paraphrase.json")
_local = os.path.expanduser("~/Programming/code-2026/first-lora/model")
MODEL = _local if os.path.isdir(_local) else "Qwen/Qwen2.5-0.5B-Instruct"
FIGS = os.path.join(HERE, "figs_attention")
os.makedirs(FIGS, exist_ok=True)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {dev}")

from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", output_attentions=True
).to(dev)
model.eval()
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
print(f"model: {n_layers} layers, {n_heads} heads")

# ── 加载语料 ──────────────────────────────────────────────────────────────
with open(CORPUS, encoding="utf-8") as f:
    corpus = json.load(f)

paraphrases = corpus["paraphrases"]

# ── Attention 提取 ─────────────────────────────────────────────────────────
@torch.no_grad()
def extract_attention(text):
    """提取末层 attention weights
    返回: (n_heads, seq_len, seq_len) — 末层多头注意力矩阵
    """
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
    out = model(**ids, output_attentions=True)
    # attentions: tuple of (1, n_heads, seq_len, seq_len), len = n_layers
    last_attn = out.attentions[-1][0]  # (n_heads, seq_len, seq_len)
    return last_attn.float().cpu().numpy()

@torch.no_grad()
def extract_attention_rollout(text):
    """Attention Rollout: 累积逐层 attention
    返回: (seq_len, seq_len) — 累积注意力矩阵
    """
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
    out = model(**ids, output_attentions=True)
    # 平均各头, 然后累积
    attns = [a[0].mean(0).float().cpu().numpy() for a in out.attentions]  # list of (seq_len, seq_len)
    # rollout: I + avg_attn, 然后 cumprod
    result = np.eye(attns[0].shape[0])
    for a in attns:
        result = a @ result
    return result

# ── 选取 5 个概念, 画 4 变体的 attention 热力图 ───────────────────────────
selected_cids = list(paraphrases.keys())[:5]  # 前 5 个概念

attention_metrics = {
    "minted": "2026-10-15",
    "model": "Qwen2.5-0.5B-Instruct",
    "selected_concepts": selected_cids,
    "concepts": {},
}

for cid in selected_cids:
    data = paraphrases[cid]
    sentences = data["sentences"]
    name = data.get("name_en") or data["name"]
    name_zh = data["name"]
    print(f"\n=== {cid}: {name} ({name_zh}) ===", flush=True)

    # 4 变体的末层 attention rollout
    rollouts = []
    last_attns = []
    for si, s in enumerate(sentences):
        print(f"  variant {si}: {s[:60]}...", flush=True)
        ro = extract_attention_rollout(s)
        la = extract_attention(s)  # (n_heads, seq, seq)
        rollouts.append(ro)
        last_attns.append(la.mean(0))  # 平均各头

    # 图: 4 变体的 rollout 热力图
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5), dpi=150)
    for vi in range(4):
        ax = axes[vi]
        im = ax.imshow(rollouts[vi], cmap="YlOrRd", aspect="auto")
        ax.set_title(f"Variant {vi}", fontsize=10)
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Token Position")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{cid}: {name} — Attention Rollout (4 variants)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f"fig_attn_{cid}_rollout.png"))
    plt.close(fig)

    # 图: 末层 attention 聚焦模式 (末 token 对其他 token 的关注)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for vi in range(4):
        s = sentences[vi]
        tokens = tok.tokenize(s)
        n_tok = len(tokens)
        # 末 token 对所有 token 的 attention
        last_to_all = rollouts[vi][-1, :n_tok]
        ax.plot(range(n_tok), last_to_all, "o-", ms=4, lw=1.4, label=f"Variant {vi}")
    
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Attention Weight (rollout)")
    ax.set_title(f"{cid}: {name} — Last Token Attention to Previous Tokens")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f"fig_attn_{cid}_focus.png"))
    plt.close(fig)

    # 计算变体间的 attention 相似度
    # 方法: 用末层 attention (非 rollout) 末 token 对前文的关注分布的 cosine similarity
    # 不同变体 token 长度不同, 截取末尾 min_len 个 token 的关注分布
    min_len = min(la.shape[0] for la in last_attns)
    focus_vectors = [la[-1, -min_len:] for la in last_attns]
    
    rollout_sims = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            ra = focus_vectors[a]
            rb = focus_vectors[b]
            rollout_sims[a, b] = float(np.dot(ra, rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-8))

    avg_sim = float(np.mean([rollout_sims[a, b] for a in range(4) for b in range(a+1, 4)]))
    print(f"  avg rollout similarity between variants: {avg_sim:.3f}")

    attention_metrics["concepts"][cid] = {
        "name": name,
        "sentences": sentences,
        "avg_rollout_similarity": avg_sim,
        "rollout_sims": rollout_sims.tolist(),
    }

# ── 跨概念对比 ────────────────────────────────────────────────────────────
# 取 5 个概念各 1 变体, 用末层 attention 末 token 关注分布比较
@torch.no_grad()
def extract_last_attn_focus(text):
    """提取末层 attention 末 token 对所有 token 的关注"""
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
    out = model(**ids, output_attentions=True)
    last_attn = out.attentions[-1][0]  # (n_heads, seq, seq)
    return last_attn.mean(0)[-1].float().cpu().numpy()  # 末行, 平均各头

cross_focus = []
for ci, cid in enumerate(selected_cids):
    s = paraphrases[cid]["sentences"][0]
    cross_focus.append(extract_last_attn_focus(s))

min_cross = min(len(f) for f in cross_focus)
cross_focus = [f[-min_cross:] for f in cross_focus]

cross_sims = np.zeros((5, 5))
for a in range(5):
    for b in range(5):
        cross_sims[a, b] = float(np.dot(cross_focus[a], cross_focus[b]) / 
                                  (np.linalg.norm(cross_focus[a]) * np.linalg.norm(cross_focus[b]) + 1e-8))

# 图: 跨概念 attention 相似度热力图
fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
im = ax.imshow(cross_sims, cmap="YlOrRd", vmin=0, vmax=1)
names = [paraphrases[cid].get("name_en") or paraphrases[cid]["name"] for cid in selected_cids]
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(names, fontsize=9, rotation=20)
ax.set_yticklabels(names, fontsize=9)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{cross_sims[i,j]:.2f}", ha="center", va="center", fontsize=9)
ax.set_title("Cross-concept Attention Rollout Similarity")
plt.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_attn_cross_concept.png"))
plt.close(fig)

# 同概念内 vs 跨概念
intra_sims = [attention_metrics["concepts"][cid]["avg_rollout_similarity"] for cid in selected_cids]
cross_mean = float(np.mean([cross_sims[a, b] for a in range(5) for b in range(a+1, 5)]))
intra_mean = float(np.mean(intra_sims))
print(f"\nintra-concept avg: {intra_mean:.3f}")
print(f"cross-concept avg: {cross_mean:.3f}")
print(f"margin: {intra_mean - cross_mean:.3f}")

attention_metrics["summary"] = {
    "intra_concept_avg": intra_mean,
    "cross_concept_avg": cross_mean,
    "margin": intra_mean - cross_mean,
}

out = os.path.join(HERE, "attention_metrics.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(attention_metrics, f, ensure_ascii=False, indent=1)

print(f"\nDONE — {out}")
print(f"  figs: {FIGS}/")
print(f"  intra={intra_mean:.3f} cross={cross_mean:.3f} margin={intra_mean-cross_mean:.3f}")
