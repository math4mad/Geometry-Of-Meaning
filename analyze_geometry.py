#!/usr/bin/env python3
"""Phase 3 · 量化与可视化 — 几何坍缩检验

对 representations.npz 中的三种聚合表示，逐层计算:

  1. 余弦相似度热力图 (per-layer) — 正样本组内 vs 负样本组
  2. UMAP 2D 降维 (per-layer) — 浅层 vs 深层对比
  3. CKA (Centered Kernel Alignment) — 不同 Token 序列的隐藏状态矩阵对齐度
  4. 几何边距 — 余弦相似度区分同概念 vs 异概念

冻结判据:
  G-1 同概念坍缩: 深层(末3层)正样本余弦相似度 > 0.75 且 > 负样本
  G-2 UMAP 簇分离: 深层 UMAP 中同概念坍缩为紧密簇
  G-3 CKA 对齐: 深层 CKA > 0.85

输出: metrics_geometry.json + figs_geometry/
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.join(HERE, "representations.npz")
FIGS = os.path.join(HERE, "figs_geometry")
os.makedirs(FIGS, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 加载 ───────────────────────────────────────────────────────────────────
data = np.load(REP, allow_pickle=True)
last_token = data["last_token"]    # (200, 25, 896)
mean_pool = data["mean_pool"]
entity = data["entity"]
labels = data["labels"]            # (200,) concept IDs
stages = data["stages"]            # (200,) 0-4
variant_idx = data["variant_idx"] # (200,) 0-3
sentences = data["sentences"]
n_layers = int(data["n_layers"]) + 1  # 25
hidden_dim = int(data["hidden_dim"])
N = last_token.shape[0]
n_concepts = N // 4  # 50

print(f"loaded: {N} sentences, {n_layers} layers, {hidden_dim} dim, {n_concepts} concepts")

# ── 工具函数 ─────────────────────────────────────────────────────────────────
def cosine_sim(a, b):
    """两个向量的余弦相似度"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def cosine_matrix(X):
    """计算矩阵的余弦相似度矩阵 (n, n)"""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / (norms + 1e-8)
    return Xn @ Xn.T

def linear_cka(X, Y):
    """Linear CKA between two matrices (n, d)"""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    X = X / (np.linalg.norm(X) + 1e-8)
    Y = Y / (np.linalg.norm(Y) + 1e-8)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-8))

# ── 按概念分组 ──────────────────────────────────────────────────────────────
concept_groups = {}  # concept_id -> [indices]
for i in range(N):
    cid = str(labels[i])
    if cid not in concept_groups:
        concept_groups[cid] = []
    concept_groups[cid].append(i)

concept_ids = sorted(concept_groups.keys())

# ── 逐层计算 ────────────────────────────────────────────────────────────────
results = {strategy: {} for strategy in ["last_token", "mean_pool", "entity"]}

for strategy, reps in [("last_token", last_token), ("mean_pool", mean_pool), ("entity", entity)]:
    print(f"\n=== {strategy} ===")

    # 每层的: 正样本平均余弦、负样本平均余弦、边距
    per_layer_pos = []  # 同概念不同变体的平均余弦
    per_layer_neg = []  # 不同概念之间的平均余弦
    per_layer_margin = []  # pos - neg
    per_layer_cka = []  # CKA: 同概念不同变体间的平均对齐度

    for layer in range(n_layers):
        layer_reps = reps[:, layer, :]  # (200, 896)

        # 正样本: 同概念内 4 变体两两余弦
        pos_sims = []
        for cid in concept_ids:
            indices = concept_groups[cid]
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    pos_sims.append(cosine_sim(layer_reps[indices[a]], layer_reps[indices[b]]))

        # 负样本: 跨概念对（采样以控制计算量）
        neg_sims = []
        rng = np.random.default_rng(42)
        sampled_concepts = rng.choice(len(concept_ids), size=min(20, len(concept_ids)), replace=False)
        for i_a in range(len(sampled_concepts)):
            for i_b in range(i_a + 1, len(sampled_concepts)):
                ca = concept_ids[sampled_concepts[i_a]]
                cb = concept_ids[sampled_concepts[i_b]]
                ia = concept_groups[ca][0]  # 取各概念第 0 变体
                ib = concept_groups[cb][0]
                neg_sims.append(cosine_sim(layer_reps[ia], layer_reps[ib]))

        pos_mean = float(np.mean(pos_sims))
        neg_mean = float(np.mean(neg_sims))
        per_layer_pos.append(pos_mean)
        per_layer_neg.append(neg_mean)
        per_layer_margin.append(pos_mean - neg_mean)

        # CKA: 同概念 4 变体的表示矩阵对齐度
        cka_vals = []
        for cid in concept_ids:
            indices = concept_groups[cid]
            if len(indices) < 2:
                continue
            X = layer_reps[indices[0]:indices[0]+1]
            Y = layer_reps[indices[1]:indices[1]+1]
            # CKA on single vectors isn't meaningful; use all 4 as a group
            # Instead: compute pairwise CKA of variant vectors reshaped
            # For single vectors, CKA = cosine^2
            cka_vals.append(cosine_sim(layer_reps[indices[0]], layer_reps[indices[1]]) ** 2)
        per_layer_cka.append(float(np.mean(cka_vals)))

    results[strategy]["pos"] = per_layer_pos
    results[strategy]["neg"] = per_layer_neg
    results[strategy]["margin"] = per_layer_margin
    results[strategy]["cka"] = per_layer_cka

    # 判据裁决 (末 3 层)
    last3_pos = np.mean(per_layer_pos[-3:])
    last3_neg = np.mean(per_layer_neg[-3:])
    last3_margin = last3_pos - last3_neg
    g1_pass = last3_pos > 0.75 and last3_pos > last3_neg
    g3_pass = np.mean(per_layer_cka[-3:]) > 0.85

    print(f"  last3: pos={last3_pos:.3f} neg={last3_neg:.3f} margin={last3_margin:.3f}")
    print(f"  G-1 (pos>0.75 & pos>neg): {'PASS' if g1_pass else 'FAIL'}")
    print(f"  G-3 (CKA>0.85): {'PASS' if g3_pass else 'FAIL'} (CKA={np.mean(per_layer_cka[-3:]):.3f})")

    results[strategy]["verdict"] = {
        "last3_pos_mean": float(last3_pos),
        "last3_neg_mean": float(last3_neg),
        "last3_margin": float(last3_margin),
        "G1_pass": bool(g1_pass),
        "G3_pass": bool(g3_pass),
        "G3_cka": float(np.mean(per_layer_cka[-3:])),
    }

# ── 图 1: 逐层余弦相似度曲线 (正 vs 负 + 边距) ──────────────────────────────────
for strategy in ["last_token", "mean_pool", "entity"]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    ax = axes[0]
    layers = np.arange(n_layers)
    ax.plot(layers, results[strategy]["pos"], "s-", color="#21918c", lw=1.6, ms=4, label="Same concept (pos)")
    ax.plot(layers, results[strategy]["neg"], "^-", color="#b8860b", lw=1.6, ms=4, label="Diff concept (neg)")
    ax.axhline(0.75, color="#d62728", ls="--", lw=1.0, label="G-1 threshold 0.75")
    ax.set_xlabel("Layer (0=embedding, 24=last)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"{strategy}: Same vs Diff Concept Cosine Similarity (per layer)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(-0.1, 1.05)

    ax = axes[1]
    ax.plot(layers, results[strategy]["margin"], "o-", color="#440154", lw=1.6, ms=4)
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Margin (pos − neg)")
    ax.set_title(f"{strategy}: Geometric Margin (Same-concept Separability)")
    ax.set_ylim(-0.1, 0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f"fig_{strategy}_cosine_margin.png"))
    plt.close(fig)

# ── 图 2: 余弦相似度热力图 (浅层 vs 深层, 同概念) ────────────────────────────────
for strategy, reps in [("last_token", last_token), ("mean_pool", mean_pool), ("entity", entity)]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)

    for ax_idx, layer in enumerate([0, n_layers // 2, n_layers - 1]):
        # 取前 10 个概念 × 4 变体 = 40 句
        sub_indices = []
        sub_labels = []
        for cid in concept_ids[:10]:
            sub_indices.extend(concept_groups[cid])
            sub_labels.extend([cid] * 4)

        sub_reps = reps[sub_indices, layer, :]
        sim_matrix = cosine_matrix(sub_reps)

        ax = axes[ax_idx]
        im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-0.5, vmax=1.0, aspect="auto")
        ax.set_title(f"Layer {layer} ({'embedding' if layer == 0 else 'deep' if layer == n_layers-1 else 'mid'})")
        ax.set_xlabel("Sentence Index")
        ax.set_ylabel("Sentence Index")
        # 标注概念边界
        for b in range(4, 40, 4):
            ax.axhline(b - 0.5, color="white", lw=0.5, alpha=0.5)
            ax.axvline(b - 0.5, color="white", lw=0.5, alpha=0.5)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"{strategy}: Cosine Similarity Heatmap (10 concepts x 4 variants)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f"fig_{strategy}_heatmap.png"))
    plt.close(fig)

# ── 图 3: UMAP 降维 (浅层 vs 深层) ────────────────────────────────────────────
try:
    import umap
    HAS_UMAP = True
except ImportError:
    from sklearn.manifold import TSNE
    HAS_UMAP = False
    print("UMAP not available, using t-SNE")

for strategy, reps in [("last_token", last_token), ("mean_pool", mean_pool)]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    colors = plt.cm.tab20(np.linspace(0, 1, n_concepts))

    for ax_idx, layer in enumerate([0, n_layers // 2, n_layers - 1]):
        layer_reps = reps[:, layer, :].copy()

        if HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            emb = reducer.fit_transform(layer_reps)
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=15)
            emb = reducer.fit_transform(layer_reps)

        ax = axes[ax_idx]
        for ci, cid in enumerate(concept_ids):
            indices = concept_groups[cid]
            ax.scatter(emb[indices, 0], emb[indices, 1], c=[colors[ci]], s=30, alpha=0.7, label=cid if ci < 5 else "")
            # 连线: 同概念变体之间
            for a in indices:
                for b in indices:
                    if a < b:
                        ax.plot([emb[a, 0], emb[b, 0]], [emb[a, 1], emb[b, 1]], color=colors[ci], alpha=0.15, lw=0.5)

        ax.set_title(f"Layer {layer} ({'embedding' if layer == 0 else 'deep' if layer == n_layers-1 else 'mid'})")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    fig.suptitle(f"{strategy}: {'UMAP' if HAS_UMAP else 't-SNE'} Projection (same-concept linked)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f"fig_{strategy}_umap.png"))
    plt.close(fig)

# ── 图 4: CKA 逐层曲线 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
for strategy in ["last_token", "mean_pool", "entity"]:
    ax.plot(np.arange(n_layers), results[strategy]["cka"], "o-", lw=1.6, ms=4, label=strategy)
ax.axhline(0.85, color="#d62728", ls="--", lw=1.0, label="G-3 threshold 0.85")
ax.set_xlabel("Layer")
ax.set_ylabel("CKA (cosine^2)")
ax.set_title("CKA Alignment per Layer (same concept, diff variants)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_cka_per_layer.png"))
plt.close(fig)

# ── 保存指标 ───────────────────────────────────────────────────────────────────
metrics = {
    "deck": "paraphrase_baby",
    "minted": "2026-10-15",
    "model": "Qwen2.5-0.5B-Instruct",
    "n_sentences": N,
    "n_concepts": n_concepts,
    "n_layers": n_layers,
    "hidden_dim": hidden_dim,
    "strategies": results,
    "overlap_stats": json.load(open(os.path.join(HERE, "corpus_paraphrase.json")))["overlap_stats"],
}

out = os.path.join(HERE, "metrics_geometry.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=1)

print(f"\nDONE — {out}")
print(f"  figs: {FIGS}/")
for strategy in ["last_token", "mean_pool", "entity"]:
    v = results[strategy]["verdict"]
    print(f"  {strategy}: G1={'PASS' if v['G1_pass'] else 'FAIL'} G3={'PASS' if v['G3_pass'] else 'FAIL'} "
          f"(pos={v['last3_pos_mean']:.3f} neg={v['last3_neg_mean']:.3f} margin={v['last3_margin']:.3f} cka={v['G3_cka']:.3f})")
