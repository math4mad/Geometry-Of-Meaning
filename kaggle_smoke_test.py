#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kaggle Notebook: Geometry-of-Meaning (Large Model)
# 在 Kaggle 中粘贴此文件全部内容到 cells 中运行

# === 配置 ===
# Kernel: Python 3.11 + GPU T4×2
# Internet: ON (需下载模型)
# 依赖: transformers torch umap-learn matplotlib scikit-learn

# ── Cell 1: 安装依赖 ──
!pip install -q umap-learn matplotlib

# ── Cell 2: 导入 + 设备 ──
import json, os, sys, time, io, base64
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}")
if dev == "cuda":
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem/1e9:.1f} GB)")

# ── Cell 3: 加载语料 ──
# 从 Kaggle Dataset 上传 corpus_paraphrase.json, 或内联粘贴
DS_PATH = "/kaggle/input/paraphrase-corpus/corpus_paraphrase.json"

if os.path.exists(DS_PATH):
    with open(DS_PATH) as f:
        corpus = json.load(f)
    print(f"loaded corpus from {DS_PATH}")
else:
    # fallback: 从 GitHub raw 或内联
    print("corpus not found in dataset, please upload corpus_paraphrase.json as Kaggle Dataset")
    sys.exit(1)

paraphrases = corpus["paraphrases"]
sentences = []
labels = []
stages_list = []
variant_indices = []
concept_names = []

for cid, data in paraphrases.items():
    for vi, s in enumerate(data["sentences"]):
        sentences.append(s)
        labels.append(cid)
        stages_list.append(data["stage"])
        variant_indices.append(vi)
        concept_names.append(data.get("name_en") or data["name"])

N = len(sentences)
print(f"sentences: {N} ({len(paraphrases)} concepts × 4 variants)")

# ── Cell 4: 模型 ──
# 待 ide-lola 确认地址后修改
MODEL = "Qwen/Qwen2.5-7B-Instruct"

print(f"\nloading {MODEL}...", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto",
)
model.eval()
n_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
n_heads = model.config.num_attention_heads
print(f"loaded: {n_layers}L/{hidden_dim}H/{n_heads}heads ({time.time()-t0:.0f}s)")

# ── Cell 5: 提取表示 ──
# 每隔 2 层取一层 (7B 有 28 层, 取 15 层)
LAYER_SKIP = 2
total_layers = n_layers + 1
layer_indices = list(range(0, total_layers, LAYER_SKIP))
if (total_layers - 1) not in layer_indices:
    layer_indices.append(total_layers - 1)
n_extracted = len(layer_indices)
print(f"extracting layers: {layer_indices} ({n_extracted} of {total_layers})")

# 概念分组
concept_groups = {}
for i, cid in enumerate(labels):
    if cid not in concept_groups:
        concept_groups[cid] = []
    concept_groups[cid].append(i)

@torch.no_grad()
def extract_one(text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
    out = model(**ids, output_hidden_states=True)
    hs = out.hidden_states
    last_reps = np.zeros((n_extracted, hidden_dim), dtype=np.float32)
    mean_reps = np.zeros((n_extracted, hidden_dim), dtype=np.float32)
    for j, layer in enumerate(layer_indices):
        h = hs[layer][0]
        last_reps[j] = h[-1].float().cpu().numpy()
        mean_reps[j] = h.float().mean(0).cpu().numpy()
    return last_reps, mean_reps

last_token_reps = np.zeros((N, n_extracted, hidden_dim), dtype=np.float32)
mean_pool_reps = np.zeros((N, n_extracted, hidden_dim), dtype=np.float32)

print(f"\nextracting {N} sentences...")
t1 = time.time()
for i in range(N):
    lr, mr = extract_one(sentences[i])
    last_token_reps[i] = lr
    mean_pool_reps[i] = mr
    if (i+1) % 20 == 0 or i == N-1:
        print(f"  [{i+1}/{N}] {time.time()-t1:.0f}s", flush=True)

# ── Cell 6: 几何量化 ──
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

results = {}
for strat, reps in [("last_token", last_token_reps), ("mean_pool", mean_pool_reps)]:
    per_pos, per_neg, per_margin = [], [], []
    for li in range(n_extracted):
        lr = reps[:, li, :]
        pos, neg = [], []
        for cid in concept_groups:
            idx = concept_groups[cid]
            for a in range(len(idx)):
                for b in range(a+1, len(idx)):
                    pos.append(cosine_sim(lr[idx[a]], lr[idx[b]]))
        rng = np.random.default_rng(42)
        cs = sorted(concept_groups.keys())
        samp = rng.choice(len(cs), size=min(20, len(cs)), replace=False)
        for ia in range(len(samp)):
            for ib in range(ia+1, len(samp)):
                neg.append(cosine_sim(lr[concept_groups[cs[samp[ia]]][0]],
                                      lr[concept_groups[cs[samp[ib]]][0]]))
        per_pos.append(float(np.mean(pos)))
        per_neg.append(float(np.mean(neg)))
        per_margin.append(per_pos[-1] - per_neg[-1])

    last3_pos = float(np.mean(per_pos[-3:]))
    last3_neg = float(np.mean(per_neg[-3:]))
    last3_margin = last3_pos - last3_neg
    g1 = last3_pos > 0.75 and last3_pos > last3_neg
    results[strat] = {"pos": per_pos, "neg": per_neg, "margin": per_margin,
                       "last3_pos": last3_pos, "last3_neg": last3_neg,
                       "last3_margin": last3_margin, "G1_pass": g1}
    print(f"\n{strat}: pos={last3_pos:.3f} neg={last3_neg:.3f} margin={last3_margin:.3f} G1={'PASS' if g1 else 'FAIL'}")

# ── Cell 7: 图表 ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)

# 逐层余弦
for idx, (strat, color) in enumerate([("last_token", "#21918c"), ("mean_pool", "#b8860b")]):
    ax = axes[0][idx]
    x = np.arange(n_extracted)
    ax.plot(x, results[strat]["pos"], "s-", color=color, lw=1.6, ms=4, label="Same concept (pos)")
    ax.plot(x, results[strat]["neg"], "^-", color="#666", lw=1.6, ms=4, label="Diff concept (neg)")
    ax.axhline(0.75, color="#d62728", ls="--", lw=1, label="G-1 threshold")
    ax.set_xlabel("Layer (sampled)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"{strat}: {MODEL.split('/')[-1]}")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(-0.1, 1.05)

# 逐层 margin 对比 (含 0.5B baseline)
for idx, strat in enumerate(["last_token", "mean_pool"]):
    ax = axes[1][idx]
    ax.plot(np.arange(n_extracted), results[strat]["margin"], "o-", color="#440154", lw=1.6, ms=4, label=f"{n_layers}L model")
    baseline_margin = 0.031 if strat == "last_token" else 0.049
    ax.axhline(baseline_margin, color="#999", ls="--", lw=1, label="0.5B baseline")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Margin (pos - neg)")
    ax.set_title(f"{strat}: margin vs 0.5B baseline")
    ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig("/kaggle/working/fig_smoke_large.png", dpi=150)
plt.close(fig)
print("saved: /kaggle/working/fig_smoke_large.png")

# UMAP
try:
    import umap
    HAS = True
except:
    from sklearn.manifold import TSNE
    HAS = False

fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
for idx, layer in enumerate([0, n_extracted-1]):
    lr = last_token_reps[:, layer, :]
    if HAS:
        emb = umap.UMAP(n_components=2, random_state=42, n_neighbors=15).fit_transform(lr)
    else:
        emb = TSNE(n_components=2, random_state=42, perplexity=15).fit_transform(lr)
    ax = axes[idx]
    colors = plt.cm.tab20(np.linspace(0, 1, len(concept_groups)))
    for ci, (cid, idxs) in enumerate(concept_groups.items()):
        ax.scatter(emb[idxs, 0], emb[idxs, 1], c=[colors[ci]], s=25, alpha=0.7)
    ax.set_title(f"Layer {layer_indices[layer]}")
fig.tight_layout()
fig.savefig("/kaggle/working/fig_smoke_umap.png", dpi=150)
plt.close(fig)
print("saved: /kaggle/working/fig_smoke_umap.png")

# ── Cell 8: 保存结果 ──
np.savez_compressed("/kaggle/working/smoke_reps.npz",
    last_token=last_token_reps, mean_pool=mean_pool_reps,
    labels=np.array(labels), stages=np.array(stages_list),
    variant_idx=np.array(variant_indices),
    sentences=np.array(sentences, dtype=object),
    layer_indices=np.array(layer_indices),
    n_layers=n_layers, hidden_dim=hidden_dim)

metrics = {
    "model": MODEL, "n_layers": n_layers, "hidden_dim": hidden_dim,
    "n_sentences": N, "layer_indices": layer_indices,
    "strategies": results,
    "vs_05b": {"last_token_margin": 0.031, "mean_pool_margin": 0.049},
}
with open("/kaggle/working/smoke_metrics.json", "w") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=1)
print("saved: /kaggle/working/smoke_metrics.json")
print("\n=== Summary ===")
for strat in ["last_token", "mean_pool"]:
    r = results[strat]
    print(f"  {strat}: pos={r['last3_pos']:.3f} neg={r['last3_neg']:.3f} "
          f"margin={r['last3_margin']:.3f} G1={'PASS' if r['G1_pass'] else 'FAIL'}")
