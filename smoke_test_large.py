#!/usr/bin/env python3
"""冒烟实验 · 大模型几何坍缩检验

与 Phase 2-3 的 0.5B 实验完全对齐，但支持任意 HuggingFace 模型。
用于验证：更大模型是否展现更强的同概念几何坍缩（更大 margin）。

用法:
  python3 smoke_test_large.py --model /path/to/model
  python3 smoke_test_large.py --model Qwen/Qwen2.5-1.5B-Instruct
  python3 smoke_test_large.py --model /path/to/model --max-sentences 40

设计要点:
  - 模型无关: 自动读取 num_hidden_layers / hidden_size
  - 内存安全: 支持 --max-sentences 限制, 逐句提取后释放
  - 输出格式与 Phase 2-3 兼容, 可直接用 analyze_geometry.py 分析
  - 输出: smoke_reps_{model_tag}.npz + smoke_metrics_{model_tag}.json
"""
import json, os, sys, time, argparse, hashlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus_paraphrase.json")

def parse_args():
    ap = argparse.ArgumentParser(description="大模型冒烟实验")
    ap.add_argument("--model", required=True, help="模型路径或 HuggingFace ID")
    ap.add_argument("--max-sentences", type=int, default=0, help="限制句子数 (0=全部200)")
    ap.add_argument("--layers-skip", type=int, default=1, help="每隔 N 层取一层 (减少内存)")
    ap.add_argument("--strategies", nargs="+", default=["last_token", "mean_pool"],
                    help="聚合策略: last_token mean_pool entity")
    ap.add_argument("--device", default="auto", help="auto / mps / cpu / cuda")
    return ap.parse_args()

args = parse_args()

# ── 设备选择 ──────────────────────────────────────────────────────────────
if args.device == "auto":
    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
else:
    dev = args.device
print(f"device: {dev}")

# ── 模型标签 (用于输出文件名) ──────────────────────────────────────────────
model_tag = os.path.basename(args.model.rstrip("/"))
if not model_tag:
    model_tag = args.model.replace("/", "_")
model_tag = model_tag[:30]  # 截断

# ── 加载模型 ──────────────────────────────────────────────────────────────
print(f"loading model: {args.model}...", flush=True)
t0 = time.time()
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16 if dev != "cpu" else torch.float32,
    attn_implementation="eager",
).to(dev)
model.eval()

n_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
n_heads = model.config.num_attention_heads
print(f"model loaded: {n_layers} layers, {hidden_dim} dim, {n_heads} heads ({time.time()-t0:.0f}s)", flush=True)

# ── 加载语料 ──────────────────────────────────────────────────────────────
with open(CORPUS, encoding="utf-8") as f:
    corpus = json.load(f)

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
if args.max_sentences > 0 and args.max_sentences < N:
    # 取前 max_sentences 句 (保持概念完整性: 4 句一组)
    n_concepts = args.max_sentences // 4
    keep = n_concepts * 4
    sentences = sentences[:keep]
    labels = labels[:keep]
    stages_list = stages_list[:keep]
    variant_indices = variant_indices[:keep]
    concept_names = concept_names[:keep]
    N = len(sentences)
    print(f"limited to {N} sentences ({n_concepts} concepts)")

# ── 确定要提取的层 ─────────────────────────────────────────────────────────
total_layers = n_layers + 1  # embedding + transformer
layer_indices = list(range(0, total_layers, args.layers_skip))
if (total_layers - 1) not in layer_indices:
    layer_indices.append(total_layers - 1)  # 确保末层在内
n_extracted = len(layer_indices)
print(f"extracting layers: {layer_indices} ({n_extracted} of {total_layers})", flush=True)

# ── 概念分组 ──────────────────────────────────────────────────────────────
concept_groups = {}
for i, cid in enumerate(labels):
    if cid not in concept_groups:
        concept_groups[cid] = []
    concept_groups[cid].append(i)

# ── 提取函数 ──────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_one(text, strategies):
    """提取一句话指定层的 Hidden States, 返回 dict of (n_extracted, hidden_dim)"""
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
    out = model(**ids, output_hidden_states=True)
    hs = out.hidden_states  # tuple of (1, seq_len, hidden_dim), len = total_layers

    result = {}
    for strat in strategies:
        reps = np.zeros((n_extracted, hidden_dim), dtype=np.float32)
        for j, layer in enumerate(layer_indices):
            h = hs[layer][0]  # (seq_len, hidden_dim)
            if strat == "last_token":
                reps[j] = h[-1].float().cpu().numpy()
            elif strat == "mean_pool":
                reps[j] = h.float().mean(0).cpu().numpy()
            elif strat == "entity":
                # 取中间 token (简单回退)
                mid = len(h) // 2
                reps[j] = h[mid].float().cpu().numpy()
        result[strat] = reps
    return result

# ── 批量提取 ──────────────────────────────────────────────────────────────
strategies = args.strategies
print(f"\nextracting {N} sentences × {n_extracted} layers × {len(strategies)} strategies...", flush=True)

all_reps = {s: np.zeros((N, n_extracted, hidden_dim), dtype=np.float32) for s in strategies}

t1 = time.time()
for i in range(N):
    s = sentences[i]
    reps = extract_one(s, strategies)
    for strat in strategies:
        all_reps[strat][i] = reps[strat]

    if (i + 1) % 10 == 0 or i == N - 1:
        elapsed = time.time() - t1
        print(f"  [{i+1}/{N}] {elapsed:.0f}s — {s[:50]}...", flush=True)

# ── 逐层几何量化 ───────────────────────────────────────────────────────────
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

results = {}
for strat in strategies:
    per_layer_pos = []
    per_layer_neg = []
    per_layer_margin = []

    for li in range(n_extracted):
        layer_reps = all_reps[strat][:, li, :]  # (N, hidden_dim)

        # 正样本: 同概念内变体两两余弦
        pos_sims = []
        for cid in concept_groups:
            indices = concept_groups[cid]
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    pos_sims.append(cosine_sim(layer_reps[indices[a]], layer_reps[indices[b]]))

        # 负样本: 跨概念 (采样 20 个概念)
        neg_sims = []
        rng = np.random.default_rng(42)
        cids_sorted = sorted(concept_groups.keys())
        sampled = rng.choice(len(cids_sorted), size=min(20, len(cids_sorted)), replace=False)
        for i_a in range(len(sampled)):
            for i_b in range(i_a + 1, len(sampled)):
                ia = concept_groups[cids_sorted[sampled[i_a]]][0]
                ib = concept_groups[cids_sorted[sampled[i_b]]][0]
                neg_sims.append(cosine_sim(layer_reps[ia], layer_reps[ib]))

        pos_mean = float(np.mean(pos_sims))
        neg_mean = float(np.mean(neg_sims))
        per_layer_pos.append(pos_mean)
        per_layer_neg.append(neg_mean)
        per_layer_margin.append(pos_mean - neg_mean)

    # 末 3 层裁决
    last3_pos = float(np.mean(per_layer_pos[-3:]))
    last3_neg = float(np.mean(per_layer_neg[-3:]))
    last3_margin = last3_pos - last3_neg
    g1_pass = last3_pos > 0.75 and last3_pos > last3_neg

    results[strat] = {
        "pos": per_layer_pos,
        "neg": per_layer_neg,
        "margin": per_layer_margin,
        "last3_pos": last3_pos,
        "last3_neg": last3_neg,
        "last3_margin": last3_margin,
        "G1_pass": g1_pass,
    }

    print(f"\n=== {strat} ===")
    print(f"  last3: pos={last3_pos:.3f} neg={last3_neg:.3f} margin={last3_margin:.3f}")
    print(f"  G-1: {'PASS' if g1_pass else 'FAIL'}")
    print(f"  vs 0.5B baseline: margin={last3_margin:.3f} (0.5B was ~0.031)")

# ── 保存 ──────────────────────────────────────────────────────────────────
out_npz = os.path.join(HERE, f"smoke_reps_{model_tag}.npz")
np.savez_compressed(out_npz,
    **{f"{s}": all_reps[s] for s in strategies},
    labels=np.array(labels),
    stages=np.array(stages_list),
    variant_idx=np.array(variant_indices),
    sentences=np.array(sentences, dtype=object),
    concept_names=np.array(concept_names, dtype=object),
    layer_indices=np.array(layer_indices),
    n_layers=n_layers,
    hidden_dim=hidden_dim,
)

out_json = os.path.join(HERE, f"smoke_metrics_{model_tag}.json")
metrics = {
    "minted": "2026-10-15",
    "model": args.model,
    "model_tag": model_tag,
    "n_layers": n_layers,
    "hidden_dim": hidden_dim,
    "n_heads": n_heads,
    "n_sentences": N,
    "layer_indices": layer_indices,
    "strategies": results,
    "vs_05b": {
        "last_token_margin": 0.031,
        "mean_pool_margin": 0.049,
        "note": "Qwen2.5-0.5B-Instruct baseline from Phase 3",
    },
}
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=1)

elapsed = time.time() - t0
print(f"\nDONE — {elapsed:.0f}s total")
print(f"  reps: {out_npz}")
print(f"  metrics: {out_json}")
print(f"  model: {n_layers}L/{hidden_dim}H ({n_heads} heads)")
for strat in strategies:
    r = results[strat]
    print(f"  {strat}: pos={r['last3_pos']:.3f} neg={r['last3_neg']:.3f} "
          f"margin={r['last3_margin']:.3f} G1={'PASS' if r['G1_pass'] else 'FAIL'}")
