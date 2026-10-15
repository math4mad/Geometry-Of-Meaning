#!/usr/bin/env python3
"""Phase 2 · 表示提取 — 逐层提取 Hidden States

对 corpus_paraphrase.json 中的所有句子（50 概念 × 4 变体 = 200 句），
逐层提取 Qwen2.5-0.5B-Instruct (24L) 的 Hidden States。

三种聚合策略并行:
  1. Last-token: 序列末 Token 向量
  2. Mean Pooling: 整序列均值
  3. Entity-centric: 核心实体 Token 在各层的表示

输出: representations.npz
  - last_token: (n_sentences, n_layers, hidden_dim)
  - mean_pool:  (n_sentences, n_layers, hidden_dim)
  - entity:      (n_sentences, n_layers, hidden_dim)
  - labels:      (n_sentences,)  概念 ID
  - stages:      (n_sentences,)  年龄阶
  - variant_idx: (n_sentences,)  变体编号 0-3
  - sentences:   list[str]
  - layer_names: list[str]
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus_paraphrase.json")
_local = os.path.expanduser("~/Programming/code-2026/first-lora/model")
MODEL = _local if os.path.isdir(_local) else "Qwen/Qwen2.5-0.5B-Instruct"
OUT = os.path.join(HERE, "representations.npz")

dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {dev}")

from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 加载 ───────────────────────────────────────────────────────────────────
print("loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(dev)
model.eval()
n_layers = model.config.num_hidden_layers  # 24 for Qwen2.5-0.5B
hidden_dim = model.config.hidden_size     # 896
print(f"model loaded: {n_layers} layers, {hidden_dim} dim", flush=True)

# ── 加载语料 ─────────────────────────────────────────────────────────────────
with open(CORPUS, encoding="utf-8") as f:
    corpus = json.load(f)

paraphrases = corpus["paraphrases"]
# 展平: concept_id -> 4 sentences
sentences = []
labels = []
stages = []
variant_indices = []
concept_names = []

for cid, data in paraphrases.items():
    for vi, s in enumerate(data["sentences"]):
        sentences.append(s)
        labels.append(cid)
        stages.append(data["stage"])
        variant_indices.append(vi)
        concept_names.append(data["name"])

N = len(sentences)
print(f"sentences: {N} ({len(paraphrases)} concepts × 4 variants)")

# ── 提取函数 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_hidden(text):
    """提取一句话的所有层 Hidden States
    返回: (n_layers+1, hidden_dim) — 含 embedding 层 (layer 0)
    """
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
    out = model(**ids, output_hidden_states=True)
    # hidden_states: tuple of (1, seq_len, hidden_dim), len = n_layers + 1
    hs = out.hidden_states  # (n_layers+1, 1, seq_len, hidden_dim)
    return hs

@torch.no_grad()
def extract_last_token(text):
    """Last-token representation: 末 Token 向量, 逐层"""
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
    out = model(**ids, output_hidden_states=True)
    hs = out.hidden_states  # tuple of (1, seq_len, hidden_dim)
    last_reps = np.stack([h[0, -1].float().cpu().numpy() for h in hs])  # (n_layers+1, hidden_dim)
    return last_reps

@torch.no_grad()
def extract_mean_pool(text):
    """Mean Pooling: 整序列均值, 逐层"""
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
    out = model(**ids, output_hidden_states=True)
    hs = out.hidden_states
    mean_reps = np.stack([h[0].float().mean(0).cpu().numpy() for h in hs])  # (n_layers+1, hidden_dim)
    return mean_reps

@torch.no_grad()
def extract_entity_centric(text, entity_name):
    """Entity-centric: 核心实体 Token 在各层的表示
    找到 entity_name 在 token 序列中的位置, 取该位置逐层向量
    """
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
    out = model(**ids, output_hidden_states=True)
    hs = out.hidden_states

    # 找 entity 的 token 位置
    entity_ids = tok(entity_name, add_special_tokens=False).input_ids
    input_ids = ids.input_ids[0].tolist()

    # 滑动匹配
    ent_pos = -1
    for i in range(len(input_ids) - len(entity_ids) + 1):
        if input_ids[i:i+len(entity_ids)] == entity_ids:
            ent_pos = i
            break

    if ent_pos == -1:
        # fallback: 取中间 token
        ent_pos = len(input_ids) // 2

    # 取 entity 首 token 位置的逐层向量
    ent_reps = np.stack([h[0, ent_pos].float().cpu().numpy() for h in hs])
    return ent_reps

# ── 批量提取 ─────────────────────────────────────────────────────────────────
print(f"\nextracting representations for {N} sentences...", flush=True)
t0 = time.time()

last_token_reps = np.zeros((N, n_layers + 1, hidden_dim), dtype=np.float32)
mean_pool_reps = np.zeros((N, n_layers + 1, hidden_dim), dtype=np.float32)
entity_reps = np.zeros((N, n_layers + 1, hidden_dim), dtype=np.float32)

for i in range(N):
    s = sentences[i]
    cname = concept_names[i]

    last_token_reps[i] = extract_last_token(s)
    mean_pool_reps[i] = extract_mean_pool(s)
    entity_reps[i] = extract_entity_centric(s, cname)

    if (i + 1) % 20 == 0 or i == N - 1:
        elapsed = time.time() - t0
        print(f"  [{i+1}/{N}] {elapsed:.0f}s — {sentences[i][:50]}...", flush=True)

# ── 保存 ───────────────────────────────────────────────────────────────────
np.savez_compressed(OUT,
    last_token=last_token_reps,
    mean_pool=mean_pool_reps,
    entity=entity_reps,
    labels=np.array(labels),
    stages=np.array(stages),
    variant_idx=np.array(variant_indices),
    sentences=np.array(sentences, dtype=object),
    concept_names=np.array(concept_names, dtype=object),
    layer_names=np.array([f"layer_{i}" for i in range(n_layers + 1)]),
    n_layers=n_layers,
    hidden_dim=hidden_dim,
)

elapsed = time.time() - t0
print(f"\nDONE — {OUT}")
print(f"  shapes: last_token={last_token_reps.shape}, mean_pool={mean_pool_reps.shape}, entity={entity_reps.shape}")
print(f"  time: {elapsed:.0f}s")
print(f"  layers: {n_layers+1} (embedding + {n_layers} transformer)")
print(f"  hidden_dim: {hidden_dim}")
