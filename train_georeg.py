#!/usr/bin/env python3
"""Phase 2 · 几何正则化 (Geometric Regularization) PoC — 0.5B

参考 ide-lola 建议: "在训练或微调阶段, 显式加入对比损失"

Loss 形式:
    L = L_LM + lambda * L_geo

L_geo 轻量近似 (block 内同位置 token = 正样本, 异位置 = 负样本):
    在每个训练 block 的最后一层 hidden 上, 取前 K 个 token 与后 K 个 token
    的 cosine, 前 K 之间视作正样本, 跨段视作负样本.

训练: 2 ep x 4 lambda (0, 0.05, 0.2, 1.0) x 2 pool (last_token, mean_pool)

输出:
  metrics_georeg.json
  figs_georeg/fig_margin_vs_lambda.png  (last + mean 双图)
  figs_georeg/fig_loss_vs_lambda.png

判定: H-4 (lambda>=0.2 比 baseline margin 提升 >= 0.05)
"""
import json, os, time, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus_paraphrase.json")
_local = os.path.expanduser("~/Programming/code-2026/first-lora/model")
MODEL = _local if os.path.isdir(_local) else "Qwen/Qwen2.5-0.5B-Instruct"
FIGS = os.path.join(HERE, "figs_georeg")
os.makedirs(FIGS, exist_ok=True)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {dev}", flush=True)

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

all_sentences, all_labels = [], []
for cid in concept_ids:
    for s in paraphrases[cid]["sentences"]:
        all_sentences.append(s)
        all_labels.append(cid)
N = len(all_sentences)

concept_groups = {}
for i, cid in enumerate(all_labels):
    concept_groups.setdefault(cid, []).append(i)


# ── 训练流 (geometric A 同款) ───────────────────────────────────────────
def build_stream():
    parts = []
    for ci in range(n_concepts):
        cid = concept_ids[ci]
        data = paraphrases[cid]
        name = data["name"]
        sents = data["sentences"]
        parts.append(f"{name}就是{name}。")
        for s in sents:
            parts.append(s)
    return "\n".join(parts)


# ── 表示提取 ─────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_reps_pool(lm, sentences, layer_idx=-1, pool="mean"):
    tok = AutoTokenizer.from_pretrained(MODEL)
    out_arr = []
    for s in sentences:
        ids = tok(s, return_tensors="pt", add_special_tokens=False).to(dev)
        out = lm(**ids, output_hidden_states=True)
        hs = out.hidden_states[layer_idx][0]  # (seq, dim)
        if pool == "last":
            v = hs[-1]
        elif pool == "mean":
            v = hs.mean(dim=0)
        else:
            raise ValueError(pool)
        out_arr.append(v.float().cpu().numpy())
    return np.array(out_arr)


def compute_pos_neg(reps, groups):
    pos_sims, neg_sims = [], []
    for cid, indices in groups.items():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                v1, v2 = reps[indices[a]], reps[indices[b]]
                pos_sims.append(float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)))
    rng = np.random.default_rng(42)
    cids = sorted(groups.keys())
    sampled = rng.choice(len(cids), size=min(20, len(cids)), replace=False)
    for i_a in range(len(sampled)):
        for i_b in range(i_a + 1, len(sampled)):
            ia = groups[cids[sampled[i_a]]][0]
            ib = groups[cids[sampled[i_b]]][0]
            v1, v2 = reps[ia], reps[ib]
            neg_sims.append(float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)))
    return float(np.mean(pos_sims)), float(np.mean(neg_sims))


# ── 几何正则化训练 ────────────────────────────────────────────────────────
BLOCK = 128
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
M_POS = 0.9
M_NEG = 0.5


def train_arm_geo(lm, stream_text, lambd, epochs=2, lr=1e-4, pool="mean"):
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(stream_text, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    blocks = ids[0].unfold(0, BLOCK, BLOCK).clone()
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad],
                            lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    lm.train()
    losses_lm, losses_geo = [], []
    K = 8  # 正/负样本 token 数

    for ep in range(epochs):
        ep_lm, ep_geo, nb = 0.0, 0.0, 0
        for blk in blocks:
            x = blk.unsqueeze(0)
            opt.zero_grad()
            out = lm(x, labels=x, output_hidden_states=True)
            lm_loss = out.loss
            h = out.hidden_states[-1][0]  # (seq, dim)
            v = h / (h.norm(dim=-1, keepdim=True) + 1e-8)  # (seq, dim)
            seq_len = v.size(0)
            k = min(K, seq_len // 2)
            if k >= 2:
                v_p = v[:k]  # 前半 token
                v_n = v[k:2*k]  # 后半 token
                # 正样本: 前半内部 (近似"同概念不同表达")
                cos_p = v_p @ v_p.T
                mask = torch.eye(k, device=v.device, dtype=torch.bool)
                cos_p = cos_p.masked_fill(mask, 0.0)
                # 负样本: 前半 vs 后半
                cos_n = v_p @ v_n.T
                geo_pos = torch.clamp(M_POS - cos_p, min=0.0).mean()
                geo_neg = torch.clamp(cos_n - M_NEG, min=0.0).mean()
                geo_loss = geo_pos + geo_neg
            else:
                geo_loss = torch.tensor(0.0, device=v.device)
            total = lm_loss + lambd * geo_loss
            total.backward()
            opt.step()
            ep_lm += float(lm_loss); ep_geo += float(geo_loss); nb += 1
        losses_lm.append(ep_lm / nb)
        losses_geo.append(ep_geo / nb)
        print(f"    ep={ep+1}/{epochs} lm={losses_lm[-1]:.3f} geo={losses_geo[-1]:.4f}", flush=True)
    lm.eval()
    return losses_lm, losses_geo


# ── 主流程 ────────────────────────────────────────────────────────────────
LAMBDAS = [0.0, 0.05, 0.2, 1.0]
POOLS = ["last", "mean"]
EPOCHS = 2
LR = 1e-4

stream = build_stream()
all_results = {"pools": POOLS, "lambdas": LAMBDAS, "epochs": EPOCHS, "lr": LR}

# 加载 base 一次
print("\n=== Loading base model ===", flush=True)
base_for_eval = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager",
                                                     dtype=torch.float16).to(dev)
base_for_eval.eval()
torch.mps.empty_cache() if dev == "mps" else None

for pool in POOLS:
    print(f"\n========== Pool = {pool} ==========", flush=True)
    # Baseline (no LoRA)
    reps_base = extract_reps_pool(base_for_eval, all_sentences, pool=pool)
    pos0, neg0 = compute_pos_neg(reps_base, concept_groups)
    margin0 = pos0 - neg0
    print(f"  baseline: pos={pos0:.4f} neg={neg0:.4f} margin={margin0:.4f}", flush=True)
    pool_res = {"baseline": {"pos": pos0, "neg": neg0, "margin": margin0}}

    for lam in LAMBDAS:
        print(f"\n  --- pool={pool} λ={lam} ---", flush=True)
        torch.manual_seed(13)
        m = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager",
                                                 dtype=torch.float16).to(dev)
        cfg = LoraConfig(r=16, lora_alpha=32, target_modules=TARGETS,
                         lora_dropout=0.0, task_type="CAUSAL_LM")
        lm = get_peft_model(m, cfg)
        t0 = time.time()
        losses_lm, losses_geo = train_arm_geo(lm, stream,
                                              lambd=lam, epochs=EPOCHS, lr=LR, pool=pool)
        reps = extract_reps_pool(lm, all_sentences, pool=pool)
        pos, neg = compute_pos_neg(reps, concept_groups)
        margin = pos - neg
        print(f"  λ={lam}: pos={pos:.4f} neg={neg:.4f} margin={margin:.4f}  Δ={margin-margin0:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
        pool_res[f"lambda_{lam}"] = {
            "lambd": lam, "pos": pos, "neg": neg, "margin": margin,
            "losses_lm": losses_lm, "losses_geo": losses_geo,
        }
        del lm, m
        torch.mps.empty_cache() if dev == "mps" else None

    all_results[pool] = pool_res

# ── 绘图 ──────────────────────────────────────────────────────────────────
# 图 1: margin vs lambda, 双池化
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150, sharey=True)
colors = ["#666666", "#90be6d", "#f9c74f", "#f8961e", "#d62828"]
for ax, pool in zip(axes, POOLS):
    margins = [all_results[pool]["baseline"]["margin"]] + \
              [all_results[pool][f"lambda_{l}"]["margin"] for l in LAMBDAS]
    labels_p = ["baseline"] + [f"λ={l}" for l in LAMBDAS]
    bars = ax.bar(labels_p, margins, color=colors[:len(labels_p)],
                  edgecolor="#333", linewidth=0.6)
    for i, v in enumerate(margins):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_title(f"Pool = {pool}")
    ax.set_ylabel("Margin (pos - neg cosine)")
    ax.axhline(all_results[pool]["baseline"]["margin"], ls="--", color="#888", lw=0.8)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Phase 2: Geometric Regularization — Margin vs λ (0.5B)")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_margin_vs_lambda.png"))
plt.close(fig)

# 图 2: LM loss vs epoch per lambda, 双池化
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150, sharey=True)
for ax, pool in zip(axes, POOLS):
    for lam in LAMBDAS:
        if lam == 0.0:
            continue
        losses = all_results[pool][f"lambda_{lam}"]["losses_lm"]
        ax.plot(range(1, len(losses)+1), losses, "o-", label=f"λ={lam}", lw=1.5, ms=5)
    ax.set_title(f"Pool = {pool}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LM Loss")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
fig.suptitle("LM Loss Across Lambda Sweep (0.5B)")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_loss_vs_lambda.png"))
plt.close(fig)

# 图 3: geo loss vs epoch per lambda, 双池化
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150, sharey=True)
for ax, pool in zip(axes, POOLS):
    for lam in LAMBDAS:
        if lam == 0.0:
            continue
        losses = all_results[pool][f"lambda_{lam}"]["losses_geo"]
        ax.plot(range(1, len(losses)+1), losses, "o-", label=f"λ={lam}", lw=1.5, ms=5)
    ax.set_title(f"Pool = {pool}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Geo Loss")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
fig.suptitle("Geometric Loss Across Lambda Sweep (0.5B)")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_geo_loss.png"))
plt.close(fig)

# ── H-4 判定 ─────────────────────────────────────────────────────────────
# H-4: lambda=1.0 比 baseline margin 提升 >= 0.05
h4 = {}
for pool in POOLS:
    base = all_results[pool]["baseline"]["margin"]
    best_lam = max(LAMBDAS, key=lambda l: all_results[pool][f"lambda_{l}"]["margin"])
    best_margin = all_results[pool][f"lambda_{best_lam}"]["margin"]
    h4[pool] = {
        "best_lambda": best_lam,
        "best_margin": best_margin,
        "delta": best_margin - base,
        "pass": best_margin - base >= 0.05,
    }

# ── 保存 ──────────────────────────────────────────────────────────────────
out = {
    "minted": time.strftime("%Y-%m-%d"),
    "arm": "geometry-meaning-phase2-georeg-0.5b-poc",
    "model": "Qwen2.5-0.5B-Instruct + LoRA(r=16, α=32)",
    "M_POS": M_POS,
    "M_NEG": M_NEG,
    "lr": LR,
    "epochs": EPOCHS,
    "lambdas": LAMBDAS,
    "pools": POOLS,
    "results": all_results,
    "H4": h4,
    "verdict": ("geometric regularization helps" if all(h4[p]["pass"] for p in POOLS)
                else "partial" if any(h4[p]["pass"] for p in POOLS)
                else "geometric regularization does not help at 0.5B scale"),
}
out_path = os.path.join(HERE, "metrics_georeg.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print(f"\n{'='*60}\nDONE — {out_path}")
print(f"  figs: {FIGS}/")
print(f"  H-4 verdict: {out['verdict']}")
for pool in POOLS:
    print(f"\n  [{pool}]")
    print(f"    baseline margin: {all_results[pool]['baseline']['margin']:.4f}")
    for lam in LAMBDAS:
        print(f"    λ={lam:>4}: margin={all_results[pool][f'lambda_{lam}']['margin']:.4f}  (Δ={all_results[pool][f'lambda_{lam}']['margin']-all_results[pool]['baseline']['margin']:+.4f})")
    h = h4[pool]
    print(f"    best λ={h['best_lambda']}  margin={h['best_margin']:.4f}  Δ={h['delta']:+.4f}  {'PASS' if h['pass'] else 'FAIL'}")
