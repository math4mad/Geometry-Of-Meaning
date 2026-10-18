#!/usr/bin/env python3
"""Phase 2 v2 · 跨句配对几何损失 PoC — 0.5B

新设计 (vs v1):
  旧: block 内 K token 互拉/推开 -> 局部一致性 (非概念对齐)
  新: 每步抽 micro-batch, 同概念句子 hidden 拉近, 异概念推开 (InfoNCE/SupCon)

Loss 形式:
  L = L_LM + lambda * L_geo
  L_geo: 同 batch 内, 对每个 anchor i, 把它同概念的句子作正样本, 异概念作负样本
         temperature tau=0.07, symmetric InfoNCE

训练: 2 ep x 4 lambda (0, 0.05, 0.2, 1.0) x 2 pool (last, mean)
每步: 抽 B=4 句 micro-batch (尽量覆盖多概念), 1 forward

输出:
  metrics_georeg_v2.json
  figs_georeg_v2/fig_margin_vs_lambda.png  (last + mean)
  figs_georeg_v2/fig_loss_vs_lambda.png
  figs_georeg_v2/fig_geo_loss.png

判定: H-4 (lambda >= 0.05 比 baseline margin 提升 >= 0.05)
"""
import json, os, time, random
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus_paraphrase.json")
_local = os.path.expanduser("~/Programming/code-2026/first-lora/model")
MODEL = _local if os.path.isdir(_local) else "Qwen/Qwen2.5-0.5B-Instruct"
FIGS = os.path.join(HERE, "figs_georeg_v2")
os.makedirs(FIGS, exist_ok=True)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {dev}", flush=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# ── 语料 ──────────────────────────────────────────────────────────────────
with open(CORPUS, encoding="utf-8") as f:
    corpus = json.load(f)
paraphrases = corpus["paraphrases"]
concept_ids = sorted(paraphrases.keys())
n_concepts = len(concept_ids)

# 概念 -> [句子列表]
concept_to_sents = {cid: paraphrases[cid]["sentences"] for cid in concept_ids}
all_sents = []  # (cid, text)
for cid in concept_ids:
    for s in concept_to_sents[cid]:
        all_sents.append((cid, s))


# ── 句尾 pad: 让 batch 内等长 ────────────────────────────────────────────
def batch_pool_forward(lm, sents_text, pool="mean", tok=None):
    """sents_text: List[str] -> Tensor (B, dim) on dev"""
    enc = tok(sents_text, return_tensors="pt", padding=True, truncation=True,
              max_length=64, add_special_tokens=False).to(dev)
    out = lm(**enc, output_hidden_states=True)
    h = out.hidden_states[-1]  # (B, L, dim)
    mask = enc.attention_mask.unsqueeze(-1).float()  # (B, L, 1)
    if pool == "mean":
        v = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    elif pool == "last":
        # 取每句最后一个有效 token (倒序第一个 mask=1 的位置)
        L = h.size(1)
        idx = (enc.attention_mask.sum(dim=1) - 1).clamp(min=0)  # (B,)
        v = h[torch.arange(h.size(0), device=dev), idx]
    else:
        raise ValueError(pool)
    return F.normalize(v.float(), dim=-1)  # (B, dim) 已 L2 norm


# ── 跨句 InfoNCE 损失 ────────────────────────────────────────────────────
TAU = 0.07
M_NEG = 0.5  # 备份: 负样本 cosine > 0.5 时再加 margin push


def info_nce_loss(z, labels):
    """z: (B, dim) L2-normalized; labels: (B,) concept ids
       对每个 anchor, 跨 batch 内同概念作正, 异概念作负
       symmetric: 平均 anchor 视角 + 配对视角
    """
    B = z.size(0)
    sim = z @ z.T / TAU  # (B, B)
    # mask: 自身不算
    self_mask = torch.eye(B, device=z.device, dtype=torch.bool)
    sim_masked = sim.masked_fill(self_mask, float("-inf"))

    # pos mask: 跨 batch 同概念 (label 相等, 但不是自身)
    eq = labels.unsqueeze(0).eq(labels.unsqueeze(1))  # (B, B)
    pos_mask = eq & ~self_mask  # (B, B)

    n_pos_per_anchor = pos_mask.sum(dim=1).float()  # (B,)
    # 若某 anchor 没有正样本, 跳过它
    valid_anchor = n_pos_per_anchor > 0

    # log_prob[a, j] = sim[a, j] - logsumexp(sim[a, :])
    log_denom = torch.logsumexp(sim_masked, dim=1, keepdim=True)  # (B, 1)
    log_prob = sim_masked - log_denom  # (B, B)
    # anchor a 视角: 对所有正样本 j 求 mean log_prob
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)  # (B,)
    loss_per_anchor = -pos_log_prob / n_pos_per_anchor.clamp(min=1)
    loss = loss_per_anchor[valid_anchor].mean()

    # 备份 margin push: 异概念 sim > 0.5 时加大间距
    neg_mask = ~eq & ~self_mask
    if neg_mask.any():
        cos_neg = sim * TAU  # 转回 cosine
        margin_pen = F.relu(cos_neg - M_NEG) * neg_mask.float()
        margin_loss = margin_pen.sum() / (neg_mask.sum() + 1e-6)
    else:
        margin_loss = torch.tensor(0.0, device=z.device)
    return loss + 0.1 * margin_loss, float(loss.detach()), float(margin_loss.detach())


# ── 表示提取 (eval) ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_reps_pool(lm, sentences, tok, pool="mean"):
    out_arr = []
    B = 16
    for i in range(0, len(sentences), B):
        v = batch_pool_forward(lm, sentences[i:i+B], pool=pool, tok=tok)
        out_arr.append(v.cpu().numpy())
    return np.concatenate(out_arr, axis=0)


def compute_pos_neg(reps, groups):
    pos_sims, neg_sims = [], []
    for cid, indices in groups.items():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                v1, v2 = reps[indices[a]], reps[indices[b]]
                pos_sims.append(float(np.dot(v1, v2)))
    rng = np.random.default_rng(42)
    cids = sorted(groups.keys())
    sampled = rng.choice(len(cids), size=min(20, len(cids)), replace=False)
    for i_a in range(len(sampled)):
        for i_b in range(i_a + 1, len(sampled)):
            ia = groups[cids[sampled[i_a]]][0]
            ib = groups[cids[sampled[i_b]]][0]
            v1, v2 = reps[ia], reps[ib]
            neg_sims.append(float(np.dot(v1, v2)))
    return float(np.mean(pos_sims)), float(np.mean(neg_sims))


# ── 训练流 (固定 micro-batch) ────────────────────────────────────────────
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
BATCH = 4  # 每步 micro batch 句子数
STEPS_PER_EP = 80  # 每 epoch 多少步 (LM 训练不依赖 step 数)
LR = 1e-4


def build_micro_batches(rng, n_per_concept=1):
    """每 micro batch:
       - 抽 BATCH 个不同 concept
       - 每个 concept 抽 n_per_concept 句
       目标: 让 batch 内大概率有 >= 1 对同概念 (正样本) + 多对异概念 (负样本)
    """
    cids = list(concept_to_sents.keys())
    rng.shuffle(cids)
    batches = []
    for i in range(0, len(cids), BATCH):
        chunk = cids[i:i+BATCH]
        labels, texts = [], []
        for cid in chunk:
            s = rng.choice(concept_to_sents[cid])
            labels.append(cid)
            texts.append(s)
        batches.append((labels, texts))
    return batches


def train_arm_geo(lm, lambd, epochs, pool):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad],
                            lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    lm.train()
    rng = random.Random(42)
    losses_lm, losses_geo = [], []
    for ep in range(epochs):
        batches = build_micro_batches(rng)
        ep_lm, ep_geo, nb = 0.0, 0.0, 0
        for labels, texts in batches[:STEPS_PER_EP]:
            opt.zero_grad()
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=64, add_special_tokens=False).to(dev)
            out = lm(**enc, labels=enc.input_ids, output_hidden_states=True)
            lm_loss = out.loss
            z = batch_pool_forward(lm, texts, pool=pool, tok=tok)
            label_t = torch.tensor([concept_ids.index(l) for l in labels], device=dev)
            geo_loss, geo_info, margin_info = info_nce_loss(z, label_t)
            total = lm_loss + lambd * geo_loss
            total.backward()
            opt.step()
            ep_lm += float(lm_loss); ep_geo += float(geo_info); nb += 1
        losses_lm.append(ep_lm / nb)
        losses_geo.append(ep_geo / nb)
        print(f"    ep={ep+1}/{epochs} lm={losses_lm[-1]:.3f} info_nce={losses_geo[-1]:.4f}", flush=True)
    lm.eval()
    return losses_lm, losses_geo


# ── 主流程 ────────────────────────────────────────────────────────────────
LAMBDAS = [0.0, 0.05, 0.2, 1.0]
POOLS = ["last", "mean"]
EPOCHS = 2

# Eval 用的所有句子
eval_sents = [s for _, s in all_sents]
eval_labels = [c for c, _ in all_sents]
concept_groups = {}
for i, c in enumerate(eval_labels):
    concept_groups.setdefault(c, []).append(i)

print("\n=== Loading base model for eval ===", flush=True)
base_for_eval = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager",
                                                     dtype=torch.float16).to(dev)
base_for_eval.eval()
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
torch.mps.empty_cache() if dev == "mps" else None

all_results = {"pools": POOLS, "lambdas": LAMBDAS, "epochs": EPOCHS, "lr": LR,
               "BATCH": BATCH, "STEPS_PER_EP": STEPS_PER_EP, "TAU": TAU}

for pool in POOLS:
    print(f"\n========== Pool = {pool} ==========", flush=True)
    reps_base = extract_reps_pool(base_for_eval, eval_sents, tok, pool=pool)
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
        losses_lm, losses_geo = train_arm_geo(lm, lambd=lam, epochs=EPOCHS, pool=pool)
        reps = extract_reps_pool(lm, eval_sents, tok, pool=pool)
        pos, neg = compute_pos_neg(reps, concept_groups)
        margin = pos - neg
        print(f"  λ={lam}: pos={pos:.4f} neg={neg:.4f} margin={margin:.4f}  "
              f"Δ={margin-margin0:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
        pool_res[f"lambda_{lam}"] = {
            "lambd": lam, "pos": pos, "neg": neg, "margin": margin,
            "losses_lm": losses_lm, "losses_info_nce": losses_geo,
        }
        del lm, m
        torch.mps.empty_cache() if dev == "mps" else None

    all_results[pool] = pool_res


# ── 绘图 ──────────────────────────────────────────────────────────────────
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
fig.suptitle("Phase 2 v2: Cross-sentence Geometric Reg — Margin vs λ (0.5B)")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_margin_vs_lambda.png"))
plt.close(fig)

# LM loss vs epoch
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
fig.suptitle("LM Loss Across Lambda Sweep (0.5B v2)")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_loss_vs_lambda.png"))
plt.close(fig)

# Geo InfoNCE loss
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150, sharey=True)
for ax, pool in zip(axes, POOLS):
    for lam in LAMBDAS:
        if lam == 0.0:
            continue
        losses = all_results[pool][f"lambda_{lam}"]["losses_info_nce"]
        ax.plot(range(1, len(losses)+1), losses, "o-", label=f"λ={lam}", lw=1.5, ms=5)
    ax.set_title(f"Pool = {pool}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("InfoNCE Loss")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
fig.suptitle("InfoNCE Loss Across Lambda Sweep (0.5B v2)")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_info_nce.png"))
plt.close(fig)


# ── H-4 判定 ─────────────────────────────────────────────────────────────
h4 = {}
for pool in POOLS:
    base = all_results[pool]["baseline"]["margin"]
    # 最佳 lambda: 只看 lambda>=0.05 (排除 0)
    pos_lams = [l for l in LAMBDAS if l > 0]
    best_lam = max(pos_lams, key=lambda l: all_results[pool][f"lambda_{l}"]["margin"])
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
    "arm": "geometry-meaning-phase2-georeg-v2-0.5b-poc",
    "model": "Qwen2.5-0.5B-Instruct + LoRA(r=16, α=32)",
    "loss_form": "L_LM + lambda * InfoNCE(cross-sentence, tau=0.07) + 0.1*margin_push",
    "BATCH": BATCH,
    "STEPS_PER_EP": STEPS_PER_EP,
    "TAU": TAU,
    "lr": LR,
    "epochs": EPOCHS,
    "lambdas": LAMBDAS,
    "pools": POOLS,
    "results": all_results,
    "H4": h4,
    "verdict": ("geometric regularization helps (v2 cross-sentence)"
                if all(h4[p]["pass"] for p in POOLS)
                else "partial" if any(h4[p]["pass"] for p in POOLS)
                else "geometric regularization does not help at 0.5B v2"),
}
out_path = os.path.join(HERE, "metrics_georeg_v2.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print(f"\n{'='*60}\nDONE — {out_path}")
print(f"  figs: {FIGS}/")
print(f"  H-4 verdict: {out['verdict']}")
for pool in POOLS:
    print(f"\n  [{pool}]")
    print(f"    baseline margin: {all_results[pool]['baseline']['margin']:.4f}")
    for lam in LAMBDAS:
        print(f"    λ={lam:>4}: margin={all_results[pool][f'lambda_{lam}']['margin']:.4f}  "
              f"(Δ={all_results[pool][f'lambda_{lam}']['margin']-all_results[pool]['baseline']['margin']:+.4f})")
    h = h4[pool]
    print(f"    best λ={h['best_lambda']}  margin={h['best_margin']:.4f}  "
          f"Δ={h['delta']:+.4f}  {'PASS' if h['pass'] else 'FAIL'}")
