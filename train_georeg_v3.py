#!/usr/bin/env python3
"""Phase 2 v3 · LM 流训练 + 周期性跨句几何损失 PoC — 0.5B

关键修正 (vs v2):
  v2 错: 完全用 micro-batch 训 LM, LM 分布跟 eval 严重错位, loss 6.85 异常高
  v3 正: LM 训练仍走 v1 的 stream block 流 (LM loss ~2.5)
        每 GEO_EVERY 步, 抽 micro-batch 算 InfoNCE 几何损失
        这样几何损失作用于稳定 hidden, 不破坏 LM 流

Loss 形式:
  每步 LM:     L_lm  = CrossEntropy(stream_block)   (跟 v1 一样)
  每 GEO_EVERY: L_geo = InfoNCE(micro_batch, tau=0.07)
  L = L_lm + lambda * L_geo  (几何损失按 GEO_EVERY 步累加)

训练: 2 ep x 4 lambda x 2 pool

输出: metrics_georeg_v3.json
     figs_georeg_v3/fig_margin_vs_lambda.png
     figs_georeg_v3/fig_loss_vs_lambda.png
     figs_georeg_v3/fig_info_nce.png
"""
import json, os, time, random
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus_paraphrase.json")
_local = os.path.expanduser("~/Programming/code-2026/first-lora/model")
MODEL = _local if os.path.isdir(_local) else "Qwen/Qwen2.5-0.5B-Instruct"
FIGS = os.path.join(HERE, "figs_georeg_v3")
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

concept_to_sents = {cid: paraphrases[cid]["sentences"] for cid in concept_ids}
all_sents = [(cid, s) for cid in concept_ids for s in concept_to_sents[cid]]


# ── LM 流 (跟 v1 一样) ────────────────────────────────────────────────────
def build_stream():
    parts = []
    for cid in concept_ids:
        data = paraphrases[cid]
        name = data["name"]
        parts.append(f"{name}就是{name}。")
        for s in concept_to_sents[cid]:
            parts.append(s)
    return "\n".join(parts)


# ── Micro-batch pool forward (eval + 几何损失共用) ──────────────────────
def batch_pool(lm, sents_text, pool, tok):
    enc = tok(sents_text, return_tensors="pt", padding=True, truncation=True,
              max_length=64, add_special_tokens=False).to(dev)
    out = lm(**enc, output_hidden_states=True)
    h = out.hidden_states[-1]  # (B, L, dim)
    mask = enc.attention_mask.unsqueeze(-1).float()
    if pool == "mean":
        v = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:  # last
        idx = (enc.attention_mask.sum(dim=1) - 1).clamp(min=0)
        v = h[torch.arange(h.size(0), device=dev), idx]
    return F.normalize(v.float(), dim=-1), enc


# ── InfoNCE 损失 (修正 NaN 风险) ─────────────────────────────────────────
TAU = 0.07


def info_nce_loss(z, labels):
    """z: (B, dim) L2-normed, labels: (B,) int
       修正: row 自身 mask 后必须 >=1 个非 -inf 元素才能 logsumexp
    """
    B = z.size(0)
    sim = (z @ z.T) / TAU
    self_mask = torch.eye(B, device=z.device, dtype=torch.bool)
    sim_masked = sim.masked_fill(self_mask, -1e9)  # 不用 -inf, 防 NaN

    eq = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    pos_mask = eq & ~self_mask
    n_pos = pos_mask.sum(dim=1).float()
    valid = n_pos > 0
    if not valid.any():
        return torch.tensor(0.0, device=z.device), 0.0

    # log_prob[a, j] = sim[a, j] - logsumexp(sim[a, :])   (per row)
    log_denom = torch.logsumexp(sim_masked, dim=1, keepdim=True)
    log_prob = sim_masked - log_denom
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
    loss_per_anchor = -pos_log_prob / n_pos.clamp(min=1)
    loss = loss_per_anchor[valid].mean()
    return loss, float(loss.detach())


# ── 表示提取 (eval) ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_reps(lm, sentences, tok, pool):
    out_arr = []
    B = 16
    for i in range(0, len(sentences), B):
        v, _ = batch_pool(lm, sentences[i:i+B], pool, tok)
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
            neg_sims.append(float(np.dot(reps[ia], reps[ib])))
    return float(np.mean(pos_sims)), float(np.mean(neg_sims))


# ── 训练: LM block 流 + 周期性 micro-batch 几何损失 ─────────────────────
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
BLOCK = 128
LR = 1e-4
MICRO_BATCH = 6  # 句子数
# batch 内构造: 选 4 个 concept, 其中 2 个给 2 句, 2 个给 1 句 -> 2 对正样本 + 异概念负样本
N_CONCEPT_PER_BATCH = 4
N_DUP_CONCEPT = 2  # 这 2 个 concept 各抽 2 句
SENT_PER_DUP = 2
GEO_EVERY = 4   # 每 4 个 LM block 步插 1 个 micro-batch 几何损失


def build_micro_batches(rng):
    """micro-batch 内部分 concept 重复以产生正样本对
       batch: N_CONCEPT_PER_BATCH 个 concept
              其中 N_DUP_CONCEPT 个抽 2 句 (SENT_PER_DUP=2), 其余各 1 句
       => batch 句数 = N_DUP_CONCEPT*2 + (N_CONCEPT_PER_BATCH-N_DUP_CONCEPT)*1
       => 正样本对数 = N_DUP_CONCEPT (C(2,2)=1 对/concept)
       => 负样本对: 异 concept 配对
    """
    cids = list(concept_to_sents.keys())
    rng.shuffle(cids)
    batches = []
    i = 0
    while i + N_CONCEPT_PER_BATCH <= len(cids):
        chunk = cids[i:i + N_CONCEPT_PER_BATCH]
        i += N_CONCEPT_PER_BATCH
        rng.shuffle(chunk)
        dup_set = chunk[:N_DUP_CONCEPT]
        single_set = chunk[N_DUP_CONCEPT:]
        labels, texts = [], []
        for cid in dup_set:
            sents = rng.sample(concept_to_sents[cid], SENT_PER_DUP)
            for s in sents:
                labels.append(cid); texts.append(s)
        for cid in single_set:
            labels.append(cid); texts.append(rng.choice(concept_to_sents[cid]))
        batches.append((labels, texts))
    return batches


def train_arm(lm, stream_text, lambd, epochs, pool, tok):
    ids = tok(stream_text, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    blocks = ids[0].unfold(0, BLOCK, BLOCK).clone()
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad],
                            lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    lm.train()
    rng = random.Random(42)
    micro_batches = build_micro_batches(rng)
    n_mb = len(micro_batches)
    losses_lm, losses_geo = [], []
    geo_lm_collector = []

    for ep in range(epochs):
        rng.shuffle(micro_batches)
        ep_lm, ep_geo, nb_lm, nb_geo = 0.0, 0.0, 0, 0
        step = 0
        for blk in blocks:
            opt.zero_grad()
            x = blk.unsqueeze(0)
            out = lm(x, labels=x, output_hidden_states=True)
            lm_loss = out.loss

            if lambd > 0.0 and step % GEO_EVERY == 0:
                mb = micro_batches[(step // GEO_EVERY) % n_mb]
                labels_mb, texts_mb = mb
                # 算 hidden, 算 InfoNCE
                with torch.no_grad():
                    z_pre, _ = batch_pool(lm, texts_mb, pool, tok)
                # 但要可导, 重 forward 算 LM loss
                enc = tok(texts_mb, return_tensors="pt", padding=True, truncation=True,
                          max_length=64, add_special_tokens=False).to(dev)
                out2 = lm(**enc, output_hidden_states=True)
                h = out2.hidden_states[-1]
                mask = enc.attention_mask.unsqueeze(-1).float()
                if pool == "mean":
                    v = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    idx = (enc.attention_mask.sum(dim=1) - 1).clamp(min=0)
                    v = h[torch.arange(h.size(0), device=dev), idx]
                z = F.normalize(v.float(), dim=-1)
                label_t = torch.tensor([concept_ids.index(l) for l in labels_mb], device=dev)
                geo_loss, geo_val = info_nce_loss(z, label_t)
                total = lm_loss + lambd * geo_loss
                total.backward()
                ep_geo += geo_val; nb_geo += 1
            else:
                lm_loss.backward()
                ep_geo += 0.0

            opt.step()
            ep_lm += float(lm_loss); nb_lm += 1; step += 1
        losses_lm.append(ep_lm / nb_lm)
        losses_geo.append(ep_geo / max(nb_geo, 1))
        print(f"    ep={ep+1}/{epochs} lm={losses_lm[-1]:.3f} info_nce={losses_geo[-1]:.4f}", flush=True)
    lm.eval()
    return losses_lm, losses_geo


# ── 主流程 ────────────────────────────────────────────────────────────────
LAMBDAS = [0.0, 0.05, 0.2, 1.0]
POOLS = ["last", "mean"]
EPOCHS = 2

eval_sents = [s for _, s in all_sents]
eval_labels = [c for c, _ in all_sents]
concept_groups = {}
for i, c in enumerate(eval_labels):
    concept_groups.setdefault(c, []).append(i)

print("\n=== Loading base ===", flush=True)
base = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager",
                                            dtype=torch.float16).to(dev)
base.eval()
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
torch.mps.empty_cache() if dev == "mps" else None

stream = build_stream()

all_results = {"pools": POOLS, "lambdas": LAMBDAS, "epochs": EPOCHS, "lr": LR,
               "MICRO_BATCH": MICRO_BATCH, "GEO_EVERY": GEO_EVERY, "TAU": TAU}

for pool in POOLS:
    print(f"\n========== Pool = {pool} ==========", flush=True)
    reps_base = extract_reps(base, eval_sents, tok, pool)
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
        llm, lgeo = train_arm(lm, stream, lambd=lam, epochs=EPOCHS, pool=pool, tok=tok)
        reps = extract_reps(lm, eval_sents, tok, pool)
        pos, neg = compute_pos_neg(reps, concept_groups)
        margin = pos - neg
        print(f"  λ={lam}: pos={pos:.4f} neg={neg:.4f} margin={margin:.4f}  "
              f"Δ={margin-margin0:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
        pool_res[f"lambda_{lam}"] = {
            "lambd": lam, "pos": pos, "neg": neg, "margin": margin,
            "losses_lm": llm, "losses_info_nce": lgeo,
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
fig.suptitle("Phase 2 v3: LM stream + periodic InfoNCE — Margin vs λ (0.5B)")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_margin_vs_lambda.png"))
plt.close(fig)

# LM loss
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150, sharey=True)
for ax, pool in zip(axes, POOLS):
    for lam in LAMBDAS:
        if lam == 0.0: continue
        losses = all_results[pool][f"lambda_{lam}"]["losses_lm"]
        ax.plot(range(1, len(losses)+1), losses, "o-", label=f"λ={lam}", lw=1.5, ms=5)
    ax.set_title(f"Pool = {pool}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LM Loss")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)
fig.suptitle("LM Loss Across Lambda Sweep (0.5B v3)")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_loss_vs_lambda.png"))
plt.close(fig)

# InfoNCE
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150, sharey=True)
for ax, pool in zip(axes, POOLS):
    for lam in LAMBDAS:
        if lam == 0.0: continue
        losses = all_results[pool][f"lambda_{lam}"]["losses_info_nce"]
        ax.plot(range(1, len(losses)+1), losses, "o-", label=f"λ={lam}", lw=1.5, ms=5)
    ax.set_title(f"Pool = {pool}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("InfoNCE Loss")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)
fig.suptitle("InfoNCE Loss Across Lambda Sweep (0.5B v3)")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_info_nce.png"))
plt.close(fig)


# ── H-4 判定 ─────────────────────────────────────────────────────────────
h4 = {}
for pool in POOLS:
    base = all_results[pool]["baseline"]["margin"]
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
    "arm": "geometry-meaning-phase2-georeg-v3-0.5b-poc",
    "model": "Qwen2.5-0.5B-Instruct + LoRA(r=16, α=32)",
    "loss_form": "LM stream block + every 4 steps, micro-batch cross-sentence InfoNCE (tau=0.07)",
    "MICRO_BATCH": MICRO_BATCH,
    "GEO_EVERY": GEO_EVERY,
    "TAU": TAU,
    "lr": LR,
    "epochs": EPOCHS,
    "lambdas": LAMBDAS,
    "pools": POOLS,
    "results": all_results,
    "H4": h4,
    "verdict": ("geometric regularization helps (v3 LM+InfoNCE)"
                if all(h4[p]["pass"] for p in POOLS)
                else "partial" if any(h4[p]["pass"] for p in POOLS)
                else "geometric regularization does not help at 0.5B v3"),
}
out_path = os.path.join(HERE, "metrics_georeg_v3.json")
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
