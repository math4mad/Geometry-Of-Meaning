#!/usr/bin/env python3
"""Phase 2 v2 · 7B 几何正则化 PoC — Qwen2.5-7B-Instruct + QLoRA

设计依据: design_note_7b_georeg.md
0.5B v3 (train_georeg_v3.py) 已 H-4 PASS, 现在迁移到 7B

关键变化 vs 0.5B v3:
  - 4bit QLoRA (bitsandbytes nf4) + bf16 compute
  - BLOCK 128 -> 256
  - LR 1e-4 -> 5e-5
  - Warmup 50 步
  - lambda 扫描: {0, 0.1, 0.5, 1.0} (vs {0, 0.05, 0.2, 1.0})
  - 备援 margin push 0.1*ReLU(cos_neg - 0.5)
  - 梯度裁剪 1.0
  - gradient checkpointing (省显存)
  - 监控 lm_loss: 若 ep2 > 1.5 * ep1 -> 终止该 arm

输出:
  metrics_georeg_7b.json
  figs_georeg_7b/fig_margin_vs_lambda.png
  figs_georeg_7b/fig_loss_vs_lambda.png
  figs_georeg_7b/fig_info_nce.png
"""
import json, os, time, random
import numpy as np
import torch
import torch.nn.functional as F

# ── Config ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus_paraphrase.json")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
FIGS = os.path.join(HERE, "figs_georeg_7b")
os.makedirs(FIGS, exist_ok=True)

LAMBDAS = [0.0, 0.1, 0.5, 1.0]
POOLS = ["last", "mean"]
EPOCHS = 2
LR = 5e-5
WARMUP = 50
BLOCK = 256
MICRO_BATCH = 6
N_CONCEPT_PER_BATCH = 4
N_DUP_CONCEPT = 2
SENT_PER_DUP = 2
GEO_EVERY = 4
TAU = 0.07
GRAD_CLIP = 1.0
LM_LOSS_RATIO_LIMIT = 1.5  # ep2 > 1.5 * ep1 -> abort arm

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {dev}", flush=True)
if dev == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

# ── Matplotlib (headless) ────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 库 (延迟 import, 让错误信息更清晰) ──────────────────────────────────
print("imports...", flush=True)
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ── 语料 ──────────────────────────────────────────────────────────────────
with open(CORPUS, encoding="utf-8") as f:
    corpus = json.load(f)
paraphrases = corpus["paraphrases"]
concept_ids = sorted(paraphrases.keys())

concept_to_sents = {cid: paraphrases[cid]["sentences"] for cid in concept_ids}
all_sents = [(cid, s) for cid in concept_ids for s in concept_to_sents[cid]]


# ── LM 流 ────────────────────────────────────────────────────────────────
def build_stream():
    parts = []
    for cid in concept_ids:
        data = paraphrases[cid]
        name = data["name"]
        parts.append(f"{name}就是{name}。")
        for s in concept_to_sents[cid]:
            parts.append(s)
    return "\n".join(parts)


# ── Pool forward (2 forward 共享 hidden) ──────────────────────────────────
def batch_pool(h, mask, pool):
    if pool == "mean":
        v = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:  # last
        idx = (mask.squeeze(-1).sum(dim=1) - 1).clamp(min=0)
        v = h[torch.arange(h.size(0), device=h.device), idx]
    return F.normalize(v.float(), dim=-1)


def batch_pool_forward(model, sents_text, pool, tok):
    enc = tok(sents_text, return_tensors="pt", padding=True, truncation=True,
              max_length=64, add_special_tokens=False).to(dev)
    out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[-1]
    mask = enc.attention_mask.unsqueeze(-1).float()
    return batch_pool(h, mask, pool), enc


# ── InfoNCE + 备援 margin push ───────────────────────────────────────────
M_NEG = 0.5
MARGIN_PUSH_COEF = 0.1


def info_nce_loss(z, labels):
    B = z.size(0)
    sim = (z @ z.T) / TAU
    self_mask = torch.eye(B, device=z.device, dtype=torch.bool)
    sim_masked = sim.masked_fill(self_mask, -1e9)

    eq = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    pos_mask = eq & ~self_mask
    n_pos = pos_mask.sum(dim=1).float()
    valid = n_pos > 0
    if not valid.any():
        return torch.tensor(0.0, device=z.device), 0.0, 0.0

    log_denom = torch.logsumexp(sim_masked, dim=1, keepdim=True)
    log_prob = sim_masked - log_denom
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
    loss_per_anchor = -pos_log_prob / n_pos.clamp(min=1)
    info_nce = loss_per_anchor[valid].mean()

    # 备援 margin push: 异概念 sim > 0.5 时加大间距
    neg_mask = ~eq & ~self_mask
    if neg_mask.any():
        cos_neg = sim * TAU
        margin_pen = F.relu(cos_neg - M_NEG) * neg_mask.float()
        margin_loss = margin_pen.sum() / (neg_mask.sum() + 1e-6)
    else:
        margin_loss = torch.tensor(0.0, device=z.device)

    total = info_nce + MARGIN_PUSH_COEF * margin_loss
    return total, float(info_nce.detach()), float(margin_loss.detach())


# ── 表示提取 (eval) ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_reps(model, sentences, tok, pool):
    out_arr = []
    B = 8  # 7B + 4bit, batch 8 够稳
    for i in range(0, len(sentences), B):
        v, _ = batch_pool_forward(model, sentences[i:i+B], pool, tok)
        out_arr.append(v.cpu().numpy())
        if dev == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(out_arr, axis=0)


def compute_pos_neg(reps, groups):
    pos_sims, neg_sims = [], []
    for cid, indices in groups.items():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                pos_sims.append(float(np.dot(reps[indices[a]], reps[indices[b]])))
    rng = np.random.default_rng(42)
    cids = sorted(groups.keys())
    sampled = rng.choice(len(cids), size=min(20, len(cids)), replace=False)
    for i_a in range(len(sampled)):
        for i_b in range(i_a + 1, len(sampled)):
            ia = groups[cids[sampled[i_a]]][0]
            ib = groups[cids[sampled[i_b]]][0]
            neg_sims.append(float(np.dot(reps[ia], reps[ib])))
    return float(np.mean(pos_sims)), float(np.mean(neg_sims))


# ── Micro-batch 构造 ─────────────────────────────────────────────────────
def build_micro_batches(rng):
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


# ── 单 arm 训练 ──────────────────────────────────────────────────────────
def train_arm(model, stream_text, lambd, epochs, pool, tok):
    ids = tok(stream_text, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    blocks = ids[0].unfold(0, BLOCK, BLOCK).clone()
    n_blocks = blocks.size(0)
    total_steps = n_blocks * epochs
    geo_steps = sum(1 for s in range(n_blocks * epochs)
                    if s % GEO_EVERY == 0 and lambd > 0.0)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=WARMUP,
                                            num_training_steps=total_steps)
    rng = random.Random(42)
    micro_batches = build_micro_batches(rng)
    n_mb = len(micro_batches)

    model.train()
    losses_lm, losses_info, losses_margin = [], [], []
    aborted = False

    for ep in range(epochs):
        ep_lm, ep_info, ep_margin, nb_lm, nb_geo = 0.0, 0.0, 0.0, 0, 0
        step = 0
        for blk in blocks:
            opt.zero_grad()
            x = blk.unsqueeze(0)
            out = model(x, labels=x, output_hidden_states=True)
            lm_loss = out.loss

            if lambd > 0.0 and step % GEO_EVERY == 0:
                mb = micro_batches[(step // GEO_EVERY) % n_mb]
                labels_mb, texts_mb = mb
                enc = tok(texts_mb, return_tensors="pt", padding=True, truncation=True,
                          max_length=64, add_special_tokens=False).to(dev)
                out2 = model(**enc, output_hidden_states=True)
                h = out2.hidden_states[-1]
                mask = enc.attention_mask.unsqueeze(-1).float()
                z = batch_pool(h, mask, pool)
                label_t = torch.tensor([concept_ids.index(l) for l in labels_mb], device=dev)
                geo_loss, info_val, margin_val = info_nce_loss(z, label_t)
                total = lm_loss + lambd * geo_loss
                total.backward()
                ep_info += info_val; ep_margin += margin_val; nb_geo += 1
            else:
                lm_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()
            sched.step()
            ep_lm += float(lm_loss); nb_lm += 1; step += 1

        losses_lm.append(ep_lm / nb_lm)
        losses_info.append(ep_info / max(nb_geo, 1))
        losses_margin.append(ep_margin / max(nb_geo, 1))
        print(f"    ep={ep+1}/{epochs} lm={losses_lm[-1]:.3f} "
              f"info_nce={losses_info[-1]:.4f} margin_push={losses_margin[-1]:.4f} "
              f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

        # 异常保护
        if ep == 1 and losses_lm[1] > LM_LOSS_RATIO_LIMIT * losses_lm[0]:
            print(f"    [ABORT] lm loss exploded: {losses_lm[0]:.3f} -> {losses_lm[1]:.3f}", flush=True)
            aborted = True
            break

    model.eval()
    return losses_lm, losses_info, losses_margin, aborted


# ── 加载 4bit 模型 (单次, 复用) ─────────────────────────────────────────
def load_quant_model():
    print(f"loading 4bit QLoRA model {MODEL}...", flush=True)
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, attn_implementation="eager",
        device_map="auto",
    )
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=TARGETS,
                     lora_dropout=0.0, task_type="CAUSAL_LM")
    model = get_peft_model(m, cfg)
    model.print_trainable_parameters()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    return model


# ── 主流程 ────────────────────────────────────────────────────────────────
def main():
    eval_sents = [s for _, s in all_sents]
    eval_labels = [c for c, _ in all_sents]
    concept_groups = {}
    for i, c in enumerate(eval_labels):
        concept_groups.setdefault(c, []).append(i)

    # baseline: 只加载一次, 跑两个 pool
    print("\n=== Loading 4bit model for baseline ===", flush=True)
    base_model = load_quant_model()
    base_model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    baseline_margins = {}
    for pool in POOLS:
        reps_base = extract_reps(base_model, eval_sents, tok, pool)
        pos, neg = compute_pos_neg(reps_base, concept_groups)
        margin = pos - neg
        baseline_margins[pool] = {"pos": pos, "neg": neg, "margin": margin}
        print(f"  baseline[{pool}]: pos={pos:.4f} neg={neg:.4f} margin={margin:.4f}", flush=True)
        if dev == "cuda":
            torch.cuda.empty_cache()

    # 释放 baseline 显存
    del base_model
    if dev == "cuda":
        torch.cuda.empty_cache()
    elif dev == "mps":
        torch.mps.empty_cache()

    stream = build_stream()
    all_results = {
        "pools": POOLS, "lambdas": LAMBDAS, "epochs": EPOCHS, "lr": LR,
        "BLOCK": BLOCK, "MICRO_BATCH": MICRO_BATCH, "GEO_EVERY": GEO_EVERY,
        "TAU": TAU, "GRAD_CLIP": GRAD_CLIP,
        "WARMUP": WARMUP, "MARGIN_PUSH_COEF": MARGIN_PUSH_COEF,
    }

    for pool in POOLS:
        print(f"\n========== Pool = {pool} ==========", flush=True)
        pool_res = {"baseline": baseline_margins[pool]}

        for lam in LAMBDAS:
            print(f"\n  --- pool={pool} λ={lam} ---", flush=True)
            model = load_quant_model()
            t0 = time.time()
            llm, linfo, lmargin, aborted = train_arm(
                model, stream, lambd=lam, epochs=EPOCHS, pool=pool, tok=tok)
            reps = extract_reps(model, eval_sents, tok, pool)
            pos, neg = compute_pos_neg(reps, concept_groups)
            margin = pos - neg
            base = baseline_margins[pool]["margin"]
            print(f"  λ={lam}: pos={pos:.4f} neg={neg:.4f} margin={margin:.4f}  "
                  f"Δ={margin-base:+.4f}  ({time.time()-t0:.0f}s)"
                  f"{'  [ABORTED]' if aborted else ''}", flush=True)
            pool_res[f"lambda_{lam}"] = {
                "lambd": lam, "pos": pos, "neg": neg, "margin": margin,
                "losses_lm": llm, "losses_info_nce": linfo,
                "losses_margin_push": lmargin, "aborted": aborted,
            }
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()
            elif dev == "mps":
                torch.mps.empty_cache()
        all_results[pool] = pool_res

    # ── 绘图 ─────────────────────────────────────────────────────────────
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
    fig.suptitle("Phase 2 v2: 7B + QLoRA — Margin vs λ")
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
    fig.suptitle("7B LM Loss Across Lambda Sweep")
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
    fig.suptitle("7B InfoNCE Loss Across Lambda Sweep")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_info_nce.png"))
    plt.close(fig)

    # ── H-4' 判定 ─────────────────────────────────────────────────────────
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

    # ── 保存 ──────────────────────────────────────────────────────────────
    out = {
        "minted": time.strftime("%Y-%m-%d"),
        "arm": "geometry-meaning-phase2-georeg-7b-qlora",
        "model": "Qwen2.5-7B-Instruct + QLoRA(r=16, α=32, 4bit nf4, bf16)",
        "loss_form": f"LM stream block + every {GEO_EVERY} steps, micro-batch cross-sentence "
                     f"InfoNCE (tau={TAU}) + {MARGIN_PUSH_COEF}*ReLU(cos_neg-{M_NEG})",
        "MICRO_BATCH": MICRO_BATCH, "GEO_EVERY": GEO_EVERY, "TAU": TAU,
        "BLOCK": BLOCK, "WARMUP": WARMUP, "GRAD_CLIP": GRAD_CLIP,
        "lr": LR, "epochs": EPOCHS, "lambdas": LAMBDAS, "pools": POOLS,
        "results": all_results, "H4": h4,
        "verdict": ("7B geometric regularization helps"
                    if all(h4[p]["pass"] for p in POOLS)
                    else "partial" if any(h4[p]["pass"] for p in POOLS)
                    else "7B geometric regularization does not help"),
    }
    out_path = os.path.join(HERE, "metrics_georeg_7b.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"\n{'='*60}\nDONE — {out_path}")
    print(f"  figs: {FIGS}/")
    print(f"  H-4' verdict: {out['verdict']}")
    for pool in POOLS:
        print(f"\n  [{pool}]")
        print(f"    baseline margin: {all_results[pool]['baseline']['margin']:.4f}")
        for lam in LAMBDAS:
            r = all_results[pool][f'lambda_{lam}']
            print(f"    λ={lam:>4}: margin={r['margin']:.4f}  "
                  f"(Δ={r['margin']-all_results[pool]['baseline']['margin']:+.4f})"
                  f"{'  [ABORTED]' if r['aborted'] else ''}")
        h = h4[pool]
        print(f"    best λ={h['best_lambda']}  margin={h['best_margin']:.4f}  "
              f"Δ={h['delta']:+.4f}  {'PASS' if h['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
