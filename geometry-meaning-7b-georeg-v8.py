#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry of Meaning — Phase 2 v2: 7B Geometric Regularization (Kaggle arm)
"""
import os
# 关键: 在 import torch/cuda 之前限定 GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import os as _osx
_osx.system("python -m pip uninstall -y -q torchao 2>/dev/null")
_osx.system("python -m pip install -q 'bitsandbytes==0.50.2'")
_osx.system("python -m pip show bitsandbytes | grep -i version")

import gzip
import json, os, time, base64, hashlib, traceback, random, gc
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------- 语料内嵌 (gz+b64 + sha256 自校验; Kaggle 源 1MB 上限) ----------
_CORPUS_B64 = "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"
_CORPUS_SHA = "5d9dd063d27f15a90aa11df54259bf066f27f658271ac34837e673b01bc52a31"


def _load_corpus():
    data = gzip.decompress(base64.b64decode(_CORPUS_B64))
    got = hashlib.sha256(data).hexdigest()
    assert got == _CORPUS_SHA, f"corpus sha256 mismatch: got {got} want {_CORPUS_SHA}"
    return json.loads(data.decode("utf-8"))


corpus = _load_corpus()
paraphrases = corpus["paraphrases"]
concept_ids = sorted(paraphrases.keys())
concept_to_sents = {cid: paraphrases[cid]["sentences"] for cid in concept_ids}
all_sents = [(cid, s) for cid in concept_ids for s in concept_to_sents[cid]]
print(f"corpus loaded: {len(concept_ids)} concepts, {len(all_sents)} sentences", flush=True)

# ---------- 旋钮区 ----------
LAMBDAS = [0.0, 0.5, 1.0, 2.0, 5.0]  # v8: 微调 - 砍 λ=0 (v7 端冗余), 加 λ=0.5 (细颗粒), 保留 1/2/5
POOLS = ["last", "mean"]
EPOCHS = 3  # v8: 沿用 v7 的 3 epoch
LR = 5e-5   # v8: **改回 v5/v6 的 LR 5e-5** — v7 LR=3e-5 太低导致 Δ 下降 (-0.005 vs v6)
WARMUP = 50
BLOCK = 128  # OOM 后调整: 7B 4bit + grad checkpoint 下, 256 block 太大
MICRO_BATCH = 4  # OOM 后调整: 6 句/4 concept 在 4bit 下 forward 占用 ~2GB
N_CONCEPT_PER_BATCH = 4
N_DUP_CONCEPT = 2
SENT_PER_DUP = 2
GEO_EVERY = 4
TAU = 0.07
GRAD_CLIP = 1.0
M_NEG = 0.5
MARGIN_PUSH_COEF = 0.1
LM_LOSS_RATIO_LIMIT = 1.5
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
SEED = 13

MOUNT = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"  # debug kernel 验证的实际挂载路径
QUANT = "4bit"

HERE = "/kaggle/working"
FIGS = os.path.join(HERE, "figs_georeg_7b")
os.makedirs(FIGS, exist_ok=True)
OUT_METRICS = os.path.join(HERE, "metrics_georeg_7b_kaggle_v8.json")
OUT_FULL = os.path.join(HERE, "metrics_georeg_7b_v8_full.json")  # v8: 含 all_results 的全量存档 (不被 PARTIAL 覆盖)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}", flush=True)
if dev == "cuda":
    print(f"GPU count visible: {torch.cuda.device_count()}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)


def fl(stage, **kv):
    """即算即落盘: 每阶段算完即 dump + REPORT_LINE 哨兵 (日志即幸存者)"""
    PARTIAL.update({"stage": stage, **kv})
    with open(OUT_METRICS, "w") as f:
        json.dump(PARTIAL, f, ensure_ascii=False, indent=1)
    line = f"REPORT_LINE::stage={stage}"
    for k, v in kv.items():
        line += f"  {k}={v}"
    print(line, flush=True)


PARTIAL = {
    "arm": "geometry-meaning-7b-georeg",
    "model_mount": MOUNT, "quant": QUANT,
    "lambdas": LAMBDAS, "pools": POOLS, "epochs": EPOCHS,
    "lr": LR, "block": BLOCK, "micro_batch": MICRO_BATCH,
    "geo_every": GEO_EVERY, "tau": TAU, "margin_push_coef": MARGIN_PUSH_COEF,
}


# ---------- LM 流 ----------
def build_stream():
    parts = []
    for cid in concept_ids:
        data = paraphrases[cid]
        name = data["name"]
        parts.append(f"{name}就是{name}。")
        for s in concept_to_sents[cid]:
            parts.append(s)
    return "\n".join(parts)


# ---------- Pool forward ----------
def batch_pool(h, mask, pool):
    if pool == "mean":
        v = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        # bf16 tensor 不能作 index, 转 long
        idx = (mask.squeeze(-1).sum(dim=1) - 1).clamp(min=0).long()
        v = h[torch.arange(h.size(0), device=h.device), idx]
    return F.normalize(v.float(), dim=-1)


def batch_pool_forward(model, sents_text, pool, tok):
    enc = tok(sents_text, return_tensors="pt", padding=True, truncation=True,
              max_length=64, add_special_tokens=False).to(dev)
    out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[-1]
    mask = enc.attention_mask.unsqueeze(-1).float()
    return batch_pool(h, mask, pool), enc


# ---------- InfoNCE + 备援 margin push ----------
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
    neg_mask = ~eq & ~self_mask
    if neg_mask.any():
        cos_neg = sim * TAU
        margin_pen = F.relu(cos_neg - M_NEG) * neg_mask.float()
        margin_loss = margin_pen.sum() / (neg_mask.sum() + 1e-6)
    else:
        margin_loss = torch.tensor(0.0, device=z.device)
    total = info_nce + MARGIN_PUSH_COEF * margin_loss
    return total, float(info_nce.detach()), float(margin_loss.detach())


# ---------- 表示提取 ----------
@torch.no_grad()
def extract_reps(model, sentences, tok, pool):
    out_arr = []
    B = 8
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


# ---------- Micro-batch ----------
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


# ---------- 单 arm 训练 ----------
def train_arm(model, stream_text, lambd, epochs, pool, tok):
    ids = tok(stream_text, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    blocks = ids[0].unfold(0, BLOCK, BLOCK).clone()
    n_blocks = blocks.size(0)
    total_steps = n_blocks * epochs

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

        if ep == 1 and len(losses_lm) >= 2 and \
                losses_lm[1] > LM_LOSS_RATIO_LIMIT * losses_lm[0]:
            print(f"    [ABORT] lm loss exploded: {losses_lm[0]:.3f} -> {losses_lm[1]:.3f}", flush=True)
            aborted = True
            break

    model.eval()
    return losses_lm, losses_info, losses_margin, aborted


# ---------- 加载 4bit 模型 ----------
def load_quant_model():
    print(f"loading 4bit QLoRA model {MOUNT}...", flush=True)
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    m = AutoModelForCausalLM.from_pretrained(
        MOUNT, quantization_config=bnb, attn_implementation="eager",
        device_map="auto",
    )
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True,
                                        gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=TARGETS,
                     lora_dropout=0.0, task_type="CAUSAL_LM")
    model = get_peft_model(m, cfg)
    model.print_trainable_parameters()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    return model


# ---------- 主流程 ----------
def main():
    eval_sents = [s for _, s in all_sents]
    eval_labels = [c for c, _ in all_sents]
    concept_groups = {}
    for i, c in enumerate(eval_labels):
        concept_groups.setdefault(c, []).append(i)

    fl("loading_baseline")
    base_model = load_quant_model()
    base_model.eval()
    tok = AutoTokenizer.from_pretrained(MOUNT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    baseline_margins = {}
    for pool in POOLS:
        reps_base = extract_reps(base_model, eval_sents, tok, pool)
        pos, neg = compute_pos_neg(reps_base, concept_groups)
        margin = pos - neg
        baseline_margins[pool] = {"pos": pos, "neg": neg, "margin": margin}
        fl(f"baseline_{pool}", pos=pos, neg=neg, margin=margin)
        if dev == "cuda":
            torch.cuda.empty_cache()

    del base_model
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    print(f"[mem after baseline del] allocated={torch.cuda.memory_allocated()/1e9:.2f}GB  "
          f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

    stream = build_stream()
    all_results = {"baseline": baseline_margins, "pools": POOLS, "lambdas": LAMBDAS}

    for pool in POOLS:
        print(f"\n========== Pool = {pool} ==========", flush=True)
        fl(f"pool_start_{pool}")
        pool_res = {"baseline": baseline_margins[pool]}

        for lam in LAMBDAS:
            print(f"\n  --- pool={pool} λ={lam} ---", flush=True)
            fl(f"arm_start_{pool}_lambda_{lam}")
            model = load_quant_model()
            t0 = time.time()
            try:
                llm, linfo, lmargin, aborted = train_arm(
                    model, stream, lambd=lam, epochs=EPOCHS, pool=pool, tok=tok)
                reps = extract_reps(model, eval_sents, tok, pool)
                pos, neg = compute_pos_neg(reps, concept_groups)
                margin = pos - neg
            except Exception as e:
                print(f"  [ERROR] {e}", flush=True)
                print(traceback.format_exc(), flush=True)
                pos, neg, margin = 0.0, 0.0, 0.0
                llm, linfo, lmargin, aborted = [], [], [], True
            base = baseline_margins[pool]["margin"]
            elapsed = time.time() - t0
            print(f"  λ={lam}: pos={pos:.4f} neg={neg:.4f} margin={margin:.4f}  "
                  f"Δ={margin-base:+.4f}  ({elapsed:.0f}s)"
                  f"{'  [ABORTED]' if aborted else ''}", flush=True)
            pool_res[f"lambda_{lam}"] = {
                "lambd": lam, "pos": pos, "neg": neg, "margin": margin,
                "losses_lm": llm, "losses_info_nce": linfo,
                "losses_margin_push": lmargin, "aborted": aborted, "elapsed_s": elapsed,
            }
            fl(f"arm_done_{pool}_lambda_{lam}", margin=margin, delta=margin-base,
               elapsed_s=elapsed, aborted=aborted)
            # 显式释放 (v4 OOM 后调整)
            del model
            gc.collect()
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(f"  [mem after del] allocated={torch.cuda.memory_allocated()/1e9:.2f}GB  "
                  f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
        all_results[pool] = pool_res
        fl(f"pool_done_{pool}")

    # ── 绘图 ─────────────────────────────────────────────────────────────
    try:
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
        print("fig_margin_vs_lambda.png saved", flush=True)
    except Exception as e:
        print(f"[fig] {e}", flush=True)

    # ── H-4' 判定 ─────────────────────────────────────────────────────────
    h4 = {}
    for pool in POOLS:
        base = all_results[pool]["baseline"]["margin"]
        pos_lams = [l for l in LAMBDAS if l > 0]
        best_lam = max(pos_lams, key=lambda l: all_results[pool][f"lambda_{l}"]["margin"])
        best_margin = all_results[pool][f"lambda_{best_lam}"]["margin"]
        h4[pool] = {"best_lambda": best_lam, "best_margin": best_margin,
                    "delta": best_margin - base, "pass": best_margin - base >= 0.05}

    out = {
        "arm": "geometry-meaning-7b-georeg",
        "model_mount": MOUNT, "quant": QUANT,
        "loss_form": f"LM stream + every {GEO_EVERY} steps micro-batch InfoNCE "
                     f"(tau={TAU}) + {MARGIN_PUSH_COEF}*ReLU(cos_neg-{M_NEG})",
        "lr": LR, "epochs": EPOCHS, "lambdas": LAMBDAS, "pools": POOLS,
        "BLOCK": BLOCK, "WARMUP": WARMUP, "GRAD_CLIP": GRAD_CLIP,
        "MICRO_BATCH": MICRO_BATCH, "GEO_EVERY": GEO_EVERY, "TAU": TAU,
        "results": all_results, "H4": h4,
        "verdict": ("7B geometric regularization helps"
                    if all(h4[p]["pass"] for p in POOLS)
                    else "partial" if any(h4[p]["pass"] for p in POOLS)
                    else "7B geometric regularization does not help"),
    }
    with open(OUT_METRICS, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    # v6: 额外存全量 (含 all_results) 到 OUT_FULL, 不被 fl() 覆盖
    try:
        with open(OUT_FULL, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"FULL metrics saved: {OUT_FULL}", flush=True)
    except Exception as e:
        print(f"WARNING: full save failed: {e}", flush=True)

    print(f"\n{'='*60}\nDONE — {OUT_METRICS}")
    print(f"  H-4' verdict: {out['verdict']}")
    for pool in POOLS:
        print(f"\n  [{pool}]  baseline={all_results[pool]['baseline']['margin']:.4f}")
        for lam in LAMBDAS:
            r = all_results[pool][f'lambda_{lam}']
            print(f"    λ={lam:>4}: margin={r['margin']:.4f}  (Δ={r['margin']-all_results[pool]['baseline']['margin']:+.4f})"
                  f"{'  [ABORTED]' if r['aborted'] else ''}")
        h = h4[pool]
        print(f"    best λ={h['best_lambda']}  margin={h['best_margin']:.4f}  "
              f"Δ={h['delta']:+.4f}  {'PASS' if h['pass'] else 'FAIL'}")
    fl("done", verdict=out["verdict"])


if __name__ == "__main__":
    main()
