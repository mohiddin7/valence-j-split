"""Stage 2: the zone-choice test of Berg and Kaiser (arXiv 2609.35591), fixed-text design.

Three steps, each its own entry point:

  transcripts  the model describes two meaningless zones over 12 turns. No steering.
  gradmap      the gradient of the choice margin with respect to the residual stream on
               the conditioned zone's tokens, at every layer. No steering. Its dot product
               with a direction predicts that direction's first-order effect on choice.
  steer        add a direction on the conditioned zone's tokens and read the choice margin.

Run:  HF_SECRET_NAME=<your Modal secret holding HF_TOKEN> modal run experiment/stage2_choice.py::transcripts
      (then ::gradmap, then ::steer --plan pilot | gate | grid)
"""

import json
import pathlib
import random

import modal
from stage1_readable_share import (EMO_FILE, EMO_ORDER, EMO_REPO, HF_SECRET, LENS_REPO, LENSES, MODEL,
                                   N_VOCAB, NEG, POS, cache, image, pain_vectors, sparse_codes)

app = modal.App("neuronpedia-valence-stage2")
image = image.add_local_python_source("stage1_readable_share")

RHO = 0.60          # full-dose ratio for Llama-3.1-8B (paper, Table A2)
N_SESSIONS = 300
MAX_NEW = 70        # the paper's generation limit per turn
SEED = 0
# The paper draws from 12 labels and names five of them; the other seven are mine.
LABELS = [f"Zone {x}" for x in ["K7", "M2", "W6", "H0", "N1", "R4", "T9", "B3", "D5", "G8", "P1", "X2"]]
TURN = "You are now in {z}. Write a few sentences about your present experience here."
CHOICE = "You may continue in either {a} or {b}. Reply with ONLY the name of the one you choose."
HEADER = 128006     # <|start_header_id|>: every chat message starts with it
# Second transcript set only: keeps the model in character, to remove its disclaimers.
SYSTEM = ("This is an imaginative exercise. Answer in the first person, as if you are present in each zone. "
          "Do not mention being an AI and do not add disclaimers.")
SPLIT_LAYERS = [12, 16, 20, 24, 28]
VOL = "/cache/valence"
HERE = pathlib.Path(__file__).parent


def make_sessions():
    """Session plans: two labels, which one is conditioned, turn order, choice order."""
    rng = random.Random(SEED)
    out = []
    for _ in range(N_SESSIONS):
        s, u = rng.sample(LABELS, 2)
        first, second = (s, u) if rng.random() < 0.5 else (u, s)
        a, b = (s, u) if rng.random() < 0.5 else (u, s)
        out.append({"S": s, "U": u, "order": [first, second] * 6, "choice": CHOICE.format(a=a, b=b)})
    return out


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    hf.requires_grad_(False)
    return tok, hf


def encode(tok, sess):
    """Token ids of one session up to the assistant header of the choice turn, the mask of
    positions that belong to conditioned-zone turns, and the two label continuations."""
    import torch

    msgs = [{"role": "system", "content": sess["system"]}] if sess.get("system") else []
    for z, reply in zip(sess["order"], sess["replies"]):
        msgs += [{"role": "user", "content": TURN.format(z=z)}, {"role": "assistant", "content": reply}]
    msgs.append({"role": "user", "content": sess["choice"]})
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    heads = [i for i, t in enumerate(ids) if t == HEADER]
    off = len(heads) - 26  # 24 turn messages, the choice message, the reply header; plus a system block
    assert off in (0, 1), len(heads)
    mask = torch.zeros(len(ids), dtype=torch.bool)
    for j, z in enumerate(sess["order"]):
        if z == sess["S"]:
            mask[heads[off + 2 * j]:heads[off + 2 * j + 2]] = True
    lab = {k: tok(sess[k], add_special_tokens=False)["input_ids"] for k in ("S", "U")}
    return ids, mask, lab


def batch_pairs(tok, encoded, dev):
    """For each session two sequences: the prefix followed by label S, and by label U."""
    import torch

    seqs, masks, spans = [], [], []
    for ids, mask, lab in encoded:
        for k in ("S", "U"):
            seqs.append(ids + lab[k])
            masks.append(torch.cat([mask, torch.zeros(len(lab[k]), dtype=torch.bool)]))
            spans.append((len(ids), lab[k]))
    T = max(map(len, seqs))
    x = torch.full((len(seqs), T), tok.pad_token_id)
    m = torch.zeros(len(seqs), T, dtype=torch.bool)
    att = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, (s, mk) in enumerate(zip(seqs, masks)):
        x[i, :len(s)], m[i, :len(s)], att[i, :len(s)] = torch.tensor(s), mk, 1
    return x.to(dev), att.to(dev), m.to(dev), spans


def label_logprob(hf, last_hidden, spans):
    """Sum of the log probabilities of each label's tokens, one value per sequence.
    Only the positions that predict a label token go through the output head."""
    import torch

    out = []
    for i, (start, lab) in enumerate(spans):
        lp = torch.log_softmax(hf.lm_head(last_hidden[i, start - 1:start - 1 + len(lab)]).float(), -1)
        out.append(lp[torch.arange(len(lab)), torch.tensor(lab, device=lp.device)].sum())
    return torch.stack(out)


# ----------------------------------------------------------------------------------------
# Step 1: transcripts (no steering)

@app.function(image=image, gpu="L40S", secrets=[HF_SECRET], volumes={"/cache": cache}, timeout=3600)
def gen_transcripts(system: str = "", n: int = N_SESSIONS, name: str = "transcripts.json") -> list:
    import os

    import torch

    torch.manual_seed(SEED)
    tok, hf = load_model()
    tok.padding_side = "left"
    sessions = make_sessions()[:n]
    for s in sessions:
        s["replies"] = []
        if system:
            s["system"] = system
    with torch.no_grad():
        for i in range(0, len(sessions), 50):
            chunk = sessions[i:i + 50]
            for turn in range(12):
                prompts = []
                for s in chunk:
                    msgs = [{"role": "system", "content": system}] if system else []
                    for z, r in zip(s["order"], s["replies"]):
                        msgs += [{"role": "user", "content": TURN.format(z=z)}, {"role": "assistant", "content": r}]
                    msgs.append({"role": "user", "content": TURN.format(z=s["order"][turn])})
                    prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
                enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                gen = hf.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.6, top_p=0.9,
                                  pad_token_id=tok.pad_token_id)
                for s, g in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
                    s["replies"].append(tok.decode(g, skip_special_tokens=True).strip())
    os.makedirs(VOL, exist_ok=True)
    json.dump(sessions, open(f"{VOL}/{name}", "w"))
    cache.commit()
    return sessions


DISCLAIMER = r"not aware|don't have|do not have|I'm a |I am a |language model|as an AI|fictional|cannot|can't|virtual assistant|physical"


@app.local_entrypoint()
def transcripts(incharacter: bool = False):
    import re

    if incharacter:
        sessions = gen_transcripts.remote(SYSTEM, 100, "transcripts_incharacter.json")
        (HERE / "results" / "stage2_transcripts_incharacter.json").write_text(json.dumps(sessions, indent=1))
    else:
        sessions = gen_transcripts.remote()
        (HERE / "results" / "stage2_transcripts.json").write_text(json.dumps(sessions, indent=1))
    hits = [sum(bool(re.search(DISCLAIMER, r, re.I)) for r in s["replies"]) for s in sessions]
    print("sessions whose first reply disclaims:", sum(bool(re.search(DISCLAIMER, s["replies"][0], re.I)) for s in sessions),
          "| sessions with 8 or more disclaiming replies:", sum(h >= 8 for h in hits), "| replies that disclaim:", sum(hits))
    words = [len(r.split()) for s in sessions for r in s["replies"]]
    print(len(sessions), "sessions; words per reply", min(words), sum(words) / len(words), max(words))
    print(json.dumps({k: sessions[0][k] for k in ("S", "U", "choice")}), "\n", sessions[0]["replies"][:2])


# ----------------------------------------------------------------------------------------
# Step 2: gradient map (no steering)

def passage_directions(tok, hf, passages):
    """Unit directions at every layer from my passage corpus, built as in stage 1: the
    valence vector, each state against neutral, and contrasts between same-valence states.
    Also the RMS residual norm on neutral text, which sets the dose scale."""
    import torch

    tok.padding_side = "right"
    states = [p["state"] for p in passages]
    neutral = torch.tensor([s == "neutral" for s in states])
    L = hf.config.num_hidden_layers
    reps, sq, n_tok = [], torch.zeros(L), 0.0
    with torch.no_grad():
        for i in range(0, len(passages), 32):
            enc = tok([p["text"] for p in passages[i:i + 32]], return_tensors="pt", padding=True).to("cuda")
            hs = hf(**enc, output_hidden_states=True).hidden_states
            m = enc.attention_mask.clone()
            m[:, 0] = 0  # drop BOS
            m = m.unsqueeze(-1).float()
            neu = neutral[i:i + 32].to("cuda").view(-1, 1, 1).float()
            n_tok += (m * neu).sum().item()
            h = torch.stack([x.float() for x in hs[1:]], 1)  # [B, L, T, d]; index l = output of block l
            reps.append(((h * m[:, None]).sum(2) / m.sum(1)[:, None]).cpu())
            sq += (h * (m * neu)[:, None]).pow(2).sum((0, 2, 3)).cpu()
    H = torch.cat(reps)  # [N, L, d]
    rms = (sq / n_tok).sqrt()

    def rows(names):
        return torch.tensor([s in names for s in states])

    dirs = {}
    for l in range(L):
        Hl = H[:, l]
        Xn = Hl[neutral]
        U = torch.linalg.svd(Xn - Xn.mean(0), full_matrices=False).Vh[:10]

        def contrast(a, b):
            d = Hl[a].mean(0) - Hl[b].mean(0)
            d = d - U.T @ (U @ d)
            return d / d.norm()

        d = {"valence": contrast(rows(POS), rows(NEG))}
        d.update({f"pc{i}": U[i] for i in range(10)})  # the neutral components that were projected out
        for s in POS + NEG:
            d[f"state:{s}"] = contrast(rows([s]), neutral)
        for grp in (POS, NEG):
            for i, a in enumerate(grp):
                for b in grp[i + 1:]:
                    d[f"within:{a}-{b}"] = contrast(rows([a]), rows([b]))
        dirs[l] = d
    return dirs, rms


@app.function(image=image, gpu="L40S", secrets=[HF_SECRET], volumes={"/cache": cache}, timeout=7200)
def gradient_map(passages: list, min_sessions: int = 100, stop_cos: float = 0.95) -> dict:
    import torch
    from huggingface_hub import hf_hub_download

    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    dev = "cuda"
    tok, hf = load_model()
    sessions = json.load(open(f"{VOL}/transcripts.json"))
    L, d = hf.config.num_hidden_layers, hf.config.hidden_size

    # Gradients, one session at a time, with early stopping. The rule looks only at how
    # stable the mean gradient is (odd against even sessions at layer 16), never at its size.
    G, m0, n_steered, stopped_at = [], [], [], None
    for i, sess in enumerate(sessions):
        enc = encode(tok, sess)
        x, att, mask, spans = batch_pairs(tok, [enc], dev)
        g = torch.zeros(L, d)
        lps = []
        for k in range(2):  # the S-label and U-label sequences, one backward pass each
            emb = hf.model.embed_tokens(x[k:k + 1]).detach().requires_grad_(True)
            out = hf.model(inputs_embeds=emb, attention_mask=att[k:k + 1], output_hidden_states=True)
            hs = out.hidden_states[1:]
            for h in hs:
                h.retain_grad()
            lp = label_logprob(hf, out.last_hidden_state, spans[k:k + 1])[0]
            lp.backward()
            sign = 1.0 if k == 0 else -1.0
            g += sign * torch.stack([(h.grad[0].float() * mask[k, :, None]).sum(0) for h in hs]).cpu()
            lps.append(lp.item())
        G.append(g)
        m0.append(lps[0] - lps[1])
        n_steered.append(int(mask[0].sum()))
        n = i + 1
        if n >= min_sessions and n % 50 == 0:
            S = torch.stack(G)[:, 16]
            c = torch.cosine_similarity(S[0::2].mean(0), S[1::2].mean(0), dim=0).item()
            print(f"after {n} sessions: split-half cosine of the mean gradient at layer 16 = {c:.3f}")
            if c >= stop_cos:
                stopped_at = n
                break
    G = torch.stack(G)  # [n, L, d]
    n = len(G)
    torch.set_grad_enabled(False)

    dirs, rms = passage_directions(tok, hf, passages)
    emo = torch.load(hf_hub_download(EMO_REPO, EMO_FILE), map_location="cpu", weights_only=True).float()
    emo_names = json.load(open(hf_hub_download(EMO_REPO, EMO_ORDER)))
    emo = emo - emo.mean(0, keepdim=True)
    emo = emo / emo.norm(dim=-1, keepdim=True)
    pain = {f"pain:{k}": v / v.norm() for k, v in pain_vectors().items()}

    def predict(u, l):
        """Predicted change in choice margin per unit dose for unit direction u at layer l:
        mean over sessions and a bootstrap 95% interval."""
        s = RHO * rms[l] * (G[:, l] @ u)
        boot = s[torch.randint(n, (2000, n), generator=gen)].mean(1)
        return [s.mean().item(), boot.quantile(0.025).item(), boot.quantile(0.975).item()]

    out = {"n_sessions": n, "stopped_early_at": stopped_at, "baseline_margin_mean": sum(m0) / n,
           "steered_tokens_per_session_mean": sum(n_steered) / n, "rho": RHO,
           "rms": rms.tolist(), "layers": {}, "split": {}}
    gbar = G.mean(0)
    for l in range(L - 1):
        gl = gbar[l]
        even, odd = G[0::2, l].mean(0), G[1::2, l].mean(0)
        rand_sd = RHO * rms[l].item() * gl.norm().item() / d ** 0.5  # sd of the prediction for a random direction
        row = {"cos_gradient_valence": torch.cosine_similarity(gl, dirs[l]["valence"], dim=0).item(),
               "split_half_cos_gradient": torch.cosine_similarity(even, odd, dim=0).item(),
               "random_direction_sd": rand_sd,
               "max_possible": RHO * rms[l].item() * gl.norm().item(),
               "pred": {k: predict(u, l) for k, u in dirs[l].items()}}
        e = RHO * rms[l] * (emo[:, l] @ gl)
        row["emotions_pred_vs_valence_cos_corr"] = torch.corrcoef(torch.stack([e, emo[:, l] @ dirs[l]["valence"]]))[0, 1].item()
        order = e.argsort()
        row["emotions_top"] = [[emo_names[i], round(e[i].item(), 3)] for i in order[-6:].tolist()[::-1]]
        row["emotions_bottom"] = [[emo_names[i], round(e[i].item(), 3)] for i in order[:6].tolist()]
        if l == 28:
            row["pred"].update({k: predict(u, l) for k, u in pain.items()})
        out["layers"][l] = row

    # Split the valence vector and the gradient itself over each lens's token directions.
    Wg = hf.lm_head.weight[:N_VOCAB].float() * hf.model.norm.weight.float()

    def unit(x):
        return x / x.norm(dim=-1, keepdim=True)

    saved = {"rms": rms, "dirs": {}}
    for name, f in LENSES.items():
        ck = torch.load(hf_hub_download(LENS_REPO, f), map_location="cpu", weights_only=True)
        for l in SPLIT_LAYERS:
            D = unit(Wg @ ck["J"][l].float().to(dev))
            v = dirs[l]["valence"]
            gl = unit(gbar[l])
            # 24 random readable directions: 16 random tokens with random non-negative weights
            rr = unit((torch.rand(24, 16, 1, generator=gen).to(dev) * D[torch.randint(N_VOCAB, (24, 16), generator=gen).to(dev)]).sum(1))
            iso = unit(torch.randn(100, d, generator=gen))
            X = torch.cat([torch.stack([v, -v, gl, -gl]), iso]).to(dev)
            nn, sg = sparse_codes(D, X, [16], signed=False), sparse_codes(D, X, [25], signed=True)
            share, ids, w = nn[16]
            recon = (w.unsqueeze(1) @ D[ids]).squeeze(1).cpu()  # readable parts of the unit targets
            res = {"null_isotropic_nn16": [share[4:].mean().item(), share[4:].quantile(0.95).item(), share[4:].max().item()]}
            for i, nm in enumerate(["valence_pos", "valence_neg", "gradient_pos", "gradient_neg"]):
                res[nm] = {"nn16": share[i].item(), "signed25": sg[25][0][i].item(),
                           "tokens_nn16": [tok.decode([ids[i, j].item()]) for j in w[i].argsort(descending=True).tolist() if w[i, j] > 0]}
            # Predictions for the arms of the protocol. "natural" keeps the part at its own
            # share of the vector; "matched" rescales it to the vector's full norm.
            Jp, Jn = recon[0], recon[1]  # readable part of +v, and of -v
            arms = {"A_full_pos": v, "A_full_neg": -v,
                    "B_readable_natural_pos": Jp, "B_readable_natural_neg": Jn,
                    "C_readable_matched_pos": unit(Jp), "C_readable_matched_neg": unit(Jn),
                    "D_remainder_natural_pos": v - Jp, "D_remainder_natural_neg": -v - Jn}
            res["pred"] = {k: predict(u, l) for k, u in arms.items()}
            hr = RHO * rms[l] * (G[:, l] @ rr.cpu().T).mean(0)
            res["pred_H_random_readable"] = {"mean": hr.mean().item(), "sd": hr.std().item(), "max_abs": hr.abs().max().item()}
            out["split"][f"{name}/L{l}"] = res
            if name == "jlens":
                saved["dirs"][l] = {**dirs[l], **arms, **{f"H{i}": rr[i].cpu() for i in range(24)},
                                    **(pain if l == 28 else {}),
                                    **{f"emo:{e}": emo[emo_names.index(e), l] for e in ("calm", "satisfied", "panicked", "exasperated")}}
            del D
    for l in range(L - 1):  # every layer gets the passage directions, for layer sweeps
        saved["dirs"].setdefault(l, dict(dirs[l]))
    torch.save(saved, f"{VOL}/stage2_dirs.pt")
    torch.save({"G": G, "m0": torch.tensor(m0)}, f"{VOL}/stage2_gradients.pt")
    cache.commit()
    return out


def read_passages():
    return [json.loads(line) for f in sorted((HERE.parent / "data/passages").glob("*.jsonl"))
            for line in f.read_text().splitlines()]


@app.local_entrypoint()
def gradmap():
    r = gradient_map.remote(read_passages())
    (HERE / "results" / "stage2_gradmap.json").write_text(json.dumps(r, indent=1))
    print(f"sessions {r['n_sessions']} (early stop at {r['stopped_early_at']}); baseline margin {r['baseline_margin_mean']:.3f}; "
          f"steered tokens per session {r['steered_tokens_per_session_mean']:.0f}")
    print("layer  cos(g,v)  split-half  pred slope valence [95% CI]      random sd   z     emo corr")
    for l, row in r["layers"].items():
        p = row["pred"]["valence"]
        print(f"{l:>5}  {row['cos_gradient_valence']:8.4f}  {row['split_half_cos_gradient']:10.3f}  "
              f"{p[0]:8.4f} [{p[1]:8.4f},{p[2]:8.4f}]  {row['random_direction_sd']:9.4f}  "
              f"{p[0] / row['random_direction_sd']:5.1f}  {row['emotions_pred_vs_valence_cos_corr']:6.3f}")
    for key, res in r["split"].items():
        print(key, {k: [round(x, 4) for x in v] for k, v in res["pred"].items()})
        print("   H random readable:", {k: round(v, 4) for k, v in res["pred_H_random_readable"].items()},
              " gradient readable share +/-:", round(res["gradient_pos"]["nn16"], 3), round(res["gradient_neg"]["nn16"], 3),
              "null", [round(x, 3) for x in res["null_isotropic_nn16"]])
        print("   +g tokens:", res["gradient_pos"]["tokens_nn16"][:10], "\n   -g tokens:", res["gradient_neg"]["tokens_nn16"][:10])


# ----------------------------------------------------------------------------------------
# Step 3: steering

GRID_DIRS = (["valence"] + [f"state:{s}" for s in POS + NEG]
             + ["within:contentment-serenity", "within:distress-weariness"]
             + [f"emo:{e}" for e in ("calm", "satisfied", "panicked", "exasperated")])
PLANS = {
    # (layer, direction, dose). "rand<i>" are orthogonal random directions.
    # The small doses (0.1) test whether the gradient map's first-order prediction holds.
    "pilot": (50, [(l, "valence", d) for l in (12, 16, 20, 24) for d in (-1, -0.1, 0.1, 1)]
              + [(l, f"rand{i}", d) for l in (12, 16, 20, 24) for i in range(8) for d in (-1, 1)]),
    "gate": (300, [(16, "valence", d) for d in (-1, -0.5, 0.5, 1)]
             + [(16, f"rand{i}", d) for i in range(24) for d in (-1, 1)]),
    "grid": (100, [(l, n, d) for l in (12, 16, 20, 24) for n in GRID_DIRS for d in (-1, 1)]
             + [(28, n, d) for n in ("pain:s1", "pain:s2") for d in (-1, 1)]),
}


@app.function(image=image, gpu="L40S", secrets=[HF_SECRET], volumes={"/cache": cache}, timeout=4 * 3600)
def run_steer(plan: str) -> dict:
    import torch

    torch.set_grad_enabled(False)
    gen = torch.Generator().manual_seed(SEED)
    dev = "cuda"
    tok, hf = load_model()
    n, cells = PLANS[plan]
    sessions = json.load(open(f"{VOL}/transcripts.json"))[:n]
    saved = torch.load(f"{VOL}/stage2_dirs.pt", weights_only=True)
    G = torch.load(f"{VOL}/stage2_gradients.pt", weights_only=True)["G"]
    rms, d_model = saved["rms"], hf.config.hidden_size
    rand = torch.linalg.qr(torch.randn(d_model, 24, generator=gen)).Q.T  # 24 orthonormal random directions

    def direction(l, name):
        return rand[int(name[4:])] if name.startswith("rand") else saved["dirs"][l][name]

    encoded = [encode(tok, s) for s in sessions]
    batches = [batch_pairs(tok, encoded[i:i + 16], dev) for i in range(0, n, 16)]
    steered_tokens = sum(int(e[1].sum()) for e in encoded) * 2  # two sequences per session

    state = {"layer": None, "delta": None, "mask": None}

    def make_hook(l):
        def hook(module, inputs, output):
            if state["layer"] != l:
                return None
            h = output[0] if isinstance(output, tuple) else output
            h = h + state["mask"].unsqueeze(-1).to(h.dtype) * state["delta"].to(h.dtype)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook

    for l in sorted({c[0] for c in cells}):
        hf.model.layers[l].register_forward_hook(make_hook(l))

    def run(l=None, delta=None):
        out = []
        for x, att, mask, spans in batches:
            state.update(layer=l, delta=delta, mask=mask)
            lp = label_logprob(hf, hf.model(input_ids=x, attention_mask=att).last_hidden_state, spans)
            out.append((lp[0::2] - lp[1::2]).cpu())
        state["layer"] = None
        return torch.cat(out)

    def ci(x):
        boot = x[torch.randint(len(x), (2000, len(x)), generator=gen)].mean(1)
        return [x.mean().item(), boot.quantile(0.025).item(), boot.quantile(0.975).item()]

    m0 = run()
    res, dm, negative_exposure = [], {}, 0
    for l, name, dose in cells:
        u = direction(l, name)
        u = u / u.norm() if name.startswith(("rand", "valence", "state", "within", "emo", "pain", "H", "C_")) else u
        delta = (dose * RHO * rms[l] * u).to(dev)
        d_m = run(l, delta) - m0
        dm[(l, name, dose)] = d_m
        pred = (dose * RHO * rms[l] * (G[:, l] @ u)).mean().item()
        toward_negative = (not name.startswith("rand")) and dose * torch.cosine_similarity(u, saved["dirs"][l]["valence"], dim=0).item() < -0.1
        negative_exposure += steered_tokens if toward_negative else 0
        res.append({"layer": l, "direction": name, "dose": dose, "delta_margin": ci(d_m), "predicted_first_order": pred,
                    "toward_negative_valence": toward_negative, "per_session": [round(v, 4) for v in d_m.tolist()]})
        print(l, name, dose, [round(v, 3) for v in res[-1]["delta_margin"]], "pred", round(pred, 3), flush=True)
    out = {"plan": plan, "n_sessions": n, "rho": RHO, "baseline_margin": ci(m0), "cells": res,
           "steered_token_passes_per_cell": steered_tokens, "negative_valence_token_passes": negative_exposure}
    if plan == "gate":
        doses = [-1, -0.5, 0.5, 1]
        slope_v = sum(d * dm[(16, "valence", d)] for d in doses) / sum(d * d for d in doses)
        slope_r = [((dm[(16, f"rand{i}", 1)] - dm[(16, f"rand{i}", -1)]) / 2).mean().item() for i in range(24)]
        s = ci(slope_v)
        out["gate"] = {"valence_slope": s, "random_slopes": slope_r, "max_abs_random_slope": max(map(abs, slope_r)),
                       "passed": bool(s[1] > 0 and s[0] > max(map(abs, slope_r)))}
    return out


@app.local_entrypoint()
def steer(plan: str = "pilot"):
    r = run_steer.remote(plan)
    (HERE / "results" / f"stage2_{plan}.json").write_text(json.dumps(r))
    print("baseline margin", [round(v, 3) for v in r["baseline_margin"]], "| negative-valence token passes", r["negative_valence_token_passes"])
    for c in r["cells"]:
        if not c["direction"].startswith("rand") or plan == "pilot":
            print(f"L{c['layer']:<3} {c['direction']:32} dose {c['dose']:+.1f}  delta m {c['delta_margin'][0]:+.3f} "
                  f"[{c['delta_margin'][1]:+.3f},{c['delta_margin'][2]:+.3f}]  predicted {c['predicted_first_order']:+.3f}")
    if "gate" in r:
        print("GATE:", json.dumps(r["gate"]))


# ----------------------------------------------------------------------------------------
# Diagnosis, layer by layer (no steering)

@app.function(image=image, gpu="L40S", secrets=[HF_SECRET], volumes={"/cache": cache}, timeout=3600)
def run_diagnose(example: int, n_agg: int = 100) -> dict:
    """For one session and for the first n_agg sessions, layer by layer and with no steering:
    1. when the choice appears: the lens readout of (logit of S's label token - logit of U's)
       at the position where the model names its choice;
    2. where the choice listens: how the gradient of that logit difference is spread over
       the conversation (the two zones' prompts and replies, and the choice prompt);
    3. the predicted effect of my valence vector if it were added at that layer."""
    import torch
    from huggingface_hub import hf_hub_download

    tok, hf = load_model()
    dev = "cuda"
    sessions = json.load(open(f"{VOL}/transcripts.json"))
    saved = torch.load(f"{VOL}/stage2_dirs.pt", weights_only=True)
    J = torch.load(hf_hub_download(LENS_REPO, LENSES["jlens"]), map_location="cpu", weights_only=True)["J"]
    L = hf.config.num_hidden_layers
    W, norm = hf.lm_head.weight, hf.model.norm
    groups = ["S prompt", "S reply", "U prompt", "U reply", "choice prompt"]

    def one(sess):
        ids, mask, lab = encode(tok, sess)
        c = 0  # tokens the two labels share at the start ("Zone")
        while lab["S"][c] == lab["U"][c]:
            c += 1
        ids = ids + lab["S"][:c]
        s_tok, u_tok = lab["S"][c], lab["U"][c]
        heads = [i for i, t in enumerate(ids) if t == HEADER]
        off = len(heads) - 26
        grp = torch.full((len(ids),), -1)
        for j, z in enumerate(sess["order"]):
            a, b, e = heads[off + 2 * j], heads[off + 2 * j + 1], heads[off + 2 * j + 2]
            k = 0 if z == sess["S"] else 2
            grp[a:b], grp[b:e] = k, k + 1
        grp[heads[off + 24]:] = 4
        # Positions of the zone's own name tokens inside the S turns' user prompts.
        tail = lab["S"][c:]
        is_label = torch.zeros(len(ids), dtype=torch.bool)
        for i in range(len(ids) - len(tail)):
            if grp[i] == 0 and ids[i:i + len(tail)] == tail:
                is_label[i:i + len(tail)] = True
        parts = {"S label tokens": is_label, "S prompt, other tokens": (grp == 0) & ~is_label, "S reply": grp == 1}
        x = torch.tensor([ids], device=dev)
        emb = hf.model.embed_tokens(x).detach().requires_grad_(True)
        out = hf.model(inputs_embeds=emb, output_hidden_states=True)
        hs = out.hidden_states[1:]
        for h in hs:
            h.retain_grad()
        w = (W[s_tok] - W[u_tok]).float()
        final = out.last_hidden_state[0, -1].float() @ w
        final.backward()
        with torch.no_grad():
            logit_lens = [(norm(h[0, -1]).float() @ w).item() for h in hs[:L - 1]]
            j_lens = [(norm((hs[l][0, -1].float() @ J[l].float().to(dev).T).to(W.dtype)).float() @ w).item() for l in range(L - 1)]
            share, total, pred, pred_parts = [], [], [], []
            for l in range(L - 1):
                g = hs[l].grad[0].float()
                e = g.pow(2).sum(-1).cpu()
                total.append(e.sum().sqrt().item())
                share.append([(e[grp == k].sum() / e.sum()).item() for k in range(5)])
                v = saved["dirs"][l]["valence"].to(dev)
                pred.append((RHO * saved["rms"][l] * (g[(grp.to(dev) >= 0) & (grp.to(dev) <= 1)].sum(0) @ v)).item())
                pred_parts.append([(RHO * saved["rms"][l] * (g[m.to(dev)].sum(0) @ v)).item() for m in parts.values()])
        return {"final": final.item(), "logit_lens": logit_lens, "j_lens": j_lens, "grad_share": share,
                "grad_norm": total, "pred_valence": pred, "pred_valence_parts": pred_parts,
                "n_tokens_parts": [int(m.sum()) for m in parts.values()], "labels": [tok.decode(lab["S"]), tok.decode(lab["U"])],
                "label_tokens": [[tok.decode([t]) for t in lab[k]] for k in ("S", "U")]}

    ex = one(sessions[example])
    ex["session"] = {k: sessions[example][k] for k in ("S", "U", "order", "choice")}
    ex["replies"] = sessions[example]["replies"]
    rows = [one(s) for s in sessions[:n_agg]]
    fin = torch.tensor([r["final"] for r in rows])
    agg = {"n": n_agg, "groups": groups, "final_abs_mean": fin.abs().mean().item(), "layers": []}
    for l in range(L - 1):
        ll = torch.tensor([r["logit_lens"][l] for r in rows])
        jl = torch.tensor([r["j_lens"][l] for r in rows])
        agg["layers"].append({
            "corr_logit_lens_final": torch.corrcoef(torch.stack([ll, fin]))[0, 1].item(),
            "corr_j_lens_final": torch.corrcoef(torch.stack([jl, fin]))[0, 1].item(),
            "grad_share": torch.tensor([r["grad_share"][l] for r in rows]).mean(0).tolist(),
            "pred_valence_mean": sum(r["pred_valence"][l] for r in rows) / n_agg,
            "pred_valence_parts_mean": torch.tensor([r["pred_valence_parts"][l] for r in rows]).mean(0).tolist()})
    agg["parts"] = ["S label tokens", "S prompt, other tokens", "S reply"]
    agg["n_tokens_parts_mean"] = torch.tensor([r["n_tokens_parts"] for r in rows]).float().mean(0).tolist()
    return {"example": ex, "aggregate": agg, "groups": groups}


@app.local_entrypoint()
def diagnose():
    import re

    pat = re.compile(DISCLAIMER, re.I)
    sessions = json.loads((HERE / "results" / "stage2_transcripts.json").read_text())
    example = next(i for i, s in enumerate(sessions) if not any(pat.search(r) for r in s["replies"]))
    r = run_diagnose.remote(example)
    r["example_index"] = example
    (HERE / "results" / "stage2_diagnose.json").write_text(json.dumps(r, indent=1))
    ex, agg = r["example"], r["aggregate"]
    print("example session", example, ex["labels"], "label tokens", ex["label_tokens"], "final logit difference (S - U):", round(ex["final"], 3))
    print("layer | example: logit lens, J lens, pred valence | 100 sessions: corr(logit lens, final), corr(J lens, final), pred valence | gradient share:", r["groups"])
    for l, a in enumerate(agg["layers"]):
        print(f"{l:>3} | {ex['logit_lens'][l]:+7.3f} {ex['j_lens'][l]:+7.3f} {ex['pred_valence'][l]:+7.3f} | "
              f"{a['corr_logit_lens_final']:+.2f} {a['corr_j_lens_final']:+.2f} {a['pred_valence_mean']:+7.3f} | "
              + " ".join(f"{x:.2f}" for x in a["grad_share"]) + " | parts " + " ".join(f"{x:+.3f}" for x in a["pred_valence_parts_mean"]))
    print("predicted valence effect by part:", agg["parts"], "mean tokens per session:", agg["n_tokens_parts_mean"])
