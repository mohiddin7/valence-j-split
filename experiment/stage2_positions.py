"""Stage 2, second pilot: which tokens does steering act through, and does choice order matter?

Each session is packed into one sequence: the 12 turns, then four endings that cannot see
each other (two choice orders x two labels). So every session is read under both orders.

  gradparts  no steering. The gradient of the choice margin at every position and layer,
             dotted with the valence vector (a position-by-layer map), and summed over three
             kinds of token in the conditioned zone's turns: the zone's name, the rest of the
             prompt, the reply. Also the readable share of the neutral principal components.
  steer      add a direction on one kind of token at one layer and read the choice margin.
             One container per layer.

Run after stage2_choice.py::transcripts (both sets):
      HF_SECRET_NAME=<your Modal secret holding HF_TOKEN> modal run experiment/stage2_positions.py::gradparts
      (then ::steer)
"""

import json
import pathlib

import modal
from stage1_readable_share import HF_SECRET, LENS_REPO, LENSES, N_VOCAB, cache, image, sparse_codes
from stage2_choice import (CHOICE, HEADER, RHO, SEED, TURN, VOL, batch_pairs, encode, label_logprob, load_model,
                           passage_directions, read_passages)

app = modal.App("neuronpedia-valence-stage2-positions")
image = image.add_local_python_source("stage1_readable_share", "stage2_choice")

N = 100
LAYERS = [9, 11, 12, 15, 16, 20, 24]
PARTS = ["name", "prompt_other", "reply"]  # in the conditioned zone's turns: parts 0, 1, 2
MASKS = {"reply": [2], "name": [0], "prompt_other": [1], "all": [0, 1, 2]}
SETS = {"base": "transcripts.json", "incharacter": "transcripts_incharacter.json"}
DOSES = [-1, -0.5, 0.5, 1]
HERE = pathlib.Path(__file__).parent


def pack(tok, sess):
    """One sequence for a session. Returns the token ids, position ids, the spans of the four
    endings, and for every position of the 12 turns its part (S turns: 0 name, 1 other prompt,
    2 reply; U turns: 3, 4, 5) and its bin for the map (zone, the zone's n-th turn, part).
    Endings, in order: S listed first with label S, with label U; U listed first, S, U."""
    import torch

    msgs = [{"role": "system", "content": sess["system"]}] if sess.get("system") else []
    for z, reply in zip(sess["order"], sess["replies"]):
        msgs += [{"role": "user", "content": TURN.format(z=z)}, {"role": "assistant", "content": reply}]
    lab = {k: tok(sess[k], add_special_tokens=False)["input_ids"] for k in ("S", "U")}
    c = 0  # tokens the two labels share at the start ("Zone")
    while lab["S"][c] == lab["U"][c]:
        c += 1
    full = [tok.apply_chat_template(msgs + [{"role": "user", "content": CHOICE.format(a=sess[a], b=sess[b])}],
                                    add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
            for a, b in (("S", "U"), ("U", "S"))]
    heads = [i for i, t in enumerate(full[0]) if t == HEADER]
    off = len(heads) - 26  # 24 turn messages, the choice message, the reply header; plus a system block
    assert off in (0, 1), len(heads)
    P = heads[off + 24]  # the 12 turns end here
    assert full[0][:P] == full[1][:P]
    ids, pos = list(full[0][:P]), list(range(P))
    part, bins = torch.full((P,), -1), torch.full((P,), -1)
    for j, z in enumerate(sess["order"]):
        a, b, e = heads[off + 2 * j], heads[off + 2 * j + 1], heads[off + 2 * j + 2]
        k = "S" if z == sess["S"] else "U"
        base = 0 if k == "S" else 3
        part[a:b], part[b:e] = base + 1, base + 2
        tail = lab[k][c:]
        hits = [i for i in range(a, b - len(tail) + 1) if ids[i:i + len(tail)] == tail]
        assert len(hits) == 1, (sess[k], hits)
        part[hits[0]:hits[0] + len(tail)] = base
        bins[a:e] = (base // 3) * 18 + (j // 2) * 3 + part[a:e] % 3
    ends, spans = [], []
    for o in range(2):
        for k in ("S", "U"):
            end = full[o][P:] + lab[k]
            ends.append((len(ids), len(ids) + len(end)))
            spans.append((len(ids) + len(end) - len(lab[k]), lab[k]))
            ids += end
            pos += list(range(P, P + len(end)))
    return {"ids": ids, "pos": pos, "part": part, "bins": bins, "ends": ends, "spans": spans, "P": P}


def batch(tok, packed, dev, dtype):
    """Padded tensors for a list of packed sessions. The attention mask is 4-D and additive:
    causal, and no ending may attend to another ending."""
    import torch

    B, T = len(packed), max(len(p["ids"]) for p in packed)
    x = torch.full((B, T), tok.pad_token_id)
    pos = torch.zeros(B, T, dtype=torch.long)
    part = torch.full((B, T), -1)
    allow = torch.tril(torch.ones(T, T, dtype=torch.bool)).repeat(B, 1, 1)
    for i, p in enumerate(packed):
        n = len(p["ids"])
        x[i, :n], pos[i, :n], part[i, :p["P"]] = torch.tensor(p["ids"]), torch.tensor(p["pos"]), p["part"]
        for a1, b1 in p["ends"]:
            for a2, b2 in p["ends"]:
                if a2 > a1:
                    allow[i, a2:b2, a1:b1] = False
    att = torch.zeros(B, 1, T, T, dtype=dtype).masked_fill(~allow[:, None], torch.finfo(dtype).min)
    return x.to(dev), pos.to(dev), att.to(dev), part.to(dev), [p["spans"] for p in packed]


def margins(hf, last, spans):
    """Choice margins log p(S) - log p(U), shape [B, 2]: S listed first, U listed first."""
    import torch

    out = []
    for i, sp in enumerate(spans):
        lp = label_logprob(hf, last[i:i + 1].float().expand(4, -1, -1), sp)
        out.append(torch.stack([lp[0] - lp[1], lp[2] - lp[3]]))
    return torch.stack(out)


# ----------------------------------------------------------------------------------------
# No steering: gradients by position

@app.function(image=image, gpu="L40S", secrets=[HF_SECRET], volumes={"/cache": cache}, timeout=7200)
def grad_parts(passages: list, tset: str, example: int) -> dict:
    import torch
    from huggingface_hub import hf_hub_download

    tok, hf = load_model()
    dev, dtype = "cuda", hf.dtype
    hf.lm_head.float()  # in bfloat16, logits near 20 are rounded to steps of 0.125
    L, d = hf.config.num_hidden_layers, hf.config.hidden_size
    sessions = json.load(open(f"{VOL}/{SETS[tset]}"))[:N]
    dirs, rms = passage_directions(tok, hf, passages)
    old = torch.load(f"{VOL}/stage2_dirs.pt", weights_only=True)["dirs"]
    same = min(torch.cosine_similarity(dirs[l]["valence"], old[l]["valence"], dim=0).item() for l in LAYERS)
    assert same > 0.999, same  # the valence vector is the one used in the first pilot and the gate
    V = torch.stack([dirs[l]["valence"] for l in range(L)]).to(dev)
    scale = (RHO * rms).to(dev)

    Gp = torch.zeros(len(sessions), 2, len(LAYERS), 3, d)
    heat = torch.zeros(len(sessions), 2, L, 36)
    m0, ex = [], None
    for i, sess in enumerate(sessions):
        p = pack(tok, sess)
        x, pos, att, part, spans = batch(tok, [p], dev, dtype)
        emb = hf.model.embed_tokens(x).detach().requires_grad_(True)
        out = hf.model(inputs_embeds=emb, attention_mask=att, position_ids=pos, output_hidden_states=True)
        hs = out.hidden_states[1:]
        m = margins(hf, out.last_hidden_state, spans)[0]
        m0.append(m.detach().cpu())
        P, bins = p["P"], p["bins"].to(dev)
        maps = []
        for o in range(2):
            g = torch.stack([x[0, :P].float() for x in torch.autograd.grad(m[o], hs, retain_graph=o == 0)])  # [L, P, d]
            hp = scale[:, None] * (g * V[:, None]).sum(-1)  # predicted change per unit dose, by position
            heat[i, o] = torch.zeros(L, 36, device=dev).index_add_(1, bins[bins >= 0], hp[:, bins >= 0]).cpu()
            for k in range(3):
                Gp[i, o, :, k] = g[LAYERS][:, part[0, :P] == k].sum(1).cpu()
            maps.append(hp.cpu())
        if i == example:
            ex = {"session": i, "S": sess["S"], "U": sess["U"], "tokens": [tok.decode([t]) for t in p["ids"][:P]],
                  "part": p["part"].tolist(), "map": [[[round(v, 4) for v in row] for row in mp.tolist()] for mp in maps]}
        del out, hs, m, g, hp
    torch.set_grad_enabled(False)
    m0 = torch.stack(m0)
    torch.save({"Gp": Gp, "m0": m0}, f"{VOL}/pilot2_grad_{tset}.pt")
    torch.save({"rms": rms, "dirs": {l: {k: v for k, v in dirs[l].items() if k == "valence" or k.startswith("pc")}
                                     for l in LAYERS}}, f"{VOL}/pilot2_dirs.pt")
    cache.commit()

    res = {"set": tset, "n": len(sessions), "baseline_margin_by_order": m0.mean(0).tolist(),
           "tokens_per_part_mean": [sum(int((pack(tok, s)["part"] == k).sum()) for s in sessions) / len(sessions) for k in range(3)],
           # [order][layer][zone S/U][the zone's n-th turn][name, other prompt, reply]
           "map_mean": heat.mean(0).view(2, L, 2, 6, 3).tolist(),
           "valence_pred_by_part": {l: [[(RHO * rms[l] * (Gp[:, o, li, k] @ dirs[l]["valence"])).mean().item() for k in range(3)]
                                        for o in range(2)] for li, l in enumerate(LAYERS)},
           "example": ex}
    if tset == "base":  # readable share of the neutral components, the baseline stage 1 lacked
        Wg = hf.lm_head.weight[:N_VOCAB].float() * hf.model.norm.weight.float()
        gen = torch.Generator().manual_seed(SEED)
        res["readable_share_nn16"] = {}
        for name, f in LENSES.items():
            ck = torch.load(hf_hub_download(LENS_REPO, f), map_location="cpu", weights_only=True)
            for l in (16, 24):
                D = Wg @ ck["J"][l].float().to(dev)
                D = D / D.norm(dim=-1, keepdim=True)
                pcs = torch.stack([dirs[l][f"pc{i}"] for i in range(10)])
                iso = torch.randn(100, d, generator=gen)
                X = torch.cat([dirs[l]["valence"][None], -dirs[l]["valence"][None], pcs, -pcs, iso / iso.norm(dim=-1, keepdim=True)]).to(dev)
                share = sparse_codes(D, X, [16], signed=False)[16][0].cpu()
                res["readable_share_nn16"][f"{name}/L{l}"] = {
                    "valence_pos": share[0].item(), "valence_neg": share[1].item(),
                    "neutral_pc_pos": share[2:12].tolist(), "neutral_pc_neg": share[12:22].tolist(),
                    "random_mean": share[22:].mean().item(), "random_max": share[22:].max().item()}
                del D
    return res


@app.local_entrypoint()
def gradparts():
    example = json.loads((HERE / "results" / "stage2_diagnose.json").read_text())["example_index"]
    passages = read_passages()
    for r in grad_parts.starmap([(passages, t, example if t == "base" else -1) for t in SETS]):
        (HERE / "results" / f"stage2_gradparts_{r['set']}.json").write_text(json.dumps(r))
        print(f"\n{r['set']}: {r['n']} sessions; baseline margin, S listed first {r['baseline_margin_by_order'][0]:+.3f}, "
              f"U listed first {r['baseline_margin_by_order'][1]:+.3f}; tokens per part {r['tokens_per_part_mean']}")
        print("predicted change in margin per unit dose of valence, by steered part (S first | U first):", PARTS)
        for l, row in r["valence_pred_by_part"].items():
            print(f"  L{l:<3}", " ".join(f"{v:+.3f}" for v in row[0]), "|", " ".join(f"{v:+.3f}" for v in row[1]))
        for key, s in r.get("readable_share_nn16", {}).items():
            pc = s["neutral_pc_pos"] + s["neutral_pc_neg"]
            print(f"  readable share {key}: +v {s['valence_pos']:.3f}  -v {s['valence_neg']:.3f}  neutral PCs mean {sum(pc) / 20:.3f} "
                  f"max {max(pc):.3f}  random mean {s['random_mean']:.3f} max {s['random_max']:.3f}")


# ----------------------------------------------------------------------------------------
# Steering

@app.function(image=image, gpu="L40S", secrets=[HF_SECRET], volumes={"/cache": cache}, timeout=3 * 3600)
def steer_layer(l: int, n: int = N) -> dict:
    import torch

    torch.set_grad_enabled(False)
    gen = torch.Generator().manual_seed(SEED)
    tok, hf = load_model()
    dev, dtype = "cuda", hf.dtype
    hf.lm_head.float()  # in bfloat16, logits near 20 are rounded to steps of 0.125
    saved = torch.load(f"{VOL}/pilot2_dirs.pt", weights_only=True)
    R = saved["rms"][l].item()
    rand = torch.linalg.qr(torch.randn(hf.config.hidden_size, 24, generator=gen)).Q.T  # the same draw as the first pilot
    U = {"valence": saved["dirs"][l]["valence"], **{f"pc{i}": saved["dirs"][l][f"pc{i}"] for i in range(3)},
         **{f"rand{i}": rand[i] for i in range(8)}}
    U = {k: u / u.norm() for k, u in U.items()}
    cells = [("valence", d) for d in DOSES] + [(k, d) for k in U if k != "valence" for d in (-1, 1)]
    layers = hf.model.layers

    def ci(x):
        boot = x[torch.randint(len(x), (2000, len(x)), generator=gen)].mean(1)
        return [x.mean().item(), boot.quantile(0.025).item(), boot.quantile(0.975).item()]

    def ci3(x):  # S listed first, U listed first, mean of the two orders
        return [ci(x[:, 0]), ci(x[:, 1]), ci(x.mean(1))]

    out = {"layer": l, "n_sessions": n, "rho": RHO, "rms": R, "sets": {}}
    for tset, masks in (("base", list(MASKS)), ("incharacter", ["reply"])):
        sessions = json.load(open(f"{VOL}/{SETS[tset]}"))[:n]
        Gp = torch.load(f"{VOL}/pilot2_grad_{tset}.pt", weights_only=True)["Gp"][:n, :, LAYERS.index(l)]  # [n, 2, 3, d]
        packed = [pack(tok, s) for s in sessions]
        batches = [batch(tok, packed[i:i + 16], dev, dtype) for i in range(0, n, 16)]

        # Clean pass through the whole model: baseline margins, and the residual after block l.
        grab = {}
        hook = layers[l].register_forward_hook(lambda mod, inp, o: grab.update(h=o[0] if isinstance(o, tuple) else o))
        m0, H = [], []
        for x, pos, att, part, spans in batches:
            m0.append(margins(hf, hf.model(input_ids=x, attention_mask=att, position_ids=pos, use_cache=False).last_hidden_state, spans).cpu())
            H.append(grab["h"])
        hook.remove()
        m0 = torch.cat(m0)

        # Check 1: the packed sequence gives the same margins as four ordinary sequences.
        ref = []
        for o, (a, b) in enumerate((("S", "U"), ("U", "S"))):
            enc = [encode(tok, {**s, "choice": CHOICE.format(a=s[a], b=s[b])}) for s in sessions[:4]]
            x, att, _, spans = batch_pairs(tok, enc, dev)
            lp = label_logprob(hf, hf.model(input_ids=x, attention_mask=att).last_hidden_state.float(), spans)
            ref.append((lp[0::2] - lp[1::2]).cpu())
        diff = torch.stack(ref, 1) - m0[:4]
        print("packed minus ordinary:", [round(v, 3) for v in diff.flatten().tolist()], flush=True)
        err_pack = diff.abs().max().item()

        # Everything below block l+1 is unchanged by steering after block l, so run only the blocks above.
        hf.model.layers = torch.nn.ModuleList(layers[l + 1:])

        def run(parts=None, delta=None):
            res = []
            for (x, pos, att, part, spans), h in zip(batches, H):
                if delta is not None:
                    h = h + torch.isin(part, parts).unsqueeze(-1).to(h.dtype) * delta.to(h.dtype)
                res.append(margins(hf, hf.model(inputs_embeds=h, attention_mask=att, position_ids=pos, use_cache=False).last_hidden_state, spans).cpu())
            return torch.cat(res)

        # Check 2: restarting from the stored residual reproduces the baseline.
        err_restart = (run() - m0).abs().max().item()
        print(f"L{l} {tset}: packed vs ordinary {err_pack:.4f}, restart vs full {err_restart:.4f}", flush=True)
        assert err_pack < 0.2 and err_restart < 0.05, (err_pack, err_restart)

        rows, summary, exposure = [], {}, 0
        for mk in masks:
            parts = torch.tensor(MASKS[mk], device=dev)
            n_tok = sum(int(torch.isin(p["part"], parts.cpu()).sum()) for p in packed)
            g = Gp[:, :, MASKS[mk]].sum(2)  # [n, 2, d]
            dm = {}
            for name, dose in cells:
                delta = dose * RHO * R * U[name]
                dm[(name, dose)] = x = run(parts, delta.to(dev)) - m0
                exposure += n_tok if (name == "valence" and dose < 0) else 0
                rows.append({"mask": mk, "direction": name, "dose": dose, "delta_margin": ci3(x),
                             "predicted_first_order": (g @ delta).mean(0).tolist(),
                             "per_session": [[round(v, 4) for v in r] for r in x.tolist()]})
            odd = {k: (dm[(k, 1)] - dm[(k, -1)]) / 2 for k in U}
            even = {k: (dm[(k, 1)] + dm[(k, -1)]) / 2 for k in U}
            slope = sum(d * dm[("valence", d)] for d in DOSES) / sum(d * d for d in DOSES)
            summary[mk] = {"steered_tokens_per_session": n_tok / n, "valence_slope": ci3(slope),
                           "odd": {k: ci3(v) for k, v in odd.items()}, "even": {k: ci3(v) for k, v in even.items()}}
            s = summary[mk]
            print(f"L{l} {tset} {mk}: valence slope (mean of orders) {s['valence_slope'][2][0]:+.3f}, "
                  f"even {s['even']['valence'][2][0]:+.3f}", flush=True)
        hf.model.layers = layers
        out["sets"][tset] = {"baseline_margin": ci3(m0), "check_packed_vs_ordinary": err_pack, "check_restart": err_restart,
                             "summary": summary, "cells": rows, "negative_valence_token_passes": exposure}
    return out


@app.local_entrypoint()
def steer(layers: str = "", n: int = N):
    ls = [int(v) for v in layers.split(",")] if layers else LAYERS
    results = list(steer_layer.map(ls, kwargs={"n": n}))
    name = "stage2_positions.json" if (ls == LAYERS and n == N) else "stage2_positions_smoke.json"
    (HERE / "results" / name).write_text(json.dumps(results))

    def rng(vals):
        return f"{min(vals):+.2f}..{max(vals):+.2f}"

    print("slope and odd/even parts are changes in choice margin per unit dose; S1 = S listed first, U1 = U listed first")
    print("layer set         mask          tok | valence slope S1, U1, mean [95% CI of mean]  | random odd range | "
          "valence even | random even range | neutral PC even range")
    for r in results:
        for tset, s in r["sets"].items():
            for mk, m in s["summary"].items():
                v = m["valence_slope"]
                ro = [m["odd"][f"rand{i}"][2][0] for i in range(8)]
                re_ = [m["even"][f"rand{i}"][2][0] for i in range(8)]
                pe = [m["even"][f"pc{i}"][2][0] for i in range(3)]
                print(f"L{r['layer']:<4} {tset:11} {mk:12} {m['steered_tokens_per_session']:4.0f} | {v[0][0]:+.3f} {v[1][0]:+.3f} "
                      f"{v[2][0]:+.3f} [{v[2][1]:+.3f},{v[2][2]:+.3f}] | {rng(ro)} | {m['even']['valence'][2][0]:+.3f} | {rng(re_)} | {rng(pe)}")
    print("negative-valence token passes:", sum(s["negative_valence_token_passes"] for r in results for s in r["sets"].values()))


# ----------------------------------------------------------------------------------------
# Check, no steering: with the whole model in float32, do the packed sequence and four
# ordinary sequences give the same margins? If so, their gap in bfloat16 is rounding.

@app.function(image=image, gpu="L40S", secrets=[HF_SECRET], volumes={"/cache": cache}, timeout=1800)
def check_packing(k: int = 4) -> dict:
    import torch

    torch.set_grad_enabled(False)
    tok, hf = load_model()
    hf.float()
    out = {}
    for tset, f in SETS.items():
        sessions = json.load(open(f"{VOL}/{f}"))[:k]
        x, pos, att, _, spans = batch(tok, [pack(tok, s) for s in sessions], "cuda", hf.dtype)
        m = margins(hf, hf.model(input_ids=x, attention_mask=att, position_ids=pos, use_cache=False).last_hidden_state, spans).cpu()
        ref = []
        for a, b in (("S", "U"), ("U", "S")):
            enc = [encode(tok, {**s, "choice": CHOICE.format(a=s[a], b=s[b])}) for s in sessions]
            x, att, _, spans = batch_pairs(tok, enc, "cuda")
            lp = label_logprob(hf, hf.model(input_ids=x, attention_mask=att).last_hidden_state, spans)
            ref.append((lp[0::2] - lp[1::2]).cpu())
        out[tset] = (m - torch.stack(ref, 1)).flatten().tolist()
    return out


@app.local_entrypoint()
def check():
    for tset, diff in check_packing.remote().items():
        print(tset, "packed minus ordinary, float32:", [round(v, 4) for v in diff])
