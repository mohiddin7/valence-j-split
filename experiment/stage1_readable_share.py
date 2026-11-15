"""Stage 1: how much of a valence direction can the Jacobian lens read as words?

No steering happens here. The script builds a valence direction from my passage corpus
(following Berg and Kaiser, arXiv 2609.35591, section 3.1), splits it into a sparse code
over lens token directions plus a remainder, and compares the readable share with two
null baselines. It repeats the split for three lenses, five layers and two estimators,
and for public emotion and pain vectors.

Run:  HF_SECRET_NAME=<your Modal secret holding HF_TOKEN> modal run experiment/stage1_readable_share.py::main
      (add ::pain_main for the Pain Axis vector)
"""

import json
import os
import pathlib

import modal

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_DIR = "llama3.1-8b-it"
LENSES = {
    "jlens": f"{LENS_DIR}/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt",
    "jpp": f"{LENS_DIR}/jpp/Salesforce-wikitext/Llama-3.1-8B-Instruct_jpp_lens.pt",
    "rlens": f"{LENS_DIR}/rlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_rlens.pt",
}
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
EMO_REPO = "davidafrica/functional-wellbeing"
EMO_FILE = "concept_vectors/emotions_llama-3.1-8b/emotion_vectors_raw.pt"
EMO_ORDER = "concept_vectors/emotions_llama-3.1-8b/emotions_order.json"
PAIN_URL = ("https://raw.githubusercontent.com/camberg23/Pain-axis/7c256502ed3d98e4e6379290fe7db2f93cb8d025/results/"
            "3.2_pain_vectors/pain_vectors/Llama_3.1_8B_instruct/pain_vectors.pt")

# Layers are zero-based block outputs, the lens library's convention. The paper's
# "layer 16 of 32" could be block 15 or block 16, so both are measured.
LAYERS = [15, 16, 20, 24, 28]
K_NN = [16, 25]      # non-negative sparse code sizes
K_SIGNED = [25, 100]  # signed sparse code sizes
N_NULL = 200
SEED = 0

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=5.5",
        "accelerate",
        "huggingface_hub",
        f"jlens @ git+https://github.com/anthropics/jacobian-lens@{JLENS_COMMIT}",
    )
    .env({"HF_HOME": "/cache/hf"})
    .pip_install("scipy")
)
app = modal.App("neuronpedia-valence-stage1")
cache = modal.Volume.from_name("neuronpedia-hf-cache", create_if_missing=True)
# A Modal secret that provides HF_TOKEN for the gated Llama weights.
HF_SECRET = modal.Secret.from_name(os.environ.get("HF_SECRET_NAME", "huggingface-secret"))


def _describe(obj, depth=0, limit=6):
    """Short structural description of a loaded checkpoint."""
    import torch

    pad = "  " * depth
    if isinstance(obj, torch.Tensor):
        return f"{pad}Tensor{tuple(obj.shape)} {obj.dtype}"
    if isinstance(obj, dict):
        lines = [f"{pad}dict[{len(obj)}]"]
        for k in list(obj)[:limit]:
            lines.append(f"{pad} {k!r}:")
            lines.append(_describe(obj[k], depth + 2, limit))
        return "\n".join(lines)
    if isinstance(obj, (list, tuple)):
        head = _describe(obj[0], depth + 1, limit) if obj else ""
        return f"{pad}{type(obj).__name__}[{len(obj)}]\n{head}"
    return f"{pad}{type(obj).__name__}: {str(obj)[:80]}"


@app.function(image=image, volumes={"/cache": cache}, timeout=900)
def inspect_inputs() -> str:
    """Print the structure of the third-party vector files. No model, no token.
    These files are untrusted, so they are only ever loaded with weights_only=True."""
    import urllib.request

    import torch
    from huggingface_hub import hf_hub_download

    out = []
    emo = torch.load(hf_hub_download(EMO_REPO, EMO_FILE), map_location="cpu", weights_only=True)
    out.append("EMOTIONS\n" + _describe(emo))
    order = json.load(open(hf_hub_download(EMO_REPO, EMO_ORDER)))
    out.append(f"ORDER {type(order).__name__} len={len(order)} head={str(order)[:300]}")
    urllib.request.urlretrieve(PAIN_URL, "/tmp/pain.pt")
    try:
        pain = torch.load("/tmp/pain.pt", map_location="cpu", weights_only=True)
        out.append("PAIN\n" + _describe(pain))
    except Exception as e:  # numpy payloads are refused by weights_only
        out.append(f"PAIN refused by weights_only: {str(e)[:300]}")
    for name, f in LENSES.items():
        ck = torch.load(hf_hub_download(LENS_REPO, f), map_location="cpu", weights_only=True)
        keys = list(ck["J"])
        out.append(f"LENS {name}: keys {sorted(ck)} layers {keys[0]!r}..{keys[-1]!r} "
                   f"({type(keys[0]).__name__}) provenance={str(ck.get('provenance'))[:400]}")
    cache.commit()
    return "\n\n".join(out)


@app.local_entrypoint()
def inspect():
    print(inspect_inputs.remote())


POS = ["contentment", "flow_engagement", "relief", "serenity"]
NEG = ["distress", "frustration", "weariness", "dread"]
N_VOCAB = 128000  # Llama 3.1 ids from 128000 up are special or reserved tokens


def sparse_codes(D, X, ks, signed):
    """Greedy sparse codes of each row of X over the unit-norm rows of D.

    Non-negative: pick the atom most positively aligned with the residual, then refit all
    chosen atoms by non-negative least squares. Signed: pick by absolute alignment, refit
    by least squares. Returns {k: (share of squared norm captured, atom ids, weights)}.
    """
    import numpy as np
    import torch
    from scipy.optimize import nnls

    B = X.shape[0]
    R = X.clone()
    S = torch.empty(B, 0, dtype=torch.long, device=X.device)
    Xn = X.double().cpu().numpy()
    snaps = {}
    for step in range(1, max(ks) + 1):
        C = R @ D.T
        score = C.abs() if signed else C
        if S.shape[1]:
            score.scatter_(1, S, float("-inf"))
        S = torch.cat([S, score.argmax(1, keepdim=True)], 1)
        A = D[S]  # [B, step, d]
        if signed:
            G = (A @ A.transpose(1, 2)).double()
            G += 1e-8 * torch.eye(step, dtype=G.dtype, device=G.device)
            Wt = torch.linalg.solve(G, (A @ X.unsqueeze(-1)).double()).squeeze(-1).float()
        else:
            An = A.double().cpu().numpy()
            Wt = torch.tensor(np.stack([nnls(An[b].T, Xn[b])[0] for b in range(B)]),
                              dtype=torch.float32, device=X.device)
        R = X - (Wt.unsqueeze(1) @ A).squeeze(1)
        if step in ks:
            snaps[step] = (1 - R.pow(2).sum(1) / X.pow(2).sum(1), S.clone(), Wt.clone())
    return snaps


@app.function(
    image=image,
    gpu="L40S",
    secrets=[HF_SECRET],
    volumes={"/cache": cache},
    timeout=5400,
)
def stage1(passages: list[dict], quick: bool = False) -> dict:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    dev = "cuda"
    layers = [16] if quick else LAYERS
    lenses = {"jlens": LENSES["jlens"]} if quick else LENSES
    n_null = 20 if quick else N_NULL

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token, tok.padding_side = tok.eos_token, "right"
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev).eval()
    assert tok.bos_token_id == N_VOCAB and hf.lm_head.weight.shape[0] == 128256

    # 1. One vector per passage and layer: the mean residual over its tokens, without BOS
    #    (the BOS residual is huge and nearly constant, so it would swamp the mean).
    states = [p["state"] for p in passages]
    neutral = torch.tensor([s == "neutral" for s in states])
    reps = {l: [] for l in layers}
    sq = {l: 0.0 for l in layers}
    n_neutral_tok = 0.0
    for i in range(0, len(passages), 32):
        enc = tok([p["text"] for p in passages[i:i + 32]], return_tensors="pt", padding=True).to(dev)
        assert (enc.input_ids[:, 0] == tok.bos_token_id).all()
        with torch.no_grad():
            hs = hf(**enc, output_hidden_states=True).hidden_states
        m = enc.attention_mask.clone()
        m[:, 0] = 0
        m = m.unsqueeze(-1).float()
        neu = neutral[i:i + 32].to(dev).view(-1, 1, 1).float()
        n_neutral_tok += (m * neu).sum().item()
        for l in layers:
            h = hs[l + 1].float()  # hidden_states[l + 1] is the output of block l
            reps[l].append(((h * m).sum(1) / m.sum(1)).cpu())
            sq[l] += (h * m * neu).pow(2).sum().item()
    H = {l: torch.cat(reps[l]) for l in layers}
    rms = {l: (sq[l] / n_neutral_tok) ** 0.5 for l in layers}

    def rows(names):
        return torch.tensor([s in names for s in states])

    def neutral_pcs(Hl):
        Xn = Hl[neutral]
        return torch.linalg.svd(Xn - Xn.mean(0), full_matrices=False).Vh[:10]

    def contrast(Hl, a, b, U):
        d = Hl[a].mean(0) - Hl[b].mean(0)
        return d - U.T @ (U @ d)

    def unit(x):
        return x / x.norm(dim=-1, keepdim=True)

    def auroc(p, n):
        return (p.view(-1, 1) > n.view(1, -1)).float().mean().item()

    emo = torch.load(hf_hub_download(EMO_REPO, EMO_FILE), map_location="cpu", weights_only=True).float()
    emo_names = json.load(open(hf_hub_download(EMO_REPO, EMO_ORDER)))
    emo = emo - emo.mean(0, keepdim=True)  # centre across the 171 emotions at every layer

    out = {"model": MODEL, "layers": layers, "n_passages": len(passages), "quick": quick,
           "pain_vector": "measured separately by pain_share(); see stage1_pain.json",
           "emotion_layer_assumption": "index l of the emotion file is taken as the output of block l",
           "vector": {}, "split": {}}
    V, targets = {}, {}
    for l in layers:
        Hl, U = H[l], neutral_pcs(H[l])
        v = contrast(Hl, rows(POS), rows(NEG), U)
        V[l] = v
        # Reliability of the rebuilt vector: split halves, and held-out states.
        halves = []
        for _ in range(20):
            pick = torch.rand(len(states), generator=gen) < 0.5
            a = contrast(Hl, rows(POS) & pick, rows(NEG) & pick, U)
            b = contrast(Hl, rows(POS) & ~pick, rows(NEG) & ~pick, U)
            halves.append(torch.cosine_similarity(a, b, dim=0).item())
        held = []
        for p_out, n_out in zip(POS, NEG):
            w = contrast(Hl, rows([s for s in POS if s != p_out]), rows([s for s in NEG if s != n_out]), U)
            held.append(auroc(Hl[rows([p_out])] @ w, Hl[rows([n_out])] @ w))
        cos_emo = unit(emo[:, l]) @ unit(v)
        order = cos_emo.argsort()
        out["vector"][l] = {
            "norm": v.norm().item(), "rms_residual_norm_neutral": rms[l],
            "norm_over_rms": v.norm().item() / rms[l],
            "split_half_cosine_mean": sum(halves) / len(halves),
            "held_out_state_auroc": held,
            "emotions_most_aligned": [emo_names[i] for i in order[-10:].tolist()[::-1]],
            "emotions_most_opposed": [emo_names[i] for i in order[:10].tolist()],
        }
        # Targets to split, all unit norm.
        splits = []
        idx = neutral.nonzero().squeeze(1)
        for _ in range(n_null):
            perm = idx[torch.randperm(len(idx), generator=gen)]
            a, b = torch.zeros_like(neutral), torch.zeros_like(neutral)
            a[perm[:len(idx) // 2]], b[perm[len(idx) // 2:]] = True, True
            splits.append(contrast(Hl, a, b, U))
        within = [contrast(Hl, rows([a]), rows([b]), U)
                  for grp in (POS, NEG) for i, a in enumerate(grp) for b in grp[i + 1:]]
        groups = [("valence_pos", v[None]), ("valence_neg", -v[None]),
                  ("null_isotropic", torch.randn(n_null, v.shape[0], generator=gen)),
                  ("null_neutral_split", torch.stack(splits)),
                  ("within_valence_contrasts", torch.stack(within)),
                  ("emotions", emo[:, l])]
        targets[l] = ([(n, len(x)) for n, x in groups], unit(torch.cat([x for _, x in groups])).to(dev))
    os.makedirs("/cache/valence", exist_ok=True)
    torch.save({"vectors": V, "rms": rms, "layers": layers}, "/cache/valence/v_rebuilt_quick.pt" if quick
               else "/cache/valence/v_rebuilt.pt")
    cache.commit()

    # 2. Split every target over each lens's token directions: row t is J^T (g * u_t).
    Wg = hf.lm_head.weight[:N_VOCAB].float() * hf.model.norm.weight.float()

    def stat(x):
        return {"mean": x.mean().item(), "median": x.median().item(),
                "p95": x.quantile(0.95).item(), "max": x.max().item()}

    for name, f in lenses.items():
        ck = torch.load(hf_hub_download(LENS_REPO, f), map_location="cpu", weights_only=True)
        for l in layers:
            D = unit(Wg @ ck["J"][l].float().to(dev))
            sizes, X = targets[l]
            # Self-check: a planted non-negative mix of 5 atoms must be recovered almost fully.
            planted = unit((torch.rand(5, 1, device=dev) + 0.5).T @ D[torch.randint(N_VOCAB, (5,), device=dev)])
            Xall = torch.cat([X, planted])
            nn = sparse_codes(D, Xall, K_NN, signed=False)
            sg = sparse_codes(D, Xall, K_SIGNED, signed=True)
            assert nn[K_NN[0]][0][-1] > 0.98, nn[K_NN[0]][0][-1]
            share = {f"nn{k}": nn[k][0].cpu() for k in K_NN} | {f"signed{k}": sg[k][0].cpu() for k in K_SIGNED}
            res, start = {}, 0
            for gname, n in sizes:
                sl = slice(start, start + n)
                if n == 1:
                    res[gname] = {e: s[start].item() for e, s in share.items()}
                    _, ids, w = nn[K_NN[0]]
                    top = w[start].argsort(descending=True)
                    res[gname]["tokens_nn16"] = [[tok.decode([ids[start, j].item()]), round(w[start, j].item(), 3)]
                                                 for j in top.tolist() if w[start, j] > 0]
                    res[gname]["tokens_most_aligned"] = [tok.decode([t]) for t in (D @ X[start]).topk(15).indices.tolist()]
                else:
                    res[gname] = {e: stat(s[sl]) for e, s in share.items()}
                start += n
            for null in ("null_isotropic", "null_neutral_split"):
                off = sum(n for g, n in sizes[:[g for g, _ in sizes].index(null)])
                cnt = dict(sizes)[null]
                for side in ("valence_pos", "valence_neg"):
                    for e, s in share.items():
                        ge = (s[off:off + cnt] >= res[side][e]).sum().item()
                        res[side][f"p_{e}_vs_{null}"] = (1 + ge) / (1 + cnt)
            out["split"][f"{name}/L{l}"] = res
            del D
    return out


@app.local_entrypoint()
def main(quick: bool = False):
    here = pathlib.Path(__file__).parent
    passages = [json.loads(line) for f in sorted((here.parent / "data/passages").glob("*.jsonl"))
                for line in f.read_text().splitlines()]
    assert len(passages) == 544, len(passages)
    result = stage1.remote(passages, quick)
    path = here / "results" / ("stage1_quick.json" if quick else "stage1.json")
    path.write_text(json.dumps(result, indent=1))
    for l, v in result["vector"].items():
        print(f"L{l}: |v|/rms {v['norm_over_rms']:.3f}  split-half cos {v['split_half_cosine_mean']:.3f}  "
              f"held-out AUROC {[round(a, 3) for a in v['held_out_state_auroc']]}")
        print("   aligned:", v["emotions_most_aligned"], "\n   opposed:", v["emotions_most_opposed"])
    print(f"{'lens/layer':12} {'est':9} {'+v':>6} {'-v':>6} {'iso mean/p95':>14} {'split mean/p95':>15} {'within':>7} {'emotions':>8}")
    for key, r in result["split"].items():
        for e in [f"nn{k}" for k in K_NN] + [f"signed{k}" for k in K_SIGNED]:
            print(f"{key:12} {e:9} {r['valence_pos'][e]:6.3f} {r['valence_neg'][e]:6.3f} "
                  f"{r['null_isotropic'][e]['mean']:6.3f}/{r['null_isotropic'][e]['p95']:.3f}  "
                  f"{r['null_neutral_split'][e]['mean']:6.3f}/{r['null_neutral_split'][e]['p95']:.3f}  "
                  f"{r['within_valence_contrasts'][e]['mean']:7.3f} {r['emotions'][e]['mean']:8.3f}")
        print("   +v tokens:", r["valence_pos"]["tokens_nn16"][:10])
        print("   -v tokens:", r["valence_neg"]["tokens_nn16"][:10])


PAIN_LAYER = 28  # recorded in the file, with final-token extraction


def pain_vectors():
    """The two Pain Axis vectors (Tagliabue, Dung and Berg, arXiv 2609.16247; MIT).

    The checkpoint's pickle holds numpy objects that torch's safe loader refuses, and I do
    not unpickle files I did not write. So this reads the two raw float32 storages directly.
    I found which storage is which by listing the pickle's opcodes without running them.
    """
    import io
    import urllib.request
    import zipfile

    import numpy as np
    import torch

    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(PAIN_URL).read()))
    vecs = {n: torch.tensor(np.frombuffer(z.read(f"pain_vectors/data/{i}"), dtype="<f4").copy())
            for i, n in enumerate(["s2", "s1"])}
    assert all(v.shape == (4096,) and v.isfinite().all() for v in vecs.values())
    return vecs


@app.function(image=image, gpu="L40S", secrets=[HF_SECRET], volumes={"/cache": cache}, timeout=3600)
def pain_share() -> dict:
    """Readable share of the Pain Axis vectors at their own layer, with the stage 1 method."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    gen = torch.Generator().manual_seed(SEED)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev).eval()
    Wg = hf.lm_head.weight[:N_VOCAB].float() * hf.model.norm.weight.float()

    def unit(x):
        return x / x.norm(dim=-1, keepdim=True)

    pv = pain_vectors()
    names = ["s1_pos", "s1_neg", "s2_pos", "s2_neg"]
    X = unit(torch.cat([torch.stack([pv["s1"], -pv["s1"], pv["s2"], -pv["s2"]]),
                        torch.randn(N_NULL, 4096, generator=gen)])).to(dev)
    out = {"layer": PAIN_LAYER, "norms": {k: v.norm().item() for k, v in pv.items()},
           "cos_s1_s2": torch.cosine_similarity(pv["s1"], pv["s2"], dim=0).item(), "split": {}}
    try:  # my rebuilt valence vector, kept on a private volume; positive pole = positive valence
        v = torch.load("/cache/valence/v_rebuilt.pt", weights_only=True)["vectors"][PAIN_LAYER]
        out["cos_with_my_valence_vector"] = {k: torch.cosine_similarity(p, v, dim=0).item() for k, p in pv.items()}
    except FileNotFoundError:
        pass
    for name, f in LENSES.items():
        ck = torch.load(hf_hub_download(LENS_REPO, f), map_location="cpu", weights_only=True)
        D = unit(Wg @ ck["J"][PAIN_LAYER].float().to(dev))
        nn = sparse_codes(D, X, K_NN, signed=False)
        sg = sparse_codes(D, X, K_SIGNED, signed=True)
        share = {f"nn{k}": nn[k][0].cpu() for k in K_NN} | {f"signed{k}": sg[k][0].cpu() for k in K_SIGNED}
        res = {"null_isotropic": {e: {"mean": s[4:].mean().item(), "p95": s[4:].quantile(0.95).item(),
                                      "max": s[4:].max().item()} for e, s in share.items()}}
        _, ids, w = nn[K_NN[0]]
        for i, n in enumerate(names):
            res[n] = {e: s[i].item() for e, s in share.items()}
            res[n]["tokens_nn16"] = [[tok.decode([ids[i, j].item()]), round(w[i, j].item(), 3)]
                                     for j in w[i].argsort(descending=True).tolist() if w[i, j] > 0]
        out["split"][f"{name}/L{PAIN_LAYER}"] = res
        del D
    return out


@app.local_entrypoint()
def pain_main():
    result = pain_share.remote()
    (pathlib.Path(__file__).parent / "results" / "stage1_pain.json").write_text(json.dumps(result, indent=1))
    print("cos s1,s2:", round(result["cos_s1_s2"], 3), " cos with my valence vector:", result.get("cos_with_my_valence_vector"))
    for key, r in result["split"].items():
        for n in ("s1_pos", "s1_neg", "s2_pos", "s2_neg"):
            print(f"{key:10} {n:7} nn16 {r[n]['nn16']:.3f} nn25 {r[n]['nn25']:.3f} signed100 {r[n]['signed100']:.3f}   "
                  f"null nn16 {r['null_isotropic']['nn16']['mean']:.3f}/{r['null_isotropic']['nn16']['p95']:.3f}")
            print("     ", [t for t, _ in r[n]["tokens_nn16"]][:12])
