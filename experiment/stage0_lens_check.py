"""Stage 0: load Llama-3.1-8B-Instruct and its Jacobian lens on Modal, check that the
lens reads sensibly, and measure forward-pass speed. No steering happens here.

Run:  HF_SECRET_NAME=<your Modal secret holding HF_TOKEN> modal run experiment/stage0_lens_check.py
"""

import json
import os
import pathlib

import modal

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"  # anthropics/jacobian-lens

# A prompt I ran on Neuronpedia's lens page, and the readouts the site listed for it
# across layers 9-31.
WALK_PROMPT = "Your last five answers were all rejected as wrong. Try again: what is 17 + 26?"
WALK_READOUTS = {"correct", "answer", "assistant", "calculator", "incorrect", "answers",
                 "calculation", "hopefully", "maybe", "wrong", "accurate"}

PLAIN_TEXT = (
    "The Amazon River in South America is the largest river by discharge volume of water "
    "in the world, and the disputed longest river system in the world in comparison to the "
    "Nile. The headwaters of the Apurimac River on Nevado Mismi had been considered for "
    "nearly a century as the most distant source of the Amazon basin, until a 2014 study "
    "found it to be the headwaters of the Mantaro River in Peru."
)

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
)
app = modal.App("neuronpedia-valence-stage0")
cache = modal.Volume.from_name("neuronpedia-hf-cache", create_if_missing=True)
# A Modal secret that provides HF_TOKEN for the gated Llama weights.
HF_SECRET = modal.Secret.from_name(os.environ.get("HF_SECRET_NAME", "huggingface-secret"))


@app.function(
    image=image,
    gpu="L40S",
    secrets=[HF_SECRET],
    volumes={"/cache": cache},
    timeout=1800,
)
def stage0() -> dict:
    import time
    from collections import Counter

    import torch
    from jlens.hf import from_hf
    from jlens.lens import JacobianLens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out: dict = {"model": MODEL, "lens_file": LENS_FILE, "jlens_commit": JLENS_COMMIT}
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model = from_hf(hf, tok)
    lens = JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    cache.commit()
    out["lens"] = {"n_prompts": lens.n_prompts, "d_model": lens.d_model,
                   "source_layers": [lens.source_layers[0], lens.source_layers[-1]],
                   "n_layers_model": model.n_layers}

    # 1. Depth check: the Jacobian lens should agree with the model's own next-token
    #    choice more often than the plain logit lens at mid depth, and more with depth.
    layers = [8, 16, 24, lens.source_layers[-1]]
    j_logits, m_logits, _ = lens.apply(model, PLAIN_TEXT, layers=layers)
    l_logits, _, _ = lens.apply(model, PLAIN_TEXT, layers=layers, use_jacobian=False)
    top = m_logits.argmax(-1)
    agree = {l: {"jacobian": round((j_logits[l].argmax(-1) == top).float().mean().item(), 3),
                 "logit": round((l_logits[l].argmax(-1) == top).float().mean().item(), 3)}
             for l in layers}
    out["top1_agreement_with_model"] = agree
    assert agree[layers[-1]]["jacobian"] > agree[8]["jacobian"], agree
    assert agree[16]["jacobian"] >= agree[16]["logit"], agree

    # 2. Read-direction identity (Day, section 3.3): d_t = J^T (g * u_t) must give the
    #    same token ranking as the library's own readout. Stage 1 builds its dictionary
    #    from these directions, so this must hold before anything else.
    g = hf.model.norm.weight.float()
    W = hf.lm_head.weight.float()
    J16 = lens.jacobians[16].to(W.device)
    D16 = (W * g) @ J16  # [vocab, d_model]; row t is the read direction of token t
    h = torch.randn(lens.d_model, device=W.device)
    mine = D16 @ h
    theirs = model.unembed(lens.transport(h, 16)).float()
    r = torch.corrcoef(torch.stack([mine, theirs]))[0, 1].item()
    out["read_direction_identity_r"] = r
    out["dictionary"] = {"shape": list(D16.shape), "gb_fp32": round(D16.numel() * 4 / 1e9, 2)}
    assert r > 0.9999, r

    # 3. Walkthrough prompt: which tokens does the lens surface across layers 9-31?
    chat = tok.apply_chat_template([{"role": "user", "content": WALK_PROMPT}],
                                   tokenize=False, add_generation_prompt=True)
    chat = chat.removeprefix(tok.bos_token)  # the lens adapter adds BOS itself
    walk_layers = [l for l in lens.source_layers if l >= 9]
    w_logits, _, _ = lens.apply(model, chat, layers=walk_layers)
    counts: Counter = Counter()
    for l in walk_layers:
        for ids in w_logits[l].topk(5, dim=-1).indices.tolist():
            counts.update(tok.decode([i]).strip().lower() for i in ids)
    common = [w for w, _ in counts.most_common(60) if w.isalpha()]
    out["walkthrough_top_tokens"] = common[:40]
    out["walkthrough_overlap_with_site"] = sorted(WALK_READOUTS & set(common))

    # 4. Speed: forward passes over 1,100-token sequences with a hook at layer 16,
    #    the shape of one experiment session.
    ids = tok(PLAIN_TEXT, return_tensors="pt").input_ids[0]
    seq = ids.repeat(1100 // len(ids) + 1)[:1100]
    hook = hf.model.layers[16].register_forward_hook(lambda m, i, o: None)
    speed = {}
    for bs in (8, 32):
        batch = seq.unsqueeze(0).repeat(bs, 1).to("cuda")
        with torch.no_grad():
            hf(batch)  # warm-up
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                hf(batch)
            torch.cuda.synchronize()
        speed[f"batch_{bs}_tokens_per_s"] = round(3 * bs * 1100 / (time.time() - t0))
    hook.remove()
    out["speed"] = speed
    out["gpu_gb_peak"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    return out


@app.local_entrypoint()
def main():
    result = stage0.remote()
    path = pathlib.Path(__file__).parent / "results" / "stage0.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
