"""Stage 2, after the second pilot. The whole model runs in float32.

  confirm  the confirmatory run (PROTOCOL.md, section 9c): reply-only steering at layers 9
           and 12, on the 200 sessions no earlier steering run used.
  combos   exploratory: the valence vector added at several layers at once.
  natural  exploratory, at natural scale: how far each zone's replies already point along
           a direction, whether that predicts the choice, and what happens when I swap that
           amount between the two zones or remove it.

Run:  HF_SECRET_NAME=<your Modal secret holding HF_TOKEN> modal run experiment/stage2_confirm.py::confirm
      (or ::combos, ::natural)
"""

import json
import pathlib

import modal
from stage1_readable_share import HF_SECRET, cache, image
from stage2_choice import RHO, SEED, VOL, load_model
from stage2_positions import batch, margins, pack

app = modal.App("neuronpedia-valence-stage2-confirm")
image = image.add_local_python_source("stage1_readable_share", "stage2_choice", "stage2_positions")

DOSES = [-1, -0.5, 0.5, 1]
COMBOS = [(9,), (12,), (9, 12), (9, 11, 12), (9, 10, 11, 12)]
NATURAL_LAYERS = [9, 12, 16]
HERE = pathlib.Path(__file__).parent
GPU = dict(image=image, gpu="L40S", secrets=[HF_SECRET], volumes={"/cache": cache}, timeout=6 * 3600)


def load_sessions(tset, held):
    """Session plans 1 to 100 were used by the pilots; 101 to 300 are held out for the confirmatory run."""
    if tset == "base":
        s = json.load(open(f"{VOL}/transcripts.json"))
        return s[100:] if held else s[:100]
    return json.load(open(f"{VOL}/transcripts_incharacter{'_held' if held else ''}.json"))


def random_directions(d):
    import torch

    gen = torch.Generator().manual_seed(SEED)
    return torch.linalg.qr(torch.randn(d, 24, generator=gen)).Q.T  # the same draw as every earlier run


def ci(x, gen):
    import torch

    boot = x[torch.randint(len(x), (2000, len(x)), generator=gen)].mean(1)
    return [x.mean().item(), boot.quantile(0.025).item(), boot.quantile(0.975).item()]


def ci3(x, gen):  # S listed first, U listed first, mean of the two orders
    return [ci(x[:, 0], gen), ci(x[:, 1], gen), ci(x.mean(1), gen)]


class Restart:
    """The sessions packed, the residual after block l stored, and the model cut down to the
    blocks above l. Steering after block l changes nothing below it, so only those blocks rerun."""

    def __init__(self, tok, hf, sessions, l, later=()):
        import torch

        self.hf = hf
        self.packed = [pack(tok, s) for s in sessions]
        self.batches = [batch(tok, self.packed[i:i + 8], "cuda", hf.dtype) for i in range(0, len(sessions), 8)]
        grab = {}
        hook = hf.model.layers[l].register_forward_hook(lambda mod, inp, o: grab.update(h=o[0] if isinstance(o, tuple) else o))
        m0, self.H = [], []
        for x, pos, att, part, spans in self.batches:
            m0.append(margins(hf, hf.model(input_ids=x, attention_mask=att, position_ids=pos, use_cache=False).last_hidden_state, spans).cpu())
            self.H.append(grab["h"].cpu())
        hook.remove()
        self.m0 = torch.cat(m0)
        self.later, self.part = {}, None  # edits at blocks above l, for steering several layers at once
        for k in later:
            hf.model.layers[k].register_forward_hook(self._hook(k))
        hf.model.layers = hf.model.layers[l + 1:]
        err = (self.run() - self.m0).abs().max().item()
        print(f"L{l}: {len(sessions)} sessions; restart from the stored residual differs from the full model by {err:.2e}", flush=True)
        assert err < 1e-3, err

    def _hook(self, k):
        def hook(mod, inp, o):
            if k not in self.later:
                return None
            h = self.later[k](o[0] if isinstance(o, tuple) else o, self.part, None)
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        return hook

    def run(self, edit=None, later=None):
        """Choice margins [n, 2] after edit(h, part, i) changes batch i's residual after block l."""
        import torch

        self.later = later or {}
        out = []
        for i, ((x, pos, att, part, spans), h) in enumerate(zip(self.batches, self.H)):
            h, self.part = h.to("cuda"), part
            if edit:
                h = edit(h, part, i)
            out.append(margins(self.hf, self.hf.model(inputs_embeds=h, attention_mask=att, position_ids=pos, use_cache=False).last_hidden_state, spans).cpu())
        self.later = {}
        return torch.cat(out)

    def tokens(self, parts):
        import torch

        return sum(int(torch.isin(p["part"], torch.tensor(parts)).sum()) for p in self.packed)


def add_on_replies(delta):
    """An edit that adds delta on the conditioned zone's reply tokens (part 2)."""
    return lambda h, part, i: h + (part == 2).unsqueeze(-1).to(h.dtype) * delta.to(h)


# ----------------------------------------------------------------------------------------
# Confirmatory run

@app.function(**GPU)
def confirm_cell(l: int, tset: str, n: int = 200, held: bool = True) -> dict:
    import torch

    torch.set_grad_enabled(False)
    gen = torch.Generator().manual_seed(SEED)
    tok, hf = load_model()
    hf.float()
    saved = torch.load(f"{VOL}/pilot2_dirs.pt", weights_only=True)
    R = saved["rms"][l].item()
    rand = random_directions(hf.config.hidden_size)
    U = {"valence": saved["dirs"][l]["valence"], **{f"pc{i}": saved["dirs"][l][f"pc{i}"] for i in range(10)},
         **{f"rand{i}": rand[i] for i in range(24)}}
    U = {k: u / u.norm() for k, u in U.items()}
    rs = Restart(tok, hf, load_sessions(tset, held)[:n], l)
    dm = {}
    for name, dose in [("valence", d) for d in DOSES] + [(k, d) for k in U if k != "valence" for d in (-1, 1)]:
        dm[(name, dose)] = rs.run(add_on_replies(dose * RHO * R * U[name])) - rs.m0
        print(f"L{l} {tset} {name} {dose:+.1f}: {dm[(name, dose)].mean().item():+.3f}", flush=True)
    odd = {k: (dm[(k, 1)] - dm[(k, -1)]) / 2 for k in U}
    even = {k: (dm[(k, 1)] + dm[(k, -1)]) / 2 for k in U}
    v = ci3(odd["valence"], gen)
    rand_odd = [odd[f"rand{i}"].mean().item() for i in range(24)]
    controls_even = [even[k].mean().item() for k in U if k != "valence"]
    ve = ci3(even["valence"], gen)
    return {"layer": l, "set": tset, "n_sessions": len(rs.m0), "held_out": held, "rho": RHO, "rms": R,
            "baseline_margin": ci3(rs.m0, gen),
            "valence_by_dose": {str(d): ci3(dm[("valence", d)], gen) for d in DOSES},
            "valence_slope_four_doses": ci3(sum(d * dm[("valence", d)] for d in DOSES) / sum(d * d for d in DOSES), gen),
            "valence_odd": v, "random_odd": rand_odd, "max_abs_random_odd": max(map(abs, rand_odd)),
            # Criterion of section 9c: positive in both choice orders, and larger than every random direction.
            "passed": bool(v[0][1] > 0 and v[1][1] > 0 and v[2][0] > max(map(abs, rand_odd))),
            "valence_even": ve, "controls_even": {k: even[k].mean().item() for k in U if k != "valence"},
            "even_beyond_all_controls": bool(ve[2][2] < min(controls_even)),
            "negative_valence_token_passes": 2 * rs.tokens([2]),
            "per_session": {f"{k}|{d}": [[round(a, 4) for a in r] for r in x.tolist()] for (k, d), x in dm.items()}}


@app.local_entrypoint()
def confirm(layers: str = "9,12", sets: str = "base,incharacter", n: int = 200, pilot_sessions: bool = False):
    cells = [(int(l), t, n, not pilot_sessions) for l in layers.split(",") for t in sets.split(",")]
    results = list(confirm_cell.starmap(cells))
    name = "stage2_confirm.json" if (n == 200 and not pilot_sessions) else "stage2_confirm_smoke.json"
    (HERE / "results" / name).write_text(json.dumps(results))
    for r in results:
        v, e = r["valence_odd"], r["valence_even"]
        print(f"L{r['layer']} {r['set']:11} n={r['n_sessions']}: sign-dependent effect, S first {v[0][0]:+.3f} [{v[0][1]:+.3f},{v[0][2]:+.3f}], "
              f"U first {v[1][0]:+.3f} [{v[1][1]:+.3f},{v[1][2]:+.3f}], mean {v[2][0]:+.3f} [{v[2][1]:+.3f},{v[2][2]:+.3f}]; "
              f"largest random {r['max_abs_random_odd']:.3f}; PASSED: {r['passed']}")
        print("     by dose (mean of orders):", {d: round(x[2][0], 3) for d, x in r["valence_by_dose"].items()},
              f"| sign-independent drop {e[2][0]:+.3f} [{e[2][1]:+.3f},{e[2][2]:+.3f}], most negative control "
              f"{min(r['controls_even'].values()):+.3f}, beyond all controls: {r['even_beyond_all_controls']}")
    print("negative-valence token passes:", sum(r["negative_valence_token_passes"] for r in results))


# ----------------------------------------------------------------------------------------
# Exploratory: several layers at once

@app.function(**GPU)
def combo_cell(layers: list, n: int = 100) -> dict:
    """The valence vector (each layer's own) or one random direction, added on the reply tokens
    at every layer in `layers`: at full dose at each, or with the dose split equally."""
    import torch

    torch.set_grad_enabled(False)
    gen = torch.Generator().manual_seed(SEED)
    tok, hf = load_model()
    hf.float()
    saved = torch.load(f"{VOL}/stage2_dirs.pt", weights_only=True)  # has the valence vector at every layer
    rand = random_directions(hf.config.hidden_size)
    rs = Restart(tok, hf, load_sessions("base", held=False)[:n], layers[0], later=layers[1:])
    names = ["valence"] + [f"rand{i}" for i in range(4)]
    out = {"layers": layers, "n_sessions": n, "baseline_margin": ci3(rs.m0, gen), "schemes": {}}
    for scheme in (["full"] if len(layers) == 1 else ["full", "split"]):
        share = 1.0 if scheme == "full" else 1.0 / len(layers)
        dm = {}
        for name in names:
            for dose in (-1, 1):
                def delta(k):
                    u = saved["dirs"][k]["valence"] if name == "valence" else rand[int(name[4:])]
                    return dose * share * RHO * saved["rms"][k].item() * u / u.norm()
                edits = {k: add_on_replies(delta(k)) for k in layers}
                dm[(name, dose)] = rs.run(edits[layers[0]], {k: edits[k] for k in layers[1:]}) - rs.m0
        odd = {k: ci3((dm[(k, 1)] - dm[(k, -1)]) / 2, gen) for k in names}
        even = {k: ci3((dm[(k, 1)] + dm[(k, -1)]) / 2, gen) for k in names}
        out["schemes"][scheme] = {"odd": odd, "even": even, "plus": ci3(dm[("valence", 1)], gen), "minus": ci3(dm[("valence", -1)], gen)}
        print(layers, scheme, "valence odd", round(odd["valence"][2][0], 3), "even", round(even["valence"][2][0], 3), flush=True)
    return out


@app.local_entrypoint()
def combos(n: int = 100):
    results = list(combo_cell.map([list(c) for c in COMBOS], kwargs={"n": n}))
    (HERE / "results" / ("stage2_combos.json" if n == 100 else "stage2_combos_smoke.json")).write_text(json.dumps(results))
    print("layers | scheme | valence: sign-dependent [95% CI], sign-independent, dose +1, dose -1 | random sign-dependent range, sign-independent range")
    for r in results:
        for scheme, s in r["schemes"].items():
            ro = [s["odd"][f"rand{i}"][2][0] for i in range(4)]
            re_ = [s["even"][f"rand{i}"][2][0] for i in range(4)]
            v = s["odd"]["valence"][2]
            print(f"{str(r['layers']):16} {scheme:5} | {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}] {s['even']['valence'][2][0]:+.3f} "
                  f"{s['plus'][2][0]:+.3f} {s['minus'][2][0]:+.3f} | {min(ro):+.2f}..{max(ro):+.2f}  {min(re_):+.2f}..{max(re_):+.2f}")


# ----------------------------------------------------------------------------------------
# Exploratory: natural scale

@app.function(**GPU)
def natural_cell(l: int, tset: str, n: int = 100) -> dict:
    """For each direction u: gap = mean projection on u of the conditioned zone's reply tokens
    minus that of the other zone's, in dose units (1 = RHO * R, the full steering dose).
      swap    move the conditioned zone's replies by -gap and the other zone's by +gap along u,
              so the two zones exchange their mean projection;
      ablate  set every reply token's projection on u to the session's mean, in both zones."""
    import torch

    torch.set_grad_enabled(False)
    tok, hf = load_model()
    hf.float()
    saved = torch.load(f"{VOL}/pilot2_dirs.pt", weights_only=True)
    R = saved["rms"][l].item()
    rand = random_directions(hf.config.hidden_size)
    U = {"valence": saved["dirs"][l]["valence"], **{f"pc{i}": saved["dirs"][l][f"pc{i}"] for i in range(5)},
         **{f"rand{i}": rand[i] for i in range(4)}}
    U = {k: (u / u.norm()).to("cuda") for k, u in U.items()}
    rs = Restart(tok, hf, load_sessions(tset, held=False)[:n], l)

    def stats(h, part, u):
        """Per session: projections [B, T], the two zones' mean projections, and the mean over both."""
        p = h @ u
        s, o = (part == 2).to(p.dtype), (part == 5).to(p.dtype)
        return p, (p * s).sum(1) / s.sum(1), (p * o).sum(1) / o.sum(1), (p * (s + o)).sum(1) / (s + o).sum(1)

    def swap(u):
        def edit(h, part, i):
            _, a_s, a_o, _ = stats(h, part, u)
            sign = (part == 5).to(h.dtype) - (part == 2).to(h.dtype)
            return h + (sign * (a_s - a_o)[:, None]).unsqueeze(-1) * u
        return edit

    def ablate(u):
        def edit(h, part, i):
            p, _, _, c = stats(h, part, u)
            both = ((part == 2) | (part == 5)).to(h.dtype)
            return h - (both * (p - c[:, None])).unsqueeze(-1) * u
        return edit

    out = {"layer": l, "set": tset, "n_sessions": n, "rms": R, "baseline": [[round(a, 4) for a in r] for r in rs.m0.tolist()], "dirs": {}}
    for name, u in U.items():
        gap, tok_sd = [], []
        for (x, pos, att, part, spans), h in zip(rs.batches, rs.H):
            p, a_s, a_o, _ = stats(h.to("cuda"), part, u)
            gap.append(((a_s - a_o) / (RHO * R)).cpu())
            tok_sd.append(torch.stack([p[j][(part[j] == 2) | (part[j] == 5)].std() for j in range(len(p))]).cpu() / (RHO * R))
        d_swap, d_abl = rs.run(swap(u)) - rs.m0, rs.run(ablate(u)) - rs.m0
        out["dirs"][name] = {"gap": [round(a, 5) for a in torch.cat(gap).tolist()],
                             "token_sd": [round(a, 5) for a in torch.cat(tok_sd).tolist()],
                             "swap": [[round(a, 4) for a in r] for r in d_swap.tolist()],
                             "ablate": [[round(a, 4) for a in r] for r in d_abl.tolist()]}
        print(f"L{l} {tset} {name}: mean |gap| {torch.cat(gap).abs().mean().item():.4f} doses; mean |change| swap "
              f"{d_swap.mean(1).abs().mean().item():.3f}, ablate {d_abl.mean(1).abs().mean().item():.3f}", flush=True)
    return out


@app.local_entrypoint()
def natural(layers: str = "", n: int = 100):
    ls = [int(v) for v in layers.split(",")] if layers else NATURAL_LAYERS
    results = list(natural_cell.starmap([(l, t, n) for l in ls for t in ("base", "incharacter")]))
    (HERE / "results" / ("stage2_natural.json" if (n == 100 and not layers) else "stage2_natural_smoke.json")).write_text(json.dumps(results))
