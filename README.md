# valence-j-split

Does the part of a valence state that a language model can put into words drive its choices, or does the rest?

I split a valence direction in Llama-3.1-8B-Instruct into the part the Jacobian lens can read as tokens and the remainder, and test which part moves a later choice. The full plan is in [PROTOCOL.md](PROTOCOL.md).

This builds on two papers:

- Gurnee et al., "Verbalizable Representations Form a Global Workspace in Language Models" (arXiv 2607.15495), for the Jacobian lens and the J-space.
- Berg and Kaiser, "Language Models Act on Hidden Valence" (arXiv 2609.35591), for the valence vector and the zone-choice test.

## Status

| Stage | What | Steering | Status |
| :--- | :--- | :--- | :--- |
| 0 | Load the model and lens, check the lens, measure speed | none | Done |
| 1 | How much of the valence direction the lens can read | none | Done |
| 2 | Gate: reproduce the zone-choice effect with the full vector | yes | Not started |
| 3 | Readable part against remainder, on choice and on text | yes | Not started |

No steering experiment has been run yet. The tag `protocol-v1` dates the protocol before stage 2.

## Stage 1 result

Share of each direction's squared norm captured by 16 lens tokens, at layer 16:

| Direction | Jacobian lens | J++ lens | R lens |
| :--- | :--- | :--- | :--- |
| Valence, negative side | 13.7% | 9.6% | 11.6% |
| Valence, positive side | 6.9% | 6.6% | 7.6% |
| 171 public emotion vectors (mean) | 12.2% | 11.3% | 11.2% |
| Contrast between two same-valence states | 7.7% | 6.9% | 7.9% |
| Random contrast between neutral passages | 4.6% | 6.4% | 6.2% |
| Random direction | 3.0% | 4.6% | 4.4% |

- The negative side of valence is readable: it beats the neutral-contrast baseline under every lens and at every layer I measured (15, 16, 20, 24, 28). Its tokens include "frustrating", "worse" and "debacle".
- The positive side is not clearly readable: it beats that baseline with the Jacobian lens but not with the J++ lens, and it is no higher than a contrast between two positive states.
- The Pain Axis vectors, at their own layer 28, have a readable share of 6.5–6.9% against 3.6% for random directions, with tokens such as "crippling", "pain" and "hurt". They are almost orthogonal to my valence vector (cosine −0.03 and −0.06).
- A random direction already gets 3–5%, because the dictionary has 128,000 tokens. Only comparisons with the baselines mean anything.

Limits: one passage corpus, one seed, and a valence vector I rebuilt from the paper's description, not the authors' own.

## Layout

- `PROTOCOL.md`: the design, fixed parameters, arms, predictions and limits.
- `data/passages/`: 544 short first-person passages (8 affective states and neutral) used to build the valence vector, with the spec they were written to.
- `experiment/stage0_lens_check.py`, `experiment/stage1_readable_share.py`: the code.
- `experiment/results/`: outputs of each run.

## Running it

The scripts run on [Modal](https://modal.com) with one L40S GPU. You need access to `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face and a Modal secret that provides `HF_TOKEN`.

```
HF_SECRET_NAME=<your secret> modal run experiment/stage0_lens_check.py
HF_SECRET_NAME=<your secret> modal run experiment/stage1_readable_share.py::main
HF_SECRET_NAME=<your secret> modal run experiment/stage1_readable_share.py::pain_main
```

## What I do not publish

Steering vectors. Berg and Kaiser withhold theirs because related code was reused to induce negative states in models, and I follow them. The passages, code and result numbers are here; anyone with model access can rebuild the vector.
