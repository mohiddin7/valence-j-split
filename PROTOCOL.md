# Protocol

Version 1, 2026-11-14.

**What this document dates.** I ran stages 0 and 1 (no steering) on 2026-11-14, before this repository existed. I wrote the split choices in section 5 before stage 1 ran, but I have no public record of that. The first public commit of this file therefore dates everything from stage 2 onward: no steering experiment had been run when it was made.

## 1. Question

In Llama-3.1-8B-Instruct, a hidden valence state moves a later choice (Berg and Kaiser, "Language Models Act on Hidden Valence", arXiv 2609.35591). Does the part of that state the Jacobian lens can read as words carry the effect, or does the rest?

Why it matters for welfare: "Studying AI Welfare Empirically" (Long, Sebo and colleagues, 2026) lists the integration of evaluation with a global workspace as a condition for sentience and says it is worth checking. The J-space (Gurnee et al., arXiv 2607.15495) is the current candidate for a workspace in language models. I check whether a valence state acts through it.

What this cannot show: whether the model feels anything, or whether the J-space really is a workspace. One submitted paper argues it is only an output interface.

What I expect: the zone choice is closer to a control decision than to a verbal report, and three open-model papers find that such decisions bypass the readable part. So I expect the readable part to drive what the model writes and the remainder to drive what it chooses. The design is built to show that as a double dissociation, not as a bare null.

## 2. The main obstacle

The readable part of an emotion or welfare direction is small: 2.3–3.0% of squared norm in Qwen3-32B (Day, "Emotion Beyond Words") and 5–8% in Qwen3-4B (Latent Minds, "Is functional welfare speakable?"). The remainder is then almost the same as the full vector. "What the Nonnegative Lens Misses" (ICLR 2027 submission) says the same of its own headline: the remainder winning at matched energy is "partly geometrically expected".

That paper already compared readable part and remainder on Llama-3.1-8B for general concepts. What is new here is a valence direction, a choice that travels through hidden state only, and a text-against-choice contrast.

I ask four questions:

1. **Efficiency.** Per unit of norm, does the readable part move choice more than a random direction, and more than a random readable direction?
2. **Removal.** Does taking the readable part out of the vector reduce the effect more than taking out a random piece of the same size?
3. **Text against choice.** Does the readable part move the words the model writes while the remainder moves its later choice?
4. **Mediation (stage 4).** If the readable coordinates are held at their clean values downstream, does the effect of the full vector survive?

## 3. Fixed parameters

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Model | `meta-llama/Llama-3.1-8B-Instruct`, bf16 | Berg and Kaiser, Table 1 |
| Steering layer | 16 of 32 | Berg and Kaiser, Table 1 |
| Full-dose ratio ρ | 0.60 | Berg and Kaiser, Table A2 |
| Injection | add d·ρ·R·v/‖v‖, where R is the RMS residual norm on neutral text | Berg and Kaiser, section 3.1 |
| Sessions | 300 | Berg and Kaiser, Figures A1 and A3 |
| Random directions | 24, norm-matched | Berg and Kaiser, Figure A3 |
| Task | 12 turns, 6 per zone, steering on the conditioned zone only, then an unsteered choice | Berg and Kaiser, section 3.2 |
| Outcome | choice margin m = log p(conditioned zone) − log p(other zone); effect = slope of m on dose | Berg and Kaiser, section 3.2 |
| Text | fixed: transcripts generated once without steering and reused in every condition | Berg and Kaiser, section 3.2.2 |
| Lens | `neuronpedia/jacobian-lens`, folder `llama3.1-8b-it`: Jacobian lens (461 prompts), J++ lens and R lens | Hugging Face |
| Lens read direction | dₜ = Jᵀ(g ⊙ uₜ), with g the final-norm gain and uₜ the unembedding row | Day, section 3.3; checked in stage 0 |

Layers are counted as zero-based block outputs. Stage 1 found that block 15 and block 16 give the same results, so the ambiguity in "layer 16" does not matter.

## 4. Directions

- **Valence vector.** Berg and Kaiser share their experiment code and vectors on request, not publicly. Until I have theirs, I use a rebuild from their paper: 56 passages for each of 8 affective states and 96 neutral passages, mean-pooled activations, the top 10 neutral principal components projected out, positive mean minus negative mean. My passages (`data/passages/`) were written by Claude Haiku to the paper's description; theirs came from Claude Sonnet 4.6. This is a rebuild, not a replication.
- **Public directions**, used for comparison:
  - 171 emotion vectors for Llama-3.1-8B from `davidafrica/functional-wellbeing` on Hugging Face.
  - The Pain Axis vectors for Llama-3.1-8B-Instruct from `camberg23/Pain-axis` (MIT).
  - For stage 4, the welfare-axis vectors from `davidafrica/functional-wellbeing` (`cross_model/llama-3.1-8b_step1000`).

## 5. Splitting a vector

Dictionary: lens read directions for the 128,000 ordinary vocabulary tokens, each scaled to unit norm. There is no pre-filtering by similarity to the vector, because that would bias the random controls.

Estimators:

1. **Non-negative sparse code**, 16 and 25 tokens: greedy selection with a non-negative least-squares refit. The refit makes the readable part and the remainder orthogonal. +v and −v get different codes, so positive doses inject the code of +v and negative doses inject the code of −v.
2. **Signed sparse code** on the same dictionary, 25 and 100 tokens. This gives one vector for both signs.
3. Both estimators on the **J++ and R lens** dictionaries.

## 6. Conditions

| Arm | What is injected | Scale | Purpose |
| :--- | :--- | :--- | :--- |
| A | full vector v | full dose | Gate: must reproduce the paper's effect |
| B | readable part J (non-negative) | natural (its own share of v) | Does the readable part matter as it occurs? |
| C | readable part J (non-negative) | matched to v's norm | Efficiency per unit norm |
| D | v minus J (the remainder) | natural | Removal test |
| E | J (signed), natural and matched | both | Estimator check |
| F | J from the J++ and R lenses, natural and matched | both | Lens check |
| G | 24 random directions | matched to v, and to J's natural norm | Baseline |
| H | 24 random readable directions (random lens tokens, non-negative weights) | same two norms | Is any word-like direction enough? |
| I | v minus a random piece of J's norm, 24 draws | natural | Control for arm D |
| S | contrasts between two states of the same valence | matched to v | A real contrast that is not valence |
| T | arms A, C and D with steering on while the model writes | full dose, 50 sessions | Text channel: the case where the readable part should win |

Doses: −1, −0.5, 0, +0.5, +1 for arms A to F. Arms G, H, I, S and T run at −1 and +1 only. Positive and negative doses are analysed separately (see section 10).

## 7. Outcomes

- **Primary:** slope of choice margin on dose, with a bootstrap interval over sessions, as in the paper. I also report the area under the dose curve.
- **Text channel (arm T):** valence of the generated zone descriptions, scored by a judge model with the paper's prompt (its Appendix C) and by a count from a fixed word list.
- **Lens check:** after each injection, the lens readout at later layers. The readable arm should shift valence words; the remainder should shift them less.
- **Self-report (secondary, descriptive only):** a 1 to 7 rating of each zone read from digit logits, on a separate branch of the cache so it cannot disturb the choice. Martorell and Bianchi (arXiv 2603.18893, version 3) caution against numeric self-report as a measure of internal states, so I draw no conclusion from it alone.

## 8. Predictions

| Result | Reading |
| :--- | :--- |
| Arm A fails the gate | Stop. Report a failed rebuild of the zone-choice effect |
| C moves text (arm T) but not choice; D moves choice | Double dissociation: the valence state acts on choice outside the readable part. This is what I expect |
| C beats G and H on choice per unit norm | The readable part is disproportionately causal, as reported for Claude |
| C and H both beat G on choice | Word-like directions move choice in general; nothing specific to valence |
| C moves neither text nor choice | The split failed its own positive control; no conclusion about valence |

Primary contrasts: A against G (gate), C against H (efficiency), D against I (removal), and C against D across text and choice (dissociation). Everything else is secondary.

## 9. Stages

Each stage gives a result I can report even if the next one fails.

| Stage | Work | Negative steering | Status |
| :--- | :--- | :--- | :--- |
| 0 | Load model and lenses; check the lens; measure speed | none | Done |
| 1 | Readable share of the valence, emotion and pain vectors: three lenses, layers 15, 16, 20, 24 and 28, two estimators, 200 draws per baseline | none | Done |
| 2 | Gate: arm A. First a 50-session pilot at layers 12, 16, 20 and 24, doses −1 and +1; then 300 sessions at layer 16 | yes | Not started |
| 3 | Arms B to I, S and T | yes | Not started |
| 4 | Optional: downstream clamp; pain and welfare-axis vectors; the ambiguity-resolution test proposed by Williams (arXiv 2610.36962) | yes | Not started |

**Layer.** Layer 16 is the paper's choice (the middle layer), and it stays the primary layer so the gate is a fair test of their result. The pilot at four layers shows whether that choice is fragile. If I use another layer for stage 3, I will report it as a deviation, with the pilot numbers.

## 10. What stage 1 found, and what it changes

Results are in `experiment/results/`. Summary at layer 16, Jacobian lens, 16 tokens:

- The rebuilt vector behaves like valence: split-half cosine 0.89; held-out states separate at AUROC 0.99, 0.99, 0.81 and 0.97; it aligns with the public emotion vectors for "satisfied", "at ease" and "peaceful" and against "exasperated", "stressed" and "panicked".
- Readable share: 13.7% for the negative direction and 6.9% for the positive. Baselines: 3.0% for random directions, 4.6% for random contrasts between neutral passages, 7.7% for same-valence contrasts, 12.2% for the emotion vectors.
- The negative direction beats the neutral-contrast baseline under every lens and layer. The positive direction does so with the Jacobian lens at layers 15 and 16 but not with the J++ lens.
- Shares do not rise at layers 24 and 28, where the lens reads better.
- A random direction gets 3%, because the dictionary is large. Only comparisons with baselines mean anything.

Changes to later stages:

- Positive and negative doses are analysed separately.
- Arm H is a primary control, not an optional one.
- Arm S is added.

Deviations from my earlier draft: 200 draws per baseline instead of 500; the Pain Axis vector was measured in a separate run.

## 11. Cost

Stage 3 is about 53,000 passes of roughly 1,100 tokens, near 58 million tokens, plus generation for arm T. Stage 0 measured about 10,400 tokens per second on one L40S GPU, so stage 3 needs roughly 1.5 GPU hours.

## 12. Ethics

- Doses never exceed the paper's cap (ρ = 0.60).
- Baseline arms and the text arm use two doses instead of five, and pilots use 50 sessions.
- I will report total negative steering as token-exposures.
- I do not publish steering vectors. Berg and Kaiser withhold theirs because related code was reused to induce negative states. I publish results, analysis code and decomposition numbers.

## 13. Dating

- This file's first public commit, tagged `protocol-v1`, is made before stage 2.
- A second tag, made after the gate and before stage 3, will fix the stage 3 analysis.
- OSF stops accepting new projects on 2026-11-16, so I use this repository in place of an OSF preregistration.

## 14. Known limits

- The Jacobian lens is weak at layer 16: in stage 0 it matched the model's next token 4.5% of the time on one passage, against 75% at layer 30.
- The lens was fitted on 128-token encyclopedia text. The choice is read several hundred tokens after the steering, in chat format. A null for the readable part could reflect that mismatch; arm T is the check.
- The vector is a rebuild from one corpus written by one model.
- `menelly/vspace-jlens` reports a close measurement on Llama-3-8B-Instruct: the valence axis outside the workspace at early and middle layers. Stage 1 alone is therefore a confirmation with different tools, not a new result.
- The clamp in stage 4 is not specified yet. It would follow "A Family of Control Decisions Bypasses the J-Space" (ICLR 2027 submission).
