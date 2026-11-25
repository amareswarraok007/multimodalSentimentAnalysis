# Methodology (as implemented in the notebooks)

## Three experimental tracks

| Track | Notebooks | Models / conditions |
|---|---|---|
| A. Text preprocessing | NB01 (v1), NB09, NB10 | RoBERTa-base fine-tuned per run (128 tokens, 2 epochs, batch 32, lr 2·10⁻⁵ encoder / 10⁻⁴ new layers). R1 conventional cleaning (lowercase; URLs, emojis, digits, punctuation and stop-words incl. *not/no/never* removed); R2 minimally processed text; R3 ECCM cue flags (+ PMI domain anchoring on Amazon); R4 = R3 + Group-DRO (Amazon) |
| B. Text + audio + video fusion | NB02v2–NB08b | MOSEI (primary) and MOSI (replication), MMSA `aligned_50` features: text 768-d, audio 74-d (MOSI 5-d), video 35-d (MOSI 20-d), 50 steps. M1 text-only; M2 RMSR adaptation; M3/M4 fixed fusion (M4 with modality dropout); 13 single-change ablations; re-fitted heads H1–H4; all 7 input combinations at test |
| C. Text + photo | NB09 | Amazon Reviews 2023 (Fashion, Beauty), frozen RoBERTa (selected R3 backbone) + frozen CLIP ViT-B/32 image features; heads T0 text, IMG photo, CAT concatenation, F1 RMSR adaptation, F2 token cross-attention, F3 pooled attention, SHUF shuffled-photo control; photos hidden for 0/30/50/70/100% of test reviews |

## Fusion models (Track B)

Notation: modality m ∈ {t, a, v}; x̄ₘ = mean of modality m over real word steps; D = 128.

**M2 — equation-level adaptation of EMO-FAR's RMSR (Eq. 10–14):**

- (1) zₘ = Wₘ x̄ₘ + bₘ
- (2) qₘ = σ(uₘᵀ zₘ); γ = softmax(q_t, q_a, q_v). Because qₘ ∈ (0, 1), γₘ ∈ [1/(1+2e), e/(e+2)] = [0.155, 0.576]:
  a missing modality still receives at least 15.5% of the weight.
- (3) h = γ_t z_t + γ_a W_v z_a + γ_v W_v z_v; ŷ = softmax(W_c h + b_c). With one pooled vector per modality the
  attention of Eq. 13 is a softmax over a single score, i.e. 1.

No Eq. 15 alignment loss in M2; pre-extracted features instead of the paper's encoders.

**M3/M4 — fixed fusion:** per-modality Linear + LayerNorm to D = 128; text→audio and text→video multi-head
cross-attention (4 heads) with residual; masked mean pooling; gate w = softmax(g + log a) with g = W_g[h_t; h_a; h_v]
and availability a ∈ {0, 1}³ (a missing modality gets weight 0); classifier Linear–ReLU–Linear. M4 adds modality
dropout: aₘ ~ Bernoulli(1 − p), p = 0.3, at least one modality kept.

**Training:** AdamW (lr 5·10⁻⁴, weight decay 10⁻⁴), batch 64, up to 30 epochs, early stopping on validation macro-F1
(patience 6), gradient clipping 1.0, class-weighted cross-entropy with weights c_k = N/(3N_k).
Eq. 15 variants (A0 on M2, B6 on M4) add λ₁·L_align with L_align = (1/B) Σᵢ Σ_{m∈{a,v}} [1 − cos(z_t⁽ⁱ⁾, z_m⁽ⁱ⁾)],
λ₁ = 0.3 (the paper's value; not tuned). B6 counts only pairs where both modalities are present.

**Re-fitted heads (NB06/NB07):** M4's encoders frozen; H1 shared head with M4's exposure; H2 shared head with all 7
input combinations equally often; H3 = H2 + FiLM adjustment; H4 one head per combination.

## Metrics

- Macro-F1 = (1/K) Σ_k 2TP_k / (2TP_k + FP_k + FN_k) — main metric (classes are imbalanced).
- Acc-2 (MOSEI/MOSI): sign agreement on clips with a non-zero score.
- ECE = Σ_b (|B_b|/N) · |acc(B_b) − conf(B_b)| (confidence bins).
- Worst-category F1 and category gap (Amazon: Fashion vs Beauty), parameters, measured latency (NB09 protocol: batch 64,
  2 warm-up passes, median of 10, fp16 encoders, download/pre-processing/tokenisation excluded).

## Statistics

- Effect Δ = mean over seeds of (score_with − score_without), in percentage points.
- Paired bootstrap resampling seeds and test groups (videos for MOSEI/MOSI, products for Amazon, reviews for
  Flipkart/IMDB/Yelp); 95% CI = 2.5th–97.5th percentiles; p = 2(min{#Δ* ≤ 0, #Δ* ≥ 0} + 1)/(B + 1).
- Holm correction within families fixed before results: NB08 26 primary tests per dataset (91 secondary); NB09 text
  family 4, fusion family 6, secondary family; NB10 primary 6, negation-subset 6. NB07 had no correction in its
  pre-registered rule; NB11 adds a post-hoc Holm over its 9 checks.
- Decision rule: HELPS / HURTS only if Holm p < .05 and |Δ| ≥ 1 pp.

## Evidence labels (NB11)

| Label | Meaning |
|---|---|
| Supported | Holm p < .05 and \|Δ\| ≥ 1 pp; for NB07 checks: check met and post-hoc Holm p < .05 |
| Nominal only | NB07 check met without correction but not after post-hoc Holm |
| Bounded | 95% CI inside ±1 pp — not a formal equivalence test |
| Inconclusive | no detectable effect; the CI still allows ≥ 1 pp — not evidence of no effect |
| Exploratory | MOSEI diagnostics NB03–NB06 (test set reused; raw p) |
| Descriptive | measured, not tested |
| Hypothesis | proposed explanation without a causal test |
