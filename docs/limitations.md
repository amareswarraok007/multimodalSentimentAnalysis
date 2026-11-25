# Limitations, negative results and open questions

Generated from Notebook 11 table `L01_limitations_negatives_unresolved` (final run, 25 Nov 2026).

## Design limitations

- 3 seeds per configuration; seed variance is part of every interval.
- The MOSEI test set was examined repeatedly in NB02–NB08, so MOSEI-only effects are partly exploratory.
- MOSI's test set is small (686 clips), so intervals are wide.
- MOSEI/MOSI audio and video features are word-aligned to the transcript, so "text missing" still assumes a transcript existed.
- Amazon: photo-containing reviews only, star-derived labels, balanced classes, simulated missing photos, 12,000 reviews.
- Amazon fusion heads use frozen encoders; the published EMO-FAR model (its encoders and its latency/fairness losses) was not reproduced.
- Fairness was measured across two product categories only, not across groups of people.
- NB10's bootstrap resamples individual reviews (no product or business IDs in the data); intervals may be too narrow.
- The Flipkart copy is mostly English (not code-mixed); Yelp test class balance 40/20/40; IMDB has two classes; ECCM domain anchoring is inactive on these three datasets.
- Latency was measured only for the Amazon models, under one protocol.
- RAM of the experiment machines was not recorded.
- The Eq. 15 analysis (NB08b) is associational; its criterion K4 can only detect a near-singular projection for M2/A0.
- Amazon and NB10 prediction files store no sample identifiers; their order rests on saved row order and exact score recomputation.
- NB07 had no multiplicity correction in its pre-registered rule; the Holm correction applied in NB11 is post hoc.

## Null, bounded or inconclusive results

- ECCM vs minimally processed text: bounded ×3, inconclusive ×1.
- Group-DRO: bounded (macro-F1), inconclusive (worst-category F1).
- Photo fusion vs text, and between fusion designs: bounded ×3, inconclusive ×3; real vs shuffled photos and hidden photos: bounded.
- Eight single architectural changes between M2 and M4: inconclusive ×9, bounded ×7 (two measures each).
- Token vs pooled attention (M4 vs B3): bounded.
- Separate per-combination heads (H4) vs one re-fitted shared head (H2): inconclusive (NB07 marked this "confirmed" on non-significance).

## Partly supported

- Re-fitted shared head (H2) vs M4's own output, text missing (MOSI): audio+video supported after post-hoc Holm; video nominal only.

## Negative results for base-paper components

- Eq. 15 (published form) lowered MOSEI mean-of-7 macro-F1 by 6.97 pp; MOSI: opposite direction.
- Conventional (paper-style) cleaning scored lower than minimally processed text on all four datasets.

## Open questions

- Does the modality-dropout effect replicate beyond MOSEI? (MOSI: same direction, not confirmed.)
- Does Eq. 15 cause the neutral-heavy output without text? (Needs a λ₁ sweep with retraining; not run.)
- Does ECCM help on genuinely code-mixed review text? (Not tested.)
- Do photos help when the review text is very short or ambiguous? (Not tested.)
- How much does the head re-fit result overlap with published work (e.g. SCBC)? Publication details and method must be verified before any claim.

## Note

NB06's saved verdict label "EXPOSURE EXPLAINS IT" does not match its own numbers; the comparisons are used, the label is not.
