# Experiment index

| Notebook | Question | Design | Output folder | Key result (evidence label) |
|---|---|---|---|---|
| NB01 (v1, superseded) | Does ECCM preprocessing and fairness training help on Amazon? | first single-seed version | `emofar_results/` root files | superseded by NB09 |
| NB02 v1 (superseded) | First MOSEI fusion run | earlier code | — | superseded by NB02v2 |
| NB02v2 | Baselines M1–M4 on MOSEI, all input combinations | 3 seeds, CPU | `mosei_v2_cpu` | M4 63.45 ± 0.50, M2 62.27 ± 0.53 macro-F1 with all inputs (descriptive) |
| NB03 | Which modalities does M4's gate use? Do saved models reload exactly? | diagnostics, no training | `mosei_v3_diagnostics` | mean gate weight text 0.816, video 0.112, audio 0.071 (descriptive); reload max difference 0 |
| NB04 | Do dedicated unimodal models beat M4 without text? | U1–U3 vs M4 subsets | `mosei_v4_capacity_utility` | V +2.41, A+V +1.98 pp, Holm-3 p ≈ .02 (exploratory) |
| NB05 | Representation or classifier problem? | probes, late fusion | `mosei_v5_representation_diagnostics` | late fusion M1+U3 vs M4 +0.42 pp, raw p .33 (exploratory) |
| NB06 | Does re-fitting the classifier head recover text-missing performance? | heads H1–H4 on frozen M4 | `mosei_v6_head_vs_exposure` | H2 − M4: V +1.72 (raw p .037), A+V +1.38 (raw p .079) (exploratory) |
| NB07 | Pre-registered confirmation on MOSI | checks C1–C5 | `mosi_v7_confirmation` | H2 − M4: A+V +6.58 supported; V +3.59 nominal only (post-hoc Holm) |
| NB08 | Which single changes from M2 to M4 matter? | 13 ablations × 2 measures, MOSEI + MOSI | `v8_ablations` | dropout +2.84 (supported, MOSEI); Eq. 15 −6.97 (supported harm, MOSEI); others no supported effect |
| NB08b | Is Eq. 15 associated with representation/prediction collapse? | pre-registered criteria K1–K6, no training | `v8_ablations/eq15_check` | MOSEI "mixed pattern"; MOSI "alignment not strong" (association only) |
| NB09 | Do photos help Amazon sentiment? Does cleaning hurt? Does ECCM/Group-DRO help? | R1–R4 text runs + 7 heads, hidden-photo masks, latency | `amazon_v9` | cleaning hurts +8.13 (supported); photo fusion +0.06 (bounded); +71% params, +44% latency |
| NB10 | Do the preprocessing findings generalise? | R1–R3 on Flipkart, IMDB, Yelp | `text_v10` | cleaning hurts on all three (supported); ECCM no benefit |
| NB11 | Consolidate, validate and export | no training | `final_v11` | 347 checks OK; figures, tables, claim-to-evidence table |
