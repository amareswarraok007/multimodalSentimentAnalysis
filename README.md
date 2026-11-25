# Multimodal Sentiment Analysis: An Empirical Investigation of Fusion, Robustness and Efficiency

DSC 501 CEDS course project · M.Tech Data Science · SRM University-AP · November 2026  
Author: Kothapalli Amareswar Rao · Guide: Dr. M. Krishna Siva Prasad, Department of Computer Science and Engineering

This repository contains the notebooks, saved statistical results and reproduction instructions for a
**controlled empirical evaluation** of the EMO-FAR framework (Nithya & Rajesh Kanna, *Engineering Applications of
Artificial Intelligence* 181 (2026) 115701, doi:10.1016/j.engappai.2026.115701).

**What this project is — and is not.** We re-implemented EMO-FAR's components at equation level on public data and
tested them one change at a time on six datasets, with statistical tests fixed for each experiment before it ran. It is an evaluation and
architectural-adaptation study. It does **not** propose a new algorithm, and it does **not** reproduce the published
EMO-FAR end to end (its linked Amazon dataset is binary and text-only, which does not match the paper's description).

## Main results (test macro-F1; Δ = with − without, percentage points; 3 seeds)

| Question | Result | Evidence |
|---|---|---|
| Conventional cleaning vs minimally processed text | minimal text higher on Amazon +8.13, Flipkart +4.68, Yelp +3.46, IMDB +1.39 | Supported ×4 (Holm p ≤ .001) |
| ECCM cue-based preprocessing vs minimal text | −0.14, +0.60, −0.11, −0.12 | Bounded ×3 (CI within ±1 pp; not an equivalence test), inconclusive ×1 |
| Modality dropout on the RMSR adaptation (MOSEI, mean of 7 input combinations) | +2.84 [+1.17, +4.33] | Supported on MOSEI; MOSI same direction, not confirmed |
| Eq. 15 alignment loss on the RMSR adaptation (MOSEI) | −6.97 [−8.25, −5.92] | Supported harm on MOSEI; MOSI opposite direction |
| Eight other single architectural changes | — | No supported effect |
| Re-fitted classifier head on frozen features, MOSI, text missing | audio+video +6.58 [+3.41, +10.30]; video +3.59 [+0.45, +6.71] | A+V supported (post-hoc Holm p .004); video nominal only (Holm p .238) |
| Photo fusion vs text only (Amazon) | +0.06 [−0.74, +0.89]; +71% parameters, +44% latency | Bounded; cost measured |
| Group-DRO (Amazon, two product categories) | macro-F1 +0.01; worst-category F1 +0.10 | Bounded / inconclusive |

The complete claim-to-evidence table, limitations and every figure/table are produced by Notebook 11
(`results/final_v11/`, see `tables/C01_claim_to_evidence.csv`, `tables/L01_limitations_negatives_unresolved.csv`
and `INDEX.md`). Evidence labels are defined in `docs/methodology.md`.

## Repository layout

```
notebooks/            the executed Colab notebooks, unchanged (outputs kept as run evidence)
  NB01 … NB11, NB08b  one folder per notebook; *_v1 files are superseded first versions
results/
  statistical_tests/  saved test tables copied from the experiment folders (byte-checked, see MANIFEST.md5)
  final_v11/          Notebook 11 outputs: figures (PDF + 300-dpi PNG), tables (CSV + LaTeX), records, INDEX
configs/              recorded hyperparameters, hardware/software and dataset hashes
docs/                 methodology, dataset access, reproducibility, limitations, experiment index
references/           references.bib (verified entries)
scripts/              check_repo.py (repository checks); nb11_source.py (Notebook 11 source)
requirements.txt      package versions recorded by the runs
```

## Reproducing the results

Full instructions: `docs/reproducibility.md`. In short:

1. **Data** (not included; see `docs/dataset_access.md`): place the files under `MyDrive/datasets/` exactly as listed.
2. **Run order** (Google Colab, Drive mounted): NB02v2 → NB03 → NB04 → NB05 → NB06 → NB07 → NB08 → NB08b → NB09 → NB10 → NB11.
   NB02v2–NB08b run on CPU (4 threads); NB09 and NB10 need a GPU (runs used an NVIDIA L4); NB11 runs on CPU.
   Each notebook writes to its own folder under `MyDrive/emofar_results/` and reads earlier folders read-only.
3. **Seeds**: 42, 43, 44 everywhere. CPU thread count changes the last digits of the fusion results
   (see `docs/reproducibility.md`), so use 4 threads to match the saved numbers.
4. **Consolidate**: Notebook 11 re-validates every source file (347 checks in the final run: 0 missing,
   0 inconsistent), recomputes saved scores from saved predictions and regenerates all figures and tables.
   A second complete run printed identical tables.

The notebooks use absolute Google Drive paths (`/content/drive/MyDrive/...`). They are kept exactly as run; to use
other paths, change the path constants in the first code cell of each notebook.

## Known issues

- Notebook 11 needed a one-line fix for pandas 2.2 (`fillna` on a categorical column); the version here includes it.
- NB06 saved a verdict label ("EXPOSURE EXPLAINS IT") that its own numbers do not support; NB11 uses the comparisons, not the label.
- NB07's pre-registered rule had no multiplicity correction; NB11 adds a post-hoc Holm correction over its 9 checks.
- NB01 and NB02 v1 are superseded (single seed / earlier code) and are not used for any reported result.
- An older Eq. 15 check folder (`eq15_check1_before…`) exists in the results Drive; only `v8_ablations/eq15_check` is used.
- Amazon and text-dataset prediction files store no sample identifiers; their order relies on saved row order and exact
  score recomputation (see `docs/limitations.md`).
