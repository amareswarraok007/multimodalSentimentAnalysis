# Reproducibility

## Environment (recorded by the runs; NB11 record R02)

| Notebooks | Hardware | Software |
|---|---|---|
| NB02v2–NB08b (fusion, MOSEI/MOSI) | Colab CPU, 4 PyTorch threads (Intel Xeon @ 2.20 GHz; NB05 ran on AMD EPYC 7B12) | Python 3.13.18, PyTorch 2.14.1+cpu, NumPy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1 |
| NB09, NB10 (text, photos) | Colab NVIDIA L4 GPU | PyTorch 2.14.1+cu130, Transformers 5.22.1 (Python version not recorded) |
| NB11 (consolidation) | Colab CPU | Colab default packages (its own versions are saved in `records/this_consolidation_run.json`) |

RAM was not recorded by any run. `requirements.txt` lists the recorded versions.

## Google Drive layout expected by the notebooks

```
MyDrive/
├── datasets/                       see docs/dataset_access.md
└── emofar_results/                 created by the notebooks
    ├── mosei_v2_cpu/               NB02v2  (models M1–M4, predictions, logs)
    ├── mosei_v3_diagnostics/       NB03
    ├── mosei_v4_capacity_utility/  NB04
    ├── mosei_v5_representation_diagnostics/  NB05
    ├── mosei_v6_head_vs_exposure/  NB06
    ├── mosi_v7_confirmation/       NB07
    ├── v8_ablations/               NB08  (+ eq15_check/ from NB08b)
    ├── amazon_v9/                  NB09
    ├── text_v10/                   NB10
    └── final_v11/                  NB11  (figures, tables, records, INDEX)
```

## Run order and dependencies

| Step | Notebook | Reads | Device | Recorded run time |
|---|---|---|---|---|
| 1 | NB02v2 MOSEI fusion (CPU) | MOSEI data | CPU | see notebook output |
| 2 | NB03 diagnostics | `mosei_v2_cpu` | CPU | see notebook output |
| 3 | NB04 capacity / utility | `mosei_v2_cpu`, `mosei_v3_diagnostics` | CPU | see notebook output |
| 4 | NB05 representation diagnostics | `mosei_v2_cpu`, `mosei_v3_diagnostics`, `mosei_v4_capacity_utility`, MOSEI label files | CPU | see notebook output |
| 5 | NB06 head vs exposure | `mosei_v2_cpu`, `mosei_v4_capacity_utility`, `mosei_v5_representation_diagnostics` | CPU | see notebook output |
| 6 | NB07 MOSI confirmation | MOSI data; settings copied read-only from `mosei_v2_cpu` | CPU | 21 models in 213 s, 33 heads in 14 s |
| 7 | NB08 ablations | MOSEI, MOSI data; reuses `mosei_v2_cpu` / `mosi_v7_confirmation` reference models only if their saved settings match | CPU | see notebook output |
| 8 | NB08b Eq. 15 check (no training) | `v8_ablations` | CPU | see notebook output |
| 9 | NB09 Amazon text + photo | Amazon data | L4 GPU | see notebook output |
| 10 | NB10 Flipkart / IMDB / Yelp | text datasets | L4 GPU | 27 runs in about 200 min |
| 11 | NB11 consolidation (no training) | every folder above, read-only | CPU | 442 s (first run), 185 s (second run) |

NB01 and NB02 v1 are superseded and are not needed.

## Seeds and settings

Seeds 42, 43 and 44 for every configuration. Full settings are saved by each run (`run_info*.json`, training logs)
and collected by NB11 in `records/all_run_configs_verbatim.json` and table `R03_hyperparameters`. Summary:
`configs/hyperparameters.md`.

## Determinism

- NB08 retrained M2 seed 42 and reproduced the NB02v2 weights bit-identically (record `reference_check.json`).
- NB08b reloaded the NB08 models and reproduced their saved predictions (max difference 3.6·10⁻⁷).
- CPU thread count changes the last digits: the NB07 sandbox preview (2 threads) and the official run (4 threads)
  gave identical verdicts but slightly different numbers. Use 4 threads to match.
- NB11 is deterministic in practice: a second complete run printed identical tables (the output files were not compared byte by byte).

## Integrity checks performed by NB11 (final run)

- 347 source checks OK, 0 missing, 0 inconsistent.
- Test-sample identifiers identical, in order, between NB02v2/NB04/NB05/NB06 and NB08 (MOSEI) and NB07 and NB08 (MOSI).
- Saved macro-F1 recomputed exactly from saved predictions for MOSEI, MOSI, Amazon, Flipkart, IMDB and Yelp.
- Amazon split fingerprint equals Notebook 1's; NB10 split fingerprints and source hashes match.

## What could not be verified

- RAM of the experiment machines (not recorded).
- Sample order of Amazon and text-dataset predictions by identifier (prediction files store none); it relies on the
  saved row order (NB09 asserted it at run time) and on exact recomputation of saved scores.

## Checking this repository

```
python scripts/check_repo.py
```

It checks that every notebook is valid JSON with executed cells, that no data, checkpoints or secrets are committed,
that the copied statistical files match `results/statistical_tests/MANIFEST.md5`, and — once `results/final_v11/`
has been copied in from Drive — that every file listed in its `INDEX.csv` is present.
