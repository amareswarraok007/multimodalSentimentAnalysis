# Saved statistical results (copied from the experiment folders)

| File | Copied from (Google Drive `emofar_results/`) |
|---|---|
| NB07_mosi_v7_confirmation__checks.csv | `mosi_v7_confirmation/checks.csv` |
| NB07_mosi_v7_confirmation__CONFIRMATION.json | `mosi_v7_confirmation/CONFIRMATION.json` (same content; whitespace re-formatted) |
| NB08_v8_ablations__primary_tests.csv | `v8_ablations/primary_tests.csv` |
| NB08b_eq15_check__EQ15_CHECK.json | `v8_ablations/eq15_check/EQ15_CHECK.json` (same content; whitespace re-formatted) |
| NB09_amazon_v9__tests.csv | `amazon_v9/tests.csv` |
| NB09_amazon_v9__results_per_seed.csv | `amazon_v9/results_per_seed.csv` |
| NB09_amazon_v9__latency_params_b64.csv | `amazon_v9/latency_params_b64.csv` |
| NB10_text_v10__tests.csv | `text_v10/tests.csv` |
| NB10_text_v10__results_per_seed.csv | `text_v10/results_per_seed.csv` |

The seven CSV files were checked byte-for-byte against the Drive copies by size; `MANIFEST.md5` lets you re-check them.
Column meanings: `effect_pp` = with − without in percentage points; `lo_pp`/`hi_pp` = 95% bootstrap CI; `p` = raw
two-sided bootstrap p; `p_holm` = Holm-adjusted within the family named in `family` (NB09/NB10) or the 26 primary
tests of a dataset (NB08).
