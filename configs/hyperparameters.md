# Recorded settings (from run_info*.json and training logs; NB11 table R03)

| Notebooks | Settings |
|---|---|
| NB02v2 (MOSEI) | D_MODEL 128; EPOCHS 30; PATIENCE 6; BATCH_SIZE 64; LR 5e-4; MOD_DROPOUT 0.3; mask seeds 7–11; bootstrap 10,000; missing rates 0.3/0.5/0.7 |
| NB03–NB06 (MOSEI) | NB02v2 settings reused (n_train 16,326) |
| NB07 (MOSI) | D_MODEL 128; EPOCHS 30; PATIENCE 6; BATCH_SIZE 64; LR 5e-4; MOD_DROPOUT 0.3; 4 threads |
| NB08 (MOSEI, MOSI) | as NB07; lambda1 (Eq. 15) 0.3 |
| NB09 (Amazon) | roberta-base; 2,000 reviews per class per category; MAX_LEN 128; EPOCHS 2; BATCH_SIZE 32; LR encoder 2e-5, new layers 1e-4; Group-DRO eta 0.01; photo dropout 0.3; head width 256; head epochs ≤ 30, patience 5, lr 1e-3, batch 64; CLIP openai/clip-vit-base-patch32 (frozen) |
| NB10 (Flipkart, IMDB, Yelp) | roberta-base; MAX_LEN 128; EPOCHS 2; BATCH_SIZE 32; LR encoder 2e-5, new layers 1e-4 |

Fixed in code (not in the saved settings) for NB02v2–NB08: AdamW (weight decay 1e-4), inverse-frequency class-weighted
cross-entropy, gradient clipping 1.0, early stopping on validation macro-F1. Seeds 42, 43, 44 everywhere.
The complete saved settings are in `results/final_v11/records/all_run_configs_verbatim.json`.
