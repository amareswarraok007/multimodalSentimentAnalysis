# Dataset access

No dataset is redistributed in this repository. Download each source yourself, respect its licence/terms, and place
it under `MyDrive/datasets/` as shown. The notebooks verify sizes, schemas and (where recorded) SHA-256 hashes.

| Dataset | Source | Place under `MyDrive/datasets/` | Used by |
|---|---|---|---|
| CMU-MOSEI, `aligned_50.pkl` (MMSA processed features) | MMSA release of CMU-MOSEI (Zadeh et al., 2018) | `CMU-MOSEI/aligned_50.pkl` (+ `label.csv`, `CMU_MOSEI_Labels.csd` for NB05) | NB02v2–NB08b |
| CMU-MOSI, `aligned_50.pkl` | Hugging Face `tamb2203579/CMU-MOSI` (MMSA processed features; Zadeh et al., 2016) | `CMU-MOSI/Processed/aligned_50.pkl` | NB07, NB08, NB08b, NB11 |
| Amazon Reviews 2023 | Hugging Face `McAuley-Lab/Amazon-Reviews-2023`, files `raw/review_categories/Amazon_Fashion.jsonl` and `All_Beauty.jsonl` (Hou et al., 2024); photos are fetched from the image URLs in the reviews | `Amazon-Reviews-2023/raw/review_categories/` | NB01, NB09 |
| Flipkart | Hugging Face `KayEe/flipkart_sentiment_analysis` (`train.json`, `test.json`) | `Flipkart/` (NB10 downloads it if missing) | NB10 |
| IMDB | Hugging Face `stanfordnlp/imdb`, `plain_text/{train,test}-00000-of-00001.parquet` (Maas et al., 2011) | `IMDB/` (downloaded if missing) | NB10 |
| Yelp | Hugging Face `Yelp/yelp_review_full`, `yelp_review_full/{train,test}-00000-of-00001.parquet` (Zhang et al., 2015) | `Yelp/` (downloaded if missing) | NB10 |

## Recorded file hashes (SHA-256)

| File | SHA-256 |
|---|---|
| CMU-MOSEI `aligned_50.pkl` | `45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791` (matches the official MMSA value) |
| CMU-MOSI `aligned_50.pkl` | `d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9` |
| Flipkart `train.json` / `test.json` | `4f53e0a45bd5b61fcb7cd43884c1561122ce17d4aea9334b551055a99ffa9943` / `9cb489adce35ba808b3d898a7907b02329a652ffaae08aff7f7c11706b1ab801` |
| IMDB train / test parquet | `db47d16b5c297cc0dd625e519c81319c24c9149e70e8496de5475f6fa928342c` / `b52e26e2f872d282ffac460bf9770b25ac6f102cda0e6ca7158df98c94e8b3da` |
| Yelp train / test parquet | `da3ca3dcc52b1e2e44195b53668610d645a59010a985dd45aa12e1109cd44601` / `bf06d5969bff93ecd4a6d4b330643761ce108b42e9b8a894cf82c9df02a08540` |

All NB10 source hashes matched at run time (NB11 record R04).

## Labels and splits (as produced by the notebooks)

| Dataset | Labels | Split (train / valid / test) | How the split is made |
|---|---|---|---|
| MOSEI | sign of the sentiment score; 0 = neutral | 16,326 / 1,871 / 4,659 | official MMSA split |
| MOSI | sign of the sentiment score | 1,284 / 229 / 686 | official MMSA split |
| Amazon | 1–2★ negative, 3★ neutral, 4–5★ positive; reviews with a customer photo; balanced per class and category | 8,367 / 1,800 / 1,833 | by product (no product in two splits); split fingerprint `c3a53bf9…` checked by NB09 and NB11 |
| Flipkart | negative / neutral / positive | 40,000 / 5,000 / 5,000 | official files pooled (they overlap), conflicting and duplicate texts removed, ≥ 3 words, stratified sample of 50,000 |
| IMDB | negative / positive | 22,414 / 2,490 / 24,678 | duplicates removed; test texts also in train removed; train split 90/10 |
| Yelp | 1–2★ negative, 3★ neutral, 4–5★ positive | 79,999 / 10,001 / 10,000 | 90,000 from official train → train/valid; 10,000 from official test |

## The base paper's own data links

The Kaggle Amazon set linked by EMO-FAR (`bittlingmayer/amazonreviews`) is binary and text-only, while the paper
describes text + image data with three classes; the paper's results therefore cannot be rebuilt from its stated data.
This was checked by downloading samples on 22 Nov 2026.
