# Notebook 11 outputs (final consolidated results)

This folder holds the outputs of the final Notebook 11 run (25 November 2026):

- `INDEX.csv`, `INDEX.md` — the list of every table and figure, what it shows and which saved files it comes from
- `tables/` — 28 tables, each as CSV and LaTeX, plus the data behind figures F08 and F11
- `figures/` — 13 figures, each as vector PDF and 300-dpi PNG
- `records/` — run configurations, the source-validation report and the reproducibility checks

They are copied unchanged from the Google Drive folder `MyDrive/emofar_results/final_v11/`.
Running `python scripts/check_repo.py --strict` confirms that every file listed in `INDEX.csv` is present.

LaTeX tables need `\usepackage{booktabs,tabularx,xltabular,rotating,pdflscape,graphicx}`.
