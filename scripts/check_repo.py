#!/usr/bin/env python3
"""Repository checks for multimodal-sentiment-evaluation.

Run from the repository root:

    python scripts/check_repo.py            # results/final_v11 missing -> reported as INCOMPLETE
    python scripts/check_repo.py --strict   # results/final_v11 missing -> FAIL

What it checks (standard library only, no data or GPU needed):
  1. every notebook under notebooks/ is valid nbformat-4 JSON, every non-empty code cell has an
     execution count, and no saved output is an error traceback;
  2. no dataset, checkpoint, archive or oversized file is present;
  3. no access token, private key or e-mail address appears in any text file or notebook;
  4. every file in results/statistical_tests/ matches MANIFEST.md5;
  5. once results/final_v11/ has been copied from Drive, every file named in INDEX.csv exists and is non-empty;
  6. relative Markdown links in README.md, docs/ and the folder READMEs point to files that exist.

What it does NOT check: that the notebooks re-run, that the datasets are available, or that re-running
reproduces the saved numbers. Those need the data and Colab; see docs/reproducibility.md.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_EXT = {".pkl", ".pickle", ".csd", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".parquet", ".jsonl",
                 ".h5", ".hdf5", ".safetensors", ".bin", ".onnx", ".zip", ".tar", ".gz", ".7z", ".rar"}
FORBIDDEN_DIRS = {"datasets", "emofar_results", ".ipynb_checkpoints", "__pycache__", ".venv", "venv"}
MAX_BYTES = 10 * 1024 * 1024          # GitHub rejects > 100 MB; anything > 10 MB here is suspicious
TEXT_EXT = {".py", ".ipynb", ".md", ".txt", ".csv", ".json", ".tex", ".bib", ".cfg", ".toml", ".yaml", ".yml", ""}

SECRET_PATTERNS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "OpenAI/Anthropic-style key": re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}\b"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "hard-coded credential": re.compile(r"(?i)\b(?:api[_-]?key|secret|password|token)\b\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"),
    "e-mail address": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
}
# Addresses that are part of public metadata, not personal data (extend only after review).
EMAIL_ALLOW = {"noreply@github.com"}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, check, status, detail):
        self.rows.append((check, status, detail))

    def counts(self):
        out = {}
        for _, s, _ in self.rows:
            out[s] = out.get(s, 0) + 1
        return out


def repo_files():
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and ".git" not in p.relative_to(ROOT).parts:
            yield p


def rel(p):
    return p.relative_to(ROOT).as_posix()


def check_notebooks(rep):
    nbs = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    if not nbs:
        rep.add("notebooks", "FAIL", "no notebooks found under notebooks/")
        return
    for p in nbs:
        try:
            nb = json.loads(p.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            rep.add("notebooks", "FAIL", f"{rel(p)}: not valid JSON ({e})")
            continue
        if nb.get("nbformat") != 4 or not isinstance(nb.get("cells"), list):
            rep.add("notebooks", "FAIL", f"{rel(p)}: not an nbformat-4 notebook")
            continue
        code = [c for c in nb["cells"] if c.get("cell_type") == "code" and "".join(c.get("source", "")).strip()]
        unexec = [i for i, c in enumerate(code) if not c.get("execution_count")]
        errors = [i for i, c in enumerate(code)
                  if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
        if not code:
            rep.add("notebooks", "FAIL", f"{rel(p)}: no code cells")
        elif unexec:
            rep.add("notebooks", "FAIL", f"{rel(p)}: {len(unexec)} of {len(code)} non-empty code cells not executed")
        elif errors:
            rep.add("notebooks", "FAIL", f"{rel(p)}: error output saved in {len(errors)} cell(s)")
        else:
            rep.add("notebooks", "PASS", f"{rel(p)}: {len(code)} code cells, all executed, no error outputs")


def check_artefacts(rep):
    bad = []
    for p in repo_files():
        parts = set(p.relative_to(ROOT).parts[:-1])
        if p.suffix.lower() in FORBIDDEN_EXT:
            bad.append(f"{rel(p)} (extension {p.suffix})")
        elif parts & FORBIDDEN_DIRS:
            bad.append(f"{rel(p)} (inside {', '.join(sorted(parts & FORBIDDEN_DIRS))}/)")
        elif p.stat().st_size > MAX_BYTES:
            bad.append(f"{rel(p)} ({p.stat().st_size / 1e6:.1f} MB)")
    if bad:
        for b in bad:
            rep.add("data/checkpoints", "FAIL", b)
    else:
        total = sum(p.stat().st_size for p in repo_files())
        rep.add("data/checkpoints", "PASS",
                f"no data, checkpoint or archive files; no file > {MAX_BYTES // 2**20} MB; repository {total / 1e6:.1f} MB")


def check_secrets(rep):
    hits = []
    me = Path(__file__).resolve()
    for p in repo_files():
        if p.resolve() == me or p.suffix.lower() not in TEXT_EXT:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pat in SECRET_PATTERNS.items():
            for m in pat.finditer(text):
                s = m.group(0)
                if name == "e-mail address" and (s.lower() in EMAIL_ALLOW or s.lower().endswith((".png", ".pdf"))):
                    continue
                line = text.count("\n", 0, m.start()) + 1
                shown = s[:6] + "…" if name != "e-mail address" else s
                hits.append(f"{rel(p)}:{line}: {name} ({shown})")
    if hits:
        for h in hits:
            rep.add("secrets", "FAIL", h)
    else:
        rep.add("secrets", "PASS", "no tokens, keys, credentials or e-mail addresses found")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_manifest(rep):
    d = ROOT / "results" / "statistical_tests"
    man = d / "MANIFEST.md5"
    if not man.exists():
        rep.add("statistical records", "FAIL", "results/statistical_tests/MANIFEST.md5 missing")
        return
    listed = set()
    ok = 0
    for line in man.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        listed.add(name)
        f = d / name
        if not f.exists():
            rep.add("statistical records", "FAIL", f"{name}: listed in MANIFEST.md5 but missing")
        elif md5(f) != digest:
            rep.add("statistical records", "FAIL", f"{name}: checksum differs from MANIFEST.md5")
        else:
            ok += 1
    extra = sorted(p.name for p in d.iterdir()
                   if p.is_file() and p.name not in listed and p.name not in {"MANIFEST.md5", "README.md"})
    for e in extra:
        rep.add("statistical records", "WARN", f"{e}: present but not listed in MANIFEST.md5")
    if ok == len(listed) and listed:
        rep.add("statistical records", "PASS", f"{ok}/{len(listed)} files match MANIFEST.md5")


def check_final_v11(rep, strict):
    d = ROOT / "results" / "final_v11"
    idx = d / "INDEX.csv"
    if not idx.exists():
        rep.add("final_v11", "FAIL" if strict else "INCOMPLETE",
                "results/final_v11/INDEX.csv not present - copy MyDrive/emofar_results/final_v11/ here first")
        return
    with open(idx, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "files" not in rows[0]:
        rep.add("final_v11", "FAIL", "INDEX.csv has no 'files' column")
        return
    sub = {"table": "tables", "figure": "figures", "figure data": "tables", "record": "records"}
    missing, empty, n = [], [], 0
    for r in rows:
        for name in [x.strip() for x in r["files"].split(",") if x.strip()]:
            n += 1
            cands = [d / name] if "/" in name else [d / sub.get(r.get("type", ""), ""), d / "tables",
                                                    d / "figures", d / "records", d]
            if "/" not in name:
                cands = [c / name for c in cands]
            hit = next((c for c in cands if c.is_file()), None)
            if hit is None:
                missing.append(f"{r['id']}: {name}")
            elif hit.stat().st_size == 0:
                empty.append(f"{r['id']}: {name}")
    for m in missing:
        rep.add("final_v11", "FAIL", f"listed in INDEX.csv but missing: {m}")
    for e in empty:
        rep.add("final_v11", "FAIL", f"empty file: {e}")
    flagged = [r["id"] for r in rows if str(r.get("validation status", "")).upper().startswith("CHECK")]
    if flagged:
        rep.add("final_v11", "WARN", f"{len(flagged)} INDEX entries carry a CHECK validation status: "
                                     f"{', '.join(flagged[:6])}{' …' if len(flagged) > 6 else ''}")
    if not missing and not empty:
        types = {}
        for r in rows:
            types[r.get("type", "?")] = types.get(r.get("type", "?"), 0) + 1
        summary = ", ".join(f"{v} {k}" for k, v in sorted(types.items()))
        rep.add("final_v11", "PASS", f"all {n} files listed in INDEX.csv present ({summary})")


LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def check_links(rep):
    docs = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), *sorted(ROOT.glob("*/README.md")),
            *sorted(ROOT.glob("*/*/README.md"))]
    broken, n = [], 0
    for doc in docs:
        if not doc.exists():
            continue
        for m in LINK.finditer(doc.read_text(encoding="utf-8")):
            target = m.group(1).split("#")[0]
            if not target or re.match(r"^[a-z]+:", target):
                continue
            n += 1
            if not (doc.parent / target).exists():
                broken.append(f"{rel(doc)} -> {target}")
    for b in broken:
        rep.add("doc links", "FAIL", b)
    if not broken:
        rep.add("doc links", "PASS", f"{n} relative links checked, none broken")


def check_required(rep):
    need = ["README.md", "requirements.txt", ".gitignore", "docs/methodology.md", "docs/dataset_access.md",
            "docs/reproducibility.md", "docs/limitations.md", "docs/experiment_index.md",
            "references/references.bib", "configs/hyperparameters.md", "configs/dataset_sha256.json"]
    miss = [n for n in need if not (ROOT / n).is_file()]
    for m in miss:
        rep.add("required files", "FAIL", f"{m} missing")
    if not miss:
        rep.add("required files", "PASS", f"{len(need)} required files present")
    try:
        json.loads((ROOT / "configs" / "dataset_sha256.json").read_text())
    except Exception as e:  # noqa: BLE001
        rep.add("required files", "FAIL", f"configs/dataset_sha256.json is not valid JSON ({e})")
    if not (ROOT / "LICENSE").exists():
        rep.add("required files", "WARN", "no LICENSE file - choose a licence before making the repository public")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="treat a missing results/final_v11/ as a failure")
    args = ap.parse_args()
    rep = Report()
    check_required(rep)
    check_notebooks(rep)
    check_artefacts(rep)
    check_secrets(rep)
    check_manifest(rep)
    check_final_v11(rep, args.strict)
    check_links(rep)
    width = max(len(c) for c, _, _ in rep.rows)
    for c, s, d in rep.rows:
        print(f"[{s:10s}] {c:<{width}}  {d}")
    cnt = rep.counts()
    print("\nSummary: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
    print("Not checked here: re-execution of the notebooks and agreement of re-run numbers with the saved files "
          "(needs the datasets and Colab; see docs/reproducibility.md).")
    sys.exit(1 if cnt.get("FAIL") else 0)


if __name__ == "__main__":
    main()
