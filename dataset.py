"""
dataset.py — Run once to generate data/splits.csv.

Scans DATA_ROOT, builds the full image index, performs a case-level
stratified train/val/test split (80/10/10), and verifies there is no
patient leakage between splits.

Usage:
    python dataset.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_ROOT, SPLITS_CSV, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, RANDOM_SEED

# ── Category definitions (exact folder names) ──────────────────────────────
CATEGORIES = {
    "Distale UA-Aufnahme mit HG mit Fraktur":
        {"region": "distal",   "label": "fracture", "label_id": 1},
    "Distale UA-Aufnahme mit HG normale Anatomie":
        {"region": "distal",   "label": "normal",   "label_id": 0},
    "Komplette UA-Aufnahme mit EB und HG mit Fraktur":
        {"region": "complete", "label": "fracture", "label_id": 1},
    "Komplette UA-Aufnahme mit EB und HG normale Anatomie":
        {"region": "complete", "label": "normal",   "label_id": 0},
    "Proximale UA-Aufnahme mit EB mit Fraktur":
        {"region": "proximal", "label": "fracture", "label_id": 1},
    "Proximale UA-Aufnahme mit EB normale Anatomie":
        {"region": "proximal", "label": "normal",   "label_id": 0},
}


def build_records() -> pd.DataFrame:
    """Walk DATA_ROOT and return a DataFrame of all images with metadata."""
    records = []
    for cat_name, meta in CATEGORIES.items():
        cat_dir = DATA_ROOT / cat_name
        if not cat_dir.exists():
            print(f"WARNING: directory not found: {cat_dir}")
            continue
        for case_dir in sorted(cat_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            images = sorted(
                f for f in case_dir.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg")
            )
            for img_path in images:
                records.append({
                    "path":     str(img_path),
                    "case_id":  case_dir.name,
                    "category": cat_name,
                    "region":   meta["region"],
                    "label":    meta["label"],
                    "label_id": meta["label_id"],
                })
    return pd.DataFrame(
        records,
        columns=["path", "case_id", "category", "region", "label", "label_id"],
    )


def case_level_stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split at CASE level to prevent patient data leakage.

    Stratification key: region × label (6 unique strata).
    Ensures each stratum is proportionally represented in all three splits.

    Two-step approach:
      Step 1 — carve out test (10%)
      Step 2 — split remainder into train (80%) and val (10%)
    """
    cases = (
        df[["case_id", "region", "label", "label_id"]]
        .drop_duplicates("case_id")
        .reset_index(drop=True)
    )
    cases["stratum"] = cases["region"] + "_" + cases["label"]

    # val fraction relative to the train+val pool = 0.10 / 0.90 ≈ 0.111
    val_relative_frac = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)

    train_val, test = train_test_split(
        cases,
        test_size=TEST_FRAC,
        stratify=cases["stratum"],
        random_state=RANDOM_SEED,
    )
    train, val = train_test_split(
        train_val,
        test_size=val_relative_frac,
        stratify=train_val["stratum"],
        random_state=RANDOM_SEED,
    )

    split_map = (
        {cid: "train" for cid in train["case_id"]}
        | {cid: "val"   for cid in val["case_id"]}
        | {cid: "test"  for cid in test["case_id"]}
    )

    df = df.copy()
    df["split"] = df["case_id"].map(split_map)
    return df


def verify_split(df: pd.DataFrame) -> None:
    """Print split statistics and assert zero patient leakage."""
    print("\n── Case counts per split / region / label ────────────────────")
    summary = (
        df.drop_duplicates("case_id")
          .groupby(["split", "region", "label"])["case_id"]
          .count()
          .reset_index()
          .rename(columns={"case_id": "cases"})
    )
    print(summary.to_string(index=False))

    print("\n── Image counts per split ────────────────────────────────────")
    print(df.groupby("split")["path"].count().to_string())

    print("\n── Leakage check ─────────────────────────────────────────────")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ids_a = set(df[df["split"] == a]["case_id"])
        ids_b = set(df[df["split"] == b]["case_id"])
        overlap = ids_a & ids_b
        assert len(overlap) == 0, (
            f"LEAKAGE DETECTED: {len(overlap)} case_ids appear in both '{a}' and '{b}'!"
        )
    print("PASSED — no case_id appears in multiple splits.")


def main() -> None:
    print(f"Scanning: {DATA_ROOT}")
    df = build_records()

    if df.empty:
        found = []
        if DATA_ROOT.exists():
            found = [str(p) for p in DATA_ROOT.iterdir()]
        raise FileNotFoundError(
            f"\nNo images found — DATA_ROOT does not contain any expected category folders.\n"
            f"DATA_ROOT = {DATA_ROOT}\n"
            f"  exists : {DATA_ROOT.exists()}\n"
            f"  contents: {found or '(empty or does not exist)'}\n"
            f"\nExpected sub-folders (exact names):\n"
            + "\n".join(f"  • {k}" for k in CATEGORIES)
            + "\n\nFix: set the DATA_ROOT env var to the correct path, e.g.:\n"
            f"  os.environ['DATA_ROOT'] = '/kaggle/input/<image-slug>/alle Bilder'"
        )

    print(f"Found {len(df):,} images across {df['case_id'].nunique():,} cases")

    df = case_level_stratified_split(df)
    verify_split(df)

    SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_cols = ["path", "case_id", "category", "region", "label", "label_id", "split"]
    df[out_cols].to_csv(SPLITS_CSV, index=False)
    print(f"\nSaved → {SPLITS_CSV}")


if __name__ == "__main__":
    main()
