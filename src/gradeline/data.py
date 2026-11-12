"""Loading, cleaning and splitting the vinho verde physicochemical tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

RAW_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
FEATURES = [c.replace(" ", "_") for c in RAW_FEATURES] + ["is_red"]

# Grades 3 and 9 are too rare to model on their own (30 and 5 lots), so the
# scale is collapsed to five ordered classes.
GRADE_LABELS = ["<=4", "5", "6", "7", ">=8"]
LOW_TIER_MAX = 1  # class index 1 is grade 5; classes 0 and 1 form the low tier


def collapse_grade(quality: pd.Series | np.ndarray) -> np.ndarray:
    """Map raw 3..9 panel scores to class indices 0..4."""
    q = np.asarray(quality)
    return np.clip(q, 4, 8) - 4


def load_raw(data_dir: Path | None = None) -> pd.DataFrame:
    """Read both csv files and stack them with an ``is_red`` flag."""
    d = Path(data_dir) if data_dir else DATA_DIR
    frames = []
    for kind, flag in (("red", 1), ("white", 0)):
        df = pd.read_csv(d / f"winequality-{kind}.csv", sep=";")
        df.columns = [c.replace(" ", "_") for c in df.columns]
        df["is_red"] = flag
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact repeated rows. The same lot appears several times in the files."""
    return df.drop_duplicates().reset_index(drop=True)


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the feature frame and the collapsed grade vector."""
    return df[FEATURES].copy(), collapse_grade(df["quality"].to_numpy())


def split(df: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    """60/20/20 split, stratified on grade and colour."""
    strata = collapse_grade(df["quality"]).astype(str) + "_" + df["is_red"].astype(str)
    train, rest = train_test_split(df, test_size=0.4, stratify=strata, random_state=seed)
    strata_rest = strata.loc[rest.index]
    val, test = train_test_split(rest, test_size=0.5, stratify=strata_rest, random_state=seed)
    return {k: v.reset_index(drop=True) for k, v in
            {"train": train, "val": val, "test": test}.items()}
