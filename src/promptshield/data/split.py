"""Leakage-safe train/val/test split grouped by template."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def grouped_split(
    df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
) -> pd.DataFrame:
    groups = df["template_id"]
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(outer.split(df, groups=groups))
    trainval = df.iloc[trainval_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    train_idx, val_idx = next(inner.split(trainval, groups=trainval["template_id"]))

    split = pd.Series("train", index=df.index)
    split.iloc[test_idx] = "test"
    split.loc[trainval.index[val_idx]] = "val"
    return df.assign(split=split.values)
