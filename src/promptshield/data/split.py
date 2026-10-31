"""Train/val/test splits.

* ``grouped_split``: NER data. Rows generated from the same template share a
  template_id and must stay in one split, otherwise test scores are inflated.
* ``stratified_split``: secret-detector data. Values are random so there is no
  value leakage; stratifying by generator ``kind`` keeps every kind represented
  in each split. Generalisation to unseen formats is measured separately on the
  out-of-distribution challenge set.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


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


def stratified_split(
    df: pd.DataFrame, by: str, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
) -> pd.DataFrame:
    trainval, test = train_test_split(df.index, test_size=test_size, stratify=df[by], random_state=seed)
    train, val = train_test_split(
        trainval, test_size=val_size / (1 - test_size), stratify=df.loc[trainval, by], random_state=seed
    )
    split = pd.Series("train", index=df.index)
    split.loc[val] = "val"
    split.loc[test] = "test"
    return df.assign(split=split.values)
