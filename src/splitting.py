"""Temporal per-user train/test split for implicit-feedback evaluation.

A random split would leak future listens into training and inflate scores. Splitting
each user's history by timestamp (train on their earlier plays, hold out their most
recent ones) is the standard, defensible approach for implicit-feedback recommenders.
"""
from __future__ import annotations

import pandas as pd


def temporal_split(
    events: pd.DataFrame,
    user_col: str,
    item_col: str,
    timestamp_col: str,
    test_frac: float = 0.2,
    min_events_per_user: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per user, sort by timestamp and hold out the most recent `test_frac` of plays.

    Users with fewer than `min_events_per_user` total plays are dropped entirely —
    too little signal to meaningfully evaluate, and this threshold is reported in
    notebooks/01_data_audit.ipynb so the eval population is transparent.
    """
    # Subset to only the columns needed before sorting — sorting the full wide
    # dataframe (e.g. 14 columns incl. artist_name/country/etc. for the raw LastFM
    # data) to reorder just 13M rows can exhaust memory on a large PyArrow-backed
    # frame; sorting a 3-column slice is far cheaper and the extra columns aren't
    # needed downstream of this function anyway.
    events = events[[user_col, item_col, timestamp_col]].copy()

    counts = events.groupby(user_col).size()
    keep_users = counts[counts >= min_events_per_user].index
    events = events[events[user_col].isin(keep_users)].sort_values(
        [user_col, timestamp_col], kind="mergesort"
    )

    train_parts, test_parts = [], []
    for _, group in events.groupby(user_col, sort=False):
        n_test = max(1, int(len(group) * test_frac))
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])

    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, test


def held_out_as_dict(test: pd.DataFrame, user_col: str, item_col: str) -> dict:
    return test.groupby(user_col)[item_col].apply(set).to_dict()
