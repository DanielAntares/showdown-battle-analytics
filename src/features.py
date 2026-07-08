"""Assemble the model-ready table from the processed parquet files.

Joins per-game metadata (ratings, upload time) onto the turn snapshots, adds
differential and momentum features, encodes categoricals, and provides the
time-based split: games uploaded on/after `test_split_date` are the test set,
so the model is always evaluated on games strictly newer than anything it
trained on.

`mirror_raw` produces the p1<->p2 reflection of a raw table (label flipped),
used to augment training data — every position is also seen from the other
player's seat, which doubles the sample and enforces symmetry.
"""

import pandas as pd

from src.common import load_config

CATEGORICAL = [
    "p1_active_species",
    "p2_active_species",
    "p1_active_status",
    "p2_active_status",
    "weather",
    "terrain",
]
NOT_FEATURES = ["replay_id", "label_p1_win", "uploadtime"]


def load_raw() -> pd.DataFrame:
    """Turn snapshots joined with per-game ratings; no derived features yet."""
    cfg = load_config()
    processed = cfg["paths"]["processed"]
    turns = pd.read_parquet(processed / "turns.parquet")
    games = pd.read_parquet(processed / "games.parquet")
    df = turns.merge(
        games[["id", "p1_rating", "p2_rating", "uploadtime"]],
        left_on="replay_id",
        right_on="id",
    ).drop(columns="id")
    return df


def mirror_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Swap the two players' columns and flip the label."""
    swapped = {}
    for col in raw.columns:
        if col.startswith("p1_"):
            swapped[col] = "p2_" + col[3:]
        elif col.startswith("p2_"):
            swapped[col] = "p1_" + col[3:]
    out = raw.rename(columns=swapped)[raw.columns]
    out["label_p1_win"] = ~raw["label_p1_win"]
    return out


def add_derived(df: pd.DataFrame, per_game: bool = True) -> pd.DataFrame:
    """Differentials + momentum. Set per_game=False for a single-game frame."""
    df["rating_diff"] = (df.p1_rating - df.p2_rating).fillna(0)
    df["rating_mean"] = (df.p1_rating + df.p2_rating) / 2
    df["hp_diff"] = df.p1_hp_total - df.p2_hp_total
    df["fainted_diff"] = df.p1_fainted - df.p2_fainted
    df["healthy_diff"] = df.p1_healthy - df.p2_healthy
    hp = df["hp_diff"]
    grouped = df.groupby("replay_id")["hp_diff"] if per_game else hp
    df["hp_momentum_1"] = (hp - grouped.shift(1)).fillna(0)
    df["hp_momentum_3"] = (hp - grouped.shift(3)).fillna(0)
    # Explicit Pokédex features (base stats of the actives, STAB type advantage
    # via src/pokedex.py) were tested and rejected on validation: 0.5906 -> 0.5927
    # log loss overall, worse even on rare-species turns (0.6343 -> 0.6415) — the
    # species categoricals already subsume stats/typing for anything seen in
    # training. Don't re-add without beating that bar; src/pokedex.py stays for Phase 5.
    return df


def build_features(
    raw: pd.DataFrame, levels: dict | None = None, per_game: bool = True
) -> tuple[pd.DataFrame, list[str], dict]:
    """Derived features + encodings. Pass `levels` to reuse category mappings
    (mirrored/augmented frames and inference must share the training levels)."""
    df = add_derived(raw.copy(), per_game=per_game)
    df = df.drop(columns=["p1_rating", "p2_rating"])
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype("int8")
    if levels is None:
        # species/status levels shared across the p1/p2 twin columns so the
        # same Pokémon gets the same code on either side of a mirrored row
        species = sorted(set(df.p1_active_species) | set(df.p2_active_species))
        status = sorted(set(df.p1_active_status) | set(df.p2_active_status))
        levels = {
            "p1_active_species": species, "p2_active_species": species,
            "p1_active_status": status, "p2_active_status": status,
            "weather": sorted(df.weather.unique()),
            "terrain": sorted(df.terrain.unique()),
        }
    for col in CATEGORICAL:
        df[col] = pd.Categorical(df[col], categories=levels[col])
    features = [c for c in df.columns if c not in NOT_FEATURES]
    return df, features, levels


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    """Convenience: raw -> full feature table (original orientation only)."""
    df, features, _ = build_features(load_raw())
    return df, features


def time_split(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: True for test rows (games newer than the split date)."""
    cutoff = pd.Timestamp(load_config()["test_split_date"]).timestamp()
    return df.uploadtime >= cutoff
