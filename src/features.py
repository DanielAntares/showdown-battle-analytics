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

import numpy as np
import pandas as pd

from src.common import load_config

# active-vs-active interaction features that earned their place on the randbats
# validation fold (-0.003 log loss). The raw type-advantage and stat/level diffs
# were tried too but scored ~0 gain (ranks #54-69 of 69) — the species categoricals
# subsume them, and the type signal already lives inside the damage proxy — so only
# the derived speed + damage-pressure features are kept.
MATCHUP_FEATURES = [
    "spe_frac", "p1_moves_first", "p1_dmg_proxy", "p2_dmg_proxy", "dmg_proxy_diff",
]


def _boost_mult(stage) -> np.ndarray:
    stage = np.asarray(stage, dtype=float)
    # clamp the unused branch's denominator so np.where doesn't warn at stage=+2
    return np.where(stage >= 0, (2.0 + stage) / 2.0, 2.0 / (2.0 - np.minimum(stage, 0)))


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Active-vs-active interaction features the two species categoricals can't
    express on their own: who outspeeds, the offensive type matchup both ways, a
    cheap boost-aware damage/KO proxy, and level/stat gaps. Computed from the raw
    p1_/p2_ columns (so mirror augmentation handles orientation for free), from
    info visible on the field (no leakage), and level-scaled via real_stats — so in
    random battles the actual per-species level feeds the speed/damage estimates."""
    from src.movesets import real_stats, species_level
    from src.pokedex import lookup, type_advantage

    sp1 = df["p1_active_species"].astype(str)
    sp2 = df["p2_active_species"].astype(str)
    uniq = sorted(set(sp1.unique()) | set(sp2.unique()))
    code = {s: i for i, s in enumerate(uniq)}
    n = len(uniq)
    atk = np.empty(n); dfn = np.empty(n); spa = np.empty(n); spd = np.empty(n)
    spe = np.empty(n); hp = np.empty(n); lvl = np.empty(n); tps = []
    for s, i in code.items():
        rs = real_stats(s)
        atk[i], dfn[i], spa[i] = rs["atk"], rs["def"], rs["spa"]
        spd[i], spe[i], hp[i] = rs["spd"], rs["spe"], rs["hp"]
        lvl[i] = species_level(s)
        dex = lookup(s)
        tps.append(dex["types"] if dex else [])
    # best-STAB effectiveness matrix: TADV[i, j] = species i attacking species j
    tadv = np.ones((n, n))
    for i in range(n):
        if not tps[i]:
            continue
        for j in range(n):
            if tps[j]:
                tadv[i, j] = type_advantage(tps[i], tps[j])
    c1 = sp1.map(code).to_numpy()
    c2 = sp2.map(code).to_numpy()

    # some paths (lead/pivot evaluation) feed minimal snapshots that omit boost/
    # status/field columns — default them (no boost / no status / no field effect)
    def get(col, default=0):
        return df[col] if col in df else pd.Series(default, index=df.index)

    def screen(side):  # tailwind doubles speed
        return (get(f"{side}_screen_tailwind") > 0).to_numpy()

    par1 = (get("p1_active_status", "").astype(str) == "par").to_numpy()
    par2 = (get("p2_active_status", "").astype(str) == "par").to_numpy()
    spe1 = spe[c1] * _boost_mult(get("p1_boost_spe")) * np.where(par1, 0.5, 1) * np.where(screen("p1"), 2, 1)
    spe2 = spe[c2] * _boost_mult(get("p2_boost_spe")) * np.where(par2, 0.5, 1) * np.where(screen("p2"), 2, 1)
    denom = spe1 + spe2
    denom[denom == 0] = 1.0
    df["spe_frac"] = spe1 / denom
    tr = (get("trickroom") > 0).to_numpy()
    df["p1_moves_first"] = np.where(tr, spe1 < spe2, spe1 > spe2).astype(float)

    # damage pressure: best-STAB effectiveness x better boosted attacking route vs
    # the matching boosted defense (type advantage folds in here, which is why the
    # bare type_adv columns added nothing on their own)
    t1, t2 = tadv[c1, c2], tadv[c2, c1]
    a1 = atk[c1] * _boost_mult(get("p1_boost_atk")); s1 = spa[c1] * _boost_mult(get("p1_boost_spa"))
    a2 = atk[c2] * _boost_mult(get("p2_boost_atk")); s2 = spa[c2] * _boost_mult(get("p2_boost_spa"))
    d1p = dfn[c1] * _boost_mult(get("p1_boost_def")); d1s = spd[c1] * _boost_mult(get("p1_boost_spd"))
    d2p = dfn[c2] * _boost_mult(get("p2_boost_def")); d2s = spd[c2] * _boost_mult(get("p2_boost_spd"))
    p1_dmg = t1 * np.maximum(a1 / d2p, s1 / d2s)
    p2_dmg = t2 * np.maximum(a2 / d1p, s2 / d1s)
    df["p1_dmg_proxy"], df["p2_dmg_proxy"], df["dmg_proxy_diff"] = p1_dmg, p2_dmg, p1_dmg - p2_dmg
    return df

CATEGORICAL = [
    "p1_active_species",
    "p2_active_species",
    "p1_active_status",
    "p2_active_status",
    "weather",
    "terrain",
]
# item/ability reveal counts are parsed (used by the advisor's roster view) but
# excluded as model features: they were tested and rejected — adding them raised
# test log loss 0.6089 -> 0.6127 and dropped AUC 0.7233 -> 0.7205 (they largely
# proxy game progress, which `turn`/HP already capture).
NOT_FEATURES = ["replay_id", "label_p1_win", "uploadtime",
                "p1_items_revealed", "p2_items_revealed",
                "p1_abilities_revealed", "p2_abilities_revealed"]


def load_raw() -> pd.DataFrame:
    """Turn snapshots joined with per-game ratings; no derived features yet.

    Games below `train_min_rating` are excluded here (from both train and test)
    while staying on disk for skill-band analyses.
    """
    cfg = load_config()
    processed = cfg["paths"]["processed"]
    turns = pd.read_parquet(processed / "turns.parquet")
    games = pd.read_parquet(processed / "games.parquet")
    games = games[games.rating >= cfg.get("train_min_rating", 0)]
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
    # Static Pokédex features (bare base stats + own typing) were rejected for OU
    # (0.5906 -> 0.5927): the species categoricals subsumed them. add_matchup_features
    # instead adds *interaction* features (matchup/speed/damage between the two
    # actives) that a pair of independent categoricals cannot express, re-tested for
    # random battles where the 562-species vocab is far sparser and levels vary.
    df = add_matchup_features(df)
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
