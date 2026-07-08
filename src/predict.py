"""Inference: a replay URL/ID -> per-turn win probabilities from the saved model.

Reproduces the exact feature transformations of src/features.py for a single
game, using the category levels captured at training time (feature_meta.json)
so species codes line up with what the model learned. Species unseen in
training become missing values, which LightGBM handles natively.

Usage (smoke test):
    python -m src.predict https://replay.pokemonshowdown.com/gen9ou-2645378173
"""

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from src.common import ROOT
from src.features import CATEGORICAL, add_derived
from src.parser import parse_replay

MODEL_PATH = ROOT / "models" / "winprob_lgbm.txt"
META_PATH = ROOT / "models" / "feature_meta.json"
REPLAY_URL = "https://replay.pokemonshowdown.com/{id}.json"


def load_model() -> tuple[lgb.Booster, dict]:
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return booster, meta


def fetch_replay(id_or_url: str) -> dict:
    """Accepts a full replay URL, a bare ID, or an ID with .json/query suffixes."""
    rid = id_or_url.strip().rstrip("/").split("/")[-1].split("?")[0]
    rid = rid.removesuffix(".json")
    resp = requests.get(REPLAY_URL.format(id=rid), timeout=30)
    resp.raise_for_status()
    return resp.json()


def snapshot_features(game: dict, meta: dict) -> pd.DataFrame:
    df = pd.DataFrame(game["snapshots"])
    df["p1_rating"] = game.get("p1_rating") or np.nan
    df["p2_rating"] = game.get("p2_rating") or np.nan
    df = add_derived(df, per_game=False)  # single game: momentum via plain shift
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype("int8")
    for col in CATEGORICAL:
        df[col] = pd.Categorical(df[col], categories=meta["categories"][col])
    return df[meta["features"]]


def predict_game(game: dict, booster: lgb.Booster, meta: dict) -> pd.Series:
    """P(p1 wins) at the start of each turn, indexed by turn number."""
    features = snapshot_features(game, meta)
    probs = booster.predict(features)
    return pd.Series(probs, index=[s["turn"] for s in game["snapshots"]], name="p1_win_prob")


def describe_turn(prev: dict, curr: dict, game: dict) -> list[str]:
    """Human-readable events that happened during the turn between two snapshots."""
    names = {"p1": game["p1_name"], "p2": game["p2_name"]}
    events = []
    for side in ("p1", "p2"):
        n = names[side]
        if curr[f"{side}_fainted"] > prev[f"{side}_fainted"]:
            lost = curr[f"{side}_fainted"] - prev[f"{side}_fainted"]
            events.append(f"{n} lost {'a Pokémon' if lost == 1 else f'{lost} Pokémon'}")
        elif curr[f"{side}_active_species"] != prev[f"{side}_active_species"]:
            events.append(f"{n} switched to {curr[f'{side}_active_species']}")
        if curr[f"{side}_tera_used"] and not prev[f"{side}_tera_used"]:
            events.append(f"{n} Terastallized")
        for hazard, label in [("stealthrock", "Stealth Rock"), ("spikes", "Spikes"),
                              ("toxicspikes", "Toxic Spikes"), ("stickyweb", "Sticky Web")]:
            if curr[f"{side}_hazard_{hazard}"] > prev[f"{side}_hazard_{hazard}"]:
                events.append(f"{label} set against {n}")
    if not events:
        events.append("a big damage swing")
    return events


def key_moments(game: dict, probs: pd.Series, top: int = 4) -> list[dict]:
    """The turns with the largest win-probability swings, with what happened."""
    snaps = game["snapshots"]
    swings = []
    for i in range(len(snaps) - 1):
        delta = probs.iloc[i + 1] - probs.iloc[i]
        swings.append({
            "turn": snaps[i]["turn"],
            "delta": delta,
            "events": describe_turn(snaps[i], snaps[i + 1], game),
        })
    swings.sort(key=lambda s: abs(s["delta"]), reverse=True)
    return sorted(swings[:top], key=lambda s: s["turn"])


if __name__ == "__main__":
    booster, meta = load_model()
    game = parse_replay(fetch_replay(sys.argv[1]))
    probs = predict_game(game, booster, meta)
    print(f"{game['p1_name']} vs {game['p2_name']} — winner: {game[game['winner'] + '_name']}")
    print(f"P(p1 win): turn 1 {probs.iloc[0]:.0%} -> final {probs.iloc[-1]:.0%}")
    for m in key_moments(game, probs):
        print(f"  turn {m['turn']:>3} {m['delta']:+.0%}  {'; '.join(m['events'])}")
