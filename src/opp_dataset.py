"""Build the opponent-action training table from the raw replay corpus.

Walks every replay, extracts each free decision, featurizes its candidate
actions, and writes train/test parquet split on upload time (the same temporal
split as the win-prob model — the model is only ever tested on games newer than
any it trained on). Each row carries its decision `group` so the ranker knows
which candidates compete, plus `is_chosen`, `predicted`, and the opponent Elo
for per-band evaluation.

Usage:
    python -m src.opp_dataset            # full corpus
    python -m src.opp_dataset 8000       # first N games (quick iteration)
"""

import glob
import json
import sys
import time

import pandas as pd

from src.common import ROOT, load_config
from src.opp_features import FEATURES, featurize_decision
from src.opponent import decisions_from_log

META = ["group", "is_chosen", "predicted", "kind", "elo", "opp_elo", "is_test"]


def _rows_for_game(raw: dict, cutoff: float, gid: int):
    is_test = (raw.get("uploadtime") or 0) >= cutoff
    rid = raw.get("id", "")
    out = []
    for d in decisions_from_log(raw["log"]):
        group = f"{rid}#{d['turn']}#{d['side']}"
        for r in featurize_decision(d):
            out.append({**{k: r[k] for k in FEATURES},
                        "group": group, "is_chosen": r["is_chosen"],
                        "predicted": r["predicted"], "kind": r["_kind"],
                        "elo": d["elo"], "opp_elo": d["opp_elo"], "is_test": is_test})
    return out


def build(max_games: int | None = None) -> pd.DataFrame:
    cfg = load_config()
    cutoff = pd.Timestamp(cfg["test_split_date"]).timestamp()
    files = sorted(glob.glob(str(cfg["paths"]["raw_replays"] / "*.json")))
    if max_games:
        files = files[:max_games]

    rows, t0, kept = [], time.time(), 0
    for i, f in enumerate(files):
        try:
            raw = json.loads(open(f, encoding="utf-8").read())
            game_rows = _rows_for_game(raw, cutoff, i)
        except Exception:
            continue
        if game_rows:
            rows.extend(game_rows)
            kept += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(files)} games · {len(rows)} rows · "
                  f"{time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    for c in FEATURES:  # shrink: floats->float32, small ints stay ints
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    print(f"built {len(df)} rows from {kept} games in {time.time() - t0:.0f}s "
          f"(test rows: {int(df.is_test.sum())})")
    return df


def main():
    max_games = int(sys.argv[1]) if len(sys.argv) > 1 else None
    df = build(max_games)
    out = ROOT / "data" / "processed" / "opp_decisions.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
