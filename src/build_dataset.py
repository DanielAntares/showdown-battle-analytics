"""Parse all raw replays into a turn-level modeling dataset.

Output in data/processed/:
  turns.parquet — one row per turn-start snapshot, labeled with the game outcome
  games.parquet — one row of metadata per replay (joins, time-based splits)
  teams.parquet — one row per (replay, side, species): full team rosters,
                  the training data for teammate inference (Phase 5)

Usage:
    python -m src.build_dataset
"""

import json

import pandas as pd
from tqdm import tqdm

from src.common import load_config
from src.parser import parse_replay


def build() -> None:
    cfg = load_config()
    raw_dir = cfg["paths"]["raw_replays"]
    out_dir = cfg["paths"]["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    turn_rows, game_rows, team_rows, failures = [], [], [], []
    files = sorted(raw_dir.glob("*.json"))
    for path in tqdm(files, desc="parsing", unit="replay"):
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
            game = parse_replay(replay)
        except Exception as exc:  # noqa: BLE001 — collect, report, move on
            failures.append((path.name, repr(exc)))
            continue
        if game["winner"] is None or game["n_turns"] == 0:
            continue  # ties, forfeits before turn 1, or unfinished logs
        snapshots = game.pop("snapshots")
        teams = game.pop("teams")
        game.pop("events")  # display-time info; not part of the training tables
        game["uploadtime"] = replay.get("uploadtime")
        game_rows.append(game)
        for side, roster in teams.items():
            for species in roster:
                team_rows.append(
                    {"replay_id": game["id"], "side": side, "species": species,
                     "uploadtime": game["uploadtime"]}
                )
        for snap in snapshots:
            turn_rows.append(
                {
                    "replay_id": game["id"],
                    "label_p1_win": game["winner"] == "p1",
                    **snap,
                }
            )

    games = pd.DataFrame(game_rows)
    turns = pd.DataFrame(turn_rows)
    teams = pd.DataFrame(team_rows)
    games.to_parquet(out_dir / "games.parquet", index=False)
    turns.to_parquet(out_dir / "turns.parquet", index=False)
    teams.to_parquet(out_dir / "teams.parquet", index=False)

    print(f"\nParsed {len(games)} games -> {len(turns)} turn rows "
          f"({len(failures)} failures, {len(files) - len(games) - len(failures)} skipped)")
    if len(games):
        print(f"p1 win rate: {games['winner'].eq('p1').mean():.3f}")
        print(f"turns per game: mean {games['n_turns'].mean():.1f}, "
              f"median {games['n_turns'].median():.0f}, max {games['n_turns'].max()}")
    for name, err in failures[:10]:
        print(f"  FAIL {name}: {err}")


if __name__ == "__main__":
    build()
