"""How does skill level change the shape of a game?

Samples games from each Elo band (including the sub-1300 games retained on
disk), replays them through the win-prob model, and measures:

* volatility — mean absolute win-prob change per turn
* blunder rate — swings of >=25 points not explained by a crit/miss that turn
* luck swings — big swings where the log shows a crit or miss

Usage:
    python -m src.skill_bands            # prints the table, saves the figure
"""

import json

import matplotlib.pyplot as plt
import pandas as pd

from src.common import ROOT, load_config
from src.parser import parse_replay
from src.predict import load_model, predict_game
from src.train import GRID, INK_2, MUTED, SURFACE, style_axis

BANDS = [(1100, 1300), (1300, 1500), (1500, 1700), (1700, 2100)]
LABELS = ["1100–1299", "1300–1499", "1500–1699", "1700+"]
SEQ_BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]  # ordinal ramp (dataviz)
GAMES_PER_BAND = 600


def game_stats(game: dict, probs) -> dict:
    luck_turns = {t for t, evs in game["events"].items() if any(e["luck"] for e in evs)}
    turns = [s["turn"] for s in game["snapshots"]]
    blunders = mistakes = luck_swings = 0
    deltas = []
    for i in range(len(probs) - 1):
        d = abs(probs.iloc[i + 1] - probs.iloc[i])
        deltas.append(d)
        if turns[i] in luck_turns:
            luck_swings += d >= 0.15
        elif d >= 0.25:
            blunders += 1
        elif d >= 0.15:
            mistakes += 1
    return {"volatility": sum(deltas) / len(deltas) if deltas else 0.0,
            "blunders": blunders, "mistakes": mistakes, "luck_swings": luck_swings,
            "turns": game["n_turns"]}


def analyze(games_per_band: int = GAMES_PER_BAND) -> pd.DataFrame:
    cfg = load_config()
    games = pd.read_parquet(cfg["paths"]["processed"] / "games.parquet")
    games = games[(games.n_turns >= 5) & games.winner.notna()]
    booster, meta = load_model()
    raw_dir = cfg["paths"]["raw_replays"]

    rows = []
    for (lo, hi), label in zip(BANDS, LABELS):
        band = games[games.rating.between(lo, hi - 1)]
        sample = band.sample(min(games_per_band, len(band)), random_state=7)
        for rid in sample.id:
            replay = json.loads((raw_dir / f"{rid}.json").read_text(encoding="utf-8"))
            game = parse_replay(replay)
            probs = predict_game(game, booster, meta)
            rows.append({"band": label, **game_stats(game, probs)})
    df = pd.DataFrame(rows)
    return df.groupby("band", sort=False).agg(
        games=("band", "size"), volatility=("volatility", "mean"),
        blunders_per_game=("blunders", "mean"), mistakes_per_game=("mistakes", "mean"),
        luck_swings_per_game=("luck_swings", "mean"), mean_turns=("turns", "mean"))


def make_figure(summary: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, col, title, fmt in (
            (ax1, "blunders_per_game", "Unforced major swings (blunders) per game", "{:.2f}"),
            (ax2, "volatility", "Win-prob volatility (mean |Δ| per turn)", "{:.1%}")):
        vals = summary[col]
        ax.bar(summary.index, vals, color=SEQ_BLUES, width=0.62)
        for i, v in enumerate(vals):
            ax.annotate(fmt.format(v), (i, v), ha="center", xytext=(0, 4),
                        textcoords="offset points", fontsize=10, color=INK_2)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("Elo band")
        style_axis(ax)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "skill_bands.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nFigure saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    summary = analyze()
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    make_figure(summary)
