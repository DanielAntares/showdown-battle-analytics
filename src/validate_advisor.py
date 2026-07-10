"""Does the advisor give good advice? Validate it against real player decisions.

For sampled held-out games we reconstruct the state at the start of each
decision turn, ask the advisor for its top action, and compare it to what the
player *actually* did. Two questions:

1. **Agreement by skill.** If the advisor captures good play, stronger players
   should agree with it more often. Agreement rising with Elo is evidence the
   recommendation is meaningful (and confound-light — it doesn't depend on
   outcomes).
2. **Agreement vs. outcome.** Within a rating band, do players win more in the
   games where their choices matched the advisor more often? Suggestive, not
   causal (better positions both cause winning and make the obvious play the
   good one), so read it as correlational.

Forced replacements after a faint are excluded (they aren't decisions).

Usage:
    python -m src.validate_advisor            # default sample
    python -m src.validate_advisor --games 40 # games per Elo band
"""

import argparse
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.advisor import advise_search
from src.common import ROOT, load_config
from src.parser import parse_replay
from src.predict import load_model, snapshot_features
from src.skill_bands import BANDS, LABELS, SEQ_BLUES
from src.train import INK_2, SURFACE, style_axis

DECISION_TURNS = 10  # cap decision turns evaluated per game (speed)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def actual_action(game: dict, side: str, turn: int) -> tuple[str, str] | None:
    """(kind, target) the player chose that turn, or None if forced/none."""
    move = switch = None
    for e in game.get("events", {}).get(turn, []):
        if e["side"] != side:
            continue
        if " used " in e["text"]:
            move = e["text"].split(" used ", 1)[1]
        elif e["text"].startswith("switched to "):  # voluntary only, not "sent out"
            switch = e["text"].split("switched to ", 1)[1]
    if move:
        return ("move", _norm(move))
    if switch:
        return ("switch", _norm(switch))
    return None  # forced replacement or nothing recorded


def advisor_top(game: dict, side: str, booster, meta) -> tuple[str, str] | None:
    out = advise_search(game, side, booster, meta, snapshot_features)
    if not len(out):
        return None
    label = out.iloc[0].action
    if label.startswith("switch to "):
        return ("switch", _norm(label.split("switch to ", 1)[1]))
    return ("move", _norm(label))


def evaluate(games_per_band: int, seed: int = 11) -> pd.DataFrame:
    cfg = load_config()
    gdf = pd.read_parquet(cfg["paths"]["processed"] / "games.parquet")
    gdf = gdf[(gdf.n_turns >= 6) & gdf.winner.notna()]
    cutoff = pd.Timestamp(cfg["test_split_date"]).timestamp()
    gdf = gdf[gdf.uploadtime >= cutoff]  # held-out games only
    booster, meta = load_model()
    raw_dir = cfg["paths"]["raw_replays"]
    rng = np.random.default_rng(seed)

    rows = []
    for (lo, hi), label in zip(BANDS, LABELS):
        band = gdf[gdf.rating.between(lo, hi - 1)]
        sample = band.sample(min(games_per_band, len(band)), random_state=seed)
        for rid, winner in zip(sample.id, sample.winner):
            replay = json.loads((raw_dir / f"{rid}.json").read_text(encoding="utf-8"))
            full = parse_replay(replay)
            turns = rng.choice(range(2, full["n_turns"]),
                               size=min(DECISION_TURNS, full["n_turns"] - 2),
                               replace=False)
            for turn in sorted(int(t) for t in turns):
                state = parse_replay(replay, up_to_turn=turn)  # what the player saw
                for side in ("p1", "p2"):
                    actual = actual_action(full, side, turn)  # what they actually did
                    rec = advisor_top(state, side, booster, meta) if actual else None
                    if actual and rec:
                        rows.append({"band": label, "won": winner == side,
                                     "agree": actual == rec})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise SystemExit("no decisions collected — check replay availability")
    return df.groupby("band", sort=False).apply(
        lambda g: pd.Series({
            "decisions": len(g),
            "agreement": g.agree.mean(),
            "win_rate_when_agree": g.won[g.agree].mean(),
            "win_rate_when_deviate": g.won[~g.agree].mean(),
        }), include_groups=False)


def make_figure(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    ax.bar(summary.index, summary.agreement, color=SEQ_BLUES, width=0.6)
    for i, v in enumerate(summary.agreement):
        ax.annotate(f"{v:.0%}", (i, v), ha="center", xytext=(0, 4),
                    textcoords="offset points", fontsize=10, color=INK_2)
    ax.set_title("Player agreement with the advisor's top pick, by Elo",
                 fontsize=11, loc="left")
    ax.set_xlabel("Elo band")
    ax.set_ylabel("share of decisions matching the advisor")
    ax.set_ylim(0, max(summary.agreement) * 1.25)
    style_axis(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "advisor_validation.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nFigure saved to {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=40, help="games per Elo band")
    args = ap.parse_args()
    df = evaluate(args.games)
    summary = summarize(df)
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    make_figure(summary)


if __name__ == "__main__":
    main()
