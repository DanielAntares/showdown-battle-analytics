"""Swing classification for the skill-band analysis."""

import pandas as pd

from src.skill_bands import game_stats


def test_game_stats_classification():
    game = {
        "n_turns": 5,
        "snapshots": [{"turn": t} for t in range(1, 6)],
        "events": {2: [{"side": "p1", "text": "x took a critical hit", "luck": True}]},
    }
    # turn 1->2 blunder (-30), 2->3 big swing but crit that turn (luck), 3->4 mistake,
    # 4->5 quiet
    probs = pd.Series([0.50, 0.20, 0.50, 0.33, 0.30], index=range(1, 6))
    s = game_stats(game, probs)
    assert s["blunders"] == 1
    assert s["luck_swings"] == 1
    assert s["mistakes"] == 1
    assert 0 < s["volatility"] < 1
    assert s["turns"] == 5
