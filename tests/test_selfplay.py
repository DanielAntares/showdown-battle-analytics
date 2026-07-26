"""Self-play harness + the two policy tools it measures (mixed root,
determinization), and the Tera-safety cost."""

import numpy as np
import pandas as pd

from src.predict import load_model, snapshot_features
from src.selfplay import (ab_test, mixed_pick, new_game, play_game, pooled_advise,
                          sample_teams)

TEAM_A = ["Great Tusk", "Gholdengo", "Kyurem", "Dragonite", "Kingambit", "Samurott-Hisui"]
TEAM_B = ["Landorus-Therian", "Raging Bolt", "Ogerpon-Wellspring", "Kingambit",
          "Gholdengo", "Great Tusk"]


def test_new_game_is_a_valid_actionable_state():
    from src.advisor import advise_search, player_actions
    booster, meta = load_model()
    g = new_game(TEAM_A, TEAM_B, lead_a=0, lead_b=0)
    assert len(g["roster"]["p1"]) == 6 and len(g["roster"]["p2"]) == 6
    assert sum(m["active"] for m in g["roster"]["p1"]) == 1
    assert g["snapshots"][-1]["p1_hp_total"] == 6.0
    assert player_actions(g, "p1")
    out = advise_search(g, "p1", booster, meta, snapshot_features, pessimism=0.7)
    assert len(out) and out.worst_case.between(0, 1).all()


def test_mixed_pick_respects_epsilon():
    rng = np.random.default_rng(0)
    tbl = pd.DataFrame([{"action": "A", "worst_case": 0.50},
                        {"action": "B", "worst_case": 0.49},
                        {"action": "C", "worst_case": 0.20}])
    # epsilon 0 -> always the top
    assert all(mixed_pick(tbl, rng, 0.0) == "A" for _ in range(10))
    # epsilon 0.02 -> only A or B (C is too far below), never C
    picks = {mixed_pick(tbl, rng, 0.02) for _ in range(50)}
    assert picks <= {"A", "B"} and "C" not in picks


def test_determinization_pools_over_worlds():
    booster, meta = load_model()
    g = new_game(TEAM_A, TEAM_B)
    rng = np.random.default_rng(3)
    pooled = pooled_advise(g, "p1", booster, meta, 0.7, 3, rng, mode="fast")
    assert len(pooled) and pooled.worst_case.between(0, 1).all()
    assert list(pooled.worst_case) == sorted(pooled.worst_case, reverse=True)


def test_tera_costs_a_resource():
    """A Tera action must beat its un-Tera'd sibling by more than TERA_COST to be
    recommended — so it isn't spent for a marginal gain."""
    import src.advisor as A
    booster, meta = load_model()
    g = new_game(["Dragonite"] + TEAM_A[1:], TEAM_B)
    costed = A.advise_search(g, "p1", booster, meta, snapshot_features, pessimism=1.0)
    orig = A.TERA_COST
    try:
        A.TERA_COST = 0.0
        free = A.advise_search(g, "p1", booster, meta, snapshot_features, pessimism=1.0)
    finally:
        A.TERA_COST = orig
    c = costed[costed.action.str.startswith("Tera ")].set_index("action").worst_case
    f = free[free.action.str.startswith("Tera ")].set_index("action").worst_case
    common = c.index.intersection(f.index)
    assert len(common)
    assert (c[common] <= f[common] + 1e-9).all()       # cost never raises a Tera
    assert (c[common] < f[common] - 1e-4).any()         # and lowers at least one


def test_play_game_and_ab_produce_a_result():
    booster, meta = load_model()
    rng = np.random.default_rng(7)
    cfg = {"mode": "fast", "pessimism": 0.7}
    w = play_game(TEAM_A, TEAM_B, cfg, cfg, booster, meta, rng, max_turns=30)
    assert w in ("p1", "p2", None)
    # a tiny A/B runs end to end and returns a well-formed result
    res = ab_test(cfg, cfg, 2, booster, meta, seed=1, max_turns=25)
    assert res["A_wins"] + res["B_wins"] + res["draws"] == 2
    assert 0.0 <= res["A_winrate"] <= 1.0


def test_sample_teams_returns_full_rosters():
    teams = sample_teams(5, np.random.default_rng(0))
    assert len(teams) == 5 and all(len(t) == 6 for t in teams)
