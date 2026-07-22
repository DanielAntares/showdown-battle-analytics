"""Multi-turn deep search (Tier 3): the step() carry-forward primitive stays
self-consistent turn to turn, batched scoring is row-independent, and
deep_search returns a well-formed, sorted ranking."""

import json
from pathlib import Path

from src.parser import parse_replay
from src.predict import load_model, snapshot_features
from src.search import (deep_search, greedy_action, is_over, score_batch, step,
                        top_actions)

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))


def _load(idx=0, up_to_turn=8):
    raw = json.loads(FIXTURES[idx].read_text(encoding="utf-8"))
    return parse_replay(raw, up_to_turn=up_to_turn)


def _assert_consistent(game):
    """Every invariant step() promises to preserve about a game state."""
    for side in ("p1", "p2"):
        mons = game["roster"][side]
        snap = game["snapshots"][-1]
        assert all(0.0 <= m["hp"] <= 1.0 for m in mons)
        assert all(m["fainted"] == (m["hp"] <= 1e-6) or m["hp"] > 0 for m in mons)
        if any(not m["fainted"] for m in mons):
            assert sum(m["active"] for m in mons) == 1  # exactly one active if alive
        # aggregate snapshot fields must match the roster they summarize
        assert abs(snap[f"{side}_hp_total"] - sum(m["hp"] for m in mons)) < 1e-6
        assert snap[f"{side}_fainted"] == sum(m["fainted"] for m in mons)


def test_step_preserves_invariants_over_many_turns():
    game = _load(up_to_turn=3)
    _assert_consistent(game)
    for _ in range(12):
        if is_over(game):
            break
        a, b = greedy_action(game, "p1"), greedy_action(game, "p2")
        if not a or not b:
            break
        game = step(game, {"p1": a, "p2": b})
        _assert_consistent(game)


def test_step_returns_new_state_without_mutating_input():
    game = _load(up_to_turn=6)
    before = json.dumps(game["snapshots"][-1], sort_keys=True, default=str)
    a, b = greedy_action(game, "p1"), greedy_action(game, "p2")
    step(game, {"p1": a, "p2": b})
    after = json.dumps(game["snapshots"][-1], sort_keys=True, default=str)
    assert before == after  # original game dict untouched


def test_is_over_detects_wipe():
    game = _load(up_to_turn=6)
    assert not is_over(game)
    for m in game["roster"]["p2"]:
        m["fainted"], m["hp"] = True, 0.0
    assert is_over(game)


def test_score_batch_is_row_independent():
    """The whole point of batching: scoring N states in one model call must give
    each state exactly the score it would get alone. If momentum shift or category
    inference leaked across rows, a state's score would depend on its neighbours."""
    booster, meta = load_model()
    g = _load(up_to_turn=6)
    a, b = greedy_action(g, "p1"), greedy_action(g, "p2")
    g2 = step(g, {"p1": a, "p2": b})  # a genuinely different position

    together = score_batch([g, g2], "p1", booster, meta)
    alone = [score_batch([g], "p1", booster, meta)[0],
             score_batch([g2], "p1", booster, meta)[0]]
    assert together[0] == alone[0]
    assert together[1] == alone[1]
    assert all(0.0 <= v <= 1.0 for v in together)


def test_score_batch_is_side_symmetric():
    booster, meta = load_model()
    g = _load(up_to_turn=6)
    p1 = score_batch([g], "p1", booster, meta)[0]
    p2 = score_batch([g], "p2", booster, meta)[0]
    # material bonus can nudge both, but the win-prob halves must be complementary
    assert abs((p1 + p2) - 1.0) < 0.12


def test_deep_search_is_well_formed_and_sorted():
    booster, meta = load_model()
    game = _load(up_to_turn=8)
    out = deep_search(game, "p1", booster, meta, depth=2, rollout=2, top_k=2)
    assert len(out) >= 1
    assert set(out.columns) >= {"action", "worst_case", "average", "worst_response"}
    assert out.worst_case.between(0, 1).all()
    assert out.average.between(0, 1).all()
    assert (out.worst_case <= out.average + 1e-9).all()  # worst never beats average
    assert list(out.worst_case) == sorted(out.worst_case, reverse=True)


def test_deep_search_is_deterministic():
    booster, meta = load_model()
    game = _load(up_to_turn=8)
    kw = dict(depth=2, rollout=2, top_k=2)
    a = deep_search(game, "p1", booster, meta, **kw)
    b = deep_search(game, "p1", booster, meta, **kw)
    assert a.equals(b)


def test_deep_search_matches_advisor_action_set():
    """Deep search ranks the same root actions the 1-ply advisor considers."""
    booster, meta = load_model()
    game = _load(up_to_turn=8)
    out = deep_search(game, "p1", booster, meta, depth=1, rollout=1, top_k=3)
    root = {a["label"] for a in top_actions(game, "p1", 5)}
    assert set(out.action) <= root
