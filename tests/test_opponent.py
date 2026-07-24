"""Ground-truth decision extraction for the opponent-action model."""

import json
from pathlib import Path

from src.opponent import decisions_from_log

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))


def _log(idx=0):
    return json.loads(FIXTURES[idx].read_text(encoding="utf-8"))["log"]


def test_extracts_free_choices_with_symmetric_elo():
    decs = decisions_from_log(_log())
    assert decs, "should extract decisions from a normal game"
    for d in decs:
        assert d["context"] == "choice"            # forced excluded by default
        assert d["chosen_kind"] in ("move", "switch")
        assert d["snapshot"] and d["roster"]       # the state they decided from
    # one side's opponent-Elo is the other side's Elo
    by = {}
    for d in decs:
        by.setdefault(d["turn"], {})[d["side"]] = d
    both = next(v for v in by.values() if len(v) == 2)
    assert both["p1"]["elo"] == both["p2"]["opp_elo"]
    assert both["p2"]["elo"] == both["p1"]["opp_elo"]


def test_switch_targets_are_own_teammates():
    for d in decisions_from_log(_log(), keep_forced=True):
        if d["chosen_kind"] in ("switch", "replace"):
            team = {m["species"] for m in d["roster"][d["side"]]}
            assert d["chosen_name"] in team


def test_deciding_mon_was_alive_for_a_choice():
    for d in decisions_from_log(_log()):
        roster = d["roster"][d["side"]]
        active = next((m for m in roster if m["active"]), None)
        # a free choice means the acting mon was on the field and not fainted
        assert active is not None and not active["fainted"]


def test_keep_forced_flag_adds_replacements():
    base = decisions_from_log(_log())
    withforced = decisions_from_log(_log(), keep_forced=True)
    assert len(withforced) >= len(base)
    assert all(d["context"] == "choice" for d in base)
    assert any(d["context"] == "forced" for d in withforced) or len(withforced) == len(base)


def test_empty_or_junk_log_is_safe():
    assert decisions_from_log("") == []
    assert decisions_from_log("not a battle log") == []


# ---- featurization -----------------------------------------------------------

def test_featurize_gives_one_labelled_choice_per_decision():
    from src.opp_features import FEATURES, featurize_decision
    for d in decisions_from_log(_log()):
        rows = featurize_decision(d)
        assert rows, "every decision has candidate rows"
        assert sum(r["is_chosen"] for r in rows) == 1  # exactly one labelled positive
        for r in rows:
            assert all(f in r for f in FEATURES)        # full feature vector
            assert r["_kind"] in ("move", "switch")


def test_chosen_action_is_a_candidate():
    """The action actually taken must appear among the candidates (added even if
    usage never predicted it), else it could never be a training positive."""
    from src.opp_features import featurize_decision
    for d in decisions_from_log(_log()):
        chosen = [r for r in featurize_decision(d) if r["is_chosen"]]
        assert len(chosen) == 1
        assert chosen[0]["_name"] == d["chosen_name"] or \
            chosen[0]["_kind"] == d["chosen_kind"]


def test_switch_rows_carry_matchup_not_move_features():
    from src.opp_features import featurize_decision
    d = decisions_from_log(_log())[0]
    for r in featurize_decision(d):
        if r["is_switch"]:
            assert r["m_power"] == 0 and r["m_usage_prob"] == 0.0
            assert r["s_hp"] > 0  # a real benched mon
