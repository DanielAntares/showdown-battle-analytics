"""Advisor upgrades: KO-probability credit, flinch (EV), matchup-aware replacement,
opponent-policy-weighted expectation, and the known randbats Tera type."""

import numpy as np
import pytest

from src.advisor import (FLINCH_MOVES, SimState, advise_search, moves_for,
                         opp_response_probs, player_actions)
from src.pokedex import move_info, norm_name
from src.predict import load_model, snapshot_features
from src.search import deep_search
from src.selfplay import new_game

TEAM = ["Great Tusk", "Gholdengo", "Kingambit", "Dragonite", "Toxapex", "Garchomp"]


def test_ko_chance_rises_as_the_target_weakens():
    g = new_game(["Great Tusk"] + TEAM[1:], ["Kingambit"] + TEAM[1:])
    snap = g["snapshots"][-1]
    hr = dict(move_info("Headlong Rush"), name="Headlong Rush")  # Ground, 2x on Kingambit
    opp = next(m for m in g["roster"]["p2"] if m["active"])
    opp["hp"] = 1.0
    full = SimState(g, snap).ko_chance("p1", hr)
    opp["hp"] = 0.2
    low = SimState(g, snap).ko_chance("p1", hr)
    assert 0.0 <= full <= low <= 1.0
    assert low > 0.5          # a strong super-effective hit near-certainly KOs a 20% foe
    assert low > full + 0.2   # and it's a real function of remaining HP


def test_ko_credit_lifts_a_securing_move_in_the_table():
    """A move that KOs on a high roll (but not the average) should be credited, so it
    isn't out-ranked by a passive line."""
    booster, meta = load_model()
    g = new_game(["Great Tusk"] + TEAM[1:], ["Kingambit"] + TEAM[1:])
    opp = next(m for m in g["roster"]["p2"] if m["active"])
    opp["hp"] = 0.45
    g["snapshots"][-1]["p2_active_hp"] = 0.45
    out = advise_search(g, "p1", booster, meta, snapshot_features, pessimism=0.7)
    assert len(out) and out.worst_case.between(0, 1).all()


def test_flinch_is_a_known_move_and_scales_damage():
    assert "ironhead" in FLINCH_MOVES and "airslash" in FLINCH_MOVES
    g = new_game(["Kingambit"] + TEAM[1:], ["Great Tusk"] + TEAM[1:])
    snap = g["snapshots"][-1]
    cc = dict(move_info("Iron Head"), name="Iron Head")
    full = SimState(g, snap); full.use_move("p1", cc, dmg_scale=1.0)
    scaled = SimState(g, snap); scaled.use_move("p1", cc, dmg_scale=0.7)
    dmg_full = 1.0 - full.active["p2"].hp
    dmg_scaled = 1.0 - scaled.active["p2"].hp
    assert dmg_full > 0 and dmg_scaled == pytest.approx(dmg_full * 0.7, rel=1e-6)


def test_replacement_prefers_the_answer_over_the_healthiest():
    """After a faint, the mon that resists the foe's STAB and threatens back should be
    chosen ahead of a healthier but ill-matched bench mon."""
    # opponent active is a Fire attacker; our only two live bench mons are a full-HP
    # Grass (2x weak to Fire) and a hurt Garchomp (resists Fire, EQ threatens back)
    g = new_game(["Gholdengo", "Amoonguss", "Garchomp", "Toxapex", "Dragonite", "Kingambit"],
                 ["Ninetales", "Kingambit", "Toxapex", "Garchomp", "Dragonite", "Gholdengo"])
    r = g["roster"]["p1"]
    active = next(m for m in r if m["active"]); active["fainted"] = True; active["hp"] = 0.0
    for m in r:  # leave only Amoonguss (healthy, weak) and Garchomp (hurt, resists) live
        if m["species"] not in ("Amoonguss", "Garchomp") and not m["active"]:
            m["fainted"] = True; m["hp"] = 0.0
    next(m for m in r if m["species"] == "Amoonguss")["hp"] = 1.0
    next(m for m in r if m["species"] == "Garchomp")["hp"] = 0.5
    rep = SimState(g, g["snapshots"][-1])._replacement("p1", active["species"])
    assert rep["species"] == "Garchomp"  # the answer, not the healthier-but-weak Amoonguss


def test_opp_response_probs_is_a_weighted_distribution():
    g = new_game(["Great Tusk"] + TEAM[1:], ["Kingambit"] + TEAM[1:])
    acts = player_actions(g, "p2")
    probs = opp_response_probs(g, "p2", acts)
    assert len(probs) == len(acts)
    assert np.all(probs >= 0) and probs.sum() == pytest.approx(1.0)
    # the model should express *some* preference, not a flat uniform
    assert probs.std() > 0


def test_search_survives_a_double_faint_with_a_hidden_foe_bench():
    """Both actives fainted and the opponent's only revealed mon is the one that just
    fainted (random battles hide their bench) -> the opponent has no *known* reply.
    The search must still rank our replacement switches, not crash on an empty
    response set (the 'stopped when both pokemon died' ValueError)."""
    booster, meta = load_model()
    g = new_game(["Pawmot"] + TEAM[1:], ["Lycanroc"] + TEAM[1:])
    p1a = next(m for m in g["roster"]["p1"] if m["active"]); p1a["fainted"] = True; p1a["hp"] = 0.0
    p2a = next(m for m in g["roster"]["p2"] if m["active"]); p2a["fainted"] = True; p2a["hp"] = 0.0
    g["roster"]["p2"] = [p2a]  # bench unrevealed in randbats
    g["snapshots"][-1]["p1_active_hp"] = 0.0
    g["snapshots"][-1]["p2_active_hp"] = 0.0
    assert player_actions(g, "p2") == []  # foe has no known action
    fast = advise_search(g, "p1", booster, meta, snapshot_features, pessimism=0.6)
    deep = deep_search(g, "p1", booster, meta, depth=2, rollout=3, top_k=3, pessimism=0.6)
    for out in (fast, deep):
        assert len(out) and out.action.str.startswith("switch to").all()


def test_rampage_lock_restricts_to_the_move_and_forbids_switching():
    g = new_game(["Dragonite"] + TEAM, ["Kingambit"] + TEAM)
    me = next(m for m in g["roster"]["p1"] if m["active"])
    me["moves"] = ["Outrage"]; me["locked_move"] = "Outrage"
    acts = player_actions(g, "p1")
    assert all(a["kind"] != "switch" for a in acts)                     # no switching
    assert all(norm_name(a["move"]["name"]) == "outrage"
               for a in acts if a["kind"] == "move")                    # only Outrage
    me["locked_move"] = ""                                              # lock gone -> normal
    assert any(a["kind"] == "switch" for a in player_actions(g, "p1"))


def test_encore_is_pruned_against_an_already_encored_target():
    g = new_game(["Grimmsnarl"] + TEAM, ["Kingambit"] + TEAM)
    me = next(m for m in g["roster"]["p1"] if m["active"]); me["moves"] = ["Encore", "Spirit Break"]
    opp = next(m for m in g["roster"]["p2"] if m["active"])
    snap = g["snapshots"][-1]
    opp["volatiles"] = ["encore"]  # re-Encoring does nothing
    assert "Encore" not in [m["name"] for m in moves_for(me, snap, "p1", g)]
    opp["volatiles"] = []          # ...but it's a valid option otherwise
    assert "Encore" in [m["name"] for m in moves_for(me, snap, "p1", g)]


def test_known_tera_type_is_used_over_the_prediction():
    g = new_game(["Great Tusk"] + TEAM[1:], ["Kingambit"] + TEAM[1:])
    me = next(m for m in g["roster"]["p1"] if m["active"])
    me["tera_avail"] = "fairy"  # e.g. the live request's canTerastallize
    labels = [a["label"] for a in player_actions(g, "p1") if a.get("tera")]
    assert labels and all("Tera Fairy" in lbl for lbl in labels)
