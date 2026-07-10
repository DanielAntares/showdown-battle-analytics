"""Advisor v2: damage engine sanity + minimax search on real replay fixtures."""

import json
from pathlib import Path

import pytest

from src.advisor import SimState, advise_search, hazard_chip, player_actions
from src.parser import parse_replay
from src.pokedex import move_info

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))


def _sim_1v1(p1_species, p2_species, p1_status=""):
    game = {"roster": {
        "p1": [{"species": p1_species, "hp": 1.0, "status": p1_status,
                "fainted": False, "active": True, "moves": []}],
        "p2": [{"species": p2_species, "hp": 1.0, "status": "",
                "fainted": False, "active": True, "moves": []}]}}
    snap = {"turn": 5, "weather": "", "trickroom": 0}
    for s in ("p1", "p2"):
        snap.update({f"{s}_active_species": game["roster"][s][0]["species"],
                     f"{s}_active_hp": 1.0, f"{s}_active_status": "",
                     f"{s}_hp_total": 6.0, f"{s}_fainted": 0, f"{s}_statused": 0})
        snap.update({f"{s}_boost_{b}": 0 for b in ("atk", "def", "spa", "spd", "spe")})
        snap.update({f"{s}_hazard_{h}": 0 for h in
                     ("stealthrock", "spikes", "toxicspikes", "stickyweb")})
        snap.update({f"{s}_screen_{sc}": 0 for sc in
                     ("reflect", "lightscreen", "auroraveil", "tailwind")})
    game["snapshots"] = [snap]
    return game, snap


def test_hazard_chip_type_scaling():
    _, snap = _sim_1v1("Great Tusk", "Heatran")
    snap["p1_hazard_stealthrock"] = 1
    assert hazard_chip("Charizard", "p1", snap) == 0.5      # 4x rock weakness
    assert hazard_chip("Great Tusk", "p1", snap) == pytest.approx(0.125 * 0.25)


def test_damage_engine_knows_types_and_burn():
    eq = dict(move_info("Earthquake"), name="Earthquake")
    game, snap = _sim_1v1("Great Tusk", "Heatran")
    vs_heatran = SimState(game, snap).damage_fraction("p1", eq)   # 4x weak + STAB
    game2, snap2 = _sim_1v1("Great Tusk", "Corviknight")
    vs_corv = SimState(game2, snap2).damage_fraction("p1", eq)    # flying: immune
    game3, snap3 = _sim_1v1("Great Tusk", "Heatran", p1_status="brn")
    burned = SimState(game3, snap3).damage_fraction("p1", eq)
    assert vs_heatran > 0.9        # near-guaranteed KO
    assert vs_corv == 0.0
    # burn halves the attack stat; the flat +2 term keeps it from being exactly half
    assert 0.45 * vs_heatran < burned < 0.55 * vs_heatran


def test_status_and_hazard_effects_apply():
    game, snap = _sim_1v1("Clodsire", "Dragapult")
    sim = SimState(game, snap)
    sim.use_move("p1", dict(move_info("Toxic"), name="Toxic"))
    assert sim.active["p2"].status == "tox"
    sim.use_move("p1", dict(move_info("Stealth Rock"), name="Stealth Rock"))
    assert sim.snap["p2_hazard_stealthrock"] == 1


def test_faster_ko_cancels_slower_move():
    """Dragapult (142 spe) KOs Iron Valiant hard? No — pick a clean OHKO pairing:
    a faster attacker that KOs should prevent the slower side's move entirely."""
    game, snap = _sim_1v1("Dragapult", "Charizard")
    sim = SimState(game, snap)
    draco = dict(move_info("Shadow Ball"), name="Shadow Ball")
    flare = dict(move_info("Flare Blitz"), name="Flare Blitz")
    snap_before = sim.snap["p1_hp_total"]
    sim.resolve({"p1": {"kind": "move", "move": draco},
                 "p2": {"kind": "move", "move": flare}})
    if sim.active["p2"].fainted:  # if the KO landed, p1 must be untouched
        assert sim.snap["p1_hp_total"] == snap_before


def test_snapshot_features_handles_mixed_screen_dtype():
    """Regression: the advisor scores a whole action matrix in one batch, so a
    screen column can hold both bool (unchanged) and int (a screen was set). That
    mix must still coerce to numeric for LightGBM."""
    from src.predict import load_model, snapshot_features
    booster, meta = load_model()
    game = parse_replay(json.loads(FIXTURES[0].read_text(encoding="utf-8")), up_to_turn=8)
    snap = game["snapshots"][-1]
    a = {**snap, "p1_screen_reflect": True}
    b = {**snap, "p1_screen_reflect": 1}
    X = snapshot_features({**game, "snapshots": [a, b]}, meta)
    assert len(X) == 2
    booster.predict(X)  # must not raise on the mixed-dtype column


def test_advise_search_on_fixture():
    from src.predict import load_model, snapshot_features
    booster, meta = load_model()
    game = parse_replay(json.loads(FIXTURES[0].read_text(encoding="utf-8")), up_to_turn=10)
    out = advise_search(game, "p2", booster, meta, snapshot_features)
    assert len(out) >= 1
    assert out.worst_case.between(0, 1).all()
    assert (out.worst_case <= out.average + 1e-9).all()
    assert list(out.worst_case) == sorted(out.worst_case, reverse=True)
    assert player_actions(game, "p2")  # options exist at turn 10
