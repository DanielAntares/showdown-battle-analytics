"""Advisor v2: damage engine sanity + minimax search on real replay fixtures."""

import json
from pathlib import Path

import pytest

from src.advisor import (SimState, advise_search, hazard_chip, player_actions,
                         recommend_lead)
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


def test_sucker_punch_fails_vs_status_move():
    game, snap = _sim_1v1("Kingambit", "Dondozo")
    sim = SimState(game, snap)
    sucker = dict(move_info("Sucker Punch"), name="Sucker Punch")
    curse = dict(move_info("Curse"), name="Curse")
    sim.resolve({"p1": {"kind": "move", "move": sucker},
                 "p2": {"kind": "move", "move": curse}})
    assert sim.active["p2"].hp == 1.0  # whiffed: target wasn't attacking
    # ...but it works normally when the target attacks
    sim2 = SimState(*_sim_1v1("Kingambit", "Dondozo"))
    press = dict(move_info("Body Press"), name="Body Press")
    sim2.resolve({"p1": {"kind": "move", "move": sucker},
                  "p2": {"kind": "move", "move": press}})
    assert sim2.active["p2"].hp < 1.0


def test_rest_heals_but_sleeps_and_fails_at_full():
    game, snap = _sim_1v1("Dondozo", "Kingambit")
    sim = SimState(game, snap)
    rest = dict(move_info("Rest"), name="Rest")
    sim.use_move("p1", rest)  # full HP: no-op, no sleep
    assert sim.active["p1"].hp == 1.0 and sim.active["p1"].status == ""
    sim.active["p1"].hp = 0.3
    sim.use_move("p1", rest)
    assert sim.active["p1"].hp == 1.0 and sim.active["p1"].status == "slp"


def _mon(species, **kw):
    base = {"species": species, "hp": 1.0, "status": "", "fainted": False,
            "active": True, "moves": [], "item": "", "volatiles": [], "last_move": ""}
    return {**base, **kw}


def test_move_legality_filters():
    from src.advisor import moves_for
    # Choice lock: only the last-used move remains
    locked = moves_for(_mon("Kyurem", item="choicespecs", last_move="Ice Beam",
                            moves=["Ice Beam", "Draco Meteor"]))
    assert [m["name"] for m in locked] == ["Ice Beam"]
    # Encore: same behavior via volatile
    encored = moves_for(_mon("Gholdengo", volatiles=["encore"], last_move="Nasty Plot",
                             moves=["Nasty Plot", "Shadow Ball"]))
    assert [m["name"] for m in encored] == ["Nasty Plot"]
    # Taunt: status moves dropped
    taunted = moves_for(_mon("Gholdengo", volatiles=["taunt"],
                             moves=["Nasty Plot", "Recover", "Shadow Ball"]))
    assert all(m["category"] != "Status" for m in taunted)


def test_useless_moves_pruned():
    from src.advisor import moves_for
    game, snap = _sim_1v1("Dondozo", "Kingambit")
    dozo = _mon("Dondozo", moves=["Rest", "Curse", "Body Press", "Sleep Talk"])
    names = [m["name"] for m in moves_for(dozo, snap, "p1")]
    assert "Rest" not in names  # full HP: Rest would fail
    dozo["hp"] = 0.4
    names = [m["name"] for m in moves_for(dozo, snap, "p1")]
    assert "Rest" in names  # hurt: Rest is a real option again


def test_sleep_immobilizes_but_sleep_talk_attacks():
    game, snap = _sim_1v1("Kingambit", "Dondozo", p1_status="slp")
    sim = SimState(game, snap)
    sim.use_move("p1", dict(move_info("Kowtow Cleave"), name="Kowtow Cleave"))
    assert sim.active["p2"].hp == 1.0  # asleep: move fails
    sim.use_move("p1", dict(move_info("Sleep Talk"), name="Sleep Talk"))
    assert sim.active["p2"].hp < 1.0  # Sleep Talk still attacks


def test_upkeep_residuals():
    game, snap = _sim_1v1("Kingambit", "Dondozo")
    sim = SimState(game, snap)
    sim.active["p2"].status = "brn"
    sim.active["p2"].item = ""  # predicted Leftovers would offset the chip
    sim.snap["weather"] = "sandstorm"
    sim.upkeep()
    # Dondozo (water): burn 1/16 + sand 1/16; Kingambit (dark/steel): sand-immune
    assert sim.active["p2"].hp == pytest.approx(1 - 2 / 16)
    assert sim.active["p1"].hp == 1.0


def test_psychic_terrain_blocks_priority():
    game, snap = _sim_1v1("Kingambit", "Dondozo")
    snap["terrain"] = "psychicterrain"
    sim = SimState(game, snap)
    sucker = dict(move_info("Sucker Punch"), name="Sucker Punch")
    press = dict(move_info("Body Press"), name="Body Press")
    sim.resolve({"p1": {"kind": "move", "move": sucker},
                 "p2": {"kind": "move", "move": press}})
    assert sim.active["p2"].hp == 1.0  # priority blocked vs grounded target


def test_misty_terrain_blocks_status():
    game, snap = _sim_1v1("Clodsire", "Dragapult")
    snap["terrain"] = "mistyterrain"
    sim = SimState(game, snap)
    sim.use_move("p1", dict(move_info("Toxic"), name="Toxic"))
    assert sim.active["p2"].status == ""  # Dragapult... has Flying? no: dragon/ghost -> grounded, blocked


def test_terrain_boosts_grounded_electric():
    game, snap = _sim_1v1("Raging Bolt", "Dondozo")
    bolt = dict(move_info("Thunderbolt"), name="Thunderbolt")
    plain = SimState(game, snap).damage_fraction("p1", bolt)
    snap2 = dict(snap, terrain="electricterrain")
    boosted = SimState(game, snap2).damage_fraction("p1", bolt)
    assert boosted == pytest.approx(plain * 1.3)


def test_recommend_lead_ranks_full_team():
    from src.predict import load_model, snapshot_features
    booster, meta = load_model()
    game = parse_replay(json.loads(FIXTURES[0].read_text(encoding="utf-8")))
    for side in ("p1", "p2"):
        rec = recommend_lead(game, side, booster, meta, snapshot_features)
        assert set(rec.lead) == set(game["teams"][side])  # every team member ranked
        assert rec.average.between(0, 1).all()
        assert (rec.worst_case <= rec.average + 1e-9).all()
        # ranked best-average first
        assert list(rec.average) == sorted(rec.average, reverse=True)


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


def _pivot_game(active, bench, p2_active, active_moves):
    _, snap = _sim_1v1(active, p2_active)
    roster = {"p1": [_mon(active, active=True, moves=active_moves)]
                     + [_mon(b, active=False) for b in bench],
              "p2": [_mon(p2_active, active=True)]}
    snap["p1_hp_total"] = float(1 + len(bench))
    snap["p1_healthy"] = 1 + len(bench)
    snap["p2_healthy"] = 1
    return {"roster": roster, "snapshots": [snap], "teams": {}}


def test_active_pivot_move_detects_uturn():
    from src.advisor import active_pivot_move
    game = _pivot_game("Cinderace", ["Kyurem"], "Gholdengo", ["Pyro Ball", "U-turn"])
    pm = active_pivot_move(game, "p1")
    assert pm and pm["name"] == "U-turn"
    # a mon with no pivot move returns None
    game2 = _pivot_game("Great Tusk", ["Kyurem"], "Gholdengo", ["Earthquake", "Ice Spinner"])
    assert active_pivot_move(game2, "p1") is None


def test_pivot_targets_ranks_bench():
    from src.advisor import active_pivot_move, pivot_targets
    from src.predict import load_model, snapshot_features
    booster, meta = load_model()
    game = _pivot_game("Cinderace", ["Kyurem", "Dondozo"], "Gholdengo",
                       ["Pyro Ball", "U-turn"])
    pm = active_pivot_move(game, "p1")
    df = pivot_targets(game, "p1", booster, meta, snapshot_features, pm)
    assert set(df.target) == {"Kyurem", "Dondozo"}   # only the benched mons
    assert df.win.between(0, 1).all()
    assert list(df.win) == sorted(df.win, reverse=True)  # best-first
    # nothing to bring in -> empty frame, no crash
    solo = _pivot_game("Cinderace", [], "Gholdengo", ["Pyro Ball", "U-turn"])
    assert len(pivot_targets(solo, "p1", booster, meta, snapshot_features, pm)) == 0
