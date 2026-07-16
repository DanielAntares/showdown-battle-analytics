"""Items, abilities, move mechanics, Tera, durations, counters in the engine."""

import pytest

from src.advisor import SimState, hazard_chip, moves_for, player_actions
from src.pokedex import move_info
from tests.test_advisor import _mon, _sim_1v1


def _mv(name):
    return dict(move_info(name), name=name)


# ---- items -------------------------------------------------------------------

def test_boots_ignore_hazards():
    _, snap = _sim_1v1("Great Tusk", "Heatran")
    snap["p1_hazard_stealthrock"] = 1
    snap["p1_hazard_spikes"] = 3
    assert hazard_chip("Charizard", "p1", snap, item="heavydutyboots") == 0.0
    assert hazard_chip("Charizard", "p1", snap) > 0.4


def test_choice_band_boosts_physical():
    game, snap = _sim_1v1("Great Tusk", "Dondozo")
    plain = SimState(game, snap)
    plain.active["p1"].item = ""
    banded = SimState(game, snap)
    banded.active["p1"].item = "choiceband"
    eq = _mv("Earthquake")
    ratio = banded.damage_fraction("p1", eq) / plain.damage_fraction("p1", eq)
    assert 1.4 < ratio <= 1.5  # ~1.5x, shy of exact due to the formula's +2 term


def test_focus_sash_survives_from_full():
    game, snap = _sim_1v1("Great Tusk", "Weavile")
    sim = SimState(game, snap)
    sim.active["p2"].item = "focussash"
    sim.active["p2"].ability = ""
    sim.use_move("p1", _mv("Close Combat"))  # would be a clean OHKO
    assert not sim.active["p2"].fainted
    assert sim.active["p2"].hp == pytest.approx(0.01)


def test_leftovers_and_toxic_ramp_in_upkeep():
    game, snap = _sim_1v1("Great Tusk", "Dondozo")
    sim = SimState(game, snap)
    sim.active["p1"].item = "leftovers"
    sim.active["p1"].hp = 0.5
    sim.active["p2"].item = ""
    sim.active["p2"].status = "tox"
    sim.active["p2"].tox_turns = 3
    sim.upkeep()
    assert sim.active["p1"].hp == pytest.approx(0.5 + 1 / 16)
    assert sim.active["p2"].hp == pytest.approx(1 - 4 / 16)


# ---- abilities ----------------------------------------------------------------

def test_ability_immunities():
    game, snap = _sim_1v1("Great Tusk", "Gholdengo")
    sim = SimState(game, snap)
    sim.active["p2"].ability = "levitate"
    assert sim.damage_fraction("p1", _mv("Earthquake")) == 0.0
    sim.active["p2"].ability = "flashfire"
    assert sim.damage_fraction("p1", _mv("Flamethrower")) == 0.0


def test_unaware_ignores_attack_boosts():
    game, snap = _sim_1v1("Kingambit", "Dondozo")
    boosted = SimState(game, snap)
    boosted.active["p1"].boosts["atk"] = 6
    boosted.active["p2"].ability = "unaware"
    flat = SimState(game, snap)
    flat.active["p2"].ability = "unaware"
    kowtow = _mv("Kowtow Cleave")
    assert boosted.damage_fraction("p1", kowtow) == pytest.approx(
        flat.damage_fraction("p1", kowtow))


def test_guts_flips_burn():
    game, snap = _sim_1v1("Great Tusk", "Dondozo", p1_status="brn")
    guts = SimState(game, snap)
    guts.active["p1"].ability = "guts"
    plain = SimState(game, snap)
    plain.active["p1"].ability = ""
    eq = _mv("Earthquake")
    ratio = guts.damage_fraction("p1", eq) / plain.damage_fraction("p1", eq)
    assert 2.7 < ratio <= 3.0  # 1.5x vs 0.5x attack, minus the +2 flat term


def test_intimidate_on_switch():
    game, snap = _sim_1v1("Great Tusk", "Dondozo")
    sim = SimState(game, snap)
    lando = _mon("Landorus-Therian", active=False, ability="Intimidate")
    sim.switch("p1", lando)
    assert sim.active["p2"].boosts["atk"] == -1


def test_supreme_overlord_scales_with_faints():
    game, snap = _sim_1v1("Kingambit", "Dondozo")
    fresh = SimState(game, snap)
    fresh.active["p1"].ability = "supremeoverlord"
    late_snap = dict(snap, p1_fainted=4)
    late = SimState(game, late_snap)
    late.active["p1"].ability = "supremeoverlord"
    kowtow = _mv("Kowtow Cleave")
    ratio = late.damage_fraction("p1", kowtow) / fresh.damage_fraction("p1", kowtow)
    assert 1.3 < ratio <= 1.4  # x1.4 attack with 4 fallen allies


# ---- move mechanics ------------------------------------------------------------

def test_body_press_uses_defense():
    game, snap = _sim_1v1("Dondozo", "Kingambit")
    sim = SimState(game, snap)
    base = sim.damage_fraction("p1", _mv("Body Press"))
    sim.active["p1"].stats["def"] *= 2
    assert sim.damage_fraction("p1", _mv("Body Press")) > base * 1.5


def test_foul_play_uses_target_attack():
    game, snap = _sim_1v1("Dondozo", "Kingambit")
    sim = SimState(game, snap)
    base = sim.damage_fraction("p1", _mv("Foul Play"))
    sim.active["p2"].boosts["atk"] = 6  # the TARGET's boost powers Foul Play
    assert sim.damage_fraction("p1", _mv("Foul Play")) > base * 2


def test_fixed_damage_and_drain_and_recoil():
    game, snap = _sim_1v1("Blissey", "Kingambit")
    sim = SimState(game, snap)
    toss = sim.damage_fraction("p1", _mv("Seismic Toss"))
    assert toss == pytest.approx(100 / sim.active["p2"].stats["hp"])
    # ...and the type chart still applies: Ghost-types are immune to it
    game2, snap2 = _sim_1v1("Blissey", "Gholdengo")
    assert SimState(game2, snap2).damage_fraction("p1", _mv("Seismic Toss")) == 0.0

    game2, snap2 = _sim_1v1("Ogerpon-Wellspring", "Great Tusk")
    sim2 = SimState(game2, snap2)
    sim2.active["p1"].hp = 0.5
    sim2.active["p1"].item = ""
    sim2.use_move("p1", _mv("Horn Leech"))
    assert sim2.active["p1"].hp > 0.5  # drained back health

    game3, snap3 = _sim_1v1("Dragonite", "Dondozo")
    sim3 = SimState(game3, snap3)
    sim3.active["p1"].item = ""
    sim3.active["p1"].ability = ""
    sim3.use_move("p1", _mv("Flare Blitz"))
    assert sim3.active["p1"].hp < 1.0  # recoil


# ---- tera ----------------------------------------------------------------------

def test_tera_action_available_and_applies():
    game, snap = _sim_1v1("Kingambit", "Dondozo")
    game["roster"]["p1"][0]["moves"] = ["Kowtow Cleave", "Sucker Punch"]
    game["snapshots"] = [snap]
    acts = player_actions(game, "p1")
    tera_acts = [a for a in acts if a.get("tera")]
    assert tera_acts, "no Tera actions offered"
    sim = SimState(game, snap)
    sim.resolve({"p1": tera_acts[0],
                 "p2": {"kind": "move", "move": _mv("Body Press")}})
    assert sim.snap["p1_tera_used"] == 1
    assert sim.active["p1"].types == [tera_acts[0]["tera"]]
    # once used, no more Tera actions
    assert not [a for a in player_actions(
        {**game, "snapshots": [dict(snap, p1_tera_used=1)]}, "p1") if a.get("tera")]


# ---- durations & counters -------------------------------------------------------

def test_screen_and_weather_expiry_in_next_snapshot():
    game, snap = _sim_1v1("Great Tusk", "Dondozo")
    snap.update(p1_screen_reflect=1, weather="raindance", turn=5)
    game["field"] = {"weather_set_turn": 1, "terrain_set_turn": 0,
                     "screen_turns": {"p1": {"reflect": 1}}}
    out = SimState(game, snap).to_snapshot()
    assert out["p1_screen_reflect"] == 0  # set turn 1, gone by turn 6
    assert out["weather"] == ""
    # a fresh screen survives
    game["field"]["screen_turns"]["p1"]["reflect"] = 4
    out2 = SimState(game, snap).to_snapshot()
    assert out2["p1_screen_reflect"] == 1


def test_sleep_counter_wakes_after_three():
    game, snap = _sim_1v1("Kingambit", "Dondozo", p1_status="slp")
    game["roster"]["p1"][0]["sleep_turns"] = 3
    sim = SimState(game, snap)
    sim.use_move("p1", _mv("Kowtow Cleave"))
    assert sim.active["p2"].hp < 1.0  # slept 3 turns: acts again


def test_pp_exhausted_move_filtered():
    mon = _mon("Heatran", moves=["Magma Storm", "Earth Power"],
               uses={"Magma Storm": 8})  # Magma Storm has 5 PP -> 8 max
    names = [m["name"] for m in moves_for(mon)]
    assert "Magma Storm" not in names
    assert "Earth Power" in names
