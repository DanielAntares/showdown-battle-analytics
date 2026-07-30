"""Weight mechanic: Low Kick / Grass Knot (target weight), Heavy Slam / Heat Crash
(user-to-target ratio), the weight-changing abilities/item, and the power-0
HP-fraction moves (Super Fang / Ruination / Endeavor) that were also unmodeled."""

import pytest

from src.advisor import SimState, weight_of
from src.pokedex import move_info
from src.selfplay import new_game

TEAM = ["Kingambit"] * 5


def _sim(attacker, defender, atk_item="", def_item="", def_ability=""):
    g = new_game([attacker] + TEAM, [defender] + TEAM)
    da = next(m for m in g["roster"]["p1"] if m["active"]); da["item"] = atk_item
    dd = next(m for m in g["roster"]["p2"] if m["active"])
    dd["item"] = def_item; dd["ability"] = def_ability
    return SimState(g, g["snapshots"][-1])


def _mv(name):
    return dict(move_info(name), name=name)


def test_weight_of_applies_abilities_and_item():
    base = weight_of("Gholdengo")                     # 30 kg
    assert weight_of("Gholdengo", "heavymetal") == pytest.approx(base * 2)
    assert weight_of("Gholdengo", "lightmetal") == pytest.approx(base * 0.5)
    assert weight_of("Gholdengo", "", "floatstone") == pytest.approx(base * 0.5)
    assert weight_of("Nonexistent-Mon") >= 0.1        # missing weight -> floor, no crash


def test_low_kick_grass_knot_scale_with_target_weight():
    # Low Kick power ladder by the *target's* weight (Copperajah 650 -> 120, light -> 20)
    heavy = _sim("Great Tusk", "Copperajah")._effective_power("p1", _mv("Low Kick"))
    light = _sim("Great Tusk", "Flutter Mane")._effective_power("p1", _mv("Grass Knot"))  # ~4 kg
    assert heavy == 120
    assert light <= 40
    assert heavy > light


def test_heavy_slam_scales_with_user_to_target_ratio():
    # Copperajah (650 kg): huge vs a feather, minimum vs an equally heavy target
    vs_light = _sim("Copperajah", "Flutter Mane")._effective_power("p1", _mv("Heavy Slam"))
    vs_heavy = _sim("Copperajah", "Copperajah")._effective_power("p1", _mv("Heavy Slam"))
    assert vs_light == 120 and vs_heavy == 40


def test_heavy_metal_and_float_stone_shift_the_ratio():
    # a Float Stone on the target halves its weight -> a higher Heavy Slam bracket
    base = _sim("Great Tusk", "Kingambit")._effective_power("p1", _mv("Heavy Slam"))
    lighter = _sim("Great Tusk", "Kingambit", def_item="floatstone")._effective_power("p1", _mv("Heavy Slam"))
    assert lighter >= base
    # Heavy Metal on the target raises its weight -> Low Kick hits harder
    n = _sim("Great Tusk", "Kingambit")._effective_power("p1", _mv("Low Kick"))
    hm = _sim("Great Tusk", "Kingambit", def_ability="heavymetal")._effective_power("p1", _mv("Low Kick"))
    assert hm >= n


def test_weight_moves_actually_deal_damage_now():
    # the whole point: power-0 in the data must not mean 0 damage in the sim
    frac = _sim("Copperajah", "Gholdengo").damage_fraction("p1", _mv("Heavy Slam"))
    assert frac > 0.0


def test_hp_fraction_moves():
    sim = _sim("Great Tusk", "Kingambit")
    opp = sim.active["p2"]; opp.hp = 0.8
    sf_acc = move_info("Super Fang")["accuracy"]  # 0.9 — folded into the EV
    assert sim.damage_fraction("p1", _mv("Super Fang")) == pytest.approx(0.5 * 0.8 * sf_acc, rel=1e-6)
    # Endeavor (100% acc) brings the target down to the user's HP (deals the difference)
    sim.active["p1"].hp = 0.3
    assert sim.damage_fraction("p1", _mv("Endeavor")) == pytest.approx(0.8 - 0.3, rel=1e-6)
    # type immunity still applies: Super Fang (Normal) does nothing to a Ghost
    ghost = _sim("Great Tusk", "Gholdengo")
    assert ghost.damage_fraction("p1", _mv("Super Fang")) == 0.0
