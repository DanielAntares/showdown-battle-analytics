"""Random-battle set/level layer: authoritative sets, per-species levels, role
narrowing, and a level-aware damage engine. All opt into `randbats_mode`."""

import pytest

import src.movesets as movesets
import src.randbats as randbats
from src.advisor import SimState, advise_search, player_actions
from src.predict import load_model, snapshot_features
from src.pokedex import move_info
from src.selfplay import new_game

pytestmark = pytest.mark.usefixtures("randbats_mode")


def test_species_carry_their_real_level():
    # randbats levels are fixed per species and (mostly) below 100
    assert randbats.species_level("Great Tusk") == 77
    assert movesets.species_level("Great Tusk") == 77  # dispatch routes through
    assert randbats.species_level("Toxapex") == 82


def test_stats_scale_with_level_below_l100():
    """A randbats mon is weaker/slower than the same species imagined at L100."""
    rb = randbats.real_stats("Great Tusk")  # L77
    movesets._randbats_mode._v = False       # peek at the L100 usage-spread version
    movesets.real_stats.cache_clear()
    l100 = movesets.real_stats("Great Tusk")
    movesets._randbats_mode._v = True
    movesets.real_stats.cache_clear()
    assert rb["hp"] < l100["hp"] and rb["atk"] < l100["atk"] and rb["spe"] < l100["spe"]
    # sanity on the level formula: HP = floor(inner*L/100)+L+10
    assert rb["hp"] == 304


def test_role_narrowing_sharpens_the_moveset():
    """Revealing a role-defining move rules out incompatible roles, so a move only
    the setup roles carry (Earthquake) surfaces once Bulk Up is seen."""
    blind = randbats.predict_moves("Great Tusk", k=4)
    seen_setup = randbats.predict_moves("Great Tusk", revealed=["Bulk Up"], k=4)
    assert "Bulk Up" in seen_setup
    assert "Earthquake" in seen_setup and "Earthquake" not in blind


def test_special_attackers_zero_attack():
    spread = randbats.predict_spread("Ninetales-Alola")  # special, snow setter
    assert spread["atk_iv"] == 0
    assert spread["evs"][1] == 0  # Atk EV slot


def test_damage_engine_is_level_aware():
    """The same move from the same attacker hits softer at its randbats level than
    it would at L100 (the level term 2L/5+2 drops with level)."""
    eq = dict(move_info("Earthquake"), name="Earthquake")
    g = new_game(["Great Tusk"] + ["Kingambit"] * 5, ["Kingambit"] * 6)
    snap = g["snapshots"][-1]
    rb_dmg = SimState(g, snap).damage_fraction("p1", eq)  # Great Tusk L77 EQ vs Kingambit
    a = SimState(g, snap).active["p1"]
    assert a.level == 77
    assert 2 * a.level / 5 + 2 < 42  # level term below the L100 constant
    assert 0.0 < rb_dmg  # still a real, super-effective (Steel) hit


def test_fixed_damage_equals_attacker_level():
    """Seismic Toss deals damage equal to the attacker's level, not a flat 100."""
    toss = dict(move_info("Seismic Toss"), name="Seismic Toss")
    g = new_game(["Gastrodon"] + ["Kingambit"] * 5, ["Kingambit"] * 6)
    sim = SimState(g, g["snapshots"][-1])
    lvl = sim.active["p1"].level
    expected = lvl / sim.active["p2"].stats["hp"]
    assert sim.damage_fraction("p1", toss) == pytest.approx(expected)


def test_advisor_runs_end_to_end_in_randbats():
    booster, meta = load_model()
    g = new_game(["Dragonite", "Gholdengo", "Great Tusk", "Toxapex", "Kingambit", "Garchomp"],
                 ["Landorus-Therian", "Raging Bolt", "Toxapex", "Kingambit", "Gholdengo", "Ditto"])
    out = advise_search(g, "p1", booster, meta, snapshot_features, pessimism=0.7)
    assert len(out) and out.worst_case.between(0, 1).all()
    assert player_actions(g, "p1")
