"""Items, abilities, move mechanics, Tera, durations, counters in the engine."""

import json
from pathlib import Path

import pytest

from src.advisor import SimState, hazard_chip, moves_for, player_actions
from src.parser import parse_replay
from src.pokedex import move_info
from tests.test_advisor import _mon, _sim_1v1

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))


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


def test_asleep_mon_cannot_attack_regardless_of_count():
    # a mon that is 'slp' at turn start is asleep, even at a high sleep count
    # (Rest/Sleep Talk loops keep re-sleeping) — only Sleep Talk acts
    game, snap = _sim_1v1("Dondozo", "Kingambit", p1_status="slp")
    game["roster"]["p1"][0]["sleep_turns"] = 3
    sim = SimState(game, snap)
    sim.use_move("p1", _mv("Body Press"))
    assert sim.active["p2"].hp == 1.0  # asleep: Body Press fails
    sim2 = SimState(game, snap)
    sim2.use_move("p1", _mv("Sleep Talk"))
    assert sim2.active["p2"].hp < 1.0  # Sleep Talk still calls a move


def test_semi_invulnerable_target_cannot_be_hit():
    game, snap = _sim_1v1("Iron Valiant", "Grafaiai")
    sim = SimState(game, snap)
    sim.active["p2"].semiinvuln = True  # Grafaiai is underground (Dig)
    assert sim.damage_fraction("p1", _mv("Close Combat")) == 0.0
    # ...but Earthquake still hits a digging target
    assert sim.damage_fraction("p1", _mv("Earthquake")) > 0.0


def test_charge_and_caller_moves_not_recommended():
    mon = _mon("Grafaiai", moves=["Dig", "Copycat", "Knock Off", "Gunk Shot"])
    names = [m["name"] for m in moves_for(mon)]
    assert "Dig" not in names and "Copycat" not in names
    assert "Knock Off" in names


def test_attacks_dropped_vs_underground_opponent():
    _, snap = _sim_1v1("Deoxys-Speed", "Grafaiai")
    game = {"roster": {
        "p1": [_mon("Deoxys-Speed", moves=["Psycho Boost", "Stealth Rock", "Spikes"])],
        "p2": [_mon("Grafaiai", active=True, volatiles=["semiinvuln"])]},
        "snapshots": [snap]}
    names = [m["name"] for m in moves_for(game["roster"]["p1"][0], snap, "p1", game)]
    assert "Psycho Boost" not in names        # would miss the underground target
    assert "Stealth Rock" in names and "Spikes" in names  # hazards still useful


def test_ability_immunity_pruned_comprehensively():
    # a predicted-Levitate opponent -> Ground moves dropped (not just Air Balloon)
    _, snap = _sim_1v1("Great Tusk", "Rotom-Wash")  # Rotom-Wash: only ability = Levitate
    game = {"roster": {
        "p1": [_mon("Great Tusk", moves=["Headlong Rush", "Ice Spinner"])],
        "p2": [_mon("Rotom-Wash", active=True)]}, "snapshots": [snap]}
    names = [m["name"] for m in moves_for(game["roster"]["p1"][0], snap, "p1", game)]
    assert "Headlong Rush" not in names   # Ground vs Levitate
    assert "Ice Spinner" in names


def test_status_move_pruned_when_target_already_statused():
    _, snap = _sim_1v1("Slowking-Galar", "Dragapult")  # non-Steel: Sludge Bomb connects
    snap["p2_active_status"] = "par"
    game = {"roster": {
        "p1": [_mon("Slowking-Galar", moves=["Thunder Wave", "Sludge Bomb"])],
        "p2": [_mon("Dragapult", active=True, status="par")]}, "snapshots": [snap]}
    names = [m["name"] for m in moves_for(game["roster"]["p1"][0], snap, "p1", game)]
    assert "Thunder Wave" not in names   # already paralyzed -> Thunder Wave would fail
    assert "Sludge Bomb" in names


def test_switch_carries_a_tempo_cost(monkeypatch):
    import src.advisor as adv
    from src.predict import load_model, snapshot_features
    booster, meta = load_model()
    game = parse_replay(json.loads(FIXTURES[0].read_text(encoding="utf-8")), up_to_turn=8)
    monkeypatch.setattr(adv, "SWITCH_COST", 0.0)
    free = adv.advise_search(game, "p1", booster, meta, snapshot_features).set_index("action")
    monkeypatch.setattr(adv, "SWITCH_COST", 0.06)
    costed = adv.advise_search(game, "p1", booster, meta, snapshot_features).set_index("action")
    switches = [a for a in free.index if a.startswith("switch")]
    assert switches, "expected some switch options in this position"
    for a in switches:  # every switch scores strictly lower once the cost applies
        assert costed.loc[a, "worst_case"] < free.loc[a, "worst_case"]


def test_choice_lock_into_immune_leaves_only_switches():
    # Kyurem (predicted Choice Specs) locked into Draco Meteor vs a Fairy -> the
    # move is immune, so moves_for returns nothing (advisor must switch)
    _, snap = _sim_1v1("Kyurem", "Hatterene")
    game = {"roster": {
        "p1": [_mon("Kyurem", last_move="Draco Meteor",
                    moves=["Draco Meteor", "Freeze-Dry", "Earth Power"])],
        "p2": [_mon("Hatterene", active=True)]}, "snapshots": [snap]}
    assert moves_for(game["roster"]["p1"][0], snap, "p1", game) == []
    # locked into a move that DOES hit -> that move is kept
    game["roster"]["p1"][0]["last_move"] = "Freeze-Dry"
    names = [m["name"] for m in moves_for(game["roster"]["p1"][0], snap, "p1", game)]
    assert names == ["Freeze-Dry"]


def test_air_balloon_grants_ground_immunity():
    game, snap = _sim_1v1("Great Tusk", "Heatran")
    sim = SimState(game, snap)
    sim.active["p2"].item = "airballoon"
    assert sim.damage_fraction("p1", _mv("Earthquake")) == 0.0  # immune while intact
    # a popped balloon (item consumed) no longer protects
    _, snap2 = _sim_1v1("Great Tusk", "Heatran")
    game2 = {"roster": {"p1": [_mon("Great Tusk", moves=["Earthquake"])],
                        "p2": [_mon("Heatran", active=True, item="", item_consumed=True)]},
             "snapshots": [snap2]}
    assert "Earthquake" in [m["name"] for m in moves_for(game2["roster"]["p1"][0], snap2, "p1", game2)]


def test_no_setup_while_dying_to_poison():
    _, snap = _sim_1v1("Zamazenta", "Landorus-Therian")
    game = {"roster": {"p1": [_mon("Zamazenta")], "p2": [_mon("Landorus-Therian", active=True)]},
            "snapshots": [snap]}
    # badly poisoned -> Iron Defense (pure setup) dropped, Body Press (attack) kept
    healthy = _mon("Zamazenta", moves=["Iron Defense", "Body Press", "Crunch"])
    assert "Iron Defense" in [m["name"] for m in moves_for(healthy, snap, "p1", game)]
    dying = _mon("Zamazenta", status="tox", moves=["Iron Defense", "Body Press", "Crunch"])
    names = [m["name"] for m in moves_for(dying, snap, "p1", game)]
    assert "Iron Defense" not in names
    assert "Body Press" in names and "Crunch" in names


def test_ko_promotes_replacement():
    """After a KO the snapshot shows the opponent's replacement, not a 0-HP
    fainted active — otherwise the win-prob model misreads the KO as bad."""
    _, snap = _sim_1v1("Kingambit", "Pikachu")
    game = {"roster": {
        "p1": [_mon("Kingambit", moves=["Kowtow Cleave"])],
        "p2": [_mon("Pikachu", active=True, hp=0.1),
               _mon("Dondozo", active=False, hp=1.0)]},
        "field": {}, "snapshots": [snap]}
    sim = SimState(game, snap)
    sim.use_move("p1", _mv("Kowtow Cleave"))
    assert sim.active["p2"].fainted
    out = sim.to_snapshot()
    assert out["p2_active_species"] == "Dondozo"   # replacement promoted
    assert out["p2_active_hp"] == 1.0
    assert out["p2_fainted"] == 1                  # the faint still counts


def test_immune_move_pruned():
    _, snap = _sim_1v1("Slowking-Galar", "Iron Treads")  # snap matches roster below
    game = {"roster": {
        "p1": [_mon("Slowking-Galar", moves=["Sludge Bomb", "Thunder Wave"])],
        "p2": [_mon("Iron Treads", active=True)]}, "snapshots": [snap]}
    names = [m["name"] for m in moves_for(game["roster"]["p1"][0], snap, "p1", game)]
    assert "Sludge Bomb" not in names  # Poison is immune vs Ground/Steel
    assert "Thunder Wave" in names


def test_immune_pruning_respects_tera():
    _, snap = _sim_1v1("Great Tusk", "Corviknight")
    game = {"roster": {
        "p1": [_mon("Great Tusk", moves=["Headlong Rush", "Ice Spinner"])],
        "p2": [_mon("Corviknight", active=True)]}, "snapshots": [snap]}
    # Corviknight (Flying/Steel): Ground-type Headlong Rush is immune (Flying)
    names = [m["name"] for m in moves_for(game["roster"]["p1"][0], snap, "p1", game)]
    assert "Headlong Rush" not in names
    # but if Corviknight has Tera'd Fighting, it loses Flying and Ground connects
    game["roster"]["p2"][0]["tera"] = "Fighting"
    names2 = [m["name"] for m in moves_for(game["roster"]["p1"][0], snap, "p1", game)]
    assert "Headlong Rush" in names2


def test_future_sight_never_recommended():
    from src.advisor import SimState
    game, snap = _sim_1v1("Slowking-Galar", "Kingambit")
    fs = _mv("Future Sight")
    sim = SimState(game, snap)
    before = sim.active["p2"].hp
    sim.use_move("p1", fs)
    assert sim.active["p2"].hp == before  # no immediate damage (delayed)
    mon = _mon("Slowking-Galar", moves=["Future Sight", "Sludge Bomb", "Chilly Reception"])
    names = [m["name"] for m in moves_for(mon)]
    assert "Future Sight" not in names  # abstained: 1-ply can't value it


def test_pp_exhausted_move_filtered():
    mon = _mon("Heatran", moves=["Magma Storm", "Earth Power"],
               uses={"Magma Storm": 8})  # Magma Storm has 5 PP -> 8 max
    names = [m["name"] for m in moves_for(mon)]
    assert "Magma Storm" not in names
    assert "Earth Power" in names
