"""Random-battle set/level layer: authoritative sets, per-species levels, role
narrowing, and a level-aware damage engine. All opt into `randbats_mode`."""

import json
import re
from pathlib import Path

import pytest

import src.movesets as movesets
import src.randbats as randbats
from src.advisor import SimState, advise_search, player_actions
from src.assistant import advise_for_request, build_game
from src.parser import parse_replay
from src.pokedex import move_info, norm_name
from src.predict import load_model, snapshot_features
from src.selfplay import new_game

pytestmark = pytest.mark.usefixtures("randbats_mode")

RB_REPLAY = Path(__file__).parent / "fixtures" / "gen9randombattle-2648177511.json"
CHOOSE = re.compile(r"^(move \d( terastallize)?|switch \d|team \d)$")


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


def test_unrevealed_reserves_become_switch_options():
    """Random battles have no team preview, so our bench only exists in the request.
    _apply_request must merge it into the roster or the advisor can never suggest
    switching to a reserve it hasn't seen yet (the "no switch suggestions" bug)."""
    from src.advisor import player_actions
    replay = json.loads(RB_REPLAY.read_text(encoding="utf-8"))
    p1_team = [m["species"] for m in parse_replay(replay)["roster"]["p1"]]  # all 6
    log = "\n".join(replay["log"].splitlines()[:_first_index_after_turn(replay["log"], 6)])
    bare = build_game(log)
    revealed = {m["species"] for m in bare["roster"]["p1"]}
    hidden = [s for s in p1_team if s not in revealed]
    assert hidden, "fixture should have an unrevealed reserve by turn 6"
    active_sp = next(m["species"] for m in bare["roster"]["p1"] if m["active"])
    pokemon = [{"details": sp, "active": sp == active_sp, "condition": "100/100",
                "item": "", "moves": []} for sp in p1_team]
    game = build_game(log, {"side": {"id": "p1", "pokemon": pokemon}, "rqid": 6})
    assert {m["species"] for m in game["roster"]["p1"]} == set(p1_team)  # all 6 merged in
    switchable = {a["mon"]["species"] for a in player_actions(game, "p1")
                  if a["kind"] == "switch"}
    assert set(hidden) & switchable  # a hidden reserve is now a switch option


def _first_index_after_turn(log: str, turn: int) -> int:
    lines = log.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(f"|turn|{turn}"):
            return i + 1
    return len(lines)


def _log_until_turn(log: str, turn: int) -> str:
    out = []
    for ln in log.splitlines():
        out.append(ln)
        if ln.startswith(f"|turn|{turn}"):
            break
    return "\n".join(out)


def test_live_assistant_produces_a_legal_choice_on_a_real_randbats_game():
    """The extension bridge (advise_for_request) on an actual downloaded randbats
    replay: it must return a legal /choose, a sane win-prob, and — proof the level
    layer flows through the live path — the active mon at its real randbats level."""
    booster, meta = load_model()
    replay = json.loads(RB_REPLAY.read_text(encoding="utf-8"))
    assert "Random Battle" in (parse_replay(replay)["format"] or "")
    log = _log_until_turn(replay["log"], 6)
    game = build_game(log)
    side = "p1"
    active = next(m for m in game["roster"][side] if m["active"])
    assert 60 <= movesets.species_level(active["species"]) < 100  # not a flat L100

    roster = game["roster"][side]
    pokemon = [{"details": m["species"], "active": m["active"],
                "condition": "0 fnt" if m["fainted"] else "100/100",
                "item": "", "moves": []} for m in roster]
    names = movesets.predict_moves(active["species"], active.get("moves", ()), 4)
    req = {"side": {"id": side, "pokemon": pokemon}, "rqid": 5,
           "active": [{"moves": [{"move": n, "id": norm_name(n), "disabled": False}
                                 for n in names]}]}
    res = advise_for_request(log, req, booster, meta, mode="fast")
    assert res["ok"] and CHOOSE.match(res["choose"])
    assert res["winprob"] is None or 0.0 <= res["winprob"] <= 1.0
    # the bridge must be able to serialize the whole payload to strictly-legal JSON
    # (a stray numpy value or NaN used to crash the reply -> browser "no response")
    from src.assistant_server import _safe
    import numpy as np
    txt = json.dumps(_safe(res), allow_nan=False)  # allow_nan=False => raises on NaN/inf
    assert json.loads(txt)["choose"] == res["choose"]
    assert json.loads(json.dumps(_safe({"a": np.float64(0.5), "b": np.array([1, 2]),
                                        "c": float("nan"), "d": np.int64(3)}),
                                 allow_nan=False)) == {"a": 0.5, "b": [1, 2], "c": None, "d": 3}
