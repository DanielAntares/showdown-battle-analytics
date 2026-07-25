"""Speed inference from observed move order: seeing who went first (at equal
priority) proves a RAW speed relation once visible modifiers are normalized
away — and the engine's turn ordering honors it over usage-guessed stats."""

from src.advisor import SimState, speed_facts
from src.parser import BattleParser, game_state
from src.pokedex import move_info

# Corviknight is predicted much slower than Dragapult (170 vs 421 effective),
# so any "Corviknight moved first" observation contradicts the stat guess.
BASE = """|player|p1|me|1|1500
|player|p2|foe|1|1500
|poke|p1|Corviknight, F|
|poke|p2|Dragapult, M|
|start
|switch|p1a: Corviknight|Corviknight, F|100/100
|switch|p2a: Dragapult|Dragapult, M|100/100
|turn|1
|move|p1a: Corviknight|Brave Bird|p2a: Dragapult
|move|p2a: Dragapult|Shadow Ball|p1a: Corviknight
|turn|2"""


def _game(log, p1_spe=0, p2_spe=0):
    p = BattleParser()
    for ln in log.splitlines():
        p.feed(ln)
    g = game_state(p)
    g["snapshots"].append(p.snapshot())
    snap = dict(g["snapshots"][-1])
    snap["p1_boost_spe"], snap["p2_boost_spe"] = p1_spe, p2_spe
    return g, snap


def _first_to_move(g, snap):
    """Race at 5% HP each: whoever the engine orders first KOs the other."""
    bb = dict(move_info("Brave Bird"), name="Brave Bird")
    sb = dict(move_info("Shadow Ball"), name="Shadow Ball")
    sim = SimState(g, snap)
    sim.active["p1"].hp = sim.active["p2"].hp = 0.05
    sim.resolve({"p1": {"kind": "move", "move": bb},
                 "p2": {"kind": "move", "move": sb}})
    if sim.active["p2"].fainted and not sim.active["p1"].fainted:
        return "p1"
    if sim.active["p1"].fainted and not sim.active["p2"].fainted:
        return "p2"
    return None


def test_observed_order_beats_stat_guess():
    g, snap = _game(BASE)
    assert speed_facts(g) == {("Corviknight", "Dragapult"): 1.0}
    assert SimState(g, snap)._observed_first() == "p1"
    assert _first_to_move(g, snap) == "p1"  # engine honors the proven order


def test_new_boost_voids_the_fact():
    """A raw fact only covers contexts it still implies: once Dragapult is at
    +2, 'Corviknight was faster at neutral' says nothing — stats decide."""
    g, snap = _game(BASE, p2_spe=2)
    assert SimState(g, snap)._observed_first() is None
    assert _first_to_move(g, snap) == "p2"


def test_boosted_observation_is_normalized():
    """Moving first WHILE at +2 proves only raw*2 > theirs (c=0.5) — it must
    never be read as 'faster at neutral' after the boost is gone."""
    boosted = BASE.replace("|turn|1", "|turn|1\n|-boost|p1a: Corviknight|spe|2")
    g, snap = _game(boosted)
    assert speed_facts(g) == {("Corviknight", "Dragapult"): 0.5}
    snap["p1_boost_spe"] = 0
    assert SimState(g, snap)._observed_first() is None
    assert _first_to_move(g, snap) == "p2"


def test_priority_moves_prove_nothing():
    prio = BASE.replace("|move|p1a: Corviknight|Brave Bird|p2a: Dragapult",
                        "|move|p1a: Corviknight|Quick Attack|p2a: Dragapult")
    g, _ = _game(prio)
    assert speed_facts(g) == {}


def test_trick_room_flips_the_observation():
    """Under Trick Room the first mover is the SLOWER one — the fact must be
    stored the right way round."""
    tr = BASE.replace("|turn|1", "|-fieldstart|move: Trick Room\n|turn|1")
    g, _ = _game(tr)
    assert speed_facts(g) == {("Dragapult", "Corviknight"): 1.0}


def test_paralysis_context_is_normalized():
    """Outspeeding a PARALYZED mon proves half as much: c = 0.5."""
    par = BASE.replace("|turn|1", "|turn|1\n|-status|p2a: Dragapult|par")
    g, _ = _game(par)
    assert speed_facts(g) == {("Corviknight", "Dragapult"): 0.5}
