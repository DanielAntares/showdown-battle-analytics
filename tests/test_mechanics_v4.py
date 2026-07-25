"""Blind-spot fixes v4: Baton Pass, boost copy/swap/invert, accuracy/evasion,
Transform, Substitute, residual volatiles (Leech Seed / Salt Cure / Curse),
Yawn, confusion, Focus Energy, Ruin abilities, Huge Power, Sharpness.

Ratio assertions use ranges: the damage formula's flat +2 term makes exact
multiplier checks fail by design (documented project gotcha)."""

from src.advisor import SimState
from src.parser import BattleParser, roster_of
from src.pokedex import move_info


def _mon(sp, **kw):
    base = {"species": sp, "hp": 1.0, "status": "", "fainted": False, "active": True,
            "moves": [], "item": "", "volatiles": [], "last_move": "",
            "acc_stage": 0, "eva_stage": 0}
    return {**base, **kw}


def _mk(p1, p2, p1kw=None, p2kw=None):
    game = {"roster": {"p1": [_mon(p1, **(p1kw or {}))], "p2": [_mon(p2, **(p2kw or {}))]}}
    snap = {"turn": 5, "weather": "", "terrain": "", "trickroom": 0}
    for s in ("p1", "p2"):
        snap.update({f"{s}_active_species": game["roster"][s][0]["species"],
                     f"{s}_active_hp": game["roster"][s][0]["hp"],
                     f"{s}_active_status": "", f"{s}_hp_total": 6.0,
                     f"{s}_fainted": 0, f"{s}_statused": 0, f"{s}_healthy": 1})
        snap.update({f"{s}_boost_{x}": 0 for x in ("atk", "def", "spa", "spd", "spe")})
        snap.update({f"{s}_hazard_{h}": 0 for h in
                     ("stealthrock", "spikes", "toxicspikes", "stickyweb")})
        snap.update({f"{s}_screen_{sc}": 0 for sc in
                     ("reflect", "lightscreen", "auroraveil", "tailwind")})
    game["snapshots"] = [snap]
    return game, snap


def _mv(n):
    return dict(move_info(n), name=n)


# ---- parser ------------------------------------------------------------------

BP_LOG = """|player|p1|a|1|1000
|player|p2|b|1|1000
|poke|p1|Scolipede, M|
|poke|p1|Kingambit, M|
|poke|p2|Blissey, F|
|start
|switch|p1a: Scolipede|Scolipede, M|100/100
|switch|p2a: Blissey|Blissey, F|100/100
|turn|1
|move|p1a: Scolipede|Swords Dance|p1a: Scolipede
|-boost|p1a: Scolipede|atk|2
|turn|2
|move|p1a: Scolipede|Substitute|p1a: Scolipede
|-start|p1a: Scolipede|Substitute
|turn|3
|move|p1a: Scolipede|Baton Pass|p1a: Scolipede
|switch|p1a: Kingambit|Kingambit, M|100/100|[from] Baton Pass
|turn|4"""


def _feed(log):
    p = BattleParser()
    for ln in log.splitlines():
        p.feed(ln)
    return p


def test_baton_pass_keeps_boosts_and_sub_normal_switch_clears():
    p = _feed(BP_LOG)
    incoming = next(m for m in roster_of(p)["p1"] if m["active"])
    assert p.sides["p1"].boosts["atk"] == 2
    assert "substitute" in incoming["volatiles"]
    # a plain switch afterwards clears everything
    p.feed("|switch|p1a: Scolipede|Scolipede, M|100/100")
    assert p.sides["p1"].boosts["atk"] == 0 and not p.sides["p1"].volatiles


def test_copy_swap_invert_boosts_and_acc_eva_stages():
    p = _feed(BP_LOG)
    p.feed("|-copyboost|p2a: Blissey|p1a: Kingambit")   # Psych Up
    assert p.sides["p2"].boosts["atk"] == 2
    p.feed("|-invertboost|p1a: Kingambit")              # Topsy-Turvy
    assert p.sides["p1"].boosts["atk"] == -2
    p.feed("|-swapboost|p1a: Kingambit|p2a: Blissey|atk")
    assert p.sides["p1"].boosts["atk"] == 2 and p.sides["p2"].boosts["atk"] == -2
    p.feed("|-unboost|p1a: Kingambit|accuracy|1")
    p.feed("|-boost|p1a: Kingambit|evasion|2")
    active = next(m for m in roster_of(p)["p1"] if m["active"])
    assert active["acc_stage"] == -1 and active["eva_stage"] == 2


def test_transform_copies_boosts_moves_ability():
    p = _feed(BP_LOG)
    p.feed("|move|p2a: Blissey|Seismic Toss|p1a: Kingambit")
    p.feed("|-transform|p2a: Blissey|p1a: Kingambit")
    assert p.sides["p2"].boosts["atk"] == p.sides["p1"].boosts["atk"]
    assert "transformed" in p.sides["p2"].volatiles


# ---- engine: substitute ------------------------------------------------------

def test_substitute_absorbs_blocks_and_breaks():
    game, snap = _mk("Kingambit", "Dondozo", p2kw={"volatiles": ["substitute"]})
    sim = SimState(game, snap)
    sim.use_move("p1", _mv("Kowtow Cleave"))
    assert sim.active["p2"].hp == 1.0          # the mon took nothing
    assert 0 < sim.active["p2"].sub_hp < 0.25  # the sub absorbed it
    sim.use_move("p1", _mv("Kowtow Cleave"))
    assert sim.active["p2"].sub_hp == 0.0      # second hit breaks it
    assert "substitute" not in sim.active["p2"].volatiles
    assert sim.active["p2"].hp == 1.0          # excess damage is lost

    game, snap = _mk("Clodsire", "Dondozo", p2kw={"volatiles": ["substitute"]})
    sim = SimState(game, snap)
    sim.use_move("p1", _mv("Toxic"))
    assert sim.active["p2"].status == ""       # status blocked by the sub


def test_substitute_move_creates_one_at_quarter_hp():
    game, snap = _mk("Kyurem", "Blissey")
    sim = SimState(game, snap)
    sim.use_move("p1", _mv("Substitute"))
    assert abs(sim.active["p1"].hp - 0.75) < 1e-9
    assert sim.active["p1"].sub_hp == 0.25
    sim.use_move("p1", _mv("Substitute"))      # already up: no double cost
    assert abs(sim.active["p1"].hp - 0.75) < 1e-9


# ---- engine: residuals & yawn --------------------------------------------------

def test_leech_seed_drains_to_the_other_side():
    game, snap = _mk("Corviknight", "Gholdengo",
                     p1kw={"volatiles": ["leechseed"], "item": "x"},
                     p2kw={"hp": 0.5, "item": "x"})
    sim = SimState(game, snap)
    sim.upkeep()
    assert abs(sim.active["p1"].hp - (1 - 1 / 8)) < 1e-9
    assert abs(sim.active["p2"].hp - (0.5 + 1 / 8)) < 1e-9


def test_salt_cure_and_ghost_curse_chip():
    game, snap = _mk("Kingambit", "Gholdengo",
                     p1kw={"volatiles": ["saltcure"], "item": "x"})
    sim = SimState(game, snap)
    sim.upkeep()
    assert abs(sim.active["p1"].hp - 0.75) < 1e-9  # steel: 1/4
    game, snap = _mk("Dondozo", "Gholdengo",
                     p1kw={"volatiles": ["curse"], "item": "x"})
    sim = SimState(game, snap)
    sim.upkeep()
    # water type: curse 1/4 (dondozo holds no item here)
    assert abs(sim.active["p1"].hp - 0.75) < 1e-9


def test_yawn_two_stage_sleep():
    game, snap = _mk("Dondozo", "Gholdengo", p1kw={"volatiles": ["yawn"], "item": "x"})
    sim = SimState(game, snap)
    sim.upkeep()
    assert sim.active["p1"].status == "" and "drowsy" in sim.active["p1"].volatiles
    sim.upkeep()
    assert sim.active["p1"].status == "slp"


# ---- engine: expected-value modifiers ------------------------------------------

def test_confusion_and_focus_energy_and_evasion():
    base = SimState(*_mk("Kingambit", "Blissey")).damage_fraction("p1", _mv("Kowtow Cleave"))
    conf = SimState(*_mk("Kingambit", "Blissey", p1kw={"volatiles": ["confusion"]})
                    ).damage_fraction("p1", _mv("Kowtow Cleave"))
    assert abs(conf - base * 2 / 3) < 1e-9
    fe = SimState(*_mk("Kingambit", "Blissey", p1kw={"volatiles": ["focusenergy"]})
                  ).damage_fraction("p1", _mv("Kowtow Cleave"))
    assert abs(fe - base * 1.25) < 1e-9
    ev = SimState(*_mk("Kingambit", "Blissey", p2kw={"eva_stage": 2})
                  ).damage_fraction("p1", _mv("Kowtow Cleave"))
    assert abs(ev - base * 0.6) < 1e-9  # -2 effective accuracy: 3/(3+2)
    # confusion also chips the user in expectation when it acts
    game, snap = _mk("Kingambit", "Blissey", p1kw={"volatiles": ["confusion"]})
    sim = SimState(game, snap)
    sim.use_move("p1", _mv("Kowtow Cleave"))
    assert sim.active["p1"].hp < 1.0


def test_ruin_abilities_huge_power_sharpness():
    vr = SimState(*_mk("Iron Valiant", "Ting-Lu", p2kw={"ability": "Vessel of Ruin"})
                  ).damage_fraction("p1", _mv("Moonblast"))
    nv = SimState(*_mk("Iron Valiant", "Ting-Lu", p2kw={"ability": "Sand Veil"})
                  ).damage_fraction("p1", _mv("Moonblast"))
    assert 0.72 < vr / nv < 0.79           # ~0.75, softened by the +2 flat term
    hp = SimState(*_mk("Azumarill", "Blissey", p1kw={"ability": "Huge Power"})
                  ).damage_fraction("p1", _mv("Play Rough"))
    np_ = SimState(*_mk("Azumarill", "Blissey", p1kw={"ability": "Thick Fat"})
                   ).damage_fraction("p1", _mv("Play Rough"))
    assert 1.9 < hp / np_ <= 2.0
    sh = SimState(*_mk("Samurott-Hisui", "Blissey", p1kw={"ability": "Sharpness"})
                  ).damage_fraction("p1", _mv("Razor Shell"))
    ns = SimState(*_mk("Samurott-Hisui", "Blissey", p1kw={"ability": "Torrent"})
                  ).damage_fraction("p1", _mv("Razor Shell"))
    assert abs(sh - ns * 1.5) < 1e-9       # frac-level multiplier: exact
    sw = SimState(*_mk("Chien-Pao", "Blissey", p1kw={"ability": "Sword of Ruin"})
                  ).damage_fraction("p1", _mv("Icicle Crash"))
    nw = SimState(*_mk("Chien-Pao", "Blissey", p1kw={"ability": "Pressure"})
                  ).damage_fraction("p1", _mv("Icicle Crash"))
    assert sw > nw * 1.15                  # defender's Def dampened 25%
