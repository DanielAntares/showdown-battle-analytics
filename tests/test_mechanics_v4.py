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


def test_good_as_gold_blocks_status_and_is_pruned():
    from src.advisor import moves_for
    game, snap = _mk("Slowking-Galar", "Gholdengo")  # GaG is Gholdengo's only ability
    sim = SimState(game, snap)
    sim.use_move("p1", _mv("Thunder Wave"))
    assert sim.active["p2"].status == ""
    me = game["roster"]["p1"][0]
    me["moves"] = ["Thunder Wave", "Sludge Bomb", "Future Sight", "Chilly Reception"]
    assert "Thunder Wave" not in [m["name"] for m in moves_for(me, snap, "p1", game)]


def test_poison_heal_heals_and_toxic_is_pruned():
    from src.advisor import moves_for
    game, snap = _mk("Gliscor", "Kingambit", p1kw={"status": "tox", "item": "toxicorb"})
    sim = SimState(game, snap)
    sim.upkeep()
    assert sim.active["p1"].hp >= 1.0 - 1e-9   # healed, not chipped
    game, snap = _mk("Clodsire", "Gliscor")
    me = game["roster"]["p1"][0]
    me["moves"] = ["Toxic", "Earthquake", "Recover", "Stealth Rock"]
    assert "Toxic" not in [m["name"] for m in moves_for(me, snap, "p1", game)]


def test_magic_bounce_reflects_and_is_pruned():
    from src.advisor import moves_for
    game, snap = _mk("Ting-Lu", "Hatterene")
    sim = SimState(game, snap)
    sim.use_move("p1", _mv("Stealth Rock"))
    assert sim.snap["p1_hazard_stealthrock"] == 1     # bounced onto the user
    assert sim.snap["p2_hazard_stealthrock"] == 0
    sim2 = SimState(game, snap)
    sim2.use_move("p1", _mv("Toxic"))
    assert sim2.active["p1"].status == "tox" and sim2.active["p2"].status == ""
    me = game["roster"]["p1"][0]
    me["moves"] = ["Stealth Rock", "Toxic", "Earthquake", "Whirlwind"]
    names = [m["name"] for m in moves_for(me, snap, "p1", game)]
    assert "Stealth Rock" not in names and "Toxic" not in names


def test_weak_armor_triggers_on_physical_only():
    game, snap = _mk("Kingambit", "Ceruledge", p2kw={"ability": "Weak Armor"})
    sim = SimState(game, snap)
    sim.use_move("p1", _mv("Kowtow Cleave"))
    assert sim.active["p2"].boosts["def"] == -1
    assert sim.active["p2"].boosts["spe"] == 2
    sim2 = SimState(game, snap)
    sim2.use_move("p1", _mv("Shadow Ball"))
    assert sim2.active["p2"].boosts["spe"] == 0


def test_dauntless_shield_defiant_and_surges_on_switch():
    # Dauntless Shield: +1 Def the moment Zamazenta lands
    game, snap = _mk("Corviknight", "Kingambit")
    game["roster"]["p1"].append(_mon("Zamazenta", active=False))
    sim = SimState(game, snap)
    sim.switch("p1", game["roster"]["p1"][1])
    assert sim.active["p1"].boosts["def"] == 1
    # Intimidate into Defiant: -1 then +2 = net +1 Atk
    game, snap = _mk("Corviknight", "Kingambit", p2kw={"ability": "Defiant"})
    game["roster"]["p1"].append(_mon("Landorus-Therian", active=False))
    sim = SimState(game, snap)
    sim.switch("p1", game["roster"]["p1"][1])
    assert sim.active["p2"].boosts["atk"] == 1
    # Drizzle sets rain; Grassy Surge sets terrain
    game, snap = _mk("Corviknight", "Kingambit")
    game["roster"]["p1"].append(_mon("Pelipper", active=False))
    sim = SimState(game, snap)
    sim.switch("p1", game["roster"]["p1"][1])
    assert sim.snap["weather"] == "raindance"
    game, snap = _mk("Corviknight", "Kingambit")
    game["roster"]["p1"].append(_mon("Rillaboom", active=False))
    sim = SimState(game, snap)
    sim.switch("p1", game["roster"]["p1"][1])
    assert sim.snap["terrain"] == "grassyterrain"


def test_swift_swim_family_doubles_speed_in_weather():
    game, snap = _mk("Barraskewda", "Corviknight")
    base = SimState(game, snap).speed("p1")
    snap2 = dict(snap, weather="raindance")
    assert abs(SimState(game, snap2).speed("p1") - 2 * base) < 1e-6


def test_damage_inference_learns_hidden_power():
    """Hits landing ~1.4x the usage-spread prediction reveal a Band/max-invest
    set; the engine should scale that attacker's future damage. Crits, faints
    and near-noise deviations must be ignored."""
    from src.advisor import damage_mults, expected_hit
    ctx = {"side": "p1", "move": "Earthquake", "crit": False, "turn": 3,
           "attacker": "Dragonite", "atk_item": "", "atk_status": "", "atk_tera": "",
           "atk_boosts": {}, "atk_fainted": 0, "defender": "Kingambit",
           "def_item": "", "def_status": "", "def_tera": "", "def_boosts": {},
           "def_screens": [], "weather": "", "terrain": ""}
    exp = expected_hit({**ctx, "def_hp": 1.0, "after": 1.0})
    assert exp > 0.3  # EQ is super-effective on Kingambit — sanity
    hot = [{**ctx, "def_hp": 1.0, "after": 1.0 - exp * 1.4},
           {**ctx, "def_hp": 0.55, "after": 0.55 - exp * 1.35}]
    mults = damage_mults({"dmg_obs": hot})
    assert 1.3 < mults[("Dragonite", "Physical")] <= 1.45
    # deadband: ordinary damage-roll variance learns nothing
    assert damage_mults({"dmg_obs": [{**ctx, "def_hp": 1.0, "after": 1.0 - exp * 1.05}]}) == {}
    # crits are excluded
    assert damage_mults({"dmg_obs": [{**ctx, "crit": True,
                                      "def_hp": 1.0, "after": 1.0 - exp * 1.5}]}) == {}
    # the engine applies the learned multiplier
    game, snap = _mk("Dragonite", "Kingambit")
    base = SimState(game, snap).damage_fraction("p1", _mv("Earthquake"))
    game["dmg_obs"] = hot
    game.pop("_dmg_mults", None)
    adj = SimState(game, snap).damage_fraction("p1", _mv("Earthquake"))
    assert 1.3 < adj / base <= 1.45


def test_parser_records_damage_observations():
    log = ("|player|p1|a|1|1000\n|player|p2|b|1|1000\n"
           "|poke|p1|Dragonite, M|\n|poke|p2|Kingambit, M|\n|start\n"
           "|switch|p1a: Dragonite|Dragonite, M|100/100\n"
           "|switch|p2a: Kingambit|Kingambit, M|100/100\n|turn|1\n"
           "|move|p1a: Dragonite|Earthquake|p2a: Kingambit\n"
           "|-damage|p2a: Kingambit|42/100\n"
           "|move|p2a: Kingambit|Sucker Punch|p1a: Dragonite\n"
           "|-crit|p1a: Dragonite\n"
           "|-damage|p1a: Dragonite|60/100\n|turn|2")
    p = _feed(log)
    obs = p.dmg_obs
    assert len(obs) == 2
    assert obs[0]["attacker"] == "Dragonite" and not obs[0]["crit"]
    assert abs(obs[0]["def_hp"] - 1.0) < 1e-9 and abs(obs[0]["after"] - 0.42) < 1e-9
    assert obs[1]["attacker"] == "Kingambit" and obs[1]["crit"]  # crit flagged


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
