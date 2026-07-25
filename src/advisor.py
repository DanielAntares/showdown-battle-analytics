"""Best-action search (advisor v2): 1-ply minimax over the joint action matrix.

For every pairing of (my action) x (opponent's plausible response) the turn is
simulated with an approximate battle engine — real damage formula at level 100,
speed/priority ordering, STAB, type chart, boosts, burn/paralysis, screens,
weather, hazard chip on switches, and common utility effects (status infliction,
stat boosts, hazard setting, healing). Each resulting state is scored by the
win-probability model; the recommended action maximizes the worst-case outcome.

Stated assumptions (visible in the UI): only information revealed in the battle
is used; unrevealed movesets fall back to typical STAB attacks; stats assume
31 IVs / 85 EVs; items, abilities, and Tera are not modeled. v3 would swap this
engine for Showdown's own simulator.
"""

import numpy as np
import pandas as pd

from src.movesets import predict_moves, real_stats, species_set
from src.pokedex import effectiveness, lookup, move_info, norm_name
from src.predict import calibrate

# priority attacks that fail outright unless the target chose an attacking move
FAILS_VS_NONATTACK = {"suckerpunch", "thunderclap"}
# two-turn moves (charge/semi-invulnerable) and move-callers: a 1-ply search
# can't value the two-turn commitment or predict what a caller copies, so the
# advisor abstains from recommending them (and won't phantom-threaten with them)
UNVALUABLE_MOVES = {
    "dig", "fly", "bounce", "dive", "phantomforce", "shadowforce", "skyattack",
    "solarbeam", "solarblade", "meteorbeam", "electroshot", "skydrop", "geomancy",
    "freezeshot", "iceburn", "razorwind", "futuresight", "doomdesire",
    "copycat", "metronome", "mirrormove", "assist", "naturepower", "mefirst",
}
# abilities granting outright immunity to a move type
ABILITY_IMMUNE = {"levitate": "ground", "flashfire": "fire", "wellbakedbody": "fire",
                  "waterabsorb": "water", "stormdrain": "water", "dryskin": "water",
                  "voltabsorb": "electric", "lightningrod": "electric",
                  "motordrive": "electric", "sapsipper": "grass",
                  "eartheater": "ground"}
SCREEN_DURATION, FIELD_DURATION = 5, 5
# tempo cost of switching (win-prob points): the free turn given up plus hazard
# chip and momentum the 1-ply search can't see. Applied symmetrically — my switch
# costs me, and the opponent switching (e.g. to deny a KO) concedes tempo to me —
# so the advisor curbs reflexive switching AND isn't fooled into thinking a KO is
# worthless just because the opponent could pivot the target out.
SWITCH_COST = 0.06
# Ogerpon's Ivy Cudgel takes the mask's type (stored as its Grass base) — without
# this the engine treats Wellspring's Ivy Cudgel as Grass and misses Water Absorb.
IVY_CUDGEL_TYPE = {"ogerponwellspring": "water", "ogerponhearthflame": "fire",
                   "ogerponcornerstone": "rock"}
# moves boosted 1.5x by Sharpness (Samurott-Hisui, Gallade, ...)
SLICING_MOVES = {
    "aircutter", "airslash", "aquacutter", "behemothblade", "bitterblade",
    "ceaselessedge", "crosspoison", "cut", "furycutter", "kowtowcleave",
    "leafblade", "mightycleave", "nightslash", "psyblade", "psychocut",
    "razorleaf", "razorshell", "sacredsword", "secretsword", "slash",
    "solarblade", "stoneaxe", "xscissor",
}
# reward for a net material lead (their faints minus mine) in the resulting
# position: the win-prob model under-values KOs — after a kill the opponent's
# next (fresh, scarier) Pokemon comes in, and the model reads that board as only
# marginally better, so a lethal move can tie or lose to a setup/hazard move on
# noise. This nudge makes securing a KO count. deep_search shares this constant.
MATERIAL_BONUS = 0.03
# phazing moves that ONLY force a switch (no damage) — useless with no switch-in
PHAZE_STATUS = {"whirlwind", "roar"}


def predicted_item(mon: dict) -> str:
    """Revealed item if known, else the species' most common item on the ladder.
    An item known to be gone (consumed/knocked/popped) stays empty."""
    if mon.get("item_consumed"):
        return ""
    if mon.get("item"):
        return norm_name(mon["item"])
    entry = species_set(mon.get("species") or "")
    return entry["item"][0][0] if entry and entry.get("item") else ""


def predicted_ability(mon: dict) -> str:
    if mon.get("ability"):
        return norm_name(mon["ability"])
    entry = species_set(mon.get("species") or "")
    return entry["ability"][0][0] if entry and entry.get("ability") else ""


def predicted_tera(mon: dict) -> str:
    if mon.get("tera"):
        return mon["tera"].lower()
    entry = species_set(mon.get("species") or "")
    return entry["tera"][0][0] if entry and entry.get("tera") else ""


def _grounded(active) -> bool:
    """Flying-types and Levitate float (Boots/Balloon stay hidden info)."""
    return "flying" not in active.types and active.ability != "levitate"

BOOST_STATS = ("atk", "def", "spa", "spd", "spe")
HAZARDS = ("stealthrock", "spikes", "toxicspikes", "stickyweb")
HAZARD_MAX = {"stealthrock": 1, "spikes": 3, "toxicspikes": 2, "stickyweb": 1}
SCREENS = ("reflect", "lightscreen", "auroraveil", "tailwind")
# types that cannot receive a given major status (independent of the move's type)
STATUS_IMMUNE = {"brn": {"fire"}, "par": {"electric"}, "frz": {"ice"},
                 "tox": {"steel", "poison"}, "psn": {"steel", "poison"}}


def status_lands(inflicts: str, move_type: str, target_types) -> bool:
    """Whether a status move would actually apply: the move's type must not be
    immune (Thunder Wave/Electric vs Ground) and the target's type must be able
    to take the status (Fire can't burn, Steel/Poison can't be poisoned, ...)."""
    tt = target_types or []
    if effectiveness(move_type, tt) == 0:
        return False
    return not any(t in STATUS_IMMUNE.get(inflicts, ()) for t in tt)


def boost_mult(stage: int) -> float:
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def _obs_mult(ctx: dict) -> float:
    """The visible speed multiplier a side had when an observation was made."""
    m = boost_mult(ctx.get("spe_stage", 0))
    if ctx.get("status") == "par":
        m *= 0.5
    if ctx.get("tailwind"):
        m *= 2.0
    return m


def speed_facts(game: dict) -> dict:
    """Proven raw-speed relations from observed move order.

    When both sides moved in one turn with equal move priority, the first mover
    was effectively faster *under that turn's visible modifiers*. Normalizing
    those away (boosts, paralysis, Tailwind; Trick Room flips who "won") yields
    a durable fact about the mons' RAW speeds — item and spread included, which
    is exactly what usage-based stat guesses get wrong (a Scarf or max-speed
    variant reveals itself the first time it outspeeds something it "shouldn't").

    Returns {(fast_species, slow_species): c} meaning raw(fast) > c · raw(slow),
    keeping the strongest (largest) c per pair. An observation made at +2 only
    proves raw×2 was faster — c=0.5 — so it can never be misread as "faster at
    neutral", per the normalization.
    """
    facts: dict = {}
    for o in game.get("speed_obs") or []:
        pr1 = (move_info(o["first_move"]) or {}).get("priority", 0) or 0
        pr2 = (move_info(o["second_move"]) or {}).get("priority", 0) or 0
        if pr1 != pr2:
            continue  # a priority mismatch explains the order by itself
        f, s = o["first"], o["second"]
        if o.get("trickroom"):
            f, s = s, f  # under Trick Room the first mover is the SLOWER one
        c = _obs_mult(o["ctx"][s]) / _obs_mult(o["ctx"][f])
        key = (o["species"][f], o["species"][s])
        facts[key] = max(facts.get(key, 0.0), c)
    return facts


def expected_hit(o: dict) -> float:
    """What the engine would have predicted for an observed hit, rebuilt from
    the observation's recorded context (boosts, status, screens, weather, Tera,
    revealed items — unrevealed ones fall back to the usage guess, which is
    exactly the belief we are testing)."""
    def _m(sp, item, status, tera, hp=1.0):
        return {"species": sp, "hp": hp, "status": status, "fainted": False,
                "active": True, "moves": [], "item": item, "tera": tera,
                "volatiles": [], "last_move": "", "acc_stage": 0, "eva_stage": 0}

    game = {"roster": {
        "p1": [_m(o["attacker"], o.get("atk_item", ""), o.get("atk_status", ""),
                  o.get("atk_tera", ""))],
        "p2": [_m(o["defender"], o.get("def_item", ""), o.get("def_status", ""),
                  o.get("def_tera", ""), hp=o.get("def_hp", 1.0))]}}
    snap = {"turn": o.get("turn", 5), "weather": o.get("weather", ""),
            "terrain": o.get("terrain", ""), "trickroom": 0,
            "p1_fainted": o.get("atk_fainted", 0), "p2_fainted": 0}
    for s, boosts in (("p1", o.get("atk_boosts") or {}), ("p2", o.get("def_boosts") or {})):
        snap.update({f"{s}_active_species": game["roster"][s][0]["species"],
                     f"{s}_active_hp": game["roster"][s][0]["hp"],
                     f"{s}_active_status": game["roster"][s][0]["status"],
                     f"{s}_hp_total": 6.0, f"{s}_statused": 0, f"{s}_healthy": 6})
        snap.update({f"{s}_boost_{b}": boosts.get(b, 0) for b in BOOST_STATS})
        snap.update({f"{s}_hazard_{h}": 0 for h in HAZARD_MAX})
        snap.update({f"{s}_screen_{sc}": 0 for sc in SCREENS})
    for sc in o.get("def_screens") or []:
        if sc in SCREENS:
            snap[f"p2_screen_{sc}"] = 1
    snap["p1_fainted"] = o.get("atk_fainted", 0)
    info = move_info(o["move"])
    if not info:
        return 0.0
    game["snapshots"] = [snap]
    return SimState(game, snap).damage_fraction("p1", dict(info, name=o["move"]))


def damage_mults(game: dict) -> dict:
    """Hidden power revealed by observed damage: per (attacker species, category)
    the median observed/expected ratio, when it deviates meaningfully.

    A Banded/max-Attack variant of a mon the usage stats call defensive shows up
    as ratios ~1.3–1.5 across its hits; the engine then scales that attacker's
    future damage. Deliberately conservative: crits, multi-hit, fixed damage and
    faint-truncated hits are skipped, a ±15% deadband absorbs damage-roll noise
    (rolls alone are ±8%), and the multiplier clamps to [0.75, 1.45]."""
    from statistics import median
    ratios: dict = {}
    for o in game.get("dmg_obs") or []:
        if o.get("crit"):
            continue
        info = move_info(o["move"])
        if not info or info.get("multihit", 1) != 1 or info.get("fixed") \
                or not info.get("power", 0):
            continue
        if o.get("after", 1.0) <= 0.011:
            continue  # faint/Sash truncated what we could observe
        exp = expected_hit(o)
        if exp < 0.04:
            continue  # too small to separate signal from rounding
        observed = o.get("def_hp", 1.0) - o.get("after", 1.0)
        if observed <= 0:
            continue
        ratios.setdefault((o["attacker"], info["category"]), []).append(observed / exp)
    out = {}
    for key, rs in ratios.items():
        m = median(rs)
        if abs(m - 1.0) > 0.15:
            out[key] = max(0.75, min(1.45, m))
    return out


def hazard_chip(species: str, side_snapshot_prefix: str, snapshot: dict,
                item: str = "", ability: str = "") -> float:
    """Fraction of max HP lost to entry hazards when this species switches in."""
    if item == "heavydutyboots":
        return 0.0
    entry = lookup(species)
    if entry is None:
        return 0.0
    types = entry["types"]
    chip = 0.0
    if snapshot[f"{side_snapshot_prefix}_hazard_stealthrock"]:
        chip += 0.125 * effectiveness("rock", types)
    if "flying" not in types and ability != "levitate":
        spikes = snapshot[f"{side_snapshot_prefix}_hazard_spikes"]
        chip += {0: 0.0, 1: 1 / 8, 2: 1 / 6, 3: 1 / 4}[spikes]
    return min(chip, 1.0)


class _Active:
    def __init__(self, mon: dict, boosts: dict | None = None):
        self.species = mon["species"]
        self.hp = mon["hp"]
        self.status = mon["status"]
        self.fainted = mon["fainted"]
        self.sleep_turns = mon.get("sleep_turns", 0)
        self.tox_turns = mon.get("tox_turns", 0)
        self.volatiles = set(mon.get("volatiles") or [])
        self.semiinvuln = "semiinvuln" in self.volatiles
        self.sub_hp = 0.25 if "substitute" in self.volatiles else 0.0
        self.acc_stage = mon.get("acc_stage", 0)
        self.eva_stage = mon.get("eva_stage", 0)
        dex = lookup(mon["species"])
        self.orig_types = dex["types"] if dex else []
        tera = (mon.get("tera") or "").lower()
        self.tera = tera  # persisted Tera type ("" if not yet used)
        self.did_tera = False  # Tera'd this turn (for multi-turn state carry-forward)
        self.types = [tera] if tera else list(self.orig_types)  # already Tera'd?
        self.stab_types = set(self.orig_types) | ({tera} if tera else set())
        self.item = predicted_item(mon)
        self.ability = predicted_ability(mon)
        self.stats = dict(real_stats(mon["species"]))  # level-100, predicted spread
        if self.item == "boosterenergy" and self.ability in ("protosynthesis", "quarkdrive"):
            best = max((s for s in BOOST_STATS), key=lambda s: self.stats[s])
            self.stats[best] = int(self.stats[best] * (1.5 if best == "spe" else 1.3))
        self.boosts = dict(boosts) if boosts else {s: 0 for s in BOOST_STATS}

    def terastallize(self, tera_type: str) -> None:
        self.types = [tera_type]
        self.stab_types = set(self.orig_types) | {tera_type}
        self.tera = tera_type
        self.did_tera = True


class SimState:
    """A rough but honest one-turn battle engine over the snapshot schema."""

    def __init__(self, game: dict, snap: dict):
        self.snap = dict(snap)
        self.game = game
        self.field = game.get("field") or {}
        self.rosters = game.get("roster") or {}
        self.active = {}
        for side in ("p1", "p2"):
            mon = next((m for m in game["roster"][side]
                        if m["species"] == snap[f"{side}_active_species"]), None)
            mon = mon or {"species": snap[f"{side}_active_species"],
                          "hp": snap[f"{side}_active_hp"],
                          "status": snap[f"{side}_active_status"], "fainted": False}
            self.active[side] = _Active(
                mon, {s: snap[f"{side}_boost_{s}"] for s in BOOST_STATS})

    def _opp(self, side: str) -> str:
        return "p2" if side == "p1" else "p1"

    def speed(self, side: str) -> float:
        a = self.active[side]
        spe = a.stats["spe"] * boost_mult(a.boosts["spe"])
        if a.item == "choicescarf":
            spe *= 1.5
        if self.snap.get(f"{side}_screen_tailwind"):
            spe *= 2
        return spe * (0.5 if a.status == "par" else 1.0)

    def _visible_spe_mult(self, side: str) -> float:
        """Current visible modifiers only — same normalization speed_facts uses."""
        a = self.active[side]
        m = boost_mult(a.boosts["spe"])
        if a.status == "par":
            m *= 0.5
        if self.snap.get(f"{side}_screen_tailwind"):
            m *= 2.0
        return m

    def _observed_first(self) -> str | None:
        """The side PROVEN to act first at equal priority under the *current*
        modifiers, from move order observed earlier in the game — or None when
        no observation covers this context (e.g. the fact was learned at +2 and
        the boost is gone). Overrides the stat-guess ordering when it fires."""
        facts = self.game.get("_speed_facts")
        if facts is None:
            facts = self.game["_speed_facts"] = speed_facts(self.game)
        if not facts:
            return None
        a, b = self.active["p1"].species, self.active["p2"].species
        ma, mb = self._visible_spe_mult("p1"), self._visible_spe_mult("p2")
        faster = None
        c = facts.get((a, b))
        if c is not None and c * ma >= mb:   # raw_a > c·raw_b ⇒ a still wins now
            faster = "p1"
        elif (c2 := facts.get((b, a))) is not None and c2 * mb >= ma:
            faster = "p2"
        if faster is None:
            return None
        if self.snap.get("trickroom"):  # slower side acts first under Trick Room
            return "p2" if faster == "p1" else "p1"
        return faster

    def switch(self, side: str, mon: dict) -> None:
        old = self.active[side]
        if old.ability == "regenerator" and not old.fainted and old.hp > 0:
            self.snap[f"{side}_hp_total"] += min(1 / 3, 1.0 - old.hp)
        incoming = _Active(mon)
        chip = hazard_chip(mon["species"], side, self.snap,
                           incoming.item, incoming.ability)
        incoming.hp = max(mon["hp"] - chip, 0.0)
        self.active[side] = incoming
        self.snap[f"{side}_hp_total"] -= min(chip, mon["hp"])
        if incoming.hp <= 0:
            incoming.fainted = True
            self.snap[f"{side}_fainted"] += 1
        elif incoming.ability == "intimidate":
            opp = self._opp(side)
            self.snap[f"{opp}_boost_atk"] = max(-6, self.snap[f"{opp}_boost_atk"] - 1)
            self.active[opp].boosts["atk"] = self.snap[f"{opp}_boost_atk"]

    def damage_fraction(self, side: str, move: dict) -> float:
        atk, dfn = self.active[side], self.active[self._opp(side)]
        if dfn.semiinvuln and norm_name(move["name"]) not in ("earthquake", "magnitude"):
            return 0.0  # underground/in-air (Dig/Fly): the attack misses
        if move["type"] == "ground" and dfn.item == "airballoon":
            return 0.0  # Air Balloon: immune to Ground moves until it pops
        if ABILITY_IMMUNE.get(dfn.ability) == move["type"]:
            return 0.0  # Levitate / Flash Fire / Water Absorb / ...
        if effectiveness(move["type"], dfn.types) == 0:
            return 0.0
        if move.get("fixed"):  # Seismic Toss / Night Shade: level = 100 damage
            dmg = 100 if move["fixed"] == "level" else float(move["fixed"])
            return dmg / dfn.stats["hp"] * move.get("accuracy", 1.0)

        physical = move["category"] == "Physical"
        weather = self.snap.get("weather", "")
        terrain = self.snap.get("terrain", "")
        # offensive stat: Foul Play uses the target's Attack; Body Press the
        # user's Defense; otherwise Atk/SpA — with the matching boost stage,
        # which Unaware defenders ignore
        off_owner = dfn if move.get("off_pokemon") == "target" else atk
        off_stat = move.get("off_stat") or ("atk" if physical else "spa")
        a_stat = off_owner.stats[off_stat]
        if dfn.ability != "unaware":
            a_stat *= boost_mult(off_owner.boosts.get(off_stat, 0))
        if atk.status and atk.ability == "guts":
            a_stat *= 1.5  # Guts: status boosts instead of hindering
        elif physical and atk.status == "brn":
            a_stat *= 0.5
        if atk.item == "choiceband" and physical:
            a_stat *= 1.5
        if atk.item == "choicespecs" and not physical:
            a_stat *= 1.5
        if atk.ability == "supremeoverlord":
            a_stat *= 1 + 0.1 * self.snap.get(f"{side}_fainted", 0)
        if off_owner is atk and physical and atk.ability in ("hugepower", "purepower"):
            a_stat *= 2
        # Ruin abilities dampen the OTHER side's stat (Ting-Lu, Gouging Fire, ...)
        if (dfn.ability == "tabletsofruin" and physical) or \
                (dfn.ability == "vesselofruin" and not physical):
            a_stat *= 0.75

        def_stat = move.get("def_stat") or ("def" if physical else "spd")
        d_stat = dfn.stats[def_stat]
        if atk.ability != "unaware":
            d_stat *= boost_mult(dfn.boosts.get(def_stat, 0))
        if (atk.ability == "swordofruin" and def_stat == "def") or \
                (atk.ability == "beadsofruin" and def_stat == "spd"):
            d_stat *= 0.75
        if dfn.item == "assaultvest" and not physical:
            d_stat *= 1.5
        if dfn.item == "eviolite":
            d_stat *= 1.5
        if weather == "sandstorm" and not physical and "rock" in dfn.types:
            d_stat *= 1.5  # sand boosts Rock-types' SpD
        if weather in ("snow", "snowscape") and physical and "ice" in dfn.types:
            d_stat *= 1.5  # snow boosts Ice-types' Def

        dmg = (42 * move["power"] * a_stat / d_stat) / 50 + 2
        frac = dmg / dfn.stats["hp"] * 0.925  # avg roll
        frac *= move.get("multihit", 1)
        frac *= 1.5 if move["type"] in atk.stab_types else 1.0
        frac *= effectiveness(move["type"], dfn.types)
        if atk.item == "lifeorb":
            frac *= 1.3
        if atk.ability == "sharpness" and norm_name(move.get("name", "")) in SLICING_MOVES:
            frac *= 1.5
        if "focusenergy" in atk.volatiles:
            frac *= 1.25  # ~50% crit chance × 1.5 crit damage, in expectation
        if "confusion" in atk.volatiles:
            frac *= 2 / 3  # 33% chance to hit itself instead
        if dfn.ability in ("multiscale", "shadowshield") and dfn.hp >= 0.999:
            frac *= 0.5
        if dfn.ability == "thickfat" and move["type"] in ("fire", "ice"):
            frac *= 0.5
        if atk.status == "par":
            frac *= 0.75  # expected value of the 25% full-paralysis chance
        if weather in ("raindance", "rain", "primordialsea"):
            frac *= {"water": 1.5, "fire": 0.5}.get(move["type"], 1.0)
        elif weather in ("sunnyday", "sun", "desolateland"):
            frac *= {"fire": 1.5, "water": 0.5}.get(move["type"], 1.0)
        if terrain and _grounded(atk):
            frac *= {"electricterrain": {"electric": 1.3},
                     "grassyterrain": {"grass": 1.3},
                     "psychicterrain": {"psychic": 1.3}}.get(terrain, {}).get(move["type"], 1.0)
        if terrain == "mistyterrain" and move["type"] == "dragon" and _grounded(dfn):
            frac *= 0.5
        opp_side = self._opp(side)
        screens = {s for s in SCREENS if self.snap[f"{opp_side}_screen_{s}"]}
        if "auroraveil" in screens or ("reflect" in screens and physical) \
                or ("lightscreen" in screens and not physical):
            frac *= 0.5
        # observed-damage correction: if this attacker's hits have measured
        # harder/softer than the usage-spread guess, trust the measurements
        mults = self.game.get("_dmg_mults")
        if mults is None:
            mults = self.game["_dmg_mults"] = damage_mults(self.game)
        frac *= mults.get((atk.species, move.get("category")), 1.0)
        # expected hit chance: base accuracy scaled by accuracy/evasion stages
        acc = move.get("accuracy", 1.0) or 1.0
        stage = max(-6, min(6, atk.acc_stage - dfn.eva_stage))
        if stage:
            acc = min(1.0, acc * ((3 + stage) / 3 if stage >= 0 else 3 / (3 - stage)))
        return frac * acc

    def _sleep_talk_proxy(self, side: str) -> dict | None:
        """Sleep Talk calls another move; approximate with the best STAB attack."""
        a, opp = self.active[side], self.active[self._opp(side)]
        if not a.types:
            return None
        best = max(a.types, key=lambda t: effectiveness(t, opp.types))
        cat = "Physical" if a.stats["atk"] >= a.stats["spa"] else "Special"
        return {"name": "(Sleep Talk)", "type": best, "category": cat,
                "power": 80, "accuracy": 1.0, "priority": 0}

    def use_move(self, side: str, move: dict) -> None:
        me, opp_side = self.active[side], self._opp(side)
        opp = self.active[opp_side]
        if me.fainted:
            return
        if norm_name(move.get("name", "")) in ("futuresight", "doomdesire"):
            return  # delayed 2 turns — no immediate effect a 1-ply search can price
        if me.status in ("slp", "frz"):
            # asleep/frozen at turn start: the move fails (can't rely on waking) —
            # only Sleep Talk still acts, calling one of the mon's other moves
            if me.status == "slp" and norm_name(move.get("name", "")) == "sleeptalk":
                move = self._sleep_talk_proxy(side)
                if move is None:
                    return
            else:
                return
        if "confusion" in me.volatiles:
            # expected self-hit: 33% chance × a ~13% typeless 40 BP hit on itself
            chip = min(0.33 * 0.13, me.hp)
            me.hp -= chip
            self.snap[f"{side}_hp_total"] -= chip
            if me.hp <= 0:
                me.fainted = True
                self.snap[f"{side}_fainted"] += 1
                return
        if norm_name(move.get("name", "")) == "rest":
            # full heal at the cost of sleeping; fails (pure no-op) at full HP
            if me.hp < 0.999:
                self.snap[f"{side}_hp_total"] += 1.0 - me.hp
                me.hp = 1.0
                if not me.status:
                    self.snap[f"{side}_statused"] += 1
                me.status = "slp"
            return
        if norm_name(move.get("name", "")) == "substitute":
            if me.sub_hp <= 0 and me.hp > 0.25:  # costs 25%; fails below that
                me.hp -= 0.25
                self.snap[f"{side}_hp_total"] -= 0.25
                me.sub_hp = 0.25
                me.volatiles.add("substitute")
            return
        if move["category"] != "Status" and (move["power"] > 0 or move.get("fixed")):
            frac = self.damage_fraction(side, move)
            if opp.sub_hp > 0 and frac > 0:
                # the substitute soaks the hit; drain/recoil/contact still key off
                # the absorbed amount, but the mon behind it is untouched
                hit = min(frac, opp.sub_hp)
                opp.sub_hp = max(0.0, opp.sub_hp - frac)
                if opp.sub_hp <= 0:
                    opp.volatiles.discard("substitute")
                dealt = 0.0
            else:
                dealt = min(frac, opp.hp)
                if (opp.item == "focussash" and opp.hp >= 0.999 and dealt >= opp.hp):
                    dealt = opp.hp - 0.01  # Sash: survive one hit from full
                hit = dealt
            opp.hp -= dealt
            self.snap[f"{opp_side}_hp_total"] -= dealt
            if opp.hp <= 0:
                opp.hp, opp.fainted = 0.0, True
                self.snap[f"{opp_side}_fainted"] += 1
            if hit > 0 and move.get("drain"):
                healed = min(hit * move["drain"], 1.0 - me.hp)
                me.hp += healed
                self.snap[f"{side}_hp_total"] += healed
            self_dmg = hit * move.get("recoil", 0)
            if hit > 0 and me.item == "lifeorb":
                self_dmg += 0.1
            if hit > 0 and move.get("contact") and opp.item == "rockyhelmet" \
                    and not opp.fainted:
                self_dmg += 1 / 6
            if self_dmg:
                lost = min(self_dmg, me.hp)
                me.hp -= lost
                self.snap[f"{side}_hp_total"] -= lost
                if me.hp <= 0:
                    me.fainted = True
                    self.snap[f"{side}_fainted"] += 1
            # meaningful secondary status (Scald burn, Nuzzle para, ...) — a
            # substitute blocks it (dealt is 0 while the sub is up)
            if (dealt > 0 and move.get("sec_status") and move.get("sec_chance", 0) >= 30
                    and not opp.status and not opp.fainted
                    and not any(t in STATUS_IMMUNE.get(move["sec_status"], ()) for t in opp.types)):
                opp.status = move["sec_status"]
                self.snap[f"{opp_side}_statused"] += 1
            return
        if move.get("inflicts") and not opp.status and not opp.fainted \
                and opp.sub_hp <= 0 \
                and status_lands(move["inflicts"], move["type"], opp.types):
            terrain = self.snap.get("terrain", "")
            terrain_blocked = _grounded(opp) and (
                terrain == "mistyterrain"
                or (terrain == "electricterrain" and move["inflicts"] == "slp"))
            if not terrain_blocked:
                opp.status = move["inflicts"]
                self.snap[f"{opp_side}_statused"] += 1
        if move.get("boosts"):
            target = me if move.get("target") == "self" else opp
            if target is opp and opp.sub_hp > 0:
                pass  # a substitute blocks stat drops aimed at the mon behind it
            else:
                for stat, amt in move["boosts"].items():
                    if stat in target.boosts:
                        target.boosts[stat] = max(-6, min(6, target.boosts[stat] + amt))
        if move.get("side_condition") in HAZARDS:
            h = move["side_condition"]
            self.snap[f"{opp_side}_hazard_{h}"] = min(
                self.snap[f"{opp_side}_hazard_{h}"] + 1, HAZARD_MAX[h])
        elif move.get("side_condition") in SCREENS:
            self.snap[f"{side}_screen_{move['side_condition']}"] = 1
        if move.get("heal"):
            healed = min(0.5, 1.0 - me.hp)
            me.hp += healed
            self.snap[f"{side}_hp_total"] += healed

    def resolve(self, actions: dict) -> None:
        """Play one turn: switches first, then moves by priority and speed."""
        movers = []
        # record what each side did (for multi-turn Choice/Encore locks + switch tracking)
        self.acted = {side: ("switch", act["mon"]["species"]) if act["kind"] == "switch"
                      else ("move", act["move"].get("name", "")) for side, act in actions.items()}
        attacking = {side: act["kind"] == "move"
                     and act["move"].get("category") != "Status"
                     and act["move"].get("power", 0) > 0
                     for side, act in actions.items()}
        for side, act in actions.items():
            if act.get("tera") and not self.snap.get(f"{side}_tera_used"):
                self.active[side].terastallize(act["tera"])
                self.snap[f"{side}_tera_used"] = 1
            if act["kind"] == "switch":
                self.switch(side, act["mon"])
            else:
                movers.append((side, act["move"]))
        trick_room = -1 if self.snap.get("trickroom") else 1
        movers.sort(key=lambda m: (m[1].get("priority", 0),
                                   trick_room * self.speed(m[0])), reverse=True)
        # observed move order beats guessed stats: if the game already proved
        # who's faster in this modifier context, honor it (equal priority only)
        if (len(movers) == 2
                and movers[0][1].get("priority", 0) == movers[1][1].get("priority", 0)
                and (first := self._observed_first()) is not None
                and movers[0][0] != first):
            movers.reverse()
        for side, move in movers:
            if (norm_name(move.get("name", "")) in FAILS_VS_NONATTACK
                    and not attacking.get(self._opp(side))):
                continue  # Sucker Punch-likes whiff when the target isn't attacking
            if (self.snap.get("terrain") == "psychicterrain"
                    and move.get("priority", 0) > 0
                    and _grounded(self.active[self._opp(side)])):
                continue  # Psychic Terrain blocks priority against grounded targets
            self.use_move(side, move)
        self.upkeep()

    def upkeep(self) -> None:
        """End-of-turn residuals: burn/poison/sand chip, volatile chip (Leech
        Seed / Salt Cure / Ghost Curse), Grassy Terrain healing, Yawn drowsiness."""
        weather = self.snap.get("weather", "")
        terrain = self.snap.get("terrain", "")
        for side, a in self.active.items():
            if a.fainted:
                continue
            delta = 0.0
            if a.ability != "magicguard":
                if a.status == "brn":
                    delta -= 1 / 16
                elif a.status == "psn":
                    delta -= 1 / 8
                elif a.status == "tox":
                    delta -= min(a.tox_turns + 1, 15) / 16  # ramping toxic counter
                if weather == "sandstorm" and not ({"rock", "ground", "steel"} & set(a.types)):
                    delta -= 1 / 16
                if "leechseed" in a.volatiles:
                    delta -= 1 / 8
                    foe = self.active[self._opp(side)]
                    if not foe.fainted:  # the seeder's side drains what we lose
                        heal = min(1 / 8, 1.0 - foe.hp)
                        foe.hp += heal
                        self.snap[f"{self._opp(side)}_hp_total"] += heal
                if "saltcure" in a.volatiles:
                    delta -= 1 / 4 if ({"water", "steel"} & set(a.types)) else 1 / 8
                if "curse" in a.volatiles:  # the Ghost-type Curse
                    delta -= 1 / 4
            if a.item == "leftovers":
                delta += 1 / 16
            elif a.item == "blacksludge":
                delta += 1 / 16 if "poison" in a.types else -1 / 8
            if terrain == "grassyterrain" and _grounded(a):
                delta += 1 / 16
            # Yawn: drowsy this turn, asleep at the end of the next one
            if "drowsy" in a.volatiles:
                a.volatiles.discard("drowsy")
                if not a.status:
                    a.status = "slp"
                    self.snap[f"{side}_statused"] += 1
            elif "yawn" in a.volatiles:
                a.volatiles.discard("yawn")
                a.volatiles.add("drowsy")
            if not delta:
                continue
            new_hp = max(0.0, min(1.0, a.hp + delta))
            self.snap[f"{side}_hp_total"] += new_hp - a.hp
            a.hp = new_hp
            if a.hp <= 0 and not a.fainted:
                a.fainted = True
                self.snap[f"{side}_fainted"] += 1

    def _replacement(self, side: str, fainted_species: str) -> dict | None:
        """The mon forced in after a faint. We can't know the real choice, so we
        assume the sensible one: the living benched Pokémon that best resists the
        opposing active's STAB (their best defensive answer), health breaking ties.
        Deterministic and realistic — far better than an arbitrary 'healthiest'."""
        bench = [m for m in self.rosters.get(side, [])
                 if not m["fainted"] and m["hp"] > 0 and m["species"] != fainted_species]
        if not bench:
            return None
        atk = self.active[self._opp(side)]
        atk_types = atk.stab_types or set(atk.types)

        def vulnerability(m):
            dex = lookup(m["species"])
            types = dex["types"] if dex else []
            return max((effectiveness(t, types) for t in atk_types), default=1.0)

        return min(bench, key=lambda m: (vulnerability(m), -m["hp"]))

    def to_snapshot(self) -> dict:
        out = dict(self.snap)
        next_turn = self.snap["turn"] + 1
        out["turn"] = next_turn
        for side in ("p1", "p2"):
            a = self.active[side]
            if a.fainted and (rep := self._replacement(side, a.species)):
                # a fainted active is replaced next turn; showing the 0-HP fainted
                # mon as "active" is out-of-distribution and wrecks the model's read
                out[f"{side}_active_species"] = rep["species"]
                out[f"{side}_active_hp"] = rep["hp"]
                out[f"{side}_active_status"] = rep["status"]
                for s in BOOST_STATS:
                    out[f"{side}_boost_{s}"] = 0
                continue
            out[f"{side}_active_species"] = a.species
            out[f"{side}_active_hp"] = max(a.hp, 0.0)
            out[f"{side}_active_status"] = a.status
            for s in BOOST_STATS:
                out[f"{side}_boost_{s}"] = a.boosts[s]
            # screens set N turns ago expire — the next-turn state must show it
            for screen, set_turn in self.field.get("screen_turns", {}).get(side, {}).items():
                if out.get(f"{side}_screen_{screen}") and \
                        next_turn - set_turn >= SCREEN_DURATION:
                    out[f"{side}_screen_{screen}"] = 0
        if out.get("weather") and self.field.get("weather_set_turn") is not None \
                and next_turn - self.field["weather_set_turn"] >= FIELD_DURATION:
            out["weather"] = ""
        if out.get("terrain") and self.field.get("terrain_set_turn") is not None \
                and next_turn - self.field["terrain_set_turn"] >= FIELD_DURATION:
            out["terrain"] = ""
        return out


def _typical_moves(species: str) -> list[dict]:
    """Last-resort STAB coverage for species absent from usage data."""
    dex = lookup(species)
    if not dex:
        return []
    category = "Physical" if dex["atk"] >= dex["spa"] else "Special"
    return [{"name": f"(likely {t} attack)", "type": t, "category": category,
             "power": 80, "accuracy": 1.0, "priority": 0} for t in dex["types"]]


def is_pure_setup(move: dict) -> bool:
    """A self-boosting move with no damage, healing, or other utility — its only
    payoff is future turns, which are wasted if the user is about to die."""
    return bool(move.get("boosts")) and move.get("category") == "Status" \
        and move.get("target", "self") == "self" \
        and not move.get("heal") and not move.get("inflicts") \
        and not move.get("side_condition")


def _defender_types(game: dict | None, snap: dict, opp: str) -> list | None:
    """The opponent active's current typing, accounting for a used Tera."""
    if not game:
        return None
    sp = snap.get(f"{opp}_active_species")
    entry = next((m for m in game["roster"][opp]
                  if m["species"] == sp and m.get("active")), None)
    if entry and entry.get("tera"):
        return [entry["tera"].lower()]
    dex = lookup(sp)
    return dex["types"] if dex else None


def moves_for(mon: dict, snap: dict | None = None, side: str | None = None,
              game: dict | None = None) -> list[dict]:
    """The moves to evaluate for a Pokémon: revealed plus likely-unrevealed ones
    (usage stats), then filtered for legality (Encore/Taunt/Choice lock) and
    obvious no-ops (healing at full HP, stacking a maxed hazard/screen)."""
    names = predict_moves(mon["species"], revealed=mon.get("moves", ()), k=4)
    moves = [dict(move_info(n), name=move_info(n)["name"]) for n in names if move_info(n)]
    if not moves:
        return _typical_moves(mon["species"])
    mask_type = IVY_CUDGEL_TYPE.get(norm_name(mon["species"]))  # Ogerpon form -> Ivy Cudgel type
    if mask_type:
        for m in moves:
            if norm_name(m["name"]) == "ivycudgel":
                m["type"] = mask_type

    uses = mon.get("uses", {})
    moves = [m for m in moves
             if uses.get(m["name"], 0) < m.get("pp", 16) * 1.6] or moves  # PP exhausted
    # abstain from two-turn / delayed / move-calling moves (see UNVALUABLE_MOVES)
    moves = [m for m in moves if norm_name(m["name"]) not in UNVALUABLE_MOVES] or moves

    volatiles = set(mon.get("volatiles", ()))
    last = norm_name(mon.get("last_move", ""))
    if "encore" in volatiles and last:  # locked into repeating the encored move
        locked = [m for m in moves if norm_name(m["name"]) == last]
        moves = locked or moves
    if "taunt" in volatiles:
        moves = [m for m in moves if m["category"] != "Status"] or moves
    disabled = norm_name(mon.get("disabled", ""))  # Disable / Cursed Body locks out one move
    if disabled:
        moves = [m for m in moves if norm_name(m["name"]) != disabled] or moves
    # Choice items lock into the last move used — via predicted_item, so we stay
    # consistent with the damage engine (which already assumes the Choice boost)
    if predicted_item(mon).startswith("choice") and last:
        locked = [m for m in moves if norm_name(m["name"]) == last]
        moves = locked or moves

    if snap is not None and side is not None:
        opp = "p2" if side == "p1" else "p1"
        opp_types = _defender_types(game, snap, opp)
        opp_entry = next((m for m in (game or {}).get("roster", {}).get(opp, [])
                          if m.get("active")), {})
        opp_underground = "semiinvuln" in (opp_entry.get("volatiles") or [])
        # the opponent's effective ability/item — same values the damage engine
        # scores with, so pruning stays consistent with what it computes as 0
        opp_ability = predicted_ability(opp_entry)
        opp_item = predicted_item(opp_entry)
        opp_status = snap.get(f"{opp}_active_status", "")
        status = mon.get("status", "")
        # does the opponent still have anything to switch in? (team preview means
        # the roster lists all six, so counting un-fainted is exact)
        opp_alive = [x for x in (game or {}).get("roster", {}).get(opp, [])
                     if not x["fainted"]]
        no_switchin = len(opp_alive) == 1  # only their active remains

        def no_effect(m):  # a status move that will simply fail
            if m["category"] != "Status" or not m.get("inflicts"):
                return False
            if opp_status:
                return True  # can't apply a fresh major status over an existing one
            # the move's type must connect AND the target's type must take the status
            # (Thunder Wave/Electric vs Ground, Will-O-Wisp vs Fire, Toxic vs Steel...)
            return not status_lands(m["inflicts"], m["type"], opp_types)

        def soft_drop(m):  # never empties the list on its own (keeps a fallback)
            n = norm_name(m["name"])
            if (m.get("heal") or n == "rest") and mon["hp"] >= 0.99:
                return True  # healing at full HP fails
            if is_pure_setup(m) and (status == "tox"
                                     or (status in ("psn", "brn") and mon["hp"] < 0.5)):
                return True  # don't set up while dying to residual damage
            if no_effect(m):
                return True  # e.g. Thunder Wave into an already-paralyzed target
            sc = m.get("side_condition")
            # hazards and phazing only pay off through a switch-in — dead moves
            # when the opponent has none left (their last Pokémon is on the field)
            if no_switchin and (sc in HAZARD_MAX or n in PHAZE_STATUS):
                return True
            return (sc in HAZARD_MAX and snap[f"{opp}_hazard_{sc}"] >= HAZARD_MAX[sc]) \
                or (sc in SCREENS and snap[f"{side}_screen_{sc}"])

        def whiffs(m):  # zero-damage attack (any immunity source) — may leave only switches
            if m["category"] == "Status" or not m.get("power", 0):
                return False
            n = norm_name(m["name"])
            if opp_underground and n not in ("earthquake", "magnitude"):
                return True
            if opp_types and effectiveness(m["type"], opp_types) == 0:
                return True  # type immunity (Ground->Flying, Dragon->Fairy, ...)
            if ABILITY_IMMUNE.get(opp_ability) == m["type"]:
                return True  # ability immunity (Levitate, Flash Fire, Water Absorb, ...)
            return m["type"] == "ground" and opp_item == "airballoon"  # item immunity

        soft = [m for m in moves if not soft_drop(m)] or moves
        moves = [m for m in soft if not whiffs(m)]  # may be empty -> switch instead
    return moves


def player_actions(game: dict, side: str) -> list[dict]:
    roster = game["roster"][side]
    snap = game["snapshots"][-1] if game.get("snapshots") else None
    me = next((m for m in roster if m["active"]), None)
    can_switch = any(not m["fainted"] and not m["active"] and m["hp"] > 0 for m in roster)
    acts = []
    if me and not me["fainted"]:
        moves = moves_for(me, snap, side, game)
        if not moves and not can_switch:
            moves = moves_for(me)  # trapped and every move whiffs: must click something
        for move in moves:
            acts.append({"kind": "move", "label": move["name"], "move": move})
        # Terastallizing is a once-per-battle action taken alongside a move
        tera = predicted_tera(me)
        if tera and snap is not None and not snap.get(f"{side}_tera_used"):
            for move in moves:
                if move["category"] != "Status":
                    acts.append({"kind": "move", "move": move, "tera": tera,
                                 "label": f"Tera {tera.title()} + {move['name']}"})
    for mon in roster:
        if not mon["fainted"] and not mon["active"] and mon["hp"] > 0:
            acts.append({"kind": "switch", "label": f"switch to {mon['species']}",
                         "mon": mon})
    return acts


def pessimism_for_elo(elo, default: float = 0.7) -> float:
    """How much to weight the opponent's *best* (worst-for-us) response over their
    average one, from their ladder rating. A strong player reliably finds the
    punish, so plan for the worst case (weight → 1); a weaker one often won't, so
    lean on the expected outcome and take the higher-value line (weight → 0.4).
    Returns a weight in [0.4, 0.92]; `default` (balanced) when the Elo is unknown."""
    if not elo:
        return default
    return float(np.clip(0.40 + (elo - 1000) * (0.92 - 0.40) / 900, 0.40, 0.92))


def _rank_by_pessimism(df: pd.DataFrame, pessimism: float) -> pd.DataFrame:
    """Order actions by a worst/average blend: pure worst-case at pessimism=1,
    pure expected value at pessimism=0. Keeps the table's columns unchanged."""
    rank = pessimism * df.worst_case + (1 - pessimism) * df.average
    return df.assign(_rank=rank).sort_values("_rank", ascending=False,
                                             ignore_index=True).drop(columns="_rank")


def advise_search(game: dict, side: str, booster, meta, snapshot_features,
                  pessimism: float = 1.0) -> pd.DataFrame:
    """Rank `side`'s actions by a worst-case / expected blend (1-ply minimax).
    `pessimism` (see pessimism_for_elo) sets how adversarial the opponent is
    assumed to be; 1.0 is the pure worst-case recommendation."""
    snap = game["snapshots"][-1]
    opp = "p2" if side == "p1" else "p1"
    mine, theirs = player_actions(game, side), player_actions(game, opp)
    if not mine:
        return pd.DataFrame(columns=["action", "worst_case", "average", "worst_response"])
    theirs = theirs or [{"kind": "move", "label": "(no options)",
                         "move": {"category": "Status", "power": 0}}]

    # simulate the whole (my action × their response) matrix, then score every
    # resulting position in ONE batched model call (much faster than per-cell).
    snapshots = []
    for a in mine:
        for b in theirs:
            sim = SimState(game, snap)
            sim.resolve({side: a, opp: b})
            snapshots.append(sim.to_snapshot())
    p1_win = calibrate(booster.predict(snapshot_features({**game, "snapshots": snapshots},
                                                         meta)), meta)
    mine_win = np.asarray(p1_win if side == "p1" else 1 - p1_win)
    # material correction: reward positions where the opponent has lost more mons
    # than me, so a move that secures a KO isn't out-ranked by a passive one.
    mat = np.array([s[f"{opp}_fainted"] - s[f"{side}_fainted"] for s in snapshots])
    mine_win = np.clip(mine_win + MATERIAL_BONUS * mat, 0.0, 1.0)
    # symmetric tempo adjustment: my switch costs me, the opponent's switch credits me
    my_switch = np.array([SWITCH_COST if a["kind"] == "switch" else 0.0 for a in mine])
    opp_switch = np.array([SWITCH_COST if b["kind"] == "switch" else 0.0 for b in theirs])
    grid = np.clip(mine_win.reshape(len(mine), len(theirs))
                   - my_switch[:, None] + opp_switch[None, :], 0.0, 1.0)

    rows = []
    for i, a in enumerate(mine):
        worst_j = int(grid[i].argmin())
        rows.append({"action": a["label"], "worst_case": float(grid[i, worst_j]),
                     "average": float(grid[i].mean()),
                     "worst_response": theirs[worst_j]["label"]})
    return _rank_by_pessimism(pd.DataFrame(rows), pessimism)


# moves that switch the user out after resolving — the follow-up mon matters as
# much as the move, so the advisor should recommend who to bring in
PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport",
               "chillyreception", "batonpass", "shedtail"}


def active_pivot_move(game: dict, side: str) -> dict | None:
    """The pivot move `side`'s active can legally use this turn, if any."""
    snap = game["snapshots"][-1] if game.get("snapshots") else None
    me = next((m for m in game["roster"][side] if m["active"]), None)
    if not me or me["fainted"]:
        return None
    for mv in moves_for(me, snap, side, game):
        if norm_name(mv["name"]) in PIVOT_MOVES:
            return mv
    return None


def pivot_targets(game: dict, side: str, booster, meta, snapshot_features,
                  pivot_move: dict | None = None) -> pd.DataFrame:
    """Rank the Pokémon `side` should bring in after a pivot move. The incoming
    mon arrives for free — the opponent already spent its turn — so each candidate
    is scored by the win-prob model on the post-pivot board with no switch tempo
    cost. Returns [target, win] best-first."""
    snap = game["snapshots"][-1]
    opp = "p2" if side == "p1" else "p1"
    base = dict(snap)
    if pivot_move is not None and pivot_move.get("power"):
        sim = SimState(game, snap)
        sim.use_move(side, pivot_move)
        opp_hp = sim.active[opp].hp
        if opp_hp > 1e-6:  # if the pivot KOs, the incoming faces an unknown mon — skip chip
            base[f"{opp}_hp_total"] = snap[f"{opp}_hp_total"] - (snap[f"{opp}_active_hp"] - opp_hp)
            base[f"{opp}_active_hp"] = max(opp_hp, 0.0)
    cands = [m for m in game["roster"][side]
             if not m["fainted"] and not m["active"] and m["hp"] > 0]
    if not cands:
        return pd.DataFrame(columns=["target", "win"])
    snaps = []
    for m in cands:
        s = dict(base)
        s[f"{side}_active_species"] = m["species"]
        s[f"{side}_active_hp"] = m["hp"]
        s[f"{side}_active_status"] = m["status"]
        snaps.append(s)
    p1 = calibrate(booster.predict(snapshot_features({**game, "snapshots": snaps}, meta)), meta)
    win = np.asarray(p1 if side == "p1" else 1 - p1)
    rows = [{"target": m["species"], "win": float(w)} for m, w in zip(cands, win)]
    return pd.DataFrame(rows).sort_values("win", ascending=False, ignore_index=True)


def _opening_snapshot(n_mine: int, n_opp: int) -> dict:
    """A fresh turn-1 board (both teams full, nothing revealed but the leads)."""
    s = {"turn": 1, "p1_active_species": "", "p2_active_species": "",
         "p1_active_hp": 1.0, "p2_active_hp": 1.0,
         "p1_active_status": "", "p2_active_status": "",
         "p1_hp_total": float(n_mine), "p2_hp_total": float(n_opp),
         "p1_fainted": 0, "p2_fainted": 0, "p1_revealed": 1, "p2_revealed": 1,
         "p1_healthy": n_mine, "p2_healthy": n_opp, "p1_statused": 0, "p2_statused": 0,
         "p1_moves_revealed": 0, "p2_moves_revealed": 0,
         "p1_tera_used": False, "p2_tera_used": False,
         "weather": "", "terrain": "", "trickroom": False}
    return s


def recommend_lead(game: dict, side: str, booster, meta, snapshot_features) -> pd.DataFrame:
    """Rank `side`'s six Pokémon as the opening lead. Each candidate is scored by
    the win-prob model on the full-team opening matchup against every one of the
    opponent's possible leads; the recommendation has the best *average* opening
    (you don't know which lead the opponent picks)."""
    opp = "p2" if side == "p1" else "p1"
    my_team, opp_team = game["teams"][side], game["teams"][opp]
    if not my_team or not opp_team:
        return pd.DataFrame(columns=["lead", "average", "worst_case", "worst_vs"])
    base = _opening_snapshot(len(my_team), len(opp_team))
    snaps = []
    for m in my_team:
        for o in opp_team:
            snaps.append({**base, f"{side}_active_species": m, f"{opp}_active_species": o})
    p1_win = calibrate(booster.predict(snapshot_features({**game, "snapshots": snaps}, meta)), meta)
    mine = np.asarray(p1_win if side == "p1" else 1 - p1_win).reshape(len(my_team), len(opp_team))

    rows = []
    for i, m in enumerate(my_team):
        j = int(mine[i].argmin())
        rows.append({"lead": m, "average": float(mine[i].mean()),
                     "worst_case": float(mine[i, j]), "worst_vs": opp_team[j]})
    return pd.DataFrame(rows).sort_values("average", ascending=False, ignore_index=True)
