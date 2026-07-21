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


def predicted_item(mon: dict) -> str:
    """Revealed item if known, else the species' most common item on the ladder."""
    if mon.get("item"):
        return norm_name(mon["item"])
    entry = species_set(mon["species"])
    return entry["item"][0][0] if entry and entry.get("item") else ""


def predicted_ability(mon: dict) -> str:
    if mon.get("ability"):
        return norm_name(mon["ability"])
    entry = species_set(mon["species"])
    return entry["ability"][0][0] if entry and entry.get("ability") else ""


def predicted_tera(mon: dict) -> str:
    if mon.get("tera"):
        return mon["tera"].lower()
    entry = species_set(mon["species"])
    return entry["tera"][0][0] if entry and entry.get("tera") else ""


def _grounded(active) -> bool:
    """Flying-types and Levitate float (Boots/Balloon stay hidden info)."""
    return "flying" not in active.types and active.ability != "levitate"

BOOST_STATS = ("atk", "def", "spa", "spd", "spe")
HAZARDS = ("stealthrock", "spikes", "toxicspikes", "stickyweb")
HAZARD_MAX = {"stealthrock": 1, "spikes": 3, "toxicspikes": 2, "stickyweb": 1}
SCREENS = ("reflect", "lightscreen", "auroraveil", "tailwind")
STATUS_IMMUNE = {"brn": "fire", "par": "electric", "tox": "steel", "psn": "steel"}


def boost_mult(stage: int) -> float:
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


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
        self.semiinvuln = "semiinvuln" in (mon.get("volatiles") or [])
        dex = lookup(mon["species"])
        self.orig_types = dex["types"] if dex else []
        tera = (mon.get("tera") or "").lower()
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


class SimState:
    """A rough but honest one-turn battle engine over the snapshot schema."""

    def __init__(self, game: dict, snap: dict):
        self.snap = dict(snap)
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

        def_stat = move.get("def_stat") or ("def" if physical else "spd")
        d_stat = dfn.stats[def_stat]
        if atk.ability != "unaware":
            d_stat *= boost_mult(dfn.boosts.get(def_stat, 0))
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
        return frac * move.get("accuracy", 1.0)

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
        if me.status in ("slp", "frz") and me.sleep_turns < 3:
            # immobilized (guaranteed wake after 3 sleep turns) — moves fail...
            if me.status == "slp" and norm_name(move.get("name", "")) == "sleeptalk":
                move = self._sleep_talk_proxy(side)  # ...except Sleep Talk
                if move is None:
                    return
            else:
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
        if move["category"] != "Status" and (move["power"] > 0 or move.get("fixed")):
            frac = self.damage_fraction(side, move)
            dealt = min(frac, opp.hp)
            if (opp.item == "focussash" and opp.hp >= 0.999 and dealt >= opp.hp):
                dealt = opp.hp - 0.01  # Sash: survive one hit from full
            opp.hp -= dealt
            self.snap[f"{opp_side}_hp_total"] -= dealt
            if opp.hp <= 0:
                opp.hp, opp.fainted = 0.0, True
                self.snap[f"{opp_side}_fainted"] += 1
            if dealt > 0 and move.get("drain"):
                healed = min(dealt * move["drain"], 1.0 - me.hp)
                me.hp += healed
                self.snap[f"{side}_hp_total"] += healed
            self_dmg = dealt * move.get("recoil", 0)
            if dealt > 0 and me.item == "lifeorb":
                self_dmg += 0.1
            if dealt > 0 and move.get("contact") and opp.item == "rockyhelmet" \
                    and not opp.fainted:
                self_dmg += 1 / 6
            if self_dmg:
                lost = min(self_dmg, me.hp)
                me.hp -= lost
                self.snap[f"{side}_hp_total"] -= lost
                if me.hp <= 0:
                    me.fainted = True
                    self.snap[f"{side}_fainted"] += 1
            # meaningful secondary status (Scald burn, Nuzzle para, ...)
            if (dealt > 0 and move.get("sec_status") and move.get("sec_chance", 0) >= 30
                    and not opp.status and not opp.fainted
                    and STATUS_IMMUNE.get(move["sec_status"]) not in opp.types):
                opp.status = move["sec_status"]
                self.snap[f"{opp_side}_statused"] += 1
            return
        if move.get("inflicts") and not opp.status and not opp.fainted \
                and STATUS_IMMUNE.get(move["inflicts"]) not in opp.types:
            terrain = self.snap.get("terrain", "")
            terrain_blocked = _grounded(opp) and (
                terrain == "mistyterrain"
                or (terrain == "electricterrain" and move["inflicts"] == "slp"))
            if not terrain_blocked:
                opp.status = move["inflicts"]
                self.snap[f"{opp_side}_statused"] += 1
        if move.get("boosts"):
            target = me if move.get("target") == "self" else opp
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
        """End-of-turn residuals: burn/poison/sand chip, Grassy Terrain healing."""
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
            if a.item == "leftovers":
                delta += 1 / 16
            elif a.item == "blacksludge":
                delta += 1 / 16 if "poison" in a.types else -1 / 8
            if terrain == "grassyterrain" and _grounded(a):
                delta += 1 / 16
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
    if norm_name(mon.get("item", "")).startswith("choice") and last:
        locked = [m for m in moves if norm_name(m["name"]) == last]
        moves = locked or moves

    if snap is not None and side is not None:
        opp = "p2" if side == "p1" else "p1"
        opp_types = _defender_types(game, snap, opp)
        opp_entry = next((m for m in (game or {}).get("roster", {}).get(opp, [])
                          if m.get("active")), {})
        opp_underground = "semiinvuln" in (opp_entry.get("volatiles") or [])
        status = mon.get("status", "")
        useful = []
        for m in moves:
            n = norm_name(m["name"])
            if (m.get("heal") or n == "rest") and mon["hp"] >= 0.99:
                continue  # healing at full HP fails / does nothing
            if is_pure_setup(m) and (status == "tox"
                                     or (status in ("psn", "brn") and mon["hp"] < 0.5)):
                continue  # don't set up while dying to residual damage
            if (m["category"] != "Status" and m.get("power", 0) > 0 and opp_types
                    and effectiveness(m["type"], opp_types) == 0):
                continue  # the opponent is immune — clicking it whiffs
            if (opp_underground and m["category"] != "Status"
                    and n not in ("earthquake", "magnitude")):
                continue  # target is underground/in-air (Dig/Fly) — attack misses
            sc = m.get("side_condition")
            if sc in HAZARD_MAX and snap[f"{opp}_hazard_{sc}"] >= HAZARD_MAX[sc]:
                continue  # hazard already at max layers
            if sc in SCREENS and snap[f"{side}_screen_{sc}"]:
                continue  # screen already up
            useful.append(m)
        moves = useful or moves
    return moves


def player_actions(game: dict, side: str) -> list[dict]:
    roster = game["roster"][side]
    snap = game["snapshots"][-1] if game.get("snapshots") else None
    me = next((m for m in roster if m["active"]), None)
    acts = []
    if me and not me["fainted"]:
        moves = moves_for(me, snap, side, game)
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


def advise_search(game: dict, side: str, booster, meta, snapshot_features) -> pd.DataFrame:
    """Rank `side`'s actions by worst-case win probability (1-ply minimax)."""
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
    mine_win = p1_win if side == "p1" else 1 - p1_win
    grid = np.asarray(mine_win).reshape(len(mine), len(theirs))

    rows = []
    for i, a in enumerate(mine):
        worst_j = int(grid[i].argmin())
        rows.append({"action": a["label"], "worst_case": float(grid[i, worst_j]),
                     "average": float(grid[i].mean()),
                     "worst_response": theirs[worst_j]["label"]})
    return pd.DataFrame(rows).sort_values("worst_case", ascending=False,
                                          ignore_index=True)


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
