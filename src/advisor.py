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

import pandas as pd

from src.pokedex import effectiveness, lookup, move_info

BOOST_STATS = ("atk", "def", "spa", "spd", "spe")
HAZARDS = ("stealthrock", "spikes", "toxicspikes", "stickyweb")
HAZARD_MAX = {"stealthrock": 1, "spikes": 3, "toxicspikes": 2, "stickyweb": 1}
SCREENS = ("reflect", "lightscreen", "auroraveil", "tailwind")
STATUS_IMMUNE = {"brn": "fire", "par": "electric", "tox": "steel", "psn": "steel"}
DEFAULT_STATS = {s: 80 for s in ("hp", "atk", "def", "spa", "spd", "spe")}


def est_stat(base: int, is_hp: bool = False) -> int:
    """Level-100 stat under a neutral 31 IV / 85 EV assumption."""
    return 2 * base + 162 if is_hp else 2 * base + 57


def boost_mult(stage: int) -> float:
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def hazard_chip(species: str, side_snapshot_prefix: str, snapshot: dict) -> float:
    """Fraction of max HP lost to entry hazards when this species switches in."""
    entry = lookup(species)
    if entry is None:
        return 0.0
    types = entry["types"]
    chip = 0.0
    if snapshot[f"{side_snapshot_prefix}_hazard_stealthrock"]:
        chip += 0.125 * effectiveness("rock", types)
    if "flying" not in types:  # crude groundedness check (Levitate is hidden info)
        spikes = snapshot[f"{side_snapshot_prefix}_hazard_spikes"]
        chip += {0: 0.0, 1: 1 / 8, 2: 1 / 6, 3: 1 / 4}[spikes]
    return min(chip, 1.0)


class _Active:
    def __init__(self, mon: dict, boosts: dict | None = None):
        self.species = mon["species"]
        self.hp = mon["hp"]
        self.status = mon["status"]
        self.fainted = mon["fainted"]
        dex = lookup(mon["species"])
        self.types = dex["types"] if dex else []
        self.stats = {k: dex[k] for k in DEFAULT_STATS} if dex else dict(DEFAULT_STATS)
        self.boosts = dict(boosts) if boosts else {s: 0 for s in BOOST_STATS}


class SimState:
    """A rough but honest one-turn battle engine over the snapshot schema."""

    def __init__(self, game: dict, snap: dict):
        self.snap = dict(snap)
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
        spe = est_stat(a.stats["spe"]) * boost_mult(a.boosts["spe"])
        return spe * (0.5 if a.status == "par" else 1.0)

    def switch(self, side: str, mon: dict) -> None:
        chip = hazard_chip(mon["species"], side, self.snap)
        self.active[side] = _Active({**mon, "hp": max(mon["hp"] - chip, 0.0)})
        self.snap[f"{side}_hp_total"] -= min(chip, mon["hp"])
        if self.active[side].hp <= 0:
            self.active[side].fainted = True
            self.snap[f"{side}_fainted"] += 1

    def damage_fraction(self, side: str, move: dict) -> float:
        atk, dfn = self.active[side], self.active[self._opp(side)]
        physical = move["category"] == "Physical"
        a_stat = est_stat(atk.stats["atk" if physical else "spa"])
        a_stat *= boost_mult(atk.boosts["atk" if physical else "spa"])
        if physical and atk.status == "brn":
            a_stat *= 0.5
        d_stat = est_stat(dfn.stats["def" if physical else "spd"])
        d_stat *= boost_mult(dfn.boosts["def" if physical else "spd"])
        dmg = (42 * move["power"] * a_stat / d_stat) / 50 + 2
        frac = dmg / est_stat(dfn.stats["hp"], is_hp=True) * 0.925  # avg roll
        frac *= 1.5 if move["type"] in atk.types else 1.0
        frac *= effectiveness(move["type"], dfn.types)
        weather = self.snap.get("weather", "")
        if weather in ("raindance", "rain"):
            frac *= {"water": 1.5, "fire": 0.5}.get(move["type"], 1.0)
        elif weather in ("sunnyday", "sun", "desolateland"):
            frac *= {"fire": 1.5, "water": 0.5}.get(move["type"], 1.0)
        opp_side = self._opp(side)
        screens = {s for s in SCREENS if self.snap[f"{opp_side}_screen_{s}"]}
        if "auroraveil" in screens or ("reflect" in screens and physical) \
                or ("lightscreen" in screens and not physical):
            frac *= 0.5
        return frac * move.get("accuracy", 1.0)

    def use_move(self, side: str, move: dict) -> None:
        me, opp_side = self.active[side], self._opp(side)
        opp = self.active[opp_side]
        if me.fainted:
            return
        if move["category"] != "Status" and move["power"] > 0:
            frac = self.damage_fraction(side, move)
            dealt = min(frac, opp.hp)
            opp.hp -= dealt
            self.snap[f"{opp_side}_hp_total"] -= dealt
            if opp.hp <= 0:
                opp.hp, opp.fainted = 0.0, True
                self.snap[f"{opp_side}_fainted"] += 1
            return
        if move.get("inflicts") and not opp.status and not opp.fainted \
                and STATUS_IMMUNE.get(move["inflicts"]) not in opp.types:
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
        for side, act in actions.items():
            if act["kind"] == "switch":
                self.switch(side, act["mon"])
            else:
                movers.append((side, act["move"]))
        trick_room = -1 if self.snap.get("trickroom") else 1
        movers.sort(key=lambda m: (m[1].get("priority", 0),
                                   trick_room * self.speed(m[0])), reverse=True)
        for side, move in movers:
            self.use_move(side, move)

    def to_snapshot(self) -> dict:
        out = dict(self.snap)
        out["turn"] = self.snap["turn"] + 1
        for side in ("p1", "p2"):
            a = self.active[side]
            out[f"{side}_active_species"] = a.species
            out[f"{side}_active_hp"] = max(a.hp, 0.0)
            out[f"{side}_active_status"] = a.status
            for s in BOOST_STATS:
                out[f"{side}_boost_{s}"] = a.boosts[s]
        return out


def _typical_moves(species: str) -> list[dict]:
    """When nothing is revealed, assume competent STAB coverage."""
    dex = lookup(species)
    if not dex:
        return []
    category = "Physical" if dex["atk"] >= dex["spa"] else "Special"
    return [{"name": f"(typical {t} attack)", "type": t, "category": category,
             "power": 80, "accuracy": 1.0, "priority": 0} for t in dex["types"]]


def player_actions(game: dict, side: str) -> list[dict]:
    roster = game["roster"][side]
    me = next((m for m in roster if m["active"]), None)
    acts = []
    if me and not me["fainted"]:
        revealed = [dict(move_info(mv), name=mv) for mv in me["moves"] if move_info(mv)]
        for move in revealed or _typical_moves(me["species"]):
            acts.append({"kind": "move", "label": move["name"], "move": move})
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

    rows = []
    for a in mine:
        outcomes = []
        for b in theirs:
            sim = SimState(game, snap)
            sim.resolve({side: a, opp: b})
            hypo = {**game, "snapshots": [sim.to_snapshot()]}
            p1_win = float(booster.predict(snapshot_features(hypo, meta))[0])
            outcomes.append((p1_win if side == "p1" else 1 - p1_win, b["label"]))
        worst = min(outcomes)
        rows.append({"action": a["label"], "worst_case": worst[0],
                     "average": sum(o[0] for o in outcomes) / len(outcomes),
                     "worst_response": worst[1]})
    return pd.DataFrame(rows).sort_values("worst_case", ascending=False,
                                          ignore_index=True)
