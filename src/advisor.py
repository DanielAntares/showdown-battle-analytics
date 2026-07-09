"""Option advisor (v1): rank a player's switches and revealed moves at any state.

Two tiers of rigor, clearly separated:

* **Switches** are scored by the win-probability model itself: build the
  hypothetical post-switch snapshot (boosts cleared, entry-hazard chip applied
  by type chart) and ask the model how the position looks. Grounded, but still
  ignores what the opponent does on the same turn.
* **Moves** are a damage heuristic over *revealed* moves only: base power x
  STAB x type effectiveness x attacker/defender base-stat ratio. It knows
  nothing about items, abilities, or exact EVs.

v2 on the roadmap replaces both with a true 1-ply search over the joint action
matrix using Showdown's open-source simulator.
"""

import pandas as pd

from src.pokedex import effectiveness, lookup, move_info

BOOST_STATS = ("atk", "def", "spa", "spd", "spe")


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


def rank_switches(game: dict, side: str, booster, meta, snapshot_features) -> pd.DataFrame:
    """Score each healthy bench Pokémon by the model's read of the post-switch state."""
    snap = game["snapshots"][-1]
    rows = []
    for mon in game["roster"][side]:
        if mon["fainted"] or mon["active"] or mon["hp"] <= 0:
            continue
        hypo = dict(snap)
        chip = hazard_chip(mon["species"], side, snap)
        hp_after = max(mon["hp"] - chip, 0.02)
        hypo[f"{side}_active_species"] = mon["species"]
        hypo[f"{side}_active_hp"] = hp_after
        hypo[f"{side}_active_status"] = mon["status"]
        hypo[f"{side}_hp_total"] = snap[f"{side}_hp_total"] - chip
        for stat in BOOST_STATS:
            hypo[f"{side}_boost_{stat}"] = 0
        hypo_game = {**game, "snapshots": [hypo]}
        p1_prob = float(booster.predict(snapshot_features(hypo_game, meta))[0])
        rows.append({
            "option": f"switch to {mon['species']}",
            "win_prob": p1_prob if side == "p1" else 1 - p1_prob,
            "hazard_chip": chip,
        })
    return pd.DataFrame(rows).sort_values("win_prob", ascending=False, ignore_index=True) \
        if rows else pd.DataFrame(columns=["option", "win_prob", "hazard_chip"])


def rank_moves(game: dict, side: str) -> pd.DataFrame:
    """Damage heuristic over the active Pokémon's *revealed* moves."""
    me = next((m for m in game["roster"][side] if m["active"]), None)
    opp_side = "p2" if side == "p1" else "p1"
    opp = next((m for m in game["roster"][opp_side] if m["active"]), None)
    if me is None or opp is None:
        return pd.DataFrame(columns=["option", "damage_score", "note"])
    my_dex, opp_dex = lookup(me["species"]), lookup(opp["species"])

    rows = []
    for move in me["moves"]:
        info = move_info(move)
        if info is None:
            continue
        if info["category"] == "Status" or info["power"] == 0:
            rows.append({"option": move, "damage_score": None, "note": "utility / status"})
            continue
        eff = effectiveness(info["type"], opp_dex["types"]) if opp_dex else 1.0
        stab = 1.5 if my_dex and info["type"] in my_dex["types"] else 1.0
        ratio = 1.0
        if my_dex and opp_dex:
            atk = my_dex["atk"] if info["category"] == "Physical" else my_dex["spa"]
            dfn = opp_dex["def"] if info["category"] == "Physical" else opp_dex["spd"]
            ratio = atk / dfn
        note = {0: "opponent is immune", 0.25: "heavily resisted", 0.5: "resisted",
                2: "super effective", 4: "4x super effective"}.get(eff, "")
        rows.append({"option": move,
                     "damage_score": round(info["power"] * eff * stab * ratio, 1),
                     "note": note})
    df = pd.DataFrame(rows)
    return df.sort_values("damage_score", ascending=False, na_position="last",
                          ignore_index=True) if len(df) else df


def advise(game: dict, side: str, booster, meta, snapshot_features) -> dict:
    """Ranked options for `side` at the game's current (last-parsed) state."""
    return {
        "switches": rank_switches(game, side, booster, meta, snapshot_features),
        "moves": rank_moves(game, side),
    }
