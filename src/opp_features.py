"""Turn each extracted decision into (state × candidate-action) feature rows for
the opponent-action ranker.

One decision becomes several rows — one per action the player could have taken
(each usage-predicted move of the active mon, plus a switch to each healthy
benched mon). Exactly one row is the action they actually chose (`is_chosen`).
Features are deliberately cheap — type chart, base stats, and usage priors, no
full damage simulation — so the whole corpus can be featurized in minutes.

`predicted` marks the rows an observer could have enumerated at play time (the
usage-predicted candidate set). The chosen move is always added so training has
a positive label, but if it was not in the predicted set that row has
predicted=False, so honest top-k accuracy (restricted to predicted rows) counts
it as a miss — we never credit the model for a move it could not have proposed.
"""

from src.advisor import PIVOT_MOVES, is_pure_setup
from src.movesets import moveset_with_probs, predict_moves, real_stats
from src.pokedex import effectiveness, lookup, move_info, norm_name

MOVE_K = 6  # how many usage-predicted moves to offer as candidates

FEATURES = [
    # shared board state
    "turn", "my_hp", "my_statused", "opp_hp", "opp_statused",
    "my_fainted", "opp_fainted", "material_diff", "my_off_boost", "my_spe_boost",
    "rocks_on_me", "i_am_faster", "n_bench",
    # shared: how much danger the active is in — the main reason to switch
    "threat_in", "threat_ko", "i_threaten", "i_ko", "best_switch_resist",
    # candidate: which kind
    "is_switch",
    # candidate: move
    "m_power", "m_type_eff", "m_is_stab", "m_is_status", "m_is_setup",
    "m_is_priority", "m_is_pivot", "m_is_heal", "m_usage_prob", "m_usage_rank",
    "m_is_revealed", "m_dmg_proxy", "m_ko",
    # candidate: switch
    "s_def_worst_eff", "s_off_best_eff", "s_hp", "s_is_full", "s_hazard_chip",
]


def _boost_mult(stage: int) -> float:
    return (2 + max(stage, 0)) / 2 if stage >= 0 else 2 / (2 - stage)


def _dmg_proxy(power, type_eff, stab, my_sp, opp_sp, physical) -> float:
    """A cheap, stat-aware damage estimate (no full formula): STAB × effectiveness
    × the relevant offense/defense stat ratio, scaled to roughly an HP fraction."""
    if not power:
        return 0.0
    mine, foe = real_stats(my_sp), real_stats(opp_sp)
    off = mine["atk"] if physical else mine["spa"]
    dfn = foe["def"] if physical else foe["spd"]
    return power * type_eff * (1.5 if stab else 1.0) * (off / dfn) / 320.0


def _best_hit(atk_sp: str, dfn_sp: str, revealed: tuple) -> float:
    """Largest damage proxy the attacker's likely moves land on the defender —
    how threatened the defender is. Uses the usage-predicted moveset."""
    atk_types = (lookup(atk_sp) or {}).get("types", [])
    dfn_types = (lookup(dfn_sp) or {}).get("types", [])
    best = 0.0
    for name in predict_moves(atk_sp, revealed, 4):
        info = move_info(name)
        power = info.get("power", 0) or 0 if info else 0
        if not power:
            continue
        eff = effectiveness(info["type"], dfn_types) if dfn_types else 1.0
        best = max(best, _dmg_proxy(power, eff, info["type"] in atk_types,
                                    atk_sp, dfn_sp, info.get("category") == "Physical"))
    return best


def _base(dec: dict) -> dict:
    """Board-state features shared by every candidate row of a decision."""
    side, snap, roster = dec["side"], dec["snapshot"], dec["roster"]
    opp = "p2" if side == "p1" else "p1"
    my_sp, opp_sp = snap[f"{side}_active_species"], snap[f"{opp}_active_species"]
    my_spe = real_stats(my_sp)["spe"] * _boost_mult(snap.get(f"{side}_boost_spe", 0))
    opp_spe = real_stats(opp_sp)["spe"] * _boost_mult(snap.get(f"{opp}_boost_spe", 0))
    bench = [m for m in roster[side]
             if not m["active"] and not m["fainted"] and m["hp"] > 0]
    opp_types = (lookup(opp_sp) or {}).get("types", [])
    my_active = next((m for m in roster[side] if m["active"]), None)
    opp_active = next((m for m in roster[opp] if m["active"]), None)
    my_hp = snap.get(f"{side}_active_hp", 1.0)
    opp_hp = snap.get(f"{opp}_active_hp", 1.0)
    # danger signals — the state features that let the model anticipate a switch
    threat_in = _best_hit(opp_sp, my_sp, tuple(opp_active["moves"]) if opp_active else ())
    i_threaten = _best_hit(my_sp, opp_sp, tuple(my_active["moves"]) if my_active else ())
    best_resist = min((max((effectiveness(t, (lookup(m["species"]) or {}).get("types", []))
                            for t in opp_types), default=2.0) for m in bench), default=2.0)
    return {
        "turn": snap.get("turn", 0),
        "my_hp": my_hp,
        "my_statused": int(bool(snap.get(f"{side}_active_status"))),
        "opp_hp": opp_hp,
        "opp_statused": int(bool(snap.get(f"{opp}_active_status"))),
        "my_fainted": snap.get(f"{side}_fainted", 0),
        "opp_fainted": snap.get(f"{opp}_fainted", 0),
        "material_diff": snap.get(f"{opp}_fainted", 0) - snap.get(f"{side}_fainted", 0),
        "my_off_boost": max(snap.get(f"{side}_boost_atk", 0), snap.get(f"{side}_boost_spa", 0)),
        "my_spe_boost": snap.get(f"{side}_boost_spe", 0),
        "rocks_on_me": snap.get(f"{side}_hazard_stealthrock", 0),
        "i_am_faster": int(my_spe > opp_spe),
        "n_bench": len(bench),
        "threat_in": threat_in,
        "threat_ko": int(threat_in >= my_hp),
        "i_threaten": i_threaten,
        "i_ko": int(i_threaten >= opp_hp),
        "best_switch_resist": best_resist,
    }, side, opp, my_sp, opp_sp, bench


def featurize_decision(dec: dict) -> list[dict]:
    """All candidate rows for one decision, each a full FEATURES dict plus meta
    (group, is_chosen, predicted, kind, name)."""
    base, side, opp, my_sp, opp_sp, bench = _base(dec)
    my_types = (lookup(my_sp) or {}).get("types", [])
    opp_types = (lookup(opp_sp) or {}).get("types", [])
    active = next((m for m in dec["roster"][side] if m["active"]), None)
    revealed = tuple(active["moves"]) if active else ()

    chosen_kind, chosen_name = dec["chosen_kind"], dec["chosen_name"]
    chosen_norm = norm_name(chosen_name)

    usage = {norm_name(n): (i, p) for i, (n, p) in enumerate(moveset_with_probs(my_sp, 12))}
    predicted_moves = predict_moves(my_sp, revealed, MOVE_K)
    move_names = list(predicted_moves)
    if chosen_kind == "move" and chosen_norm not in {norm_name(m) for m in move_names}:
        move_names.append(chosen_name)  # keep a positive label even if unusual

    rows = []
    for name in move_names:
        info = move_info(name)
        if not info:
            continue
        mtype, power = info.get("type", ""), info.get("power", 0) or 0
        eff = effectiveness(mtype, opp_types) if opp_types else 1.0
        stab = mtype in my_types
        physical = info.get("category") == "Physical"
        rank, prob = usage.get(norm_name(name), (99, 0.0))
        dmg = _dmg_proxy(power, eff, stab, my_sp, opp_sp, physical)
        rows.append({**base,
            "is_switch": 0,
            "m_power": power, "m_type_eff": eff, "m_is_stab": int(stab),
            "m_is_status": int(info.get("category") == "Status"),
            "m_is_setup": int(is_pure_setup(dict(info, name=name))),
            "m_is_priority": int((info.get("priority") or 0) > 0),
            "m_is_pivot": int(norm_name(name) in PIVOT_MOVES),
            "m_is_heal": int(bool(info.get("heal"))),
            "m_usage_prob": prob, "m_usage_rank": rank,
            "m_is_revealed": int(name in revealed),
            "m_dmg_proxy": dmg, "m_ko": int(dmg >= base["opp_hp"]),
            "s_def_worst_eff": 0.0, "s_off_best_eff": 0.0, "s_hp": 0.0,
            "s_is_full": 0, "s_hazard_chip": 0.0,
            "_kind": "move", "_name": name,
            "is_chosen": int(chosen_kind == "move" and norm_name(name) == chosen_norm),
            "predicted": name in predicted_moves})

    for m in bench:
        sp = m["species"]
        types = (lookup(sp) or {}).get("types", [])
        def_worst = max((effectiveness(t, types) for t in opp_types), default=1.0)
        off_best = max((effectiveness(t, opp_types) for t in types), default=1.0)
        chip = 0.125 * effectiveness("rock", types) if base["rocks_on_me"] else 0.0
        rows.append({**base,
            "is_switch": 1,
            "m_power": 0, "m_type_eff": 0.0, "m_is_stab": 0, "m_is_status": 0,
            "m_is_setup": 0, "m_is_priority": 0, "m_is_pivot": 0, "m_is_heal": 0,
            "m_usage_prob": 0.0, "m_usage_rank": 99, "m_is_revealed": 0,
            "m_dmg_proxy": 0.0, "m_ko": 0,
            "s_def_worst_eff": def_worst, "s_off_best_eff": off_best,
            "s_hp": m["hp"], "s_is_full": int(m["hp"] >= 0.99), "s_hazard_chip": chip,
            "_kind": "switch",
            "_name": sp,
            "is_chosen": int(chosen_kind in ("switch", "replace") and sp == chosen_name),
            "predicted": True})
    return rows
