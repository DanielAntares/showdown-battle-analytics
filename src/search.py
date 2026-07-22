"""Multi-turn move search (Tier 3): look several turns ahead instead of one.

The 1-ply advisor (src/advisor.py) simulates a single turn and scores the
result with the win-prob model. It can't see two turns out — that a KO denied
by Protect happens anyway next turn, that a setup sweep pays off later, that
hazards compound. This module adds depth.

A *full* game tree is hopeless — ~9 actions each side per turn means ~81 joint
actions, so 5 turns is ~81^5 ≈ 3.5 billion nodes. Instead we combine two ideas:

* a **depth-limited maximin tree** over the top few actions per side (real
  adversarial reasoning for the first 2–3 turns, where tactics matter most), and
* a **greedy rollout** from the tree's leaves — both sides play their best
  immediate move for a few more turns — to extend the effective horizon to ~5
  turns cheaply.

`step()` is the engine primitive both rely on: it advances the full battle state
(every Pokémon's HP/status/faint, who's active, hazards, screens, weather) one
turn, so the output can feed straight back in for the next turn.
"""

import numpy as np
import pandas as pd

from src.advisor import SimState, effectiveness, lookup, player_actions
from src.predict import calibrate, snapshot_features


def step(game: dict, actions: dict) -> dict:
    """Advance one full turn and return a NEW game dict (roster + snapshot) whose
    state can be fed back in for the next turn."""
    snap = game["snapshots"][-1]
    sim = SimState(game, snap)
    sim.resolve(actions)
    out = sim.to_snapshot()

    roster = {s: [dict(m) for m in game["roster"][s]] for s in ("p1", "p2")}
    for side in ("p1", "p2"):
        a = sim.active[side]
        kind = sim.acted.get(side, (None, None))[0]
        new_sp = out[f"{side}_active_species"]  # may be a faint-replacement
        by_sp = {m["species"]: m for m in roster[side]}
        if a.species in by_sp:  # write the turn-active mon's resulting state
            e = by_sp[a.species]
            e["hp"] = max(0.0, a.hp)
            e["status"] = a.status
            e["fainted"] = a.fainted or a.hp <= 1e-6
            e["item"] = a.item
            if a.did_tera:
                e["tera"] = a.tera
            e["sleep_turns"] = a.sleep_turns + 1 if a.status == "slp" else 0
            e["tox_turns"] = a.tox_turns + 1 if a.status == "tox" else 0
            e["last_move"] = sim.acted[side][1] if kind == "move" else ""
            if kind != "move":  # a fresh switch-in has no volatiles / lock
                e["volatiles"] = []
        for m in roster[side]:  # set the active flag; clear volatiles off the field
            was = m["active"]
            m["active"] = m["species"] == new_sp
            if (was and not m["active"]) or (m["active"] and m["species"] != a.species):
                m["volatiles"], m["last_move"] = [], ""

    # recompute aggregate snapshot fields from the roster so it's self-consistent
    for side in ("p1", "p2"):
        mons = roster[side]
        act = next((m for m in mons if m["active"]), None)
        out[f"{side}_hp_total"] = float(sum(m["hp"] for m in mons))
        out[f"{side}_fainted"] = sum(m["fainted"] for m in mons)
        out[f"{side}_healthy"] = sum(not m["fainted"] and m["hp"] >= 0.5 for m in mons)
        if act:
            out[f"{side}_active_species"] = act["species"]
            out[f"{side}_active_hp"] = act["hp"]
            out[f"{side}_active_status"] = act["status"]
    return {**game, "roster": roster, "snapshots": [out]}


def is_over(game: dict) -> bool:
    return any(all(m["fainted"] for m in game["roster"][s]) for s in ("p1", "p2"))


# ---- cheap action heuristics (used to prune/order; no model calls) -----------

def _move_value(state: dict, side: str, mv: dict) -> float:
    """Rough goodness of a move for ordering: damage fraction, KO bonus, or a
    small value for useful status moves."""
    snap = state["snapshots"][-1]
    if mv.get("category") == "Status" or not mv.get("power", 0):
        return 0.15
    sim = SimState(state, snap)
    dmg = sim.damage_fraction(side, mv)
    opp = "p2" if side == "p1" else "p1"
    return dmg + (0.5 if dmg >= snap[f"{opp}_active_hp"] else 0.0)  # reward likely KOs


def _switch_value(state: dict, side: str, mon: dict) -> float:
    """Rough goodness of a switch: how well the incoming mon resists the foe."""
    snap = state["snapshots"][-1]
    opp = "p2" if side == "p1" else "p1"
    foe = lookup(snap[f"{opp}_active_species"])
    dex = lookup(mon["species"])
    if not foe or not dex:
        return 0.2
    worst = max((effectiveness(t, dex["types"]) for t in foe["types"]), default=1.0)
    return 0.4 / (worst + 0.5)  # resisting the foe's STAB scores higher


def top_actions(state: dict, side: str, k: int) -> list[dict]:
    """The k most promising actions (cheap heuristic ordering, no model calls)."""
    acts = player_actions(state, side)
    def score(a):
        return _move_value(state, side, a["move"]) if a["kind"] == "move" \
            else _switch_value(state, side, a["mon"])
    return sorted(acts, key=score, reverse=True)[:k]


def greedy_action(state: dict, side: str) -> dict | None:
    acts = top_actions(state, side, 1)
    return acts[0] if acts else None


def _other(side: str) -> str:
    return "p2" if side == "p1" else "p1"


# ---- the search itself -------------------------------------------------------

MATERIAL_BONUS = 0.03      # match the 1-ply advisor's material correction
PESSIMISM = 0.7           # weight on the opponent's best (worst-for-us) response


def _rollout_terminal(game: dict, side: str, turns: int) -> dict:
    """Simulate `turns` greedy turns (no model) and return the terminal state to
    be scored later in a batch."""
    g, opp = game, _other(side)
    for _ in range(turns):
        if is_over(g):
            break
        a, b = greedy_action(g, side), greedy_action(g, opp)
        if not a or not b:
            break
        g = step(g, {side: a, opp: b})
    return g


def score_batch(games: list, side: str, booster, meta) -> np.ndarray:
    """Score many independent states in ONE model call (per-call DataFrame build
    is the search's dominant cost). Reuses the exact inference pipeline, then
    zeroes momentum — these are hypothetical future positions with no real turn
    history, so the cross-game shift add_derived computes here is meaningless."""
    snaps = [g["snapshots"][-1] for g in games]
    synthetic = {"snapshots": snaps,
                 "p1_rating": games[0].get("p1_rating"),
                 "p2_rating": games[0].get("p2_rating")}
    df = snapshot_features(synthetic, meta).copy()
    df["hp_momentum_1"] = 0.0
    df["hp_momentum_3"] = 0.0
    p1 = calibrate(booster.predict(df), meta)
    mine = np.asarray(p1 if side == "p1" else 1.0 - p1)
    mat = np.array([s[f"{_other(side)}_fainted"] - s[f"{side}_fainted"] for s in snaps])
    return np.clip(mine + MATERIAL_BONUS * mat, 0.0, 1.0)


def _build_tree(game: dict, side: str, depth: int, rollout: int, k: int, leaves: list):
    """Build the search tree, collecting rollout-terminal leaf states into `leaves`
    (pure simulation, no model calls). Returns a structure evaluate() reduces."""
    if depth <= 0 or is_over(game):
        leaves.append(_rollout_terminal(game, side, rollout))
        return ("leaf", len(leaves) - 1)
    opp = _other(side)
    opp_acts = top_actions(game, opp, k)
    my = []
    for a in top_actions(game, side, k):
        my.append([_build_tree(step(game, {side: a, opp: b}), side, depth - 1, rollout, k, leaves)
                   for b in opp_acts])
    return ("max", my)


def _reduce(node, scores: np.ndarray) -> float:
    if node[0] == "leaf":
        return float(scores[node[1]])
    best = 0.0
    for responses in node[1]:  # my action -> list of opponent-response subtrees
        vs = [_reduce(c, scores) for c in responses]
        best = max(best, PESSIMISM * min(vs) + (1 - PESSIMISM) * (sum(vs) / len(vs)))
    return best


def deep_search(game: dict, side: str, booster, meta, depth: int = 2,
                rollout: int = 3, top_k: int = 3) -> pd.DataFrame:
    """Rank `side`'s actions by multi-turn value. Effective horizon ≈ depth +
    rollout turns. Same table shape as the 1-ply advisor."""
    opp = _other(side)
    opp_acts = top_actions(game, opp, top_k)
    leaves: list = []
    root = []  # per root action: list of (opponent response) subtrees
    for a in top_actions(game, side, max(top_k, 5)):
        root.append([_build_tree(step(game, {side: a, opp: b}), side, depth - 1,
                                 rollout, top_k, leaves) for b in opp_acts])
    scores = score_batch(leaves, side, booster, meta) if leaves else np.array([])

    rows = []
    my_acts = top_actions(game, side, max(top_k, 5))
    for a, responses in zip(my_acts, root):
        vals = [_reduce(c, scores) for c in responses]
        rows.append({"action": a["label"], "worst_case": float(min(vals)),
                     "average": float(sum(vals) / len(vals)),
                     "worst_response": opp_acts[int(np.argmin(vals))]["label"]})
    return pd.DataFrame(rows).sort_values("worst_case", ascending=False, ignore_index=True)
