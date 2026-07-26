"""Self-play A/B strength testing, plus the two policy tools it lets us measure:
opponent determinization and a mixed-strategy root.

Inspired by the Laplace bot's practice of never shipping a strength change without
a head-to-head. Two advisor configurations play full games against each other,
adjudicated by our own engine (`step`), so this measures *relative* strength under
our model — not absolute Showdown skill, but exactly the right tool for "does
change X beat the baseline?". Starting teams are sampled from the real corpus.

    python -m src.selfplay 200          # 200 games, deep vs deep baseline
    python -m src.selfplay 200 determinize   # baseline vs +determinization

Config dict: {mode: fast|deep, pessimism: float, worlds: int, mixed: bool,
              epsilon: float, elo: int}.
"""

import json
import sys

import numpy as np
import pandas as pd

from src.advisor import advise_search, player_actions
from src.common import ROOT
from src.movesets import species_set
from src.predict import load_model, snapshot_features
from src.search import deep_search, is_over, step

BOOSTS = ("atk", "def", "spa", "spd", "spe")
HAZARDS = ("stealthrock", "spikes", "toxicspikes", "stickyweb")
SCREENS = ("reflect", "lightscreen", "auroraveil", "tailwind")


# ---- mixed-strategy root -----------------------------------------------------

def mixed_pick(table: pd.DataFrame, rng, epsilon: float = 0.0) -> str:
    """The action label to play. With epsilon>0, choose uniformly among actions
    within epsilon of the top worst_case — a deterministic bot is exploitable, so
    near-ties are broken randomly (Laplace's mixed-strategy root)."""
    if not len(table):
        return None
    top = table.worst_case.iloc[0]
    near = table[table.worst_case >= top - epsilon]
    return str(near.action.iloc[int(rng.integers(len(near)))])


# ---- opponent determinization ------------------------------------------------

def _sample_opp_worlds(game: dict, opp: str, k: int, rng) -> list[dict]:
    """`k` copies of the game, each with a different concrete guess for the
    opponent's active item + ability drawn from usage — so a single lucky/unlucky
    assumption (a hidden Scarf/Band) can't dominate the recommendation."""
    active = next((m for m in game["roster"][opp] if m["active"]), None)
    entry = species_set(active["species"]) if active else None
    if not active or not entry or active.get("item") or active.get("ability"):
        return [game]  # already revealed, or no data — nothing to sample
    items = entry.get("item") or [("", 1.0)]
    abils = entry.get("ability") or [("", 1.0)]

    def draw(dist):
        names, probs = zip(*dist)
        probs = np.array(probs, float)
        probs = probs / probs.sum() if probs.sum() else None
        return names[int(rng.choice(len(names), p=probs))]

    worlds = []
    for _ in range(k):
        roster = {s: [dict(m) for m in game["roster"][s]] for s in ("p1", "p2")}
        act = next(m for m in roster[opp] if m["active"])
        act["item"], act["ability"] = draw(items), draw(abils)
        worlds.append({**game, "roster": roster})
    return worlds


def pooled_advise(game, side, booster, meta, pessimism, worlds, rng, mode="fast"):
    """Search once per determinized world and average the tables — the pooled
    worst/average per action across plausible opponent sets."""
    opp = "p2" if side == "p1" else "p1"
    games = _sample_opp_worlds(game, opp, worlds, rng) if worlds > 1 else [game]
    tables = []
    for g in games:
        if mode == "deep":
            tables.append(deep_search(g, side, booster, meta, depth=2, rollout=3,
                                      top_k=3, pessimism=pessimism))
        else:
            tables.append(advise_search(g, side, booster, meta, snapshot_features,
                                        pessimism=pessimism))
    if len(tables) == 1:
        return tables[0]
    cat = pd.concat(tables)
    agg = (cat.groupby("action", as_index=False)
           .agg(worst_case=("worst_case", "mean"), average=("average", "mean")))
    return agg.sort_values("worst_case", ascending=False, ignore_index=True)


# ---- game construction & play ------------------------------------------------

def _entry(sp: str, active: bool) -> dict:
    return {"species": sp, "hp": 1.0, "status": "", "fainted": False, "active": active,
            "moves": [], "item": "", "ability": "", "tera": "", "volatiles": [],
            "last_move": "", "acc_stage": 0, "eva_stage": 0, "disabled": "",
            "sleep_turns": 0, "tox_turns": 0}


def new_game(team_a: list, team_b: list, lead_a=0, lead_b=0, elo=1500) -> dict:
    roster = {"p1": [_entry(sp, i == lead_a) for i, sp in enumerate(team_a)],
              "p2": [_entry(sp, i == lead_b) for i, sp in enumerate(team_b)]}
    snap = {"turn": 1, "weather": "", "terrain": "", "trickroom": 0}
    for s in ("p1", "p2"):
        act = next(m for m in roster[s] if m["active"])
        snap.update({f"{s}_active_species": act["species"], f"{s}_active_hp": 1.0,
                     f"{s}_active_status": "", f"{s}_hp_total": float(len(roster[s])),
                     f"{s}_fainted": 0, f"{s}_healthy": len(roster[s]),
                     f"{s}_statused": 0, f"{s}_tera_used": 0})
        snap.update({f"{s}_boost_{b}": 0 for b in BOOSTS})
        snap.update({f"{s}_hazard_{h}": 0 for h in HAZARDS})
        snap.update({f"{s}_screen_{sc}": 0 for sc in SCREENS})
    return {"roster": roster, "snapshots": [snap], "field": {}, "id": "selfplay",
            "format": "[Gen 9] OU", "p1_rating": elo, "p2_rating": elo,
            "p1_name": "A", "p2_name": "B", "winner": None}


def _pick(game, side, cfg, booster, meta, rng):
    active = next((m for m in game["roster"][side] if m["active"]), None)
    if active is None or active["fainted"]:
        return None
    pess = cfg.get("pessimism", 0.7)
    table = pooled_advise(game, side, booster, meta, pess, cfg.get("worlds", 1),
                          rng, mode=cfg.get("mode", "fast"))
    if not len(table):
        return None
    label = mixed_pick(table, rng, cfg.get("epsilon", 0.0)) if cfg.get("mixed") \
        else str(table.action.iloc[0])
    acts = player_actions(game, side)
    return next((a for a in acts if a["label"] == label), acts[0] if acts else None)


def play_game(team_a, team_b, cfg_a, cfg_b, booster, meta, rng, max_turns=100):
    game = new_game(team_a, team_b, int(rng.integers(6)), int(rng.integers(6)),
                    cfg_a.get("elo", 1500))
    for _ in range(max_turns):
        if is_over(game):
            break
        a, b = (_pick(game, "p1", cfg_a, booster, meta, rng),
                _pick(game, "p2", cfg_b, booster, meta, rng))
        if a is None or b is None:
            break
        game = step(game, {"p1": a, "p2": b})
    r = game["roster"]
    dead = {s: all(m["fainted"] for m in r[s]) for s in ("p1", "p2")}
    if dead["p2"] and not dead["p1"]:
        return "p1"
    if dead["p1"] and not dead["p2"]:
        return "p2"
    # timeout: adjudicate on material, then remaining HP
    fa, fb = (sum(m["fainted"] for m in r["p1"]), sum(m["fainted"] for m in r["p2"]))
    if fa != fb:
        return "p1" if fa < fb else "p2"
    ha, hb = (sum(m["hp"] for m in r["p1"]), sum(m["hp"] for m in r["p2"]))
    return "p1" if ha > hb + 1e-6 else "p2" if hb > ha + 1e-6 else None


# ---- corpus teams & the A/B --------------------------------------------------

def sample_teams(n: int, rng) -> list:
    teams = pd.read_parquet(ROOT / "data" / "processed" / "teams.parquet")
    rosters = (teams.groupby(["replay_id", "side"]).species.apply(list))
    full = [r for r in rosters if len(r) == 6]
    idx = rng.integers(0, len(full), size=n)
    return [full[i] for i in idx]


def ab_test(cfg_a, cfg_b, n_games, booster, meta, seed=0, max_turns=100):
    rng = np.random.default_rng(seed)
    pool = sample_teams(2 * n_games, rng)
    a_wins = b_wins = draws = 0
    for i in range(n_games):
        t1, t2 = pool[2 * i], pool[2 * i + 1]
        # alternate physical seat so side bias cancels out
        if i % 2 == 0:
            w, a_side, b_side = play_game(t1, t2, cfg_a, cfg_b, booster, meta, rng, max_turns), "p1", "p2"
        else:
            w, a_side, b_side = play_game(t1, t2, cfg_b, cfg_a, booster, meta, rng, max_turns), "p2", "p1"
        if w == a_side:
            a_wins += 1
        elif w == b_side:
            b_wins += 1
        else:
            draws += 1
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_games}: A {a_wins} - {b_wins} B ({draws} draws)", flush=True)
    decided = a_wins + b_wins
    wr = a_wins / decided if decided else 0.5
    se = (wr * (1 - wr) / decided) ** 0.5 if decided else 0.5
    return {"A_wins": a_wins, "B_wins": b_wins, "draws": draws, "n": n_games,
            "A_winrate": wr, "ci95": 1.96 * se}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    variant = sys.argv[2] if len(sys.argv) > 2 else ""
    booster, meta = load_model()
    base = {"mode": "fast", "pessimism": 0.7}
    if variant == "determinize":
        a, b, label = {**base, "worlds": 3}, base, "+determinization vs baseline"
    elif variant == "mixed":
        a, b, label = {**base, "mixed": True, "epsilon": 0.05}, base, "+mixed-root vs baseline"
    elif variant == "deep":
        a, b, label = {"mode": "deep", "pessimism": 0.7}, base, "deep vs fast"
    else:
        a, b, label = base, base, "baseline vs baseline (sanity: expect ~50%)"
    print(f"A/B: {label}  |  {n} games\n")
    res = ab_test(a, b, n, booster, meta)
    print(f"\nA winrate {res['A_winrate']:.1%} +/- {res['ci95']:.1%}  "
          f"(A {res['A_wins']} - {res['B_wins']} B, {res['draws']} draws)")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
