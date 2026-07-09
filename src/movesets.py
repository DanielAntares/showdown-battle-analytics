"""Predict a Pokémon's likely moves, item, ability, and EV/nature spread.

Trained on Smogon's monthly usage statistics (the "chaos" JSON), which record,
for every species at a rating baseline, the weighted frequency of each move,
item, ability, Tera type, and exact EV spread actually used on the ladder. We
distill that into a compact per-species asset (assets/usage_sets.json) and use
it two ways:

* fill in a Pokémon's probable moveset before it reveals anything, so the
  advisor never has to say "revealed moves only";
* estimate real level-100 stats from the most common EV spread + nature (IVs
  inferred: 31 everywhere, 0 Atk for purely special attackers), so the damage
  simulation uses believable speed tiers and bulk instead of a flat assumption.

Honest limitation on team context: usage stats are *marginal* per species
(aggregated across all teams), so the base prediction is not conditioned on the
rest of the team. What we can condition on is what the battle has *revealed* —
confirmed moves are always kept and excluded from the "unknown slots" guess.

Usage (rebuild the asset):
    python -m src.movesets
"""

import json

from src.common import ROOT, load_config
from src.pokedex import STATS, load_moves, lookup, norm_name

ASSET = ROOT / "assets" / "usage_sets.json"
TOP_MOVES = 8

# nature -> (boosted stat, lowered stat); neutral natures omitted
NATURES = {
    "Adamant": ("atk", "spa"), "Lonely": ("atk", "def"), "Naughty": ("atk", "spd"),
    "Brave": ("atk", "spe"), "Modest": ("spa", "atk"), "Mild": ("spa", "def"),
    "Rash": ("spa", "spd"), "Quiet": ("spa", "spe"), "Bold": ("def", "atk"),
    "Impish": ("def", "spa"), "Lax": ("def", "spd"), "Relaxed": ("def", "spe"),
    "Calm": ("spd", "atk"), "Gentle": ("spd", "def"), "Careful": ("spd", "spa"),
    "Sassy": ("spd", "spe"), "Timid": ("spe", "atk"), "Hasty": ("spe", "def"),
    "Jolly": ("spe", "spa"), "Naive": ("spe", "spd"),
}
EV_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")


# ---- runtime prediction ------------------------------------------------------

def load_sets() -> dict:
    if not ASSET.exists():
        return {}
    if not hasattr(load_sets, "_cache"):
        load_sets._cache = json.loads(ASSET.read_text(encoding="utf-8"))
    return load_sets._cache


def species_set(species: str) -> dict | None:
    return load_sets().get(norm_name(species))


def predict_moves(species: str, revealed=(), k: int = 4) -> list[str]:
    """Most likely k moves (as display names), always keeping revealed ones."""
    revealed_ids = {norm_name(m) for m in revealed}
    entry = species_set(species)
    moves = load_moves()

    def display(mid: str) -> str:
        return moves.get(mid, {}).get("name", mid)

    result = [display(m) for m in revealed_ids if m in moves]
    if entry:
        for mid, _ in entry["moves"]:
            if mid not in revealed_ids and mid in moves and len(result) < k:
                result.append(display(mid))
    return result


def moveset_with_probs(species: str, k: int = TOP_MOVES) -> list[tuple[str, float]]:
    entry = species_set(species)
    if not entry:
        return []
    moves = load_moves()
    return [(moves.get(m, {}).get("name", m), p) for m, p in entry["moves"][:k]]


def predict_spread(species: str) -> dict:
    """Most common (nature, EVs) plus an inferred Atk IV; neutral 85s if unknown."""
    entry = species_set(species)
    if entry and entry.get("spread"):
        return entry["spread"]
    return {"nature": "", "evs": [85, 85, 85, 85, 85, 85], "atk_iv": 31}


def real_stats(species: str) -> dict:
    """Level-100 stats from base + predicted EV spread + nature (IV 31, or 0 Atk)."""
    dex = lookup(species)
    if not dex:
        return {s: 160 for s in STATS}
    spread = predict_spread(species)
    evs = dict(zip(EV_ORDER, spread["evs"]))
    plus, minus = NATURES.get(spread.get("nature", ""), (None, None))
    out = {}
    for s in STATS:
        iv = spread.get("atk_iv", 31) if s == "atk" else 31
        base_val = 2 * dex[s] + iv + evs[s] // 4
        if s == "hp":
            out[s] = base_val + 110  # level-100 HP formula
        else:
            val = base_val + 5
            if s == plus:
                val = int(val * 1.1)
            elif s == minus:
                val = int(val * 0.9)
            out[s] = val
    return out


# ---- asset builder -----------------------------------------------------------

def _top(counts: dict, n: int) -> list[tuple[str, float]]:
    total = sum(counts.values()) or 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:n]
    return [(k, round(w / total, 4)) for k, w in ranked]


def _parse_spread(spread_str: str) -> dict:
    nature, _, evs = spread_str.partition(":")
    ev_list = [int(x) for x in evs.split("/")] if evs else [0] * 6
    return {"nature": nature, "evs": ev_list}


def build_asset() -> None:
    cfg = load_config()
    baseline = cfg["stats_baseline"]
    src = cfg["paths"]["raw_stats"] / f"{cfg['format']}-{cfg['stats_month']}-{baseline}.json"
    data = json.loads(src.read_text(encoding="utf-8"))["data"]

    sets = {}
    for species, mon in data.items():
        raw = mon.get("Raw count", 0)
        if raw < 200:  # ignore near-unused species
            continue
        moves = [(m, round(w / raw, 4)) for m, w in
                 sorted(mon["Moves"].items(), key=lambda kv: -kv[1])[:TOP_MOVES]
                 if m]  # drop the "" (no-move) entry
        spread_str = max(mon["Spreads"], key=mon["Spreads"].get, default="Serious:0/0/0/0/0/0")
        spread = _parse_spread(spread_str)
        # infer Atk IV: special/status attackers minimize Attack (Foul Play, confusion)
        move_cats = {load_moves().get(m, {}).get("category") for m, _ in moves}
        spread["atk_iv"] = 0 if "Physical" not in move_cats else 31

        sets[norm_name(species)] = {
            "moves": moves,
            "item": _top(mon["Items"], 3),
            "ability": _top(mon["Abilities"], 2),
            "tera": _top(mon["Tera Types"], 3),
            "spread": spread,
            "spreads": [{**_parse_spread(s), "prob": round(w / raw, 3)}
                        for s, w in sorted(mon["Spreads"].items(),
                                           key=lambda kv: -kv[1])[:3]],
        }
    ASSET.write_text(json.dumps(sets, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(sets)} species sets to {ASSET.relative_to(ROOT)} "
          f"({ASSET.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build_asset()
