"""Authoritative set data for [Gen 9] Random Battle.

Random battles don't need *usage-based* set prediction the way OU does: the team
generator's possibility space is published. Every species has a fixed level and,
per "role" (Fast Attacker, Bulky Support, Setup Sweeper, ...), the exact pool of
moves / items / abilities / Tera types it can roll. We distill that (from the
pkmn/randbats mirror of Showdown's own generator) into assets/randbats_sets.json
and expose the *same* interface movesets.py does, so the advisor and battle
engine work unchanged — with two format-specific wins:

* the real **level** (randbats mons are ~L64-100, not a flat L100), which feeds a
  level-aware damage formula and stat calc; and
* **role narrowing**: once a couple of moves are revealed we can rule out
  incompatible roles, so the predicted rest-of-set sharpens instead of averaging
  over every possibility.

Spreads follow the gen9 randbats convention: 85 EVs in every stat, 31 IVs, and no
nature (the format balances by level, not by min-maxed natures), plus the sparse
per-set overrides the data records (e.g. 0 Atk EV/IV on special attackers). Since
there is no nature to guess, the only remaining stat uncertainty is which role the
opponent rolled — and the in-battle speed-order and damage inference correct the
rest from observation.

Build the asset (needs network; run once, or when the format rotates):
    python -m src.randbats
"""

import json
from functools import lru_cache

import requests

from src.common import ROOT
from src.pokedex import STATS, load_moves, lookup, norm_name

ASSET = ROOT / "assets" / "randbats_sets.json"
DATA_URL = "https://pkmn.github.io/randbats/data/gen9randombattle.json"
EV_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")
DEFAULT_LEVEL = 100


# ---- runtime prediction ------------------------------------------------------

def load_sets() -> dict:
    if not ASSET.exists():
        return {}
    if not hasattr(load_sets, "_cache"):
        load_sets._cache = json.loads(ASSET.read_text(encoding="utf-8"))
    return load_sets._cache


def species_set(species: str) -> dict | None:
    return load_sets().get(norm_name(species))


def species_level(species: str) -> int:
    entry = species_set(species)
    return entry["level"] if entry else DEFAULT_LEVEL


def _compatible_roles(entry: dict, revealed_ids: set) -> list[dict]:
    """Roles whose move pool contains every revealed move that the species can
    actually learn — i.e. the roles still consistent with what we've seen. Falls
    back to all roles when nothing (yet) discriminates."""
    roles = list((entry.get("roles") or {}).values())
    pool = {m for r in roles for m in r["moves"]}
    constraining = revealed_ids & pool
    if not constraining:
        return roles
    compat = [r for r in roles if constraining <= set(r["moves"])]
    return compat or roles


@lru_cache(maxsize=4096)
def _predict_moves_cached(species: str, revealed: tuple, k: int) -> list[str]:
    revealed_ids = {norm_name(m) for m in revealed}
    entry = species_set(species)
    moves = load_moves()

    def display(mid: str) -> str:
        return moves.get(mid, {}).get("name", mid)

    result = [display(m) for m in revealed_ids if m in moves]
    if not entry:
        return result
    roles = _compatible_roles(entry, revealed_ids)
    # rank the still-possible moves by how many compatible roles carry them
    freq: dict[str, int] = {}
    for r in roles:
        for m in r["moves"]:
            freq[m] = freq.get(m, 0) + 1
    for mid in sorted(freq, key=lambda m: -freq[m]):
        if mid not in revealed_ids and mid in moves and len(result) < k:
            result.append(display(mid))
    return result


def predict_moves(species: str, revealed=(), k: int = 4) -> list[str]:
    """Most likely k moves (display names), keeping revealed ones and narrowing to
    the roles still consistent with them."""
    return _predict_moves_cached(species, tuple(revealed), k)


def moveset_with_probs(species: str, k: int = 8) -> list[tuple[str, float]]:
    entry = species_set(species)
    if not entry:
        return []
    moves = load_moves()
    return [(moves.get(m, {}).get("name", m), p) for m, p in entry["moves"][:k]]


def predict_spread(species: str) -> dict:
    entry = species_set(species)
    if entry and entry.get("spread"):
        return entry["spread"]
    return {"nature": "", "evs": [85] * 6, "ivs": [31] * 6, "atk_iv": 31}


@lru_cache(maxsize=4096)
def real_stats(species: str) -> dict:
    """Stats at the species' randbats level from base + the 85-EV neutral spread
    (with recorded 0-Atk/0-Spe overrides). No nature term — gen9 randbats has none."""
    dex = lookup(species)
    if not dex:
        return {s: 160 for s in STATS}
    lvl = species_level(species)
    spread = predict_spread(species)
    evs = dict(zip(EV_ORDER, spread["evs"]))
    ivs = dict(zip(EV_ORDER, spread.get("ivs") or [31] * 6))
    out = {}
    for s in STATS:
        inner = 2 * dex[s] + ivs.get(s, 31) + evs[s] // 4
        scaled = inner * lvl // 100
        out[s] = scaled + lvl + 10 if s == "hp" else scaled + 5  # neutral nature
    return out


# ---- asset builder -----------------------------------------------------------

def _apply_overrides(base: list[int], override: dict | None) -> list[int]:
    """Return EV/IV list with the data's sparse per-stat overrides applied."""
    out = list(base)
    for i, s in enumerate(EV_ORDER):
        if override and s in override and override[s] is not None:
            out[i] = override[s]
    return out


def _spread_for(mon: dict) -> dict:
    """gen9 randbats spread: 85 EVs / 31 IVs everywhere, plus the recorded
    overrides (0-Atk on special sets, partial HP, 0-Spe on Trick Room, ...)."""
    evs = _apply_overrides([85] * 6, mon.get("evs"))
    ivs = _apply_overrides([31] * 6, mon.get("ivs"))
    return {"nature": "", "evs": evs, "ivs": ivs, "atk_iv": ivs[1]}


def _role_freq(roles: dict, key: str, transform=lambda x: x) -> list[tuple[str, float]]:
    """Per-option probability = fraction of the species' roles that offer it
    (roles carry no weights in the data, so treat them as equiprobable)."""
    n = len(roles) or 1
    counts: dict[str, int] = {}
    for r in roles.values():
        for opt in r.get(key, []) or []:
            t = transform(opt)
            counts[t] = counts.get(t, 0) + 1
    return [(k, round(c / n, 4)) for k, c in sorted(counts.items(), key=lambda kv: -kv[1])]


def build_asset(src: str | None = None) -> None:
    if src and not str(src).startswith("http"):
        data = json.loads(open(src, encoding="utf-8").read())
    else:
        data = requests.get(src or DATA_URL, timeout=60).json()

    sets = {}
    for species, mon in data.items():
        roles = mon.get("roles") or {}
        norm_roles = {
            name: {
                "moves": [norm_name(m) for m in r.get("moves", [])],
                "item": [norm_name(i) for i in r.get("items", [])],
                "ability": [norm_name(a) for a in r.get("abilities", [])],
                "tera": [t.lower() for t in r.get("teraTypes", [])],
            }
            for name, r in roles.items()
        }
        sets[norm_name(species)] = {
            "level": mon["level"],
            "moves": _role_freq(roles, "moves", norm_name),
            "item": _role_freq(roles, "items", norm_name),
            "ability": _role_freq(roles, "abilities", norm_name),
            "tera": _role_freq(roles, "teraTypes", str.lower),
            "spread": _spread_for(mon),
            "roles": norm_roles,
        }
    ASSET.write_text(json.dumps(sets, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(sets)} species sets to {ASSET.relative_to(ROOT)} "
          f"({ASSET.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build_asset()
