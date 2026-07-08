"""Species knowledge: base stats and a type-effectiveness chart.

Distills Showdown's pokedex.json into a small committed asset
(assets/pokedex_min.json) so training and the deployed app share one lookup
without needing the 500 KB original at runtime. The Gen-6+ type chart is
embedded below: it is tiny, universal, and hasn't changed since 2013.

Usage (refresh the asset):
    python -m src.pokedex
"""

import json
import re
from functools import lru_cache

import requests

from src.common import ROOT

POKEDEX_URL = "https://play.pokemonshowdown.com/data/pokedex.json"
ASSET = ROOT / "assets" / "pokedex_min.json"
STATS = ("hp", "atk", "def", "spa", "spd", "spe")

# attacker -> defender -> multiplier; only non-neutral entries listed
TYPE_CHART = {
    "normal":   {"rock": 0.5, "steel": 0.5, "ghost": 0},
    "fire":     {"grass": 2, "ice": 2, "bug": 2, "steel": 2,
                 "fire": 0.5, "water": 0.5, "rock": 0.5, "dragon": 0.5},
    "water":    {"fire": 2, "ground": 2, "rock": 2,
                 "water": 0.5, "grass": 0.5, "dragon": 0.5},
    "electric": {"water": 2, "flying": 2,
                 "electric": 0.5, "grass": 0.5, "dragon": 0.5, "ground": 0},
    "grass":    {"water": 2, "ground": 2, "rock": 2, "fire": 0.5, "grass": 0.5,
                 "poison": 0.5, "flying": 0.5, "bug": 0.5, "dragon": 0.5, "steel": 0.5},
    "ice":      {"grass": 2, "ground": 2, "flying": 2, "dragon": 2,
                 "fire": 0.5, "water": 0.5, "ice": 0.5, "steel": 0.5},
    "fighting": {"normal": 2, "ice": 2, "rock": 2, "dark": 2, "steel": 2,
                 "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5,
                 "fairy": 0.5, "ghost": 0},
    "poison":   {"grass": 2, "fairy": 2,
                 "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0},
    "ground":   {"fire": 2, "electric": 2, "poison": 2, "rock": 2, "steel": 2,
                 "grass": 0.5, "bug": 0.5, "flying": 0},
    "flying":   {"grass": 2, "fighting": 2, "bug": 2,
                 "electric": 0.5, "rock": 0.5, "steel": 0.5},
    "psychic":  {"fighting": 2, "poison": 2, "psychic": 0.5, "steel": 0.5, "dark": 0},
    "bug":      {"grass": 2, "psychic": 2, "dark": 2, "fire": 0.5, "fighting": 0.5,
                 "poison": 0.5, "flying": 0.5, "ghost": 0.5, "steel": 0.5, "fairy": 0.5},
    "rock":     {"fire": 2, "ice": 2, "flying": 2, "bug": 2,
                 "fighting": 0.5, "ground": 0.5, "steel": 0.5},
    "ghost":    {"psychic": 2, "ghost": 2, "dark": 0.5, "normal": 0},
    "dragon":   {"dragon": 2, "steel": 0.5, "fairy": 0},
    "dark":     {"psychic": 2, "ghost": 2, "fighting": 0.5, "dark": 0.5, "fairy": 0.5},
    "steel":    {"ice": 2, "rock": 2, "fairy": 2,
                 "fire": 0.5, "water": 0.5, "electric": 0.5, "steel": 0.5},
    "fairy":    {"fighting": 2, "dragon": 2, "dark": 2,
                 "fire": 0.5, "poison": 0.5, "steel": 0.5},
}


def norm_name(species: str) -> str:
    """'Landorus-Therian' -> 'landorustherian' (matches pokedex.json keys)."""
    return re.sub(r"[^a-z0-9]", "", species.lower())


def effectiveness(attacker: str, defender_types: list[str]) -> float:
    mult = 1.0
    for d in defender_types:
        mult *= TYPE_CHART.get(attacker, {}).get(d, 1.0)
    return mult


def type_advantage(attacker_types: list[str], defender_types: list[str]) -> float:
    """Best STAB effectiveness the attacker has into the defender's typing."""
    return max(effectiveness(a, defender_types) for a in attacker_types)


@lru_cache(maxsize=1)
def load_pokedex() -> dict:
    return json.loads(ASSET.read_text(encoding="utf-8"))


def lookup(species: str) -> dict | None:
    """Stats+types for a species; cosmetic formes fall back to the base form."""
    dex = load_pokedex()
    entry = dex.get(norm_name(species))
    if entry is None and "-" in species:
        entry = dex.get(norm_name(species.split("-", 1)[0]))
    return entry


def build_asset() -> None:
    raw = requests.get(POKEDEX_URL, timeout=60).json()
    dex = {}
    for key, mon in raw.items():
        stats = mon.get("baseStats")
        if not stats:
            continue
        dex[key] = {s: stats[s] for s in STATS}
        dex[key]["types"] = [t.lower() for t in mon["types"]]
    ASSET.parent.mkdir(exist_ok=True)
    ASSET.write_text(json.dumps(dex, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(dex)} species to {ASSET.relative_to(ROOT)} "
          f"({ASSET.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build_asset()
