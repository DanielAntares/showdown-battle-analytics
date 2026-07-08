"""Type-chart and pokedex lookup sanity checks."""

import json
from pathlib import Path

from src.parser import parse_replay
from src.pokedex import effectiveness, lookup, type_advantage

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))


def test_type_chart():
    assert effectiveness("fire", ["grass"]) == 2
    assert effectiveness("electric", ["ground"]) == 0
    assert effectiveness("ice", ["ground", "flying"]) == 4  # Gliscor's nightmare
    assert effectiveness("water", ["water", "dragon"]) == 0.25
    assert type_advantage(["fairy", "fighting"], ["dark", "steel"]) == 4  # fighting 2x2


def test_lookup():
    lando = lookup("Landorus-Therian")
    assert lando["types"] == ["ground", "flying"] and lando["spe"] == 91
    assert lookup("Kingambit")["types"] == ["dark", "steel"]
    # cosmetic formes fall back to the base form's identical stats
    assert lookup("Gastrodon-East") == lookup("Gastrodon")
    assert lookup("Not A Pokemon") is None


def test_fixture_species_all_resolve():
    for path in FIXTURES:
        game = parse_replay(json.loads(path.read_text(encoding="utf-8")))
        for side in ("p1", "p2"):
            for species in game["teams"][side]:
                assert lookup(species) is not None, f"{path.stem}: {species}"
