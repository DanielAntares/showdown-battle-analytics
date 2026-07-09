"""Room-reference parsing (network-dependent behavior is exercised manually)."""

import pytest

from src.live import normalize_room


def test_normalize_room_variants():
    expected = "battle-gen9ou-2646246610"
    assert normalize_room("battle-gen9ou-2646246610") == expected
    assert normalize_room("https://play.pokemonshowdown.com/battle-gen9ou-2646246610") == expected
    assert normalize_room("BATTLE-GEN9OU-2646246610?p2") == expected
    assert normalize_room(">battle-gen9ou-2646246610\n|init|battle") == expected


def test_normalize_room_private_suffix():
    # private rooms carry an extra token; keep it so joins still work
    assert normalize_room("battle-gen9ou-123-abcxyz") == "battle-gen9ou-123-abcxyz"


def test_normalize_room_rejects_junk():
    with pytest.raises(ValueError):
        normalize_room("not a battle link")
