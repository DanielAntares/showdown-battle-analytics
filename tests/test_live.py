"""Room parsing and log retention (server-dependent behavior is exercised manually)."""

import pytest

from src.live import LiveBattle, normalize_room
from src.parser import parse_replay


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


def test_live_log_retention_enables_turn_review():
    lb = LiveBattle("battle-gen9ou-123", connect=False)
    lb._on_message(None, ">battle-gen9ou-123\n|player|p1|Alice||1500\n"
                         "|player|p2|Bob||1490\n|tier|[Gen 9] OU\n"
                         "|switch|p1a: X|Pikachu|100/100\n|switch|p2a: Y|Eevee|100/100\n|turn|1")
    lb._on_message(None, ">battle-gen9ou-123\n|move|p1a: X|Thunderbolt|p2a: Y\n"
                         "|-damage|p2a: Y|50/100\n|turn|2")
    game = lb.snapshot_game()
    assert game["n_turns"] == 2 and game["p1_rating"] == 1500

    # reconstruct the state at turn 1's start: Eevee not yet damaged
    at_turn_1 = parse_replay({"log": lb.raw_log()}, up_to_turn=1)
    assert at_turn_1["n_turns"] == 1
    eevee = next(m for m in at_turn_1["roster"]["p2"] if m["species"] == "Eevee")
    assert eevee["hp"] == 1.0
    # and at turn 2's start the damage is visible
    at_turn_2 = parse_replay({"log": lb.raw_log()}, up_to_turn=2)
    eevee2 = next(m for m in at_turn_2["roster"]["p2"] if m["species"] == "Eevee")
    assert eevee2["hp"] == 0.5
