"""Parser invariants checked against real replay fixtures.

Fixtures are unmodified downloads from replay.pokemonshowdown.com; every
invariant here must hold for *any* well-formed gen9ou replay, so new fixtures
can be dropped in without touching the tests.
"""

import json
from pathlib import Path

import pytest

from src.parser import BOOST_STATS, _norm_condition, parse_replay

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))


@pytest.fixture(params=FIXTURES, ids=lambda p: p.stem)
def game(request):
    replay = json.loads(request.param.read_text(encoding="utf-8"))
    return parse_replay(replay)


def test_metadata(game):
    assert game["p1_name"] and game["p2_name"]
    assert game["winner"] in ("p1", "p2")
    assert game["n_turns"] >= 1
    assert len(game["snapshots"]) == game["n_turns"]


def test_teams(game):
    for side in ("p1", "p2"):
        assert 1 <= len(game["teams"][side]) <= 6


def test_snapshot_invariants(game):
    for snap in game["snapshots"]:
        for side in ("p1", "p2"):
            assert 0 <= snap[f"{side}_fainted"] <= 6
            assert 0 <= snap[f"{side}_hp_total"] <= 6 + 1e-9
            assert 0 <= snap[f"{side}_active_hp"] <= 1 + 1e-9
            assert snap[f"{side}_revealed"] >= 1  # someone is always on the field
            for stat in BOOST_STATS:
                assert -6 <= snap[f"{side}_boost_{stat}"] <= 6
            assert 0 <= snap[f"{side}_hazard_spikes"] <= 3
            assert 0 <= snap[f"{side}_hazard_toxicspikes"] <= 2


def test_progress_is_monotonic(game):
    """Faints and reveals never decrease as the battle progresses."""
    for side in ("p1", "p2"):
        faints = [s[f"{side}_fainted"] for s in game["snapshots"]]
        reveals = [s[f"{side}_revealed"] for s in game["snapshots"]]
        assert faints == sorted(faints)
        assert reveals == sorted(reveals)


def test_turns_are_sequential(game):
    assert [s["turn"] for s in game["snapshots"]] == list(range(1, game["n_turns"] + 1))


def test_condition_normalization():
    """Conditions appear both bare and with a 'move:' prefix in the protocol."""
    assert _norm_condition("move: Stealth Rock") == "stealthrock"
    assert _norm_condition("Spikes") == "spikes"
    assert _norm_condition("move: Toxic Spikes") == "toxicspikes"
    assert _norm_condition("move: Aurora Veil") == "auroraveil"
    assert _norm_condition("move: Trick Room") == "trickroom"


def test_hazards_are_tracked():
    """Every |-sidestart| hazard in a fixture must surface in some snapshot.

    Regression test: a normalization bug once left 'move:'-prefixed hazards
    (Stealth Rock, Toxic Spikes) permanently zero while tests still passed.
    """
    for path in FIXTURES:
        replay = json.loads(path.read_text(encoding="utf-8"))
        game = parse_replay(replay)
        for hazard, marker in [("stealthrock", "move: Stealth Rock"),
                               ("toxicspikes", "move: Toxic Spikes"),
                               ("spikes", "|Spikes")]:
            if marker in replay["log"]:
                assert any(
                    s[f"p1_hazard_{hazard}"] + s[f"p2_hazard_{hazard}"] > 0
                    for s in game["snapshots"]
                ), f"{path.stem}: {hazard} appears in the log but never in a snapshot"
