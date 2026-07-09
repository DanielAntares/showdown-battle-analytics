"""Advisor behavior on real replay fixtures + hazard-chip math."""

import json
from pathlib import Path

import pytest

from src.advisor import advise, hazard_chip, rank_moves
from src.parser import parse_replay

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))


def _snap(**hazards):
    base = {"p1_hazard_stealthrock": 0, "p1_hazard_spikes": 0}
    base.update(hazards)
    return base


def test_hazard_chip_type_scaling():
    # Charizard (fire/flying): 4x rock weakness -> rocks alone cost 50%
    assert hazard_chip("Charizard", "p1", _snap(p1_hazard_stealthrock=1)) == 0.5
    # Great Tusk (ground/fighting): rocks doubly resisted -> 3.125%; grounded so spikes apply
    chip = hazard_chip("Great Tusk", "p1", _snap(p1_hazard_stealthrock=1, p1_hazard_spikes=3))
    assert chip == pytest.approx(0.125 * 0.25 + 0.25)
    # Corviknight (flying/steel): immune to spikes, resists rocks
    assert hazard_chip("Corviknight", "p1", _snap(p1_hazard_spikes=3)) == 0.0
    assert hazard_chip("", "p1", _snap()) == 0.0


def test_rank_moves_on_fixture():
    game = parse_replay(json.loads(FIXTURES[0].read_text(encoding="utf-8")))
    moves = rank_moves(game, "p1")
    if len(moves):  # active may have no revealed damaging moves at game end
        damaging = moves.dropna(subset=["damage_score"])
        assert (damaging.damage_score >= 0).all()
        assert list(damaging.damage_score) == sorted(damaging.damage_score, reverse=True)


def test_advise_full_shape():
    from src.predict import load_model, snapshot_features
    booster, meta = load_model()
    game = parse_replay(json.loads(FIXTURES[0].read_text(encoding="utf-8")), up_to_turn=10)
    assert game["n_turns"] == 10  # parse stopped at the requested turn
    out = advise(game, "p2", booster, meta, snapshot_features)
    assert set(out) == {"switches", "moves"}
    if len(out["switches"]):
        assert out["switches"].win_prob.between(0, 1).all()
        assert not out["switches"].option.str.contains("fainted").any()
