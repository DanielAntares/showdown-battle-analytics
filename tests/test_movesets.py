"""Moveset / spread prediction from usage stats."""

import pytest

from src.movesets import (NATURES, predict_moves, predict_spread, real_stats,
                          species_set)

pytestmark = pytest.mark.skipif(
    species_set("Great Tusk") is None,
    reason="usage_sets.json not built (run: python -m src.movesets)")


def test_predicts_known_set():
    moves = [m.lower() for m in predict_moves("Great Tusk", k=4)]
    assert "headlong rush".replace(" ", "") in [m.replace(" ", "") for m in moves] \
        or "rapid spin".replace(" ", "") in [m.replace(" ", "") for m in moves]
    assert len(predict_moves("Great Tusk", k=4)) == 4


def test_revealed_moves_kept_first():
    out = predict_moves("Gholdengo", revealed=["Nasty Plot"], k=4)
    assert out[0] == "Nasty Plot"
    assert len(out) == 4


def test_special_attacker_gets_zero_atk_iv():
    # Gholdengo is a special attacker -> Attack IV minimized
    assert predict_spread("Gholdengo")["atk_iv"] == 0
    # Great Tusk is physical -> full Attack IV
    assert predict_spread("Great Tusk")["atk_iv"] == 31


def test_real_stats_reflect_investment():
    tusk = real_stats("Great Tusk")
    # Jolly 252 Spe: fast; 252 Atk: strong; SpA uninvested and Jolly-hindered: low
    assert tusk["spe"] > 290
    assert tusk["atk"] > 330
    assert tusk["spa"] < tusk["atk"]


def test_nature_table_is_consistent():
    for nature, (plus, minus) in NATURES.items():
        assert plus != minus
        assert plus in ("atk", "def", "spa", "spd", "spe")


def test_unknown_species_safe_fallback():
    assert predict_moves("Not A Real Mon") == []
    spread = predict_spread("Not A Real Mon")
    assert spread["evs"] == [85, 85, 85, 85, 85, 85]
