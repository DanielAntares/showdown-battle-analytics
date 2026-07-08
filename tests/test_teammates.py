"""Teammate model behavior on a controlled synthetic corpus."""

from src.teammates import TeammateModel

ROSTERS = [
    ["Garchomp", "Rotom-Wash", "Ferrothorn"],
    ["Garchomp", "Rotom-Wash", "Clefable"],
    ["Garchomp", "Rotom-Wash", "Ferrothorn"],
    ["Weavile", "Dragapult", "Clefable"],
]


def test_cooccurrence_beats_usage():
    model = TeammateModel.fit(ROSTERS)
    ranked = model.scores(["Garchomp"]).index.tolist()
    # Rotom-Wash appears with Garchomp 3/3 times; Dragapult never does
    assert ranked[0] == "Rotom-Wash"
    assert ranked.index("Ferrothorn") < ranked.index("Dragapult")


def test_revealed_are_excluded():
    model = TeammateModel.fit(ROSTERS)
    ranked = model.scores(["Garchomp", "Rotom-Wash"]).index
    assert "Garchomp" not in ranked and "Rotom-Wash" not in ranked


def test_unknown_species_ignored():
    model = TeammateModel.fit(ROSTERS)
    assert not model.scores(["Missingno"]).empty  # falls back to usage prior


def test_predict_returns_normalized_likelihoods():
    model = TeammateModel.fit(ROSTERS)
    out = model.predict(["Garchomp"], top=3)
    assert len(out) == 3
    assert (out.relative_likelihood > 0).all()
    assert out.relative_likelihood.iloc[0] >= out.relative_likelihood.iloc[-1]
