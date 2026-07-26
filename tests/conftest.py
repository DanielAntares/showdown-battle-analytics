"""Pin the set/level provider per test, independent of config.format.

The project's config now targets random battles, so src.movesets dispatches to
src.randbats (authoritative sets, per-species levels) by default. The legacy suite
was written against standard-format semantics (level 100, Smogon usage sets), so an
autouse fixture forces OU mode for every test; randbats tests opt in via the
`randbats_mode` fixture. Both clear the memoized mode flag and the stat/move caches
so the switch actually takes effect.
"""

import pytest

import src.movesets as movesets
import src.randbats as randbats


def _set_mode(is_randbats: bool) -> None:
    movesets._randbats_mode._v = is_randbats
    movesets.real_stats.cache_clear()
    movesets._predict_moves_cached.cache_clear()
    randbats.real_stats.cache_clear()
    randbats._predict_moves_cached.cache_clear()


@pytest.fixture(autouse=True)
def ou_mode():
    """Default every test to standard-format (level-100 usage) semantics."""
    _set_mode(False)
    yield


@pytest.fixture
def randbats_mode():
    """Opt a test into random-battle semantics (authoritative sets + real levels)."""
    _set_mode(True)
    yield
    _set_mode(False)
