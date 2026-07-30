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


def test_turn_events(game):
    """Every game records per-turn actions with valid turns and sides."""
    events = game["events"]
    assert events, "no per-turn events recorded"
    assert all(1 <= t <= game["n_turns"] for t in events)
    flat = [e for evs in events.values() for e in evs]
    assert all(e["side"] in ("p1", "p2") for e in flat)
    assert any("used" in e["text"] for e in flat), "no move events"
    assert any("fainted" in e["text"] for e in flat), "no faint events"


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


# ---- user-supplied logs (private / unlisted battles) -------------------------

def _fixture_replay():
    return json.loads(FIXTURES[0].read_text(encoding="utf-8"))


def test_extract_log_from_pasted_lines():
    from src.parser import extract_log, is_battle_log
    raw = _fixture_replay()["log"]
    log = extract_log(raw)
    assert is_battle_log(log)
    # a pasted log must reproduce the fetched replay's analysis exactly
    assert parse_replay({"log": log})["n_turns"] == parse_replay(_fixture_replay())["n_turns"]


def test_extract_log_from_downloaded_replay_html():
    """Showdown's 'Download replay' embeds the (HTML-escaped) log in a script block."""
    import html as _h
    from src.parser import extract_log, is_battle_log
    raw = _fixture_replay()["log"]
    page = ('<html><body><script type="text/plain" class="battle-log-data">'
            + _h.escape(raw) + '</script></body></html>')
    log = extract_log(page)
    assert is_battle_log(log)
    got, want = parse_replay({"log": log}), parse_replay(_fixture_replay())
    assert got["n_turns"] == want["n_turns"]
    assert (got["p1_name"], got["p1_rating"]) == (want["p1_name"], want["p1_rating"])
    assert got["winner"] == want["winner"]


def test_extract_log_rejects_non_logs():
    from src.parser import extract_log, is_battle_log
    assert extract_log("") == ""
    assert not is_battle_log(extract_log("just some text\nno protocol lines"))
    # HTML chrome around the log is dropped, indentation tolerated
    assert extract_log("  |turn|1\n<div>noise</div>\n  |player|p1|a|1|1500") == \
        "|turn|1\n|player|p1|a|1|1500"


def test_disable_tracks_and_clears_the_named_move():
    from src.parser import BattleParser, roster_of
    log = ("|player|p1|a|1|1000\n|player|p2|b|1|1000\n|poke|p1|Dragonite|\n"
           "|poke|p1|Kyurem|\n|poke|p2|Blastoise|\n|start\n"
           "|switch|p1a: Dragonite|Dragonite, M|100/100\n"
           "|switch|p2a: Blastoise|Blastoise, M|100/100\n|turn|1\n"
           "|move|p1a: Dragonite|Earthquake|p2a: Blastoise\n"
           "|-start|p1a: Dragonite|Disable|Earthquake|[from] ability: Cursed Body\n|turn|2")
    p = BattleParser()
    for line in log.splitlines():
        p.feed(line)
    active = next(m for m in roster_of(p)["p1"] if m["active"])
    assert active["disabled"] == "Earthquake"
    # switching the mon out clears the disable
    for line in ["|switch|p1a: Kyurem|Kyurem|100/100", "|turn|3"]:
        p.feed(line)
    assert all(m["disabled"] == "" for m in roster_of(p)["p1"])


def test_rampage_move_locks_the_user_and_clears_on_fatigue():
    """Outrage / Thrash / Petal Dance lock the user into the move for 2-3 turns, so
    the advisor must not offer a switch until it ends (the 'suggested a switch but
    played Outrage' report)."""
    from src.parser import BattleParser, roster_of
    base = ("|player|p1|a|1|1000\n|player|p2|b|1|1000\n"
            "|poke|p1|Dragonite, M|\n|poke|p2|Blissey, F|\n|start\n"
            "|switch|p1a: Dragonite|Dragonite, M|100/100\n"
            "|switch|p2a: Blissey|Blissey, F|100/100\n|turn|1\n"
            "|move|p1a: Dragonite|Outrage|p2a: Blissey\n|turn|2\n")
    p = BattleParser()
    for ln in base.splitlines():
        p.feed(ln)
    assert next(m for m in roster_of(p)["p1"] if m["active"])["locked_move"] == "Outrage"
    # the rampage ends in self-confusion (fatigue) — the lock is over
    for ln in "|-start|p1a: Dragonite|confusion|[fatigue]\n|turn|3".splitlines():
        p.feed(ln)
    assert next(m for m in roster_of(p)["p1"] if m["active"])["locked_move"] == ""


def test_charge_move_prepare_does_not_hide_the_user():
    """Meteor Beam / Solar Beam / Sky Attack emit |-prepare| just like Dig/Fly but
    keep the user fully hittable. Flagging them semi-invulnerable (and never
    clearing it, since Power Herb fires them in one turn) made every attack read as
    a 0-damage whiff — so the advisor was forced to switch turn after turn (the
    Glimmora Meteor Beam ping-pong in gen9randombattle-2656531368)."""
    from src.parser import BattleParser, roster_of

    def prepare(species, move):
        log = (f"|player|p1|a|1|1000\n|player|p2|b|1|1000\n"
               f"|poke|p1|Persian, M|\n|poke|p2|{species}, M|\n|start\n"
               f"|switch|p1a: Persian|Persian, M|100/100\n"
               f"|switch|p2a: {species}|{species}, M|100/100\n|turn|1\n"
               f"|move|p2a: {species}|{move}||[still]\n|-prepare|p2a: {species}|{move}")
        p = BattleParser()
        for ln in log.splitlines():
            p.feed(ln)
        return next(m for m in roster_of(p)["p2"] if m["active"])["volatiles"]

    assert "semiinvuln" not in prepare("Glimmora", "Meteor Beam")  # charge, stays visible
    assert "semiinvuln" not in prepare("Venusaur", "Solar Beam")
    assert "semiinvuln" in prepare("Dugtrio", "Dig")               # truly underground
    assert "semiinvuln" in prepare("Corviknight", "Fly")


def test_from_tag_reveals_ability_and_item():
    """Abilities/items revealed via [from]/[of] tags must update the roster —
    otherwise the advisor keeps trusting the usage-predicted ability after the
    battle has shown the real one (the Water Absorb Clodsire bug)."""
    from src.parser import BattleParser, roster_of
    base = ("|player|p1|a|1|1000\n|player|p2|b|1|1000\n"
            "|poke|p1|Samurott-Hisui, M|\n|poke|p2|Clodsire, M|\n|start\n"
            "|switch|p1a: Samurott|Samurott-Hisui, M|100/100\n"
            "|switch|p2a: Clodsire|Clodsire, M|100/100\n|turn|1\n")
    # -immune form: subject owns the ability
    p = BattleParser()
    for ln in (base + "|-immune|p2a: Clodsire|[from] ability: Water Absorb").splitlines():
        p.feed(ln)
    assert next(m for m in roster_of(p)["p2"] if m["active"])["ability"] == "Water Absorb"
    # -heal form with [of] naming the ATTACKER: still the subject's ability
    p = BattleParser()
    for ln in (base + "|-heal|p2a: Clodsire|100/100|[from] ability: Water Absorb"
                      "|[of] p1a: Samurott").splitlines():
        p.feed(ln)
    assert next(m for m in roster_of(p)["p2"] if m["active"])["ability"] == "Water Absorb"
    assert next(m for m in roster_of(p)["p1"] if m["active"])["ability"] == ""
    # -damage form: the [of] mon owns the item (Rocky Helmet), not the victim
    p.feed("|-damage|p1a: Samurott|84/100|[from] item: Rocky Helmet|[of] p2a: Clodsire")
    assert next(m for m in roster_of(p)["p2"] if m["active"])["item"] == "Rocky Helmet"
    assert next(m for m in roster_of(p)["p1"] if m["active"])["item"] == ""
