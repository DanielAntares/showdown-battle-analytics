"""Live assistant: request-grounded state, legal choice mapping, full pipeline."""

import json
import re
from pathlib import Path

from src.assistant import advise_for_request, build_game, map_choice
from src.movesets import predict_moves
from src.parser import parse_replay
from src.pokedex import norm_name
from src.predict import load_model

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))
CHOOSE = re.compile(r"^(move \d( terastallize)?|switch \d|team \d)$")


def _fixture(idx=0):
    return json.loads(FIXTURES[idx].read_text(encoding="utf-8"))


def _log_until_turn(log: str, turn: int) -> str:
    lines, out = log.splitlines(), []
    for ln in lines:
        out.append(ln)
        if ln.startswith(f"|turn|{turn}"):
            break
    return "\n".join(out)


def _request_for(game: dict, side: str, rqid=7, **extra) -> dict:
    """A believable |request| built from the parsed state."""
    roster = game["roster"][side]
    pokemon = [{"details": m["species"], "active": m["active"],
                "condition": "0 fnt" if m["fainted"] else "100/100",
                "item": "", "moves": []} for m in roster]
    active = next((m for m in roster if m["active"] and not m["fainted"]), None)
    req = {"side": {"id": side, "pokemon": pokemon}, "rqid": rqid, **extra}
    if active and not extra.get("forceSwitch") and not extra.get("teamPreview"):
        names = predict_moves(active["species"], active.get("moves", ()), 4)
        req["active"] = [{"moves": [{"move": n, "id": norm_name(n), "disabled": False}
                                    for n in names]}]
    return req


def test_map_choice_moves_switches_and_tera():
    req = {"active": [{"canTerastallize": "Fire",
                       "moves": [{"move": "Pyro Ball", "id": "pyroball"},
                                 {"move": "U-turn", "id": "uturn", "disabled": True},
                                 {"move": "Sucker Punch", "id": "suckerpunch"}]}],
           "side": {"id": "p1", "pokemon": [
               {"details": "Cinderace, M", "active": True, "condition": "63/100"},
               {"details": "Kyurem", "active": False, "condition": "100/100"},
               {"details": "Dondozo, F", "active": False, "condition": "0 fnt"}]}}
    # plain move
    assert map_choice([{"action": "Pyro Ball"}], req)[0] == "move 1"
    # tera variant appends the suffix
    assert map_choice([{"action": "Tera Fire + Pyro Ball"}], req)[0] == "move 1 terastallize"
    # disabled move is skipped in favour of the next row
    assert map_choice([{"action": "U-turn"}, {"action": "Sucker Punch"}], req)[0] == "move 3"
    # switches map by species; fainted targets are never chosen
    assert map_choice([{"action": "switch to Kyurem"}], req)[0] == "switch 2"
    assert map_choice([{"action": "switch to Dondozo"}, {"action": "Pyro Ball"}],
                      req)[0] == "move 1"
    # forced switch ignores move rows entirely
    req_force = {**req, "forceSwitch": [True], "active": None}
    assert map_choice([{"action": "Pyro Ball"}, {"action": "switch to Kyurem"}],
                      req_force)[0] == "switch 2"
    # trapped: switch rows are skipped
    req_trap = json.loads(json.dumps(req))
    req_trap["active"][0]["trapped"] = True
    assert map_choice([{"action": "switch to Kyurem"}, {"action": "Pyro Ball"}],
                      req_trap)[0] == "move 1"


def test_advise_for_request_end_to_end():
    booster, meta = load_model()
    raw = _fixture()
    log = _log_until_turn(raw["log"], 8)
    game = parse_replay({"log": log})
    for side in ("p1", "p2"):
        req = _request_for(game, side)
        res = advise_for_request(log, req, booster, meta, mode="fast")
        assert res["ok"] and res["rqid"] == 7
        assert CHOOSE.match(res["choose"]), res["choose"]
        assert res["table"] and 0.0 <= (res["winprob"] or 0.5) <= 1.0


def test_wait_request_returns_no_choice():
    booster, meta = load_model()
    res = advise_for_request("", {"wait": True}, booster, meta)
    assert res["ok"] and res["choose"] is None


def test_request_grounds_our_moveset():
    """The request's real moves replace usage guesses for our active."""
    raw = _fixture()
    log = _log_until_turn(raw["log"], 8)
    game = parse_replay({"log": log})
    side = "p1"
    req = _request_for(game, side)
    req["active"] = [{"moves": [{"move": "Surf", "id": "surf"},
                                {"move": "Recover", "id": "recover"}]}]
    g2 = build_game(log, req)
    active = next(m for m in g2["roster"][side] if m["active"])
    assert active["moves"] == ["Surf", "Recover"]
