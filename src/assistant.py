"""Live assistant core: battle log + Showdown |request| JSON -> a concrete choice.

The browser extension captures, from the user's own client, (a) every protocol
line of the battle room and (b) the |request| JSON that lists the legal moves,
switches, and rqid for the pending decision. This module turns that pair into
the advisor's ranked table AND a ready-to-send `/choose ...` command:

    move 2 | move 2 terastallize | switch 3 | team 4

Because it runs on the player's own stream it works for private battles too
(the client is already in the room) — what stays impossible is spectating
someone ELSE's private game.

The request is also an accuracy upgrade: it reveals our true moveset, items and
abilities, so our own side stops being usage-guessed. Legality is enforced when
mapping the pick (disabled moves, trapped, fainted switch targets), so the
returned command is always one the simulator will accept.
"""

import json
import re

from src.advisor import (advise_search, pessimism_for_elo, recommend_lead)
from src.parser import BattleParser, game_state
from src.pokedex import norm_name
from src.predict import calibrate, snapshot_features
from src.search import deep_search

_TERA_LABEL = re.compile(r"^Tera (.+?) \+ (.+)$")


def build_game(log: str, request: dict | None = None) -> dict:
    """Parse the raw room log and synthesize the *current* state (a request can
    arrive mid-turn, e.g. a forced switch after a faint, when the last |turn|
    snapshot is already stale)."""
    parser = BattleParser()
    for line in log.splitlines():
        parser.feed(line)
    game = game_state(parser)
    game["snapshots"] = list(game["snapshots"]) + [parser.snapshot()]
    if request:
        _apply_request(game, request)
    return game


def _species_of_details(details: str) -> str:
    return (details or "").split(",")[0].strip()


def _parse_condition(cond: str) -> tuple[float, str, bool]:
    """A request's 'cur/max [status]' condition -> (hp fraction, status, fainted)."""
    cond = (cond or "").strip()
    if not cond or "fnt" in cond:
        return 0.0, "", True
    parts = cond.split()
    hp = 1.0
    if "/" in parts[0]:
        cur, _, mx = parts[0].partition("/")
        try:
            mx = float(mx)
            hp = float(cur) / mx if mx else 0.0
        except ValueError:
            hp = 1.0
    status = parts[1] if len(parts) > 1 else ""
    return max(0.0, min(1.0, hp)), status, False


def _reserve_from_request(p: dict, species: str) -> dict:
    """A roster entry for a bench Pokémon the log hasn't shown yet — random battles
    have no team preview, so our reserves only exist in the request until they're
    sent in. Same schema parser.roster_of produces."""
    hp, status, fainted = _parse_condition(p.get("condition"))
    return {"species": species, "hp": hp, "status": status, "fainted": fainted,
            "revealed": False, "active": bool(p.get("active")),
            "moves": list(p.get("moves") or []),
            "item": p.get("item") or "", "item_consumed": False,
            "ability": p.get("baseAbility") or p.get("ability") or "",
            "tera": p.get("teraType") or "", "uses": {}, "sleep_turns": 0,
            "volatiles": [], "acc_stage": 0, "eva_stage": 0, "disabled": "",
            "last_move": "", "tox_turns": 0, "future_pending": False}


def _apply_request(game: dict, request: dict) -> None:
    """Ground our own side in the request: true moves/items/abilities instead of
    usage guesses, AND add any bench Pokémon the log hasn't revealed (random battles
    skip team preview, so our reserves aren't in the parsed roster — without this the
    advisor can never suggest switching to them)."""
    side = (request.get("side") or {}).get("id")
    if side not in ("p1", "p2"):
        return
    roster = game["roster"].setdefault(side, [])
    by_species = {m["species"]: m for m in roster}
    for p in (request["side"].get("pokemon") or []):
        sp = _species_of_details(p.get("details", ""))
        mon = by_species.get(sp)
        if mon is None:  # an unrevealed reserve — bring it into the roster
            mon = _reserve_from_request(p, sp)
            roster.append(mon)
            by_species[sp] = mon
        if p.get("item"):
            mon["item"] = p["item"]
        ability = p.get("baseAbility") or p.get("ability")
        if ability:
            mon["ability"] = ability
        if not mon["active"] and p.get("moves"):
            mon["moves"] = list(p["moves"])  # ids; normalized downstream
    active = next((m for m in roster if m["active"]), None)
    act_req = (request.get("active") or [None])[0]
    if active and act_req and act_req.get("moves"):
        active["moves"] = [m.get("move") or m.get("id") for m in act_req["moves"]]
    # our real Tera type, if the request offers it (canTerastallize is the type
    # string in gen9) — so the advisor Teras to the type we'll actually get
    if active and isinstance((act_req or {}).get("canTerastallize"), str):
        active["tera_avail"] = act_req["canTerastallize"].lower()


def map_choice(rows: list, request: dict) -> tuple[str, str | None]:
    """Best legal `/choose` argument for the ranked action rows. Walks the table
    top-down and returns the first action the request allows, so the command can
    never be an illegal pick (disabled move, trapped switch, fainted target)."""
    act = (request.get("active") or [None])[0] or {}
    force = bool(request.get("forceSwitch"))
    trapped = bool(act.get("trapped"))
    can_tera = bool(act.get("canTerastallize"))
    req_moves = act.get("moves") or []
    mons = (request.get("side") or {}).get("pokemon") or []

    def move_slot(name: str) -> int | None:
        nid = norm_name(name)
        for i, m in enumerate(req_moves):
            if (m.get("id") or norm_name(m.get("move", ""))) == nid \
                    and not m.get("disabled"):
                return i + 1
        return None

    def switch_slot(species: str) -> int | None:
        for j, p in enumerate(mons):
            if _species_of_details(p.get("details", "")) == species \
                    and not p.get("active") and "fnt" not in (p.get("condition") or ""):
                return j + 1
        return None

    for r in rows:
        label = r["action"]
        if label.startswith("switch to "):
            if not trapped and (j := switch_slot(label[len("switch to "):])):
                return f"switch {j}", label
            continue
        if force:
            continue  # only a switch is legal
        tera, name = False, label
        if m := _TERA_LABEL.match(label):
            tera, name = True, m.group(2)
        if i := move_slot(name):
            return (f"move {i} terastallize" if tera and can_tera else f"move {i}"), label

    # fallbacks: any live bench mon, else the first enabled move, else struggle
    if force or not req_moves:
        j = next((j + 1 for j, p in enumerate(mons) if not p.get("active")
                  and "fnt" not in (p.get("condition") or "")), None)
        if j:
            return f"switch {j}", None
    i = next((i + 1 for i, m in enumerate(req_moves) if not m.get("disabled")), 1)
    return f"move {i}", None


def advise_for_request(log: str, request, booster, meta, mode: str = "deep") -> dict:
    """The full pipeline: parse, advise, and map to a legal command."""
    req = json.loads(request) if isinstance(request, str) else (request or {})
    if req.get("wait"):
        return {"ok": True, "waiting": True, "choose": None}
    side = (req.get("side") or {}).get("id") or "p1"
    opp = "p2" if side == "p1" else "p1"
    game = build_game(log, req)
    out = {"ok": True, "side": side, "rqid": req.get("rqid"), "mode": mode}

    try:  # win probability of the current board, from our seat
        p1 = float(calibrate(booster.predict(snapshot_features(game, meta)), meta)[-1])
        out["winprob"] = p1 if side == "p1" else 1.0 - p1
    except Exception:
        out["winprob"] = None

    if req.get("teamPreview"):
        rec = recommend_lead(game, side, booster, meta, snapshot_features)
        mons = (req.get("side") or {}).get("pokemon") or []
        slot, picked = 1, None
        if len(rec):
            best = rec.iloc[0].lead
            picked = f"lead {best}"
            slot = next((j + 1 for j, p in enumerate(mons)
                         if _species_of_details(p.get("details", "")) == best), 1)
            # same row shape as the in-battle table so the panel renders it
            out["table"] = [{"action": f"lead {r.lead}", "worst_case": r.worst_case,
                             "average": r.average, "worst_response": f"vs {r.worst_vs}"}
                            for r in rec.head(6).itertuples()]
        out["choose"], out["picked"] = f"team {slot}", picked
        return out

    import numpy as np
    from src.selfplay import pooled_advise
    rng = np.random.default_rng(req.get("rqid") or 0)
    pess = pessimism_for_elo(game.get(f"{opp}_rating"))
    worlds = int(req.get("_worlds") or 1)  # opponent determinization (opt-in)
    if worlds > 1:
        table = pooled_advise(game, side, booster, meta, pess, worlds, rng, mode=mode)
    elif mode == "deep":
        table = deep_search(game, side, booster, meta,
                            depth=2, rollout=3, top_k=3, pessimism=pess)
    else:
        table = advise_search(game, side, booster, meta, snapshot_features,
                              pessimism=pess)
    rows = table.to_dict("records")
    # auto-play sends the outright best legal action, so the panel (which shows the
    # top row + choose) always matches what actually gets played.
    out["choose"], out["picked"] = map_choice(rows, req)
    out["table"] = rows[:5]

    try:  # what the opponent is likely to do (behaviour model; optional)
        from src.opponent import predict_actions
        out["opp_pred"] = predict_actions(game, opp, top=3)
    except Exception:
        out["opp_pred"] = []
    return out
