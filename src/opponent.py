"""Opponent action model: predict what a player does next — which move, or a
switch and to which Pokémon — from the board state and their (predicted) options.

Unlike the win-probability model, which scores *positions*, this learns real
player *behavior* from the replay corpus: at every turn each side chose an action
we can read straight from the log, so ~600k genuine decisions are available as
labelled data. The pipeline is:

* `decisions_from_log` — extract one record per (turn, side): the turn-start
  state + roster (exactly what that player saw) and the action they actually
  took. Only free *choices* are kept (the active was alive and could move or
  switch); forced post-faint replacements are a separate sub-problem, flagged
  and excluded by default.
* `src/opp_features.py` turns each decision into (state × candidate-action) rows.
* `src/train_opponent.py` fits a ranker and reports top-k accuracy by Elo band.

Kept deliberately separate from the advisor so its accuracy can be measured on
its own before it is ever trusted to steer a recommendation.
"""

from src.parser import (BattleParser, _side_of, _species_of, is_battle_log,
                        roster_of)


def _split(line: str) -> list[str]:
    return line.split("|")


def decisions_from_log(log: str, keep_forced: bool = False) -> list[dict]:
    """One record per free decision in the battle. Each has: turn, side, elo,
    chosen (kind + name/target), context ('choice' or 'forced'), and the
    turn-start `snapshot` + `roster` the player decided from."""
    if not is_battle_log(log):
        return []
    parser = BattleParser()
    out: list[dict] = []
    turn = 0
    active_start: dict[str, str] = {}   # side -> species active at this turn's start
    acted: set[str] = set()             # sides whose primary action this turn is known
    fainted_before_acting: set[str] = set()

    for line in log.splitlines():
        p = _split(line)
        cmd = p[1] if len(p) > 1 else ""

        if cmd == "turn":
            parser.feed(line)
            turn = parser.turn
            active_start = {s: parser.sides[s].active for s in ("p1", "p2")}
            acted, fainted_before_acting = set(), set()
            continue

        # detect the action BEFORE feeding, so state reflects the turn's start
        if turn >= 1 and cmd in ("move", "switch", "drag", "faint"):
            side = _side_of(p[2])
            if cmd == "faint" and side not in acted:
                fainted_before_acting.add(side)
            elif cmd == "drag":
                acted.add(side)  # opponent forced it; not this side's decision
            elif cmd == "move" and side not in acted:
                _emit(out, parser, turn, side, active_start, ("move", p[3]),
                      "choice", keep_forced)
                acted.add(side)
            elif cmd == "switch" and side not in acted:
                forced = side in fainted_before_acting
                kind = "replace" if forced else "switch"
                _emit(out, parser, turn, side, active_start, (kind, _species_of(p[3])),
                      "forced" if forced else "choice", keep_forced)
                acted.add(side)

        parser.feed(line)

    return out


def _emit(out, parser, turn, side, active_start, chosen, context, keep_forced):
    if context == "forced" and not keep_forced:
        return
    # the deciding mon must have been alive at the turn's start (a real choice)
    start_sp = active_start.get(side)
    if context == "choice" and start_sp:
        mon = parser.sides[side].team.get(start_sp)
        if mon is not None and mon.fainted:
            return
    out.append({
        "turn": turn,
        "side": side,
        "elo": parser.sides[side].rating,
        "opp_elo": parser.sides["p2" if side == "p1" else "p1"].rating,
        "chosen_kind": chosen[0],
        "chosen_name": chosen[1],
        "context": context,
        "snapshot": dict(parser.snapshots[-1]) if parser.snapshots else {},
        "roster": roster_of(parser),
    })
