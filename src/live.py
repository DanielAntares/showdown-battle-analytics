"""Spectate live battles over Pokémon Showdown's WebSocket.

The live protocol is the same line format as replay logs, so a background
thread feeds each incoming line to the same BattleParser used for replays;
the UI polls `snapshot_game()` for a consistent copy of the battle state.

Public battles only — no login is needed to spectate. Rooms replay their full
history on join, so attaching mid-game catches the chart up instantly.
"""

import json
import re
import threading
import time

import websocket

from src.parser import BattleParser, game_state

WS_URL = "wss://sim3.psim.us/showdown/websocket"


def normalize_room(ref: str) -> str:
    """URL, room id, or '>room' line -> canonical room id."""
    m = re.search(r"battle-[a-z0-9]+-\d+(-[a-z0-9]+)?", ref.strip().lower())
    if not m:
        raise ValueError(f"no battle room id in {ref!r}")
    return m.group(0)


def _one_shot_query(send_line: str, response_prefix: str, timeout: float = 10) -> str | None:
    """Connect as guest, send one query after the handshake, return the response line."""
    result: dict = {}
    got = threading.Event()

    def on_message(ws, msg):
        for line in msg.split("\n"):
            if line.startswith("|challstr|"):
                ws.send(send_line)
            elif line.startswith(response_prefix):
                result["line"] = line
                got.set()

    ws = websocket.WebSocketApp(WS_URL, on_message=on_message)
    thread = threading.Thread(target=ws.run_forever, daemon=True)
    thread.start()
    got.wait(timeout)
    ws.close()
    return result.get("line")


def list_battles(format_id: str = "gen9ou") -> list[dict]:
    """Live battles for a format, rated ones first (highest Elo on top)."""
    line = _one_shot_query(f"|/cmd roomlist {format_id},none,",
                           "|queryresponse|roomlist|")
    if not line:
        return []
    rooms = json.loads(line.split("|", 3)[3]).get("rooms", {})
    battles = [{"room": rid, "min_elo": info.get("minElo", 0), **info}
               for rid, info in rooms.items()]
    return sorted(battles, key=lambda b: b["min_elo"], reverse=True)


def find_user_battle(username: str) -> str | None:
    """The battle room a user is currently playing in, if public."""
    userid = re.sub(r"[^a-z0-9]", "", username.lower())
    line = _one_shot_query(f"|/cmd userdetails {userid}",
                           "|queryresponse|userdetails|")
    if not line:
        return None
    rooms = (json.loads(line.split("|", 3)[3]) or {}).get("rooms") or {}
    for room in rooms:
        if "battle-" in room:
            return normalize_room(room)
    return None


class LiveBattle:
    """Background spectator: joins a room and keeps a BattleParser current."""

    MAX_RECONNECTS = 5

    def __init__(self, room: str, connect: bool = True):
        self.room = normalize_room(room)
        self.parser = BattleParser()
        self.log: list[str] = []  # raw room lines, kept for turn-by-turn review
        self.status = "connecting"
        self.error = ""
        self._lock = threading.Lock()
        self._ws = None
        self._closed_by_user = False
        self._reconnects = 0
        if connect:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        """Supervisor loop: (re)connect until the battle ends or we give up.

        Re-joining replays the room's full history, so on every (re)connect we
        reset the parser and log and let the replay rebuild the state cleanly."""
        while not self._closed_by_user and self._reconnects <= self.MAX_RECONNECTS:
            with self._lock:
                self.parser = BattleParser()
                self.log = []
            self._ws = websocket.WebSocketApp(
                WS_URL, on_message=self._on_message,
                on_error=self._on_error, on_close=self._on_close)
            self._ws.run_forever(ping_interval=30, reconnect=0)
            if self._closed_by_user or self.status in ("ended", "error"):
                return
            self._reconnects += 1
            self.status = "reconnecting"
            time.sleep(min(2 ** self._reconnects, 20))
        if not self._closed_by_user and self.status not in ("ended", "error"):
            self.status = "disconnected"

    def _on_message(self, ws, msg):
        lines = msg.split("\n")
        in_room = lines[0] == f">{self.room}"
        with self._lock:
            for line in lines:
                if line.startswith("|challstr|"):
                    ws.send(f"|/join {self.room}")
                    self.status = "live"
                    self._reconnects = 0  # a clean join resets the backoff
                elif in_room:
                    if line.startswith(("|win|", "|tie|")):
                        self.status = "ended"
                    elif line.startswith("|noinit|"):
                        self.status = "error"
                        self.error = "room not found (battle over or private)"
                    self.log.append(line)
                    self.parser.feed(line)

    def _on_error(self, ws, err):
        with self._lock:
            if self.status != "ended":
                self.error = str(err)

    def _on_close(self, ws, *_):
        pass  # the _run supervisor decides whether to reconnect

    def snapshot_game(self) -> dict:
        """A consistent copy of the current battle state."""
        with self._lock:
            return game_state(self.parser, id=self.room)

    def raw_log(self) -> str:
        """The battle log so far — replayable through parse_replay(up_to_turn=N)."""
        with self._lock:
            return "\n".join(self.log)

    def close(self) -> None:
        self._closed_by_user = True
        if self._ws is not None:
            self._ws.close()
