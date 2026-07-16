"""Parse Pokémon Showdown sim-protocol battle logs into per-turn state snapshots.

One snapshot is emitted at the start of every turn (at each ``|turn|N`` line),
capturing everything a spectator would know at that moment: HP, faints, status,
active Pokémon and their stat boosts, entry hazards, screens, weather, terrain,
Trick Room, and Terastallization. The eventual model predicts the game winner
from any one of these snapshots.

Protocol reference:
https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md

Known v1 limitations (all rare in Gen 9 OU): damage dealt to a Zoroark before
``|replace|`` stays credited to the disguised Pokémon; boost-copying moves
(Psych Up) are ignored.
"""

from dataclasses import dataclass, field

BOOST_STATS = ("atk", "def", "spa", "spd", "spe")
HAZARD_MAX = {"stealthrock": 1, "spikes": 3, "toxicspikes": 2, "stickyweb": 1}
SCREENS = ("reflect", "lightscreen", "auroraveil", "tailwind")


@dataclass
class Pokemon:
    species: str
    hp: float = 1.0
    status: str = ""
    fainted: bool = False
    revealed: bool = False  # actually seen in battle, not just team preview
    tera: str = ""
    moves: set = field(default_factory=set)
    uses: dict = field(default_factory=dict)  # move -> times used (PP tracking)
    sleep_turns: int = 0
    item: str = ""     # revealed held item ("" = unknown)
    ability: str = ""  # revealed ability ("" = unknown)


@dataclass
class Side:
    name: str = ""
    rating: int | None = None
    team: dict = field(default_factory=dict)  # species -> Pokemon
    nicks: dict = field(default_factory=dict)  # nickname -> species key
    active: str | None = None  # species key of the active Pokémon
    boosts: dict = field(default_factory=lambda: {s: 0 for s in BOOST_STATS})
    hazards: dict = field(default_factory=lambda: {h: 0 for h in HAZARD_MAX})
    screens: set = field(default_factory=set)
    volatiles: set = field(default_factory=set)  # encore/taunt/... on the active
    last_move: str = ""  # last move the active used since switching in
    screen_turns: dict = field(default_factory=dict)  # screen -> turn it was set
    tox_turns: int = 0  # toxic counter of the active (resets on switch)

    def active_mon(self) -> Pokemon | None:
        return self.team.get(self.active)


def _side_of(ident: str) -> str:
    """'p1a: Yanmega' or 'p2: username' -> 'p1' / 'p2'."""
    return ident[:2]


def _nick_of(ident: str) -> str:
    return ident.split(": ", 1)[1] if ": " in ident else ident


def _species_of(details: str) -> str:
    """'Zoroark, L84, M, shiny' -> 'Zoroark'; preview 'Urshifu-*' -> 'Urshifu'."""
    return details.split(",", 1)[0].strip().removesuffix("-*")


def _parse_hp(hp_str: str) -> tuple[float, str]:
    """'59/100' -> (0.59, ''); '0 fnt' -> (0.0, 'fnt'); '100/100 par' -> (1.0, 'par')."""
    parts = hp_str.strip().split(" ", 1)
    status = parts[1] if len(parts) > 1 else ""
    if parts[0] in ("0", "0.0") or status == "fnt":
        return 0.0, status
    cur, _, mx = parts[0].partition("/")
    try:
        return int(cur) / int(mx), status
    except (ValueError, ZeroDivisionError):
        return 1.0, status


def _norm_condition(cond: str) -> str:
    """'move: Stealth Rock' -> 'stealthrock'; 'Spikes' -> 'spikes'.

    Side/field conditions appear both bare and with a 'move:'/'ability:' prefix,
    so strip any prefix before normalizing.
    """
    return "".join(c for c in cond.split(":", 1)[-1].lower() if c.isalpha())


class BattleParser:
    def __init__(self) -> None:
        self.sides = {"p1": Side(), "p2": Side()}
        self.weather = ""
        self.field: set[str] = set()  # terrains, trick room, ...
        self.turn = 0
        self.tier = ""
        self.weather_set_turn = 0
        self.terrain_set_turn = 0
        self.winner: str | None = None
        self.snapshots: list[dict] = []
        self.events: dict[int, list] = {}  # turn -> what both players did

    def _event(self, side: str, text: str, luck: bool = False) -> None:
        if self.turn >= 1:  # ignore pre-battle lead switches
            self.events.setdefault(self.turn, []).append(
                {"side": side, "text": text, "luck": luck})

    # ---- roster helpers ------------------------------------------------------

    def _team_key(self, side: Side, species: str) -> str:
        """Match a switch-in species to its team-preview entry, tolerating forme
        suffixes (preview 'Urshifu' vs switch 'Urshifu-Rapid-Strike')."""
        if species in side.team:
            return species
        base = species.split("-", 1)[0]
        for key in side.team:
            if key == base or key.split("-", 1)[0] == base:
                return key
        side.team[species] = Pokemon(species)  # no preview (or unseen forme): add
        return species

    def _mon(self, ident: str) -> Pokemon | None:
        side = self.sides[_side_of(ident)]
        key = side.nicks.get(_nick_of(ident))
        return side.team.get(key) if key else None

    # ---- event handlers ------------------------------------------------------

    def _handle_switch(self, ident: str, details: str, hp_str: str) -> None:
        side = self.sides[_side_of(ident)]
        key = self._team_key(side, _species_of(details))
        side.nicks[_nick_of(ident)] = key
        mon = side.team[key]
        mon.revealed = True
        mon.hp, status = _parse_hp(hp_str)
        if status and status != "fnt":
            mon.status = status
        side.active = key
        side.boosts = {s: 0 for s in BOOST_STATS}  # switching clears boosts
        side.volatiles = set()  # ... and volatile states / move locks
        side.last_move = ""
        side.tox_turns = 0  # the toxic counter resets on switching out

    def _handle_replace(self, ident: str, details: str) -> None:
        """Zoroark's Illusion drops: the nickname's true species is revealed."""
        side = self.sides[_side_of(ident)]
        key = self._team_key(side, _species_of(details))
        side.nicks[_nick_of(ident)] = key
        side.team[key].revealed = True
        side.active = key

    def _handle_side_condition(self, side_ident: str, cond: str, start: bool) -> None:
        side = self.sides[_side_of(side_ident)]
        cond = _norm_condition(cond)
        if cond in HAZARD_MAX:
            side.hazards[cond] = min(side.hazards[cond] + 1, HAZARD_MAX[cond]) if start else 0
        elif cond in SCREENS:
            if start:
                side.screens.add(cond)
                side.screen_turns[cond] = self.turn
            else:
                side.screens.discard(cond)
                side.screen_turns.pop(cond, None)

    def feed(self, line: str) -> None:
        if not line.startswith("|"):
            return
        p = line.split("|")
        cmd = p[1]

        if cmd == "tier":
            self.tier = p[2]
        elif cmd == "player" and len(p) > 3 and p[2] in self.sides and p[3]:
            side = self.sides[p[2]]
            side.name = p[3]
            if len(p) > 5 and p[5].isdigit():
                side.rating = int(p[5])
        elif cmd == "poke":
            species = _species_of(p[3])
            self.sides[p[2]].team.setdefault(species, Pokemon(species))
        elif cmd in ("switch", "drag"):
            self._handle_switch(p[2], p[3], p[4])
            side_id = _side_of(p[2])
            species = self.sides[side_id].active
            if cmd == "drag":
                self._event(side_id, f"{species} was dragged in")
            else:
                after_faint = any(
                    e["side"] == side_id and e["text"].endswith("fainted")
                    for e in self.events.get(self.turn, []))
                verb = "sent out" if after_faint else "switched to"
                self._event(side_id, f"{verb} {species}")
        elif cmd == "replace":
            self._handle_replace(p[2], p[3])
        elif cmd == "detailschange":  # mega/forme change keeps the same team entry
            self.sides[_side_of(p[2])].nicks.setdefault(_nick_of(p[2]), _species_of(p[3]))
        elif cmd == "move":
            if mon := self._mon(p[2]):
                mon.revealed = True
                mon.moves.add(p[3])
                mon.uses[p[3]] = mon.uses.get(p[3], 0) + 1
                self.sides[_side_of(p[2])].last_move = p[3]
                self._event(_side_of(p[2]), f"{mon.species} used {p[3]}")
        elif cmd in ("-damage", "-heal", "-sethp"):
            if mon := self._mon(p[2]):
                mon.hp, status = _parse_hp(p[3])
                if status == "fnt":
                    mon.fainted = True
                elif status:
                    mon.status = status
        elif cmd == "faint":
            if mon := self._mon(p[2]):
                mon.hp, mon.fainted, mon.status = 0.0, True, ""
                self._event(_side_of(p[2]), f"{mon.species} fainted")
        elif cmd == "-crit":
            if mon := self._mon(p[2]):
                self._event(_side_of(p[2]), f"{mon.species} took a critical hit", luck=True)
        elif cmd == "-miss":
            if mon := self._mon(p[2]):
                self._event(_side_of(p[2]), f"{mon.species}'s attack missed", luck=True)
        elif cmd == "-start":
            self.sides[_side_of(p[2])].volatiles.add(_norm_condition(p[3]))
        elif cmd == "-end":
            self.sides[_side_of(p[2])].volatiles.discard(_norm_condition(p[3]))
        elif cmd in ("-item", "-enditem"):
            if mon := self._mon(p[2]):
                mon.item = p[3]  # item name; revealed by Knock Off, Boots proc, etc.
        elif cmd == "-ability":
            if mon := self._mon(p[2]):
                mon.ability = p[3]
        elif cmd == "-status":
            if mon := self._mon(p[2]):
                mon.status = p[3]
        elif cmd == "-curestatus":
            if mon := self._mon(p[2]):
                mon.status = ""
        elif cmd in ("-boost", "-unboost", "-setboost"):
            side, stat = self.sides[_side_of(p[2])], p[3]
            if stat in side.boosts:
                amount = int(p[4])
                if cmd == "-setboost":
                    side.boosts[stat] = amount
                else:
                    sign = 1 if cmd == "-boost" else -1
                    side.boosts[stat] = max(-6, min(6, side.boosts[stat] + sign * amount))
        elif cmd == "-clearboost" or cmd == "-clearnegativeboost":
            side = self.sides[_side_of(p[2])]
            for s, v in side.boosts.items():
                if cmd == "-clearboost" or v < 0:
                    side.boosts[s] = 0
        elif cmd == "-clearallboost":
            for side in self.sides.values():
                side.boosts = {s: 0 for s in BOOST_STATS}
        elif cmd == "-sidestart" or cmd == "-sideend":
            self._handle_side_condition(p[2], p[3], start=cmd == "-sidestart")
        elif cmd == "-swapsideconditions":  # Court Change
            p1, p2 = self.sides["p1"], self.sides["p2"]
            p1.hazards, p2.hazards = p2.hazards, p1.hazards
            p1.screens, p2.screens = p2.screens, p1.screens
        elif cmd == "-weather":
            name = _norm_condition(p[2])
            if name != "none" and "[upkeep]" not in line:
                self.weather_set_turn = self.turn  # fresh weather, not a tick
            self.weather = "" if name == "none" else name
        elif cmd == "-fieldstart":
            cond = _norm_condition(p[2])
            if cond.endswith("terrain"):
                self.terrain_set_turn = self.turn
            self.field.add(cond)
        elif cmd == "-fieldend":
            self.field.discard(_norm_condition(p[2]))
        elif cmd == "-terastallize":
            if mon := self._mon(p[2]):
                mon.tera = p[3]
                self._event(_side_of(p[2]), f"{mon.species} Terastallized ({p[3]})")
        elif cmd == "turn":
            self.turn = int(p[2])
            for side in self.sides.values():  # tick status counters at turn starts
                if active := side.active_mon():
                    if active.status == "slp":
                        active.sleep_turns += 1
                    elif active.status == "tox":
                        side.tox_turns += 1
            self.snapshots.append(self.snapshot())
        elif cmd == "win":
            name = p[2]
            self.winner = next(
                (sid for sid, side in self.sides.items() if side.name == name), None
            )

    # ---- output --------------------------------------------------------------

    def snapshot(self) -> dict:
        row: dict = {"turn": self.turn}
        for sid, side in self.sides.items():
            mons = list(side.team.values())
            active = side.active_mon()
            row.update(
                {
                    f"{sid}_fainted": sum(m.fainted for m in mons),
                    f"{sid}_revealed": sum(m.revealed for m in mons),
                    f"{sid}_hp_total": sum(m.hp for m in mons),
                    f"{sid}_healthy": sum(not m.fainted and m.hp >= 0.5 for m in mons),
                    f"{sid}_moves_revealed": sum(len(m.moves) for m in mons),
                    f"{sid}_items_revealed": sum(bool(m.item) for m in mons),
                    f"{sid}_abilities_revealed": sum(bool(m.ability) for m in mons),
                    f"{sid}_statused": sum(bool(m.status) and not m.fainted for m in mons),
                    f"{sid}_active_species": active.species if active else "",
                    f"{sid}_active_hp": active.hp if active else 0.0,
                    f"{sid}_active_status": active.status if active else "",
                    f"{sid}_tera_used": any(bool(m.tera) for m in mons),
                    **{f"{sid}_boost_{s}": side.boosts[s] for s in BOOST_STATS},
                    **{f"{sid}_hazard_{h}": side.hazards[h] for h in HAZARD_MAX},
                    **{f"{sid}_screen_{s}": s in side.screens for s in SCREENS},
                }
            )
        row["weather"] = self.weather
        row["terrain"] = next((f for f in self.field if f.endswith("terrain")), "")
        row["trickroom"] = "trickroom" in self.field
        return row


def game_state(parser: BattleParser, id: str | None = None,
               format: str | None = None, rating: int | None = None) -> dict:
    """The game dict downstream code consumes — from any parser (replay or live)."""
    p1, p2 = parser.sides["p1"], parser.sides["p2"]
    return {
        "id": id,
        "format": format or parser.tier,
        "rating": rating,
        "p1_name": p1.name,
        "p2_name": p2.name,
        "p1_rating": p1.rating,
        "p2_rating": p2.rating,
        "winner": parser.winner,
        "n_turns": parser.turn,
        "teams": {sid: sorted(s.team) for sid, s in parser.sides.items()},
        "roster": {
            sid: [{"species": m.species, "hp": m.hp, "status": m.status,
                   "fainted": m.fainted, "revealed": m.revealed,
                   "active": key == side.active, "moves": sorted(m.moves),
                   "item": m.item, "ability": m.ability, "tera": m.tera,
                   "uses": dict(m.uses), "sleep_turns": m.sleep_turns,
                   "volatiles": sorted(side.volatiles) if key == side.active else [],
                   "last_move": side.last_move if key == side.active else "",
                   "tox_turns": side.tox_turns if key == side.active else 0}
                  for key, m in side.team.items()]
            for sid, side in parser.sides.items()
        },
        "field": {
            "weather_set_turn": parser.weather_set_turn,
            "terrain_set_turn": parser.terrain_set_turn,
            "screen_turns": {sid: dict(s.screen_turns)
                             for sid, s in parser.sides.items()},
        },
        "snapshots": list(parser.snapshots),
        "events": dict(parser.events),
    }


def parse_replay(replay: dict, up_to_turn: int | None = None) -> dict:
    """Parse one replay JSON (as served by replay.pokemonshowdown.com/<id>.json).

    With `up_to_turn`, parsing stops at the start of that turn — the returned
    state is exactly what both players saw when picking that turn's actions.
    """
    parser = BattleParser()
    for line in replay["log"].splitlines():
        parser.feed(line)
        if up_to_turn is not None and parser.turn >= up_to_turn:
            break
    return game_state(parser, id=replay.get("id"), format=replay.get("format"),
                      rating=replay.get("rating"))
