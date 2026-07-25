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

import html as _html
import re
from dataclasses import dataclass, field

BOOST_STATS = ("atk", "def", "spa", "spd", "spe")
# volatiles Baton Pass hands to the incoming Pokémon (confusion/taunt stay behind)
_BP_PASSED = {"substitute", "leechseed", "curse", "focusenergy", "aquaring", "ingrain"}
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
    item_consumed: bool = False  # item was used/knocked/popped -> now has none
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
    acc_boosts: dict = field(default_factory=lambda: {"accuracy": 0, "evasion": 0})
    disabled: str = ""  # a move Disabled / Cursed-Body'd on the active ("" = none)
    last_move: str = ""  # last move the active used since switching in
    screen_turns: dict = field(default_factory=dict)  # screen -> turn it was set
    tox_turns: int = 0  # toxic counter of the active (resets on switch)
    future_pending: bool = False  # this side has an unresolved Future Sight/Doom Desire

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
        # observed move order: when both sides move in one turn, who went first
        # and under which visible speed modifiers — the raw material for proving
        # "their X is faster than my Y" instead of guessing from usage spreads
        self.speed_obs: list[dict] = []
        self._turn_first: tuple | None = None
        self._turn_pair_done = False
        # observed direct damage: how hard each hit actually landed, with the
        # context needed to recompute what it SHOULD have done — deviations
        # reveal hidden items/spreads (Band/Specs/max-invest vs the usage guess)
        self.dmg_obs: list[dict] = []
        self._pending_hit: dict | None = None

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
        # Baton Pass hands the incoming mon the boosts and passable volatiles;
        # Shed Tail passes only the substitute. Anything else clears everything.
        passing = _norm_condition(side.last_move) if side.last_move else ""
        if passing == "batonpass":
            side.volatiles &= _BP_PASSED
        elif passing == "shedtail":
            side.volatiles &= {"substitute"}
            side.boosts = {s: 0 for s in BOOST_STATS}
            side.acc_boosts = {"accuracy": 0, "evasion": 0}
        else:
            side.boosts = {s: 0 for s in BOOST_STATS}  # switching clears boosts
            side.acc_boosts = {"accuracy": 0, "evasion": 0}
            side.volatiles = set()  # ... and volatile states / move locks
        side.disabled = ""
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

    def _note_move_order(self, side_id: str, move: str) -> None:
        """Record who moved first when both sides act in the same turn, along
        with each side's *visible* speed modifiers at that instant (boost stage,
        paralysis, Tailwind) and Trick Room. The advisor later normalizes these
        away, so an observation like "my +2 mon went first" proves only
        raw×2 > theirs — never the bare "mine is faster"."""
        if self._turn_pair_done or self.turn < 1:
            return
        side = self.sides[side_id]
        active = side.active_mon()
        if active is None:
            return
        ctx = {"spe_stage": side.boosts["spe"], "status": active.status,
               "tailwind": "tailwind" in side.screens}
        if self._turn_first is None:
            # species captured NOW — a U-turn pivot changes side.active before
            # the opponent's move finalizes the observation
            self._turn_first = (side_id, move, ctx, side.active)
        elif self._turn_first[0] != side_id:
            f_side, f_move, f_ctx, f_species = self._turn_first
            self.speed_obs.append({
                "turn": self.turn, "first": f_side, "second": side_id,
                "first_move": f_move, "second_move": move,
                "species": {f_side: f_species, side_id: side.active},
                "ctx": {f_side: f_ctx, side_id: ctx},
                "trickroom": "trickroom" in self.field,
            })
            self._turn_pair_done = True

    def _arm_damage_obs(self, side_id: str, move: str) -> None:
        """Snapshot the attack's full context so the advisor can later recompute
        the EXPECTED damage and compare it to what actually landed."""
        atk_side = self.sides[side_id]
        def_side = self.sides["p2" if side_id == "p1" else "p1"]
        atk, dfn = atk_side.active_mon(), def_side.active_mon()
        if atk is None or dfn is None:
            self._pending_hit = None
            return
        self._pending_hit = {
            "side": side_id, "move": move, "crit": False, "turn": self.turn,
            "attacker": atk.species, "atk_item": atk.item, "atk_status": atk.status,
            "atk_tera": atk.tera, "atk_boosts": dict(atk_side.boosts),
            "atk_fainted": sum(m.fainted for m in atk_side.team.values()),
            "defender": dfn.species, "def_item": dfn.item, "def_status": dfn.status,
            "def_tera": dfn.tera, "def_boosts": dict(def_side.boosts),
            "def_screens": sorted(def_side.screens),
            "weather": self.weather,
            "terrain": next((f for f in self.field if f.endswith("terrain")), ""),
        }

    def _capture_reveals(self, cmd: str, p: list[str]) -> None:
        """Abilities/items revealed by [from]/[of] tags on any minor action —
        Water Absorb immunities, Rocky Helmet chip, Leftovers heals, Intimidate
        activations. Without this the advisor keeps trusting the usage-predicted
        ability after the battle has already shown the real one.

        [of] convention: on -damage lines the [of] Pokémon owns the effect
        (Rough Skin / Rocky Helmet hurt the attacker); elsewhere the line's
        subject owns it (absorb heals name the *attacker* in [of])."""
        subject = p[2] if len(p) > 2 and p[2][:2] in self.sides else None
        ability = item = of_ident = None
        for part in p[3:]:
            if part.startswith("[from] ability: "):
                ability = part[len("[from] ability: "):]
            elif part.startswith("ability: "):  # e.g. |-activate|MON|ability: X
                ability = part[len("ability: "):]
            elif part.startswith("[from] item: "):
                item = part[len("[from] item: "):]
            elif part.startswith("[of] "):
                of_ident = part[len("[of] "):]
        if ability is None and item is None:
            return
        owner = of_ident if (cmd == "-damage" and of_ident) else subject
        if owner and (mon := self._mon(owner)):
            if ability:
                mon.ability = ability
            if item:
                mon.item, mon.item_consumed = item, False

    def feed(self, line: str) -> None:
        if not line.startswith("|"):
            return
        p = line.split("|")
        cmd = p[1]
        if cmd.startswith("-"):
            self._capture_reveals(cmd, p)

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
                self._note_move_order(_side_of(p[2]), p[3])
                self._arm_damage_obs(_side_of(p[2]), p[3])
                if _norm_condition(p[3]) in ("futuresight", "doomdesire"):
                    self.sides[_side_of(p[2])].future_pending = True
                if "lockedmove" in line:  # emerged from Dig/Fly/... this turn
                    self.sides[_side_of(p[2])].volatiles.discard("semiinvuln")
                self._event(_side_of(p[2]), f"{mon.species} used {p[3]}")
        elif cmd == "-prepare":  # charging a two-turn move (Dig/Fly/...) -> invulnerable
            self.sides[_side_of(p[2])].volatiles.add("semiinvuln")
        elif cmd in ("-damage", "-heal", "-sethp"):
            if mon := self._mon(p[2]):
                if (cmd == "-damage" and self._pending_hit
                        and _side_of(p[2]) != self._pending_hit["side"]
                        and not any(x.startswith("[from]") for x in p[3:])):
                    o = self._pending_hit
                    self._pending_hit = None  # one observation per move
                    after, _ = _parse_hp(p[3])
                    self.dmg_obs.append({**o, "def_hp": mon.hp, "after": after})
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
            if self._pending_hit:
                self._pending_hit["crit"] = True
            if mon := self._mon(p[2]):
                self._event(_side_of(p[2]), f"{mon.species} took a critical hit", luck=True)
        elif cmd == "-miss":
            if mon := self._mon(p[2]):
                self._event(_side_of(p[2]), f"{mon.species}'s attack missed", luck=True)
        elif cmd == "-start":
            side = self.sides[_side_of(p[2])]
            cond = _norm_condition(p[3])
            side.volatiles.add(cond)
            if cond == "disable" and len(p) > 4:  # Disable/Cursed Body name one move
                side.disabled = p[4]
        elif cmd == "-end":
            cond = _norm_condition(p[3])
            self.sides[_side_of(p[2])].volatiles.discard(cond)
            if cond == "disable":
                self.sides[_side_of(p[2])].disabled = ""
            if cond in ("futuresight", "doomdesire"):  # resolved on the target's slot
                self.sides["p1" if _side_of(p[2]) == "p2" else "p2"].future_pending = False
        elif cmd == "-item":
            if mon := self._mon(p[2]):
                mon.item, mon.item_consumed = p[3], False  # item revealed / gained
        elif cmd == "-enditem":
            if mon := self._mon(p[2]):
                mon.item, mon.item_consumed = "", True  # consumed / knocked / popped
        elif cmd == "-ability":
            if mon := self._mon(p[2]):
                mon.ability = p[3]
        elif cmd == "-status":
            if mon := self._mon(p[2]):
                mon.status = p[3]
                if p[3] == "slp":
                    mon.sleep_turns = 0  # fresh sleep (e.g. a new Rest) resets the count
        elif cmd == "-curestatus":
            if mon := self._mon(p[2]):
                mon.status = ""
        elif cmd in ("-boost", "-unboost", "-setboost"):
            side, stat = self.sides[_side_of(p[2])], p[3]
            table = side.boosts if stat in side.boosts else \
                (side.acc_boosts if stat in side.acc_boosts else None)
            if table is not None:
                amount = int(p[4])
                if cmd == "-setboost":
                    table[stat] = amount
                else:
                    sign = 1 if cmd == "-boost" else -1
                    table[stat] = max(-6, min(6, table[stat] + sign * amount))
        elif cmd == "-clearboost" or cmd == "-clearnegativeboost":
            side = self.sides[_side_of(p[2])]
            for table in (side.boosts, side.acc_boosts):
                for s, v in table.items():
                    if cmd == "-clearboost" or v < 0:
                        table[s] = 0
        elif cmd == "-clearallboost":
            for side in self.sides.values():
                side.boosts = {s: 0 for s in BOOST_STATS}
                side.acc_boosts = {"accuracy": 0, "evasion": 0}
        elif cmd == "-copyboost":  # |-copyboost|USER|SOURCE (Psych Up)
            if len(p) > 3 and p[2][:2] in self.sides and p[3][:2] in self.sides:
                user, src = self.sides[_side_of(p[2])], self.sides[_side_of(p[3])]
                user.boosts = dict(src.boosts)
                user.acc_boosts = dict(src.acc_boosts)
        elif cmd == "-swapboost":  # Guard Swap / Power Swap / Heart Swap
            if len(p) > 3 and p[2][:2] in self.sides and p[3][:2] in self.sides:
                a, b = self.sides[_side_of(p[2])], self.sides[_side_of(p[3])]
                names = ([s.strip() for s in p[4].split(",")]
                         if len(p) > 4 and p[4] and not p[4].startswith("[")
                         else list(BOOST_STATS) + ["accuracy", "evasion"])
                for s in names:
                    if s in a.boosts:
                        a.boosts[s], b.boosts[s] = b.boosts[s], a.boosts[s]
                    elif s in a.acc_boosts:
                        a.acc_boosts[s], b.acc_boosts[s] = b.acc_boosts[s], a.acc_boosts[s]
        elif cmd == "-invertboost":  # Topsy-Turvy
            side = self.sides[_side_of(p[2])]
            side.boosts = {s: -v for s, v in side.boosts.items()}
            side.acc_boosts = {s: -v for s, v in side.acc_boosts.items()}
        elif cmd == "-transform":  # partial: copy boosts + known moves + ability
            if len(p) > 3 and (user := self._mon(p[2])) and (tgt := self._mon(p[3])):
                us, ts = self.sides[_side_of(p[2])], self.sides[_side_of(p[3])]
                us.boosts, us.acc_boosts = dict(ts.boosts), dict(ts.acc_boosts)
                user.moves |= set(tgt.moves)
                if tgt.ability:
                    user.ability = tgt.ability
                us.volatiles.add("transformed")
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
            self._turn_first, self._turn_pair_done = None, False
            self._pending_hit = None
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


def roster_of(parser: BattleParser) -> dict:
    """The per-side team state (each mon's hp/status/moves/active flag/...) at the
    parser's current position. Reused to snapshot the roster mid-battle, not only
    at the end (the action-prediction dataset needs it at every turn)."""
    return {
        sid: [{"species": m.species, "hp": m.hp, "status": m.status,
               "fainted": m.fainted, "revealed": m.revealed,
               "active": key == side.active, "moves": sorted(m.moves),
               "item": m.item, "item_consumed": m.item_consumed,
               "ability": m.ability, "tera": m.tera,
               "uses": dict(m.uses), "sleep_turns": m.sleep_turns,
               "volatiles": sorted(side.volatiles) if key == side.active else [],
               "acc_stage": side.acc_boosts["accuracy"] if key == side.active else 0,
               "eva_stage": side.acc_boosts["evasion"] if key == side.active else 0,
               "disabled": side.disabled if key == side.active else "",
               "last_move": side.last_move if key == side.active else "",
               "tox_turns": side.tox_turns if key == side.active else 0,
               "future_pending": side.future_pending if key == side.active else False}
              for key, m in side.team.items()]
        for sid, side in parser.sides.items()
    }


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
        "roster": roster_of(parser),
        "field": {
            "weather_set_turn": parser.weather_set_turn,
            "terrain_set_turn": parser.terrain_set_turn,
            "screen_turns": {sid: dict(s.screen_turns)
                             for sid, s in parser.sides.items()},
        },
        "snapshots": list(parser.snapshots),
        "events": dict(parser.events),
        "speed_obs": list(parser.speed_obs),
        "dmg_obs": list(parser.dmg_obs),
    }


_LOG_SCRIPT = re.compile(
    r'<script[^>]*class="[^"]*battle-log-data[^"]*"[^>]*>(.*?)</script>', re.S | re.I)


def extract_log(text: str) -> str:
    """Pull a battle log out of text the user supplies directly, so a battle that
    can't be fetched (private, unlisted, never uploaded) can still be analyzed.

    Accepts either the raw '|'-prefixed protocol lines copied from the client, or
    the HTML file Showdown's "Download replay" button produces, which embeds the
    same log in a <script class="battle-log-data"> block.
    """
    if not text:
        return ""
    if m := _LOG_SCRIPT.search(text):
        text = _html.unescape(m.group(1))
    lines = [ln for ln in (t.rstrip() for t in text.splitlines())
             if ln.lstrip().startswith("|")]
    return "\n".join(ln.lstrip() for ln in lines)


def is_battle_log(log: str) -> bool:
    """A log we can actually parse — it must identify players and have a turn."""
    return "|player|" in log and "|turn|" in log


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
