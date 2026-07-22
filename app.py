"""Streamlit demo: win-probability timelines for replays + teammate inference.

Run with:
    streamlit run app.py
"""

import random
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.advisor import advise_search, recommend_lead
from src.common import ROOT
from src.movesets import moveset_with_probs, predict_spread, species_set
from src.live import LiveBattle, find_user_battle, list_battles
from src.parser import parse_replay
from src.search import deep_search
from src.predict import (actions_by_turn, fetch_replay, key_moments, load_model,
                         predict_game, snapshot_features, turn_story, user_replays)
from src.teammates import TeammateModel

# chart chrome + series colors from the validated reference palette (dataviz skill)
SURFACE, GRID, BASELINE, INK_2, MUTED = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#52514e", "#898781"
BLUE, AQUA = "#2a78d6", "#1baf7a"

EXAMPLES = ["gen9ou-2645435074", "gen9ou-2645436173", "gen9ou-2645435369"]
BAND_RANGES = {"1100–1299": (1100, 1300), "1300–1499": (1300, 1500),
               "1500–1699": (1500, 1700), "1700+": (1700, 2100)}
GAMES_PARQUET = ROOT / "data" / "processed" / "games.parquet"

st.set_page_config(page_title="Showdown Win Probability", page_icon="📈", layout="centered")


@st.cache_resource
def cached_model():
    return load_model()


@st.cache_resource
def cached_teammates() -> TeammateModel:
    return TeammateModel.load()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_raw(replay_ref: str) -> dict:
    return fetch_replay(replay_ref)


@st.cache_data(ttl=3600, show_spinner=False)
def analyze(replay_ref: str):
    game = parse_replay(fetch_raw(replay_ref))
    booster, meta = cached_model()
    probs = predict_game(game, booster, meta)
    return game, probs


def render_advisor(game: dict, names: dict, key_prefix: str) -> None:
    """Best-action search: 1-ply minimax, or optional multi-turn deep search."""
    c1, c2 = st.columns(2)
    with c1:
        side_name = st.radio("Options for", [names["p1"], names["p2"]],
                             horizontal=True, key=f"{key_prefix}_side")
    with c2:
        mode = st.radio("Look-ahead", ["Fast (1 turn)", "Deep (~5 turns)"],
                        horizontal=True, key=f"{key_prefix}_depth",
                        help="Deep search reasons several turns ahead — setup payoff, "
                             "delayed KOs a Protect only postpones, compounding hazard "
                             "chip — but takes a couple of seconds.")
    side = "p1" if side_name == names["p1"] else "p2"
    booster, meta = cached_model()
    if mode.startswith("Deep"):
        with st.spinner("searching several turns ahead…"):
            out = deep_search(game, side, booster, meta, depth=2, rollout=3, top_k=3)
        engine_note = ("multi-turn maximin search (horizon ≈ 5 turns): a depth-2 "
                       "adversarial tree over each side's most promising moves, extended "
                       "by a greedy rollout, with every turn played out on the damage "
                       "engine and the resulting positions scored by the win-prob model.")
    else:
        with st.spinner("simulating action matrix…"):
            out = advise_search(game, side, booster, meta, snapshot_features)
        engine_note = ("1-ply minimax over every (action × opponent response) pair, each "
                       "simulated with a level-100 damage engine (STAB, type chart, boosts, "
                       "burn/paralysis, screens, weather, hazard chip) and scored by the "
                       "win-probability model.")
    if not len(out):
        st.caption("no legal actions to evaluate (active Pokémon fainted or unknown)")
        return
    best = out.iloc[0]
    st.success(f"**Best action: {best.action}** — {best.worst_case:.0%} win probability "
               f"even against the opponent's best response ({best.worst_response})")
    st.dataframe(out.assign(
        worst_case=lambda d: d.worst_case.map("{:.0%}".format),
        average=lambda d: d.average.map("{:.0%}".format)),
        hide_index=True, width="stretch")
    st.caption("⚠️ " + engine_note + " Unrevealed moves and EV/nature spreads are "
               "predicted from ladder usage stats (see the sets below).")

    with st.expander("🔮 Predicted sets in play (from ladder usage)"):
        for s in (side, "p2" if side == "p1" else "p1"):
            active = next((m for m in game["roster"][s] if m["active"]), None)
            if active:
                who = "This side" if s == side else "Opponent"
                st.markdown(f"*{who}'s active* — " +
                            predicted_set_md(active["species"], active.get("moves", ())))
        st.caption("Predictions are species-level from Smogon usage at a rating "
                   "baseline; they are not conditioned on the specific team, but ✓ "
                   "marks moves this battle has already revealed.")


EV_LABELS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]


def predicted_set_md(species: str, revealed=()) -> str:
    """A markdown block: likely moves (with usage %), item, ability, spread."""
    entry = species_set(species)
    if not entry:
        return f"**{species}** — not enough ladder data to predict a set."
    moves = moveset_with_probs(species, 6)
    rev = {m.lower().replace(" ", "").replace("-", "") for m in revealed}
    move_lines = []
    for name, prob in moves:
        seen = "✓ " if name.lower().replace(" ", "").replace("-", "") in rev else ""
        move_lines.append(f"{seen}{name} ({prob:.0%})")
    spread = predict_spread(species)
    evs = " / ".join(f"{v} {lab}" for v, lab in zip(spread["evs"], EV_LABELS) if v)
    nat = spread.get("nature", "")
    item = entry["item"][0][0] if entry.get("item") else "?"
    ability = entry["ability"][0][0] if entry.get("ability") else "?"
    tera = entry["tera"][0][0].title() if entry.get("tera") else "?"
    atk_iv = " · 0 Atk IV" if spread.get("atk_iv") == 0 else ""
    return (f"**{species}** — likely {item}, {ability}, Tera {tera}  \n"
            f"Moves: {' · '.join(move_lines)}  \n"
            f"Spread: {nat} {evs}{atk_iv}")


def render_lead_suggestion(game: dict, names: dict) -> None:
    """Which Pokémon each side should start with, given both full teams."""
    booster, meta = cached_model()
    cols = st.columns(2)
    for col, side in zip(cols, ("p1", "p2")):
        rec = recommend_lead(game, side, booster, meta, snapshot_features)
        if not len(rec):
            continue
        actual = (game["snapshots"][0][f"{side}_active_species"]
                  if game.get("snapshots") else None)
        best = rec.iloc[0].lead
        note = ""
        if actual:
            note = ("  \n✅ led the recommended pick" if actual == best
                    else f"  \n(actually led **{actual}**)")
        col.markdown(f"**{names[side]}** → lead **{best}**{note}")
        show = rec.assign(average=lambda d: d.average.map("{:.0%}".format),
                          worst_case=lambda d: d.worst_case.map("{:.0%}".format))
        show = show.rename(columns={"lead": "Lead", "average": "avg win%",
                                    "worst_case": "worst%", "worst_vs": "worst vs"})
        col.dataframe(show, hide_index=True, width="stretch")
    st.caption("Best lead = highest average win probability across the opponent's six "
               "possible leads (the opening matchup scored by the win-prob model). It "
               "assumes any of their team could lead; 'worst vs' is the reply it fears most.")


def render_key_moments(game: dict, probs, names: dict) -> None:
    for m in key_moments(game, probs):
        towards = names["p1" if m["delta"] > 0 else "p2"]
        if m["luck"]:
            tag = f"🎲 luck involved: {'; '.join(m['luck'])}"
        elif m["severity"] == "major":
            tag = f"⚠️ major swing against {names[m['against']]} — possible blunder"
        elif m["severity"] == "big":
            tag = f"big swing against {names[m['against']]} — possible mistake"
        else:
            tag = ""
        lines = [f"**Turn {m['turn']}** · `{m['delta']:+.0%}` toward **{towards}**"
                 + (f" — {tag}" if tag else "")]
        for side in ("p1", "p2"):
            if m["story"][side]:
                lines.append(f"&nbsp;&nbsp;&nbsp;{names[side]}: "
                             f"{'; '.join(m['story'][side])}")
        st.markdown("  \n".join(lines))


def render_turn_review(state_at, game: dict, probs, names: dict, key_prefix: str) -> None:
    """Post-hoc analysis of any single turn: prediction, actions, and the advisor
    as of that turn's start (using only what was revealed by then)."""
    turn = int(st.number_input("Review turn", min_value=1, max_value=game["n_turns"],
                               value=game["n_turns"], key=f"{key_prefix}_turn"))
    p_before = probs.loc[turn]
    delta = probs.loc[turn + 1] - p_before if turn + 1 in probs.index else None
    c1, c2 = st.columns(2)
    c1.metric(f"P({names['p1']} wins) at turn {turn} start", f"{p_before:.0%}",
              f"{delta:+.0%} over this turn" if delta is not None else None)
    c1.caption(confidence_cue(p_before, turn))
    board = board_state_line(game["snapshots"][turn - 1], names, game.get("field"))
    if board:
        st.caption(f"🧭 board at turn {turn} start: {board}")
    story = turn_story(game, turn)
    played = [f"**{names[s]}**: {'; '.join(story[s])}" for s in ("p1", "p2") if story[s]]
    if story["luck"]:
        played.append(f"🎲 {'; '.join(story['luck'])}")
    c2.markdown("  \n".join(played) if played else "*no actions recorded this turn*")

    st.markdown(f"**Advisor — options available at the start of turn {turn}:**")
    render_advisor(state_at(turn), names, key_prefix=f"{key_prefix}_adv")


def winprob_figure(game: dict, probs) -> go.Figure:
    snaps = game["snapshots"]
    actions = actions_by_turn(game)
    hover = [f"{s['p1_active_species']} vs {s['p2_active_species']}"
             + (f"<br>{actions[s['turn']]}" if actions.get(s["turn"]) else "")
             for s in snaps]
    moments = key_moments(game, probs)

    fig = go.Figure()
    fig.add_hline(y=0.5, line=dict(color=BASELINE, dash="dash", width=1))
    fig.add_trace(go.Scatter(
        x=list(probs.index), y=list(probs.values), mode="lines+markers",
        line=dict(color=BLUE, width=2), marker=dict(size=5),
        name="win probability", text=hover,
        hovertemplate="turn %{x} · %{text}<br>P1 win: %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[m["turn"] for m in moments], y=[probs.loc[m["turn"]] for m in moments],
        mode="markers", name="key moments", hoverinfo="skip",
        marker=dict(size=13, symbol="circle-open", color=AQUA, line=dict(width=2.5))))

    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=430,
        margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
        font=dict(color=INK_2),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        xaxis=dict(title="turn", gridcolor=GRID, linecolor=BASELINE,
                   tickcolor=BASELINE, tickfont=dict(color=MUTED), zeroline=False),
        yaxis=dict(title=f"P({game['p1_name']} wins)", range=[-0.02, 1.02],
                   tickformat=".0%", gridcolor=GRID, linecolor=BASELINE,
                   tickfont=dict(color=MUTED), zeroline=False))
    return fig


def called_from_turn(game: dict, probs) -> int | None:
    """First turn from which the model backed the actual winner without flipping."""
    right_side = (probs >= 0.5) == (game["winner"] == "p1")
    if right_side.all():
        return int(probs.index[0])
    if not right_side.iloc[-1]:
        return None
    last_wrong = right_side[~right_side].index[-1]
    later = [t for t in probs.index if t > last_wrong]
    return int(later[0]) if later else None


WEATHER_LABEL = {"raindance": "🌧 Rain", "primordialsea": "🌧 Heavy Rain",
                 "sunnyday": "☀️ Sun", "desolateland": "☀️ Harsh Sun",
                 "sandstorm": "🌪 Sandstorm", "snow": "❄️ Snow", "snowscape": "❄️ Snow"}
STATUS_LABEL = {"brn": "🔥 burned", "par": "⚡ paralyzed", "slp": "💤 asleep",
                "frz": "🧊 frozen", "psn": "☠️ poisoned", "tox": "☠️ badly poisoned"}


def board_state_line(snap: dict, names: dict, field: dict | None = None) -> str:
    """Everything the model knows about the board right now, in one line."""
    field = field or {}

    def left(set_turn, duration=5):
        if set_turn is None:
            return ""
        remain = duration - (snap["turn"] - set_turn)
        return f" ({remain}t left)" if 0 < remain <= duration else ""

    bits = []
    if snap.get("weather"):
        bits.append(WEATHER_LABEL.get(snap["weather"], snap["weather"])
                    + left(field.get("weather_set_turn")))
    if snap.get("terrain"):
        bits.append("🌐 " + snap["terrain"].replace("terrain", "").title() + " Terrain"
                    + left(field.get("terrain_set_turn")))
    if snap.get("trickroom"):
        bits.append("🔄 Trick Room")
    for s in ("p1", "p2"):
        if snap.get(f"{s}_active_status"):
            label = STATUS_LABEL.get(snap[f"{s}_active_status"], snap[f"{s}_active_status"])
            bits.append(f"{snap[f'{s}_active_species']} is {label}")
        hz = []
        if snap.get(f"{s}_hazard_stealthrock"):
            hz.append("Rocks")
        if snap.get(f"{s}_hazard_spikes"):
            hz.append(f"Spikes×{snap[f'{s}_hazard_spikes']}")
        if snap.get(f"{s}_hazard_toxicspikes"):
            hz.append(f"T.Spikes×{snap[f'{s}_hazard_toxicspikes']}")
        if snap.get(f"{s}_hazard_stickyweb"):
            hz.append("Web")
        if hz:
            bits.append(f"hazards vs {names[s]}: {', '.join(hz)}")
        screen_turns = (field.get("screen_turns") or {}).get(s, {})
        screens = [lbl + left(screen_turns.get(n))
                   for n, lbl in (("reflect", "Reflect"), ("lightscreen", "Light Screen"),
                                  ("auroraveil", "Aurora Veil"), ("tailwind", "Tailwind"))
                   if snap.get(f"{s}_screen_{n}")]
        if screens:
            bits.append(f"{names[s]}: {', '.join(screens)} up")
    return " · ".join(bits)


def confidence_cue(prob: float, turn: int) -> str:
    """A rough, honest confidence read: the model measurably firms up as a game
    progresses (turn-1–5 accuracy ~57% → ~72% late) and as the position gets
    lopsided. Combine phase and decisiveness into a qualitative label."""
    decisiveness = abs(prob - 0.5) * 2      # 0 (coin flip) … 1 (near-certain)
    phase = min(turn / 20, 1.0)
    score = 0.5 * decisiveness + 0.5 * phase
    if score > 0.6:
        return "🟢 high confidence"
    if score > 0.35:
        return "🟡 moderate confidence"
    return "🔴 low confidence — early / close game, treat as a coin-flip-ish read"


def set_example():
    st.session_state.replay_ref = random.choice(EXAMPLES)


@st.cache_data(show_spinner=False)
def game_index() -> pd.DataFrame | None:
    """Local corpus index for the Elo-band example picker (None on cloud deploys)."""
    if not GAMES_PARQUET.exists():
        return None
    return pd.read_parquet(GAMES_PARQUET, columns=["id", "rating"])


def pick_band_example():
    idx = game_index()
    lo, hi = BAND_RANGES[st.session_state.ex_band]
    pool = idx[idx.rating.between(lo, hi - 1)]
    if len(pool):
        st.session_state.replay_ref = pool.id.sample(1).iloc[0]


@st.cache_data(ttl=300, show_spinner=False)
def cached_user_replays(username: str) -> list:
    return user_replays(username)


def _ago(uploadtime: int) -> str:
    secs = max(0, int(time.time()) - int(uploadtime))
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= n:
            return f"{secs // n}{unit} ago"
    return "just now"


def pick_history_replay(replay_id: str):
    st.session_state.replay_ref = replay_id


def render_history_browser() -> None:
    st.text_input("Showdown username", key="history_user",
                  placeholder="search your uploaded replays")
    if st.button("Find replays") and st.session_state.get("history_user", "").strip():
        st.session_state.history_query = st.session_state.history_user.strip()
    query = st.session_state.get("history_query")
    if not query:
        return
    try:
        reps = cached_user_replays(query)
    except requests.HTTPError:
        st.error("Couldn't reach the replay server — try again in a moment.")
        return
    if not reps:
        st.caption(f"No public replays found for **{query}**. Only replays that were "
                   "uploaded (“Upload and share replay” after a game) are searchable.")
        return
    st.caption(f"{len(reps)} uploaded replays for **{query}** — click one to analyze:")
    for r in reps[:30]:
        p1, p2 = (r.get("players") or ["?", "?"])[:2]
        rating = f"{r['rating']}" if r.get("rating") else "unrated"
        label = f"{p1} vs {p2}  ·  {rating}  ·  {_ago(r['uploadtime'])}"
        st.button(label, key=f"hist_{r['id']}", width="stretch",
                  on_click=pick_history_replay, args=(r["id"],))


def render_replay_analyzer() -> None:
    st.text_input("Replay URL or ID", key="replay_ref",
                  placeholder="https://replay.pokemonshowdown.com/gen9ou-...")
    if game_index() is not None:
        c1, c2 = st.columns([1, 2])
        c1.selectbox("Elo band", list(BAND_RANGES), index=3, key="ex_band",
                     label_visibility="collapsed")
        c2.button("Random example from this Elo band", on_click=pick_band_example,
                  help="Compare how chaotic low-ladder games look vs high-ladder ones")
    else:
        st.button("Try an example replay", on_click=set_example)

    with st.expander("🗂️ Battle history — find a replay by username"):
        render_history_browser()

    ref = st.session_state.get("replay_ref", "").strip()
    if not ref:
        return
    try:
        game, probs = analyze(ref)
    except requests.HTTPError:
        st.error("Couldn't fetch that replay — check the URL/ID "
                 "(private replays can't be fetched).")
        return

    if "OU" not in (game["format"] or ""):
        st.warning(f"This is a {game['format']} game; the model was trained on "
                   "[Gen 9] OU, so treat these probabilities with extra skepticism.")

    p1, p2 = game["p1_name"], game["p2_name"]
    st.subheader(f"{p1} vs {p2}")
    st.caption(f"{game['format']} · rating ~{game['rating'] or 'unrated'} · "
               f"{game['n_turns']} turns")
    st.plotly_chart(winprob_figure(game, probs), width="stretch")

    if game["winner"]:
        winner_name = game[f"{game['winner']}_name"]
        final_read = probs.iloc[-1] if game["winner"] == "p1" else 1 - probs.iloc[-1]
        called = called_from_turn(game, probs)
        c1, c2, c3 = st.columns(3)
        c1.metric("Actual winner", winner_name)
        c2.metric("Model's final read", f"{final_read:.0%}",
                  help="Win probability given to the actual winner at the start of the final turn")
        c3.metric("Called it from", f"turn {called}" if called else "missed it",
                  help="First turn from which the model backed the winner without flipping again")

    st.subheader("Key moments")
    names = {"p1": p1, "p2": p2}
    render_key_moments(game, probs, names)

    with st.expander("Teams (as revealed in this replay)"):
        t1, t2 = st.columns(2)
        t1.markdown(f"**{p1}**\n\n" + "\n".join(f"- {s}" for s in game["teams"]["p1"]))
        t2.markdown(f"**{p2}**\n\n" + "\n".join(f"- {s}" for s in game["teams"]["p2"]))

    with st.expander("🚀 Lead suggestion — who should have started?"):
        render_lead_suggestion(game, names)

    with st.expander("🔎 Turn review — prediction, actions, and the advisor at any turn"):
        render_turn_review(lambda t: parse_replay(fetch_raw(ref), up_to_turn=t),
                           game, probs, names, key_prefix="replay_rev")

    st.caption("Probabilities are the model's read at the *start* of each turn. "
               "Data: public replays from replay.pokemonshowdown.com.")


def render_team_predictor() -> None:
    st.markdown("Seen part of a team — scouting an opponent, mid-battle reveals — and "
                "wondering what's in the back? Pick the known members; the model ranks "
                "the most likely hidden teammates from ladder co-occurrence patterns.")
    model = cached_teammates()
    options = sorted(model.usage, key=model.usage.get, reverse=True)
    revealed = st.multiselect("Known team members (1–5)", options, max_selections=5)
    if not revealed:
        return
    top = model.predict(revealed, top=10)
    st.dataframe(
        top.rename(columns={"species": "Likely teammate",
                            "relative_likelihood": "Relative likelihood"}),
        column_config={"Relative likelihood": st.column_config.ProgressColumn(
            format="percent", min_value=0, max_value=float(top.relative_likelihood.max()))},
        hide_index=True, width="stretch")
    st.caption("Held-out evaluation: given 3 known members, the top-ranked guess is one "
               "of the 3 hidden teammates 41% of the time (usage-only baseline: 18%), "
               "and 58% of the hidden team appears in the top 10.")

    st.divider()
    st.markdown("#### Predicted sets")
    st.caption("The likely moves, item, ability, Tera, and EV/nature spread for any "
               "Pokémon — what you'd expect before it reveals anything.")
    known_plus_top = revealed + [t for t in top.species if t not in revealed][:3]
    for species in known_plus_top:
        st.markdown(predicted_set_md(species))


def _start_live(ref: str) -> None:
    """Resolve a battle link/room id/username and start spectating it."""
    room = None
    if "battle-" in ref:
        room = ref
    elif ref:
        with st.spinner(f"looking up {ref}'s current battle…"):
            room = find_user_battle(ref)
        if room is None:
            st.error(f"Couldn't find a public battle for '{ref}'. Either they aren't "
                     "in a battle right now, or the battle is hidden/private — hidden "
                     "battles can't be found by username. If it's your own private "
                     "game, paste the full battle link from your browser instead "
                     "(including the secret suffix after the battle number).")
            return
    if not room:
        return
    if "live" in st.session_state:
        st.session_state.live.close()
    try:
        st.session_state.live = LiveBattle(room)
    except ValueError as exc:
        st.error(str(exc))


@st.fragment(run_every="3s")
def live_panel() -> None:
    live = st.session_state.get("live")
    if live is None:
        return
    game = live.snapshot_game()
    icon = {"live": "🔴", "ended": "🏁", "connecting": "⏳",
            "reconnecting": "🔄", "disconnected": "⚠️"}.get(live.status, "⚠️")
    p1, p2 = game["p1_name"] or "p1", game["p2_name"] or "p2"
    st.caption(f"{icon} {live.status} · `{live.room}` · {p1} vs {p2}")
    if live.status == "error":
        st.error(live.error or "connection error")
        return
    if live.status == "reconnecting":
        st.warning("Connection dropped — reconnecting…")
    if live.status == "disconnected":
        st.warning("Disconnected after repeated retries. Press Stop and try again.")
    if game["n_turns"] < 1:
        if game["teams"]["p1"] and game["teams"]["p2"]:
            st.info("Team preview — no turns yet. Recommended leads:")
            render_lead_suggestion(game, {"p1": p1, "p2": p2})
        else:
            st.info("Connected — waiting for team preview…")
        return
    if "OU" not in (game["format"] or "OU"):
        st.warning(f"This looks like {game['format']}, not Gen 9 OU — the model was "
                   "trained on OU, so predictions here are unreliable.")
    booster, meta = cached_model()
    probs = predict_game(game, booster, meta)
    c1, c2 = st.columns(2)
    delta = f"{probs.iloc[-1] - probs.iloc[-2]:+.0%} last turn" if len(probs) > 1 else None
    c1.metric(f"P({p1} wins) right now", f"{probs.iloc[-1]:.0%}", delta)
    c2.metric("Turn", game["n_turns"])
    st.caption(confidence_cue(probs.iloc[-1], game["n_turns"]))
    board = board_state_line(game["snapshots"][-1], {"p1": p1, "p2": p2},
                             game.get("field"))
    if board:
        st.caption(f"🧭 {board}")
    st.plotly_chart(winprob_figure(game, probs), width="stretch")
    if live.status == "ended" and game["winner"]:
        st.success(f"Battle over — {game[game['winner'] + '_name']} won. "
                   "Use the panels below to review what decided it.")
    names = {"p1": p1, "p2": p2}
    fainted_active = [s for s in ("p1", "p2") if any(
        m["active"] and m["fainted"] for m in game["roster"][s])]
    forced = bool(fainted_active) and live.status == "live"
    with st.expander("🧠 Advisor — right now (live state)", expanded=forced):
        if forced:
            st.warning(f"💀 {', '.join(names[s] for s in fainted_active)} must pick a "
                       "replacement — the ranked switch-ins are below.")
        render_advisor(game, names, key_prefix="live_now")
    with st.expander("📌 Key moments so far", expanded=live.status == "ended"):
        render_key_moments(game, probs, names)
    with st.expander("🔎 Turn review — scrub back through the battle",
                     expanded=live.status == "ended"):
        render_turn_review(lambda t: parse_replay({"log": live.raw_log()}, up_to_turn=t),
                           game, probs, names, key_prefix="live_rev")


def render_live_spectator() -> None:
    st.markdown("Watch any **public ladder battle** with a live-updating win-probability "
                "chart. Paste a battle link, or a username to find their current game — "
                "or just grab the highest-rated battle happening right now.")
    ref = st.text_input("Battle link / room id / username", key="live_ref",
                        placeholder="https://play.pokemonshowdown.com/battle-gen9ou-…  or  someusername")
    c1, c2, c3, c4 = st.columns([1, 1.4, 1, 1])
    if c1.button("Watch", type="primary") and ref.strip():
        _start_live(ref.strip())
    min_elo = c2.selectbox("min Elo", ["any", "1300+", "1500+", "1700+"], index=2,
                           key="live_elo", label_visibility="collapsed")
    if c3.button("Watch random live battle"):
        with st.spinner("fetching the live battle list…"):
            battles = list_battles()
        floor = 0 if min_elo == "any" else int(min_elo.rstrip("+"))
        pool = [b for b in battles if b["min_elo"] >= floor]
        if pool:
            _start_live(random.choice(pool[:8])["room"])
        else:
            st.error(f"No live battle at {min_elo} right now — try a lower floor.")
    if "live" in st.session_state and c4.button("Stop watching"):
        st.session_state.live.close()
        del st.session_state["live"]
        st.rerun()
    live_panel()


st.title("Pokémon Showdown — Win Probability")
st.caption(
    "Turn-by-turn win probability for ranked Gen 9 OU battles. LightGBM trained on "
    "~14k games rated 1300+; evaluated on strictly newer games (log loss 0.609, AUC 0.72).")

tab_replay, tab_live, tab_team = st.tabs(
    ["📈 Replay analyzer", "🔴 Live spectator", "🔮 Team predictor"])
with tab_replay:
    render_replay_analyzer()
with tab_live:
    render_live_spectator()
with tab_team:
    render_team_predictor()
