"""Streamlit demo: win-probability timelines for replays + teammate inference.

Run with:
    streamlit run app.py
"""

import random

import plotly.graph_objects as go
import requests
import streamlit as st

from src.advisor import advise
from src.live import LiveBattle, find_user_battle, list_battles
from src.parser import parse_replay
from src.predict import (actions_by_turn, fetch_replay, key_moments, load_model,
                         predict_game, snapshot_features, turn_story)
from src.teammates import TeammateModel

# chart chrome + series colors from the validated reference palette (dataviz skill)
SURFACE, GRID, BASELINE, INK_2, MUTED = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#52514e", "#898781"
BLUE, AQUA = "#2a78d6", "#1baf7a"

EXAMPLES = ["gen9ou-2645435074", "gen9ou-2645436173", "gen9ou-2645435369"]

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
    """Ranked switches (model-scored) and revealed moves (damage heuristic)."""
    side_name = st.radio("Options for", [names["p1"], names["p2"]],
                         horizontal=True, key=f"{key_prefix}_side")
    side = "p1" if side_name == names["p1"] else "p2"
    booster, meta = cached_model()
    out = advise(game, side, booster, meta, snapshot_features)

    active = next((m["species"] for m in game["roster"][side] if m["active"]), "?")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Switch options** — scored by the win-prob model")
    if len(out["switches"]):
        sw = out["switches"].assign(
            win_prob=lambda d: d.win_prob.map("{:.0%}".format),
            hazard_chip=lambda d: d.hazard_chip.map("{:.0%}".format))
        c1.dataframe(sw, hide_index=True, width="stretch")
    else:
        c1.caption("no healthy bench Pokémon to switch to")
    c2.markdown(f"**{active}'s revealed moves** — damage heuristic")
    if len(out["moves"]):
        c2.dataframe(out["moves"], hide_index=True, width="stretch")
    else:
        c2.caption("no moves revealed yet for the active Pokémon")
    st.caption("⚠️ v1 heuristic: uses only information revealed in this battle; doesn't "
               "model the opponent's simultaneous choice, hidden items, or abilities. "
               "Switch scores come from the win-probability model on the post-switch "
               "state; move scores are power × STAB × type chart × stat ratio.")


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


def set_example():
    st.session_state.replay_ref = random.choice(EXAMPLES)


def render_replay_analyzer() -> None:
    st.text_input("Replay URL or ID", key="replay_ref",
                  placeholder="https://replay.pokemonshowdown.com/gen9ou-...")
    st.button("Try an example replay", on_click=set_example)

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


def _start_live(ref: str) -> None:
    """Resolve a battle link/room id/username and start spectating it."""
    room = None
    if "battle-" in ref:
        room = ref
    elif ref:
        with st.spinner(f"looking up {ref}'s current battle…"):
            room = find_user_battle(ref)
        if room is None:
            st.error(f"Couldn't find a public battle for '{ref}' — are they "
                     "currently playing, with battles visible?")
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
    icon = {"live": "🔴", "ended": "🏁", "connecting": "⏳"}.get(live.status, "⚠️")
    p1, p2 = game["p1_name"] or "p1", game["p2_name"] or "p2"
    st.caption(f"{icon} {live.status} · `{live.room}` · {p1} vs {p2}")
    if live.status == "error":
        st.error(live.error)
        return
    if game["n_turns"] < 1:
        st.info("Connected — waiting for the first turn…")
        return
    booster, meta = cached_model()
    probs = predict_game(game, booster, meta)
    c1, c2 = st.columns(2)
    delta = f"{probs.iloc[-1] - probs.iloc[-2]:+.0%} last turn" if len(probs) > 1 else None
    c1.metric(f"P({p1} wins) right now", f"{probs.iloc[-1]:.0%}", delta)
    c2.metric("Turn", game["n_turns"])
    st.plotly_chart(winprob_figure(game, probs), width="stretch")
    if live.status == "ended" and game["winner"]:
        st.success(f"Battle over — {game[game['winner'] + '_name']} won. "
                   "Use the panels below to review what decided it.")
    names = {"p1": p1, "p2": p2}
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
    c1, c2, c3 = st.columns(3)
    if c1.button("Watch", type="primary") and ref.strip():
        _start_live(ref.strip())
    if c2.button("Top rated battle now"):
        with st.spinner("fetching the live battle list…"):
            battles = list_battles()
        if battles:
            _start_live(battles[0]["room"])
        else:
            st.error("Couldn't fetch the live battle list — try again in a moment.")
    if "live" in st.session_state and c3.button("Stop watching"):
        st.session_state.live.close()
        del st.session_state["live"]
        st.rerun()
    live_panel()


st.title("Pokémon Showdown — Win Probability")
st.caption(
    "Turn-by-turn win probability for ranked Gen 9 OU battles. LightGBM trained on "
    "~14k games rated 1300+; evaluated on strictly newer games (log loss 0.608, AUC 0.72).")

tab_replay, tab_live, tab_team = st.tabs(
    ["📈 Replay analyzer", "🔴 Live spectator", "🔮 Team predictor"])
with tab_replay:
    render_replay_analyzer()
with tab_live:
    render_live_spectator()
with tab_team:
    render_team_predictor()
