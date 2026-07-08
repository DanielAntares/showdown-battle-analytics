"""Streamlit demo: win-probability timelines for replays + teammate inference.

Run with:
    streamlit run app.py
"""

import random

import plotly.graph_objects as go
import requests
import streamlit as st

from src.parser import parse_replay
from src.predict import fetch_replay, key_moments, load_model, predict_game
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
def analyze(replay_ref: str):
    game = parse_replay(fetch_replay(replay_ref))
    booster, meta = cached_model()
    probs = predict_game(game, booster, meta)
    return game, probs


def winprob_figure(game: dict, probs) -> go.Figure:
    snaps = game["snapshots"]
    hover = [f"{s['p1_active_species']} vs {s['p2_active_species']}" for s in snaps]
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
    for m in key_moments(game, probs):
        towards = p1 if m["delta"] > 0 else p2
        st.markdown(f"**Turn {m['turn']}** · `{m['delta']:+.0%}` toward **{towards}** — "
                    f"{'; '.join(m['events'])}")

    with st.expander("Teams (as revealed in this replay)"):
        t1, t2 = st.columns(2)
        t1.markdown(f"**{p1}**\n\n" + "\n".join(f"- {s}" for s in game["teams"]["p1"]))
        t2.markdown(f"**{p2}**\n\n" + "\n".join(f"- {s}" for s in game["teams"]["p2"]))

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
               "of the 3 hidden teammates 35% of the time (usage-only baseline: 19%), "
               "and half the hidden team appears in the top 10.")


st.title("Pokémon Showdown — Win Probability")
st.caption(
    "Turn-by-turn win probability for ranked Gen 9 OU battles. LightGBM trained on "
    "~15k rated ladder games; evaluated on strictly newer games (log loss 0.601, AUC 0.73).")

tab_replay, tab_team = st.tabs(["📈 Replay analyzer", "🔮 Team predictor"])
with tab_replay:
    render_replay_analyzer()
with tab_team:
    render_team_predictor()
