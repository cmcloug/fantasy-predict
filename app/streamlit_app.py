import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from fantasy_fb import compare, evaluate, predict
from fantasy_fb.build import build_team_games
from fantasy_fb.features_v2 import build_v2
from fantasy_fb.paths import PROCESSED
from fantasy_fb.season_view import position_r2, r2, team_seasons

st.set_page_config(page_title="Fantasy points vs. wins", layout="wide")


@st.cache_data
def load_tg():
    return build_team_games()


@st.cache_data
def load_backtest():
    cached = PROCESSED / "backtest_v2_preds.parquet"
    return pd.read_parquet(cached) if cached.exists() else compare.backtest(build_v2())


tg = load_tg()
st.title("Fantasy points vs. NFL wins")
explore, positions, luck, model_tab, picks_tab = st.tabs(
    ["Explore", "Positions", "Lucky / unlucky", "Model vs. baselines", "This week's picks"])

with explore:
    c1, c2 = st.columns([1, 3])
    fmt = c1.radio("Scoring", ["std", "half", "ppr"], index=1, format_func={"std": "Standard", "half": "Half-PPR", "ppr": "PPR"}.get)
    seasons = c1.slider("Seasons", int(tg.season.min()), int(tg.season.max()) - 1, (2010, int(tg.season.max()) - 1))
    team_filter = c1.multiselect("Highlight teams", sorted(tg.team.unique()))
    done = tg[(tg.season >= seasons[0]) & (tg.season <= seasons[1])]
    ts = team_seasons(done, fmt)
    ts["highlight"] = ts.team.isin(team_filter) if team_filter else True
    with c2:
        metric_cols = st.columns(2)
        for col, (x, label) in zip(metric_cols, [("fp_for", "Fantasy points scored / game"), ("fp_diff", "Fantasy point DIFFERENTIAL / game")]):
            fig = px.scatter(ts, x=x, y="win_pct", color="highlight", hover_data=["team", "season", "wins"],
                             trendline="ols", trendline_scope="overall", labels={x: label, "win_pct": "Win %"},
                             title=f"{label}  (R² = {r2(ts[x], ts['win_pct']):.2f})")
            fig.update_layout(showlegend=False)
            col.plotly_chart(fig, use_container_width=True)
        st.caption("Scoring only offense explains about a third of the variation in wins; adding what the defense "
                   "allowed (the differential) explains much more. Neither is a prediction yet — see 'Model vs. baselines'.")

with positions:
    pr = position_r2(done).melt("position", var_name="side", value_name="R²")
    st.plotly_chart(px.bar(pr, x="position", y="R²", color="side", barmode="group",
                           title="How closely each position group's points track win % (team-seasons)"), use_container_width=True)

with luck:
    ts_all = team_seasons(tg[tg.season < tg.season.max()], "half")
    season_pick = st.selectbox("Season", sorted(ts_all.season.unique(), reverse=True))
    s = ts_all[ts_all.season == season_pick].sort_values("luck_wins")
    s["color"] = s.luck_wins.apply(lambda v: "Overperformed" if v > 0 else "Underperformed")
    st.plotly_chart(px.bar(s, x="luck_wins", y="team", color="color", orientation="h", height=800,
                           title="Actual wins minus wins expected from fantasy differential"), use_container_width=True)

with model_tab:
    preds = load_backtest()
    hist = preds[preds.season <= 2025]
    table = evaluate.summarize(hist)
    st.subheader("Walk-forward backtest, 2018-2025 (each season predicted from earlier seasons only)")
    st.dataframe(table.style.format({"accuracy": "{:.1%}", "brier": "{:.4f}", "log_loss": "{:.4f}"}), use_container_width=True)
    st.caption("Lower Brier / log loss is better. Vegas is the benchmark to approach, not a baseline to beat.")
    model_name = st.selectbox("Calibration for model", table.model.tolist(), index=0)
    d = hist[hist.model == model_name].copy()
    d["bin"] = pd.cut(d.p_home, [0, .3, .4, .5, .6, .7, .8, 1])
    cal = d.groupby("bin", observed=True).agg(predicted=("p_home", "mean"), actual=("home_win", "mean"), n=("home_win", "size")).reset_index()
    fig = px.scatter(cal, x="predicted", y="actual", size="n", range_x=[0, 1], range_y=[0, 1], title="Calibration (closer to the diagonal is better)")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

with picks_tab:
    st.caption("Model: logistic regression on fantasy differential, point differential, EPA and starting-QB quality (no Vegas input).")
    p = predict.weekly_picks()
    st.subheader(f"Season {int(p.season[0])}, week {int(p.week[0])}")
    st.dataframe(p[["home", "away", "pick", "confidence", "spread", "vegas_pick"]]
                 .style.format({"confidence": "{:.0%}", "spread": "{:+.1f}"}), use_container_width=True)
    card = predict.scorecard()
    if len(card):
        st.subheader("Scorecard (picks logged before kickoff)")
        st.metric("Model", f"{card.correct.mean():.0%}", f"{int(card.correct.sum())}/{len(card)}")
        st.metric("Vegas favorite", f"{card.vegas_correct.mean():.0%}")
