"""Predict the next unplayed week and log picks (with a timestamp) before kickoff."""
import datetime as dt

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import ingest
from .features_v2 import build_v2
from .paths import PROCESSED

PICKS_LOG = PROCESSED / "picks_log.csv"
MODEL_NAME = "v2: fantasy + point diff + EPA + starting QB"
COLS = ["d_fp_diff_blend", "d_pd_blend", "d_epa_blend", "d_qb"]  # independent of Vegas; spread shown alongside


def weekly_picks(season=None, week=None) -> pd.DataFrame:
    gf = build_v2(include_future=True)
    played = gf[gf["home_win"].notna() & (gf["home_win"] != 0.5)]
    todo = gf[gf["home_win"].isna()]
    if season is None:
        nxt = todo.sort_values(["season", "week"]).iloc[0]
        season, week = nxt["season"], nxt["week"]
    games = todo[(todo["season"] == season) & (todo["week"] == week)].copy()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)).fit(played[COLS].fillna(0), played["home_win"].round())
    games["p_home"] = model.predict_proba(games[COLS].fillna(0))[:, 1]
    games["pick"] = games.apply(lambda r: r["home"] if r["p_home"] >= 0.5 else r["away"], axis=1)
    games["confidence"] = games["p_home"].where(games["p_home"] >= 0.5, 1 - games["p_home"])
    games["vegas_pick"] = games.apply(lambda r: r["home"] if r["spread"] > 0 else (r["away"] if r["spread"] < 0 else ""), axis=1)
    return games[["season", "week", "game_id", "home", "away", "pick", "confidence", "p_home", "spread", "vegas_pick"]].reset_index(drop=True)


def log_picks(picks: pd.DataFrame) -> int:
    """Append picks once per game_id; never overwrite an earlier (pre-kickoff) pick."""
    picks = picks.assign(logged_at=dt.datetime.now().isoformat(timespec="seconds"), model=MODEL_NAME)
    if PICKS_LOG.exists():
        old = pd.read_csv(PICKS_LOG)
        picks = picks[~picks["game_id"].isin(old["game_id"])]
        picks = pd.concat([old, picks], ignore_index=True) if len(picks) else old
    PROCESSED.mkdir(parents=True, exist_ok=True)
    picks.to_csv(PICKS_LOG, index=False)
    return len(picks)


def scorecard() -> pd.DataFrame:
    """Grade logged picks against final scores."""
    if not PICKS_LOG.exists():
        return pd.DataFrame()
    log = pd.read_csv(PICKS_LOG)
    sc = ingest.load_schedules()[["game_id", "home_team", "home_score", "away_score"]]
    m = log.merge(sc, on="game_id", how="left")
    m = m[m["home_score"].notna()].copy()
    m["winner"] = m.apply(lambda r: r["home"] if r["home_score"] > r["away_score"] else r["away"], axis=1)
    m["correct"] = m["pick"] == m["winner"]
    m["vegas_correct"] = m["vegas_pick"] == m["winner"]
    return m


if __name__ == "__main__":
    ingest.load_player_stats(refresh=True); ingest.load_schedules(refresh=True)
    p = weekly_picks()
    n = log_picks(p)
    print(f"Season {int(p.season[0])} week {int(p.week[0])} picks (log now has {n} rows)")
    print(p.round(3).to_string(index=False))
