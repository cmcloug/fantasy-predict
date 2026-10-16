"""Team strength ratings computed only from games played before each game."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def ridge_ratings(tg: pd.DataFrame, target: str, alpha: float = 10.0, prev_weight: float = 0.5) -> pd.Series:
    """Opponent-adjusted rating per team-game row (aligned to tg.index).

    For each (season, week) fit: target ~ offense[team] + defense[opponent] + home, using last season
    (down-weighted) and this season's earlier weeks. Rating = offense strength + defense strength.
    """
    teams = sorted(tg.team.unique()); n = len(teams); ix = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(tg), 2 * n + 1))
    r = np.arange(len(tg))
    X[r, tg.team.map(ix).values] = 1
    X[r, n + tg.opponent.map(ix).values] = 1
    X[r, 2 * n] = tg.is_home.values
    y = tg[target].values
    season, week = tg.season.values, tg.week.values
    out = np.zeros(len(tg))
    for (s, w), grp in tg.groupby(["season", "week"]).indices.items():
        cur = (season == s) & (week < w)
        prev = season == s - 1
        mask = (cur | prev) & ~np.isnan(y)
        if mask.sum() < 40:
            continue
        wts = np.where(cur[mask], 1.0, prev_weight)
        model = Ridge(alpha=alpha).fit(X[mask], y[mask] - np.average(y[mask], weights=wts), sample_weight=wts)
        rating = model.coef_[:n] - model.coef_[n:2 * n]
        out[grp] = rating[tg.team.map(ix).values[grp]]
    return pd.Series(out, index=tg.index)


def elo_ratings(tg: pd.DataFrame, actual: str, k: float = 0.1, hfa: float = 1.0, regress: float = 1 / 3) -> pd.DataFrame:
    """Margin-based Elo in the units of `actual` (home-minus-away diff). Returns pre-game home/away ratings."""
    g = tg[tg.is_home == 1].sort_values(["season", "week", "game_id"])
    diff = (g[actual]).values
    rating, rows, last_season = {}, [], None
    for (s, w), grp in g.groupby(["season", "week"], sort=True):
        if s != last_season and last_season is not None:
            rating = {t: v * (1 - regress) for t, v in rating.items()}
        last_season = s
        pending = []
        for gid, h, a, d in zip(grp.game_id, grp.team, grp.opponent, grp[actual]):
            rh, ra = rating.get(h, 0.0), rating.get(a, 0.0)
            rows.append((gid, rh, ra))
            if not np.isnan(d):
                err = d - (rh - ra + hfa)
                pending.append((h, a, k * err))
        for h, a, upd in pending:
            rating[h] = rating.get(h, 0.0) + upd
            rating[a] = rating.get(a, 0.0) - upd
    return pd.DataFrame(rows, columns=["game_id", f"elo_home_{actual}", f"elo_away_{actual}"])


def qb_ratings(tg: pd.DataFrame, schedules: pd.DataFrame, qb_games: pd.DataFrame, shrink: float = 300.0) -> pd.DataFrame:
    """Pre-game shrunk EPA/dropback of the starting QB, aligned to tg.index.

    The starter is the QB listed for the game (known at kickoff); if missing (future games),
    the team's most recent starter is used.
    """
    sc = schedules.set_index("game_id")
    d = tg[["game_id", "team", "season", "week", "is_home"]].copy()
    d["orig"] = d.index
    d["qb_id"] = np.where(d.is_home == 1, d.game_id.map(sc["home_qb_id"]), d.game_id.map(sc["away_qb_id"]))
    d = d.sort_values(["team", "season", "week"])
    d["qb_id"] = d.groupby("team")["qb_id"].ffill().fillna("unknown")
    d["t_prev"] = d.season * 100 + d.week - 1  # as of the end of the previous week

    q = qb_games.merge(tg[["game_id", "season", "week"]].drop_duplicates("game_id"), on="game_id")
    q["t"] = q.season * 100 + q.week
    q = q.groupby(["qb_id", "t"], as_index=False)[["dropbacks", "epa_sum"]].sum().sort_values("t")
    q["tot_db"] = q.groupby("qb_id")["dropbacks"].cumsum()
    q["tot_epa"] = q.groupby("qb_id")["epa_sum"].cumsum()

    m = pd.merge_asof(d.sort_values("t_prev"), q[["qb_id", "t", "tot_db", "tot_epa"]],
                      left_on="t_prev", right_on="t", by="qb_id", direction="backward")
    m["qb_rating"] = m["tot_epa"].fillna(0) / (m["tot_db"].fillna(0) + shrink)
    m["qb_db_prior"] = m["tot_db"].fillna(0)
    return m.set_index("orig")[["qb_rating", "qb_db_prior"]].reindex(tg.index)
