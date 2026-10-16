"""Pre-game features. Every feature for a game uses only earlier games (shift(1))."""
import numpy as np
import pandas as pd

from .build import build_team_games

STATS = ["fp_half", "fp_against_half", "fp_diff_half", "points_for", "points_against"]
PRIOR_WEIGHT = 4  # games' worth of weight given to last season's average early in a season


def _team_features(tg: pd.DataFrame, fmt: str = "half") -> pd.DataFrame:
    tg = tg.sort_values(["team", "season", "week"]).copy()
    d = f"fp_diff_{fmt}"
    tg["pd"] = tg["points_for"] - tg["points_against"]
    g = tg.groupby(["team", "season"])

    # Season-to-date means over earlier games only
    tg["n_prior"] = g.cumcount()
    for col in [d, "pd", "win"]:
        tg[f"{col}_sea"] = g[col].transform(lambda s: s.shift(1).expanding().mean())
        tg[f"{col}_l4"] = g[col].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())

    # Last season's full-year average (known before the season starts)
    last = tg.groupby(["team", "season"])[[d, "pd", "win"]].mean()
    last.index = pd.MultiIndex.from_arrays([last.index.get_level_values(0), last.index.get_level_values(1) + 1])
    last.columns = [f"{c}_prev" for c in last.columns]
    tg = tg.merge(last.reset_index(), on=["team", "season"], how="left")

    # Blend: early in a season lean on last year (shrunk toward 0), fade it out as games accumulate
    for col in [d, "pd"]:
        prev = tg[f"{col}_prev"].fillna(0) * 0.6  # regress last year toward league average
        n = tg["n_prior"]
        sea = tg[f"{col}_sea"].fillna(0)
        tg[f"{col}_blend"] = (sea * n + prev * PRIOR_WEIGHT) / (n + PRIOR_WEIGHT)
    return tg.rename(columns={f"{d}_{k}": f"fp_diff_{k}" for k in ("sea", "l4", "prev", "blend")})


FEATURES = ["fp_diff_blend", "fp_diff_l4", "pd_blend", "win_sea"]


def build_game_features(fmt: str = "half", include_future: bool = False, tg: pd.DataFrame = None) -> pd.DataFrame:
    """One row per game: home-minus-away features plus outcome."""
    tg = _team_features(build_team_games(include_future=include_future) if tg is None else tg, fmt)
    home = tg[tg["is_home"] == 1]
    away = tg[tg["is_home"] == 0]
    m = home[["game_id", "season", "week", "team", "opponent", "win", "points_for", "points_against",
              "spread", "div_game", "rest", "opp_rest", "n_prior"] + FEATURES].merge(
        away[["game_id"] + FEATURES], on="game_id", suffixes=("", "_away"))
    for f in FEATURES:
        m[f"d_{f}"] = m[f].fillna(0) - m[f"{f}_away"].fillna(0)
    m["rest_diff"] = (m["rest"] - m["opp_rest"]).clip(-7, 7)
    m["margin"] = m["points_for"] - m["points_against"]
    m = m.rename(columns={"team": "home", "opponent": "away", "win": "home_win"})
    return m.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


if __name__ == "__main__":
    gf = build_game_features()
    print(gf.shape)
    print(gf[["season", "week", "home", "away", "home_win"] + [f"d_{f}" for f in FEATURES]].tail(5))
