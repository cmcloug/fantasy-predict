"""Team-season aggregates used by the explore charts."""
import numpy as np
import pandas as pd


def team_seasons(tg: pd.DataFrame, fmt: str = "half") -> pd.DataFrame:
    g = tg.groupby(["season", "team"])
    ts = g.agg(games=("win", "size"), wins=("win", "sum"),
               fp_for=(f"fp_{fmt}", "mean"), fp_against=(f"fp_against_{fmt}", "mean"),
               fp_diff=(f"fp_diff_{fmt}", "mean"), pts_diff=("points_for", "mean"),
               pts_against=("points_against", "mean")).reset_index()
    ts["win_pct"] = ts["wins"] / ts["games"]
    ts["pts_diff"] = ts["pts_diff"] - ts["pts_against"]
    # Wins "expected" from fantasy differential (simple linear fit), and luck = actual - expected
    slope, icpt = np.polyfit(ts["fp_diff"], ts["win_pct"], 1)
    ts["exp_win_pct"] = (icpt + slope * ts["fp_diff"]).clip(0, 1)
    ts["luck_wins"] = (ts["win_pct"] - ts["exp_win_pct"]) * ts["games"]
    return ts


def r2(x, y) -> float:
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def position_r2(tg: pd.DataFrame) -> pd.DataFrame:
    g = tg.groupby(["season", "team"]).mean(numeric_only=True).reset_index()
    rows = []
    for pos in ["QB", "RB", "WR", "TE"]:
        rows.append({"position": pos, "scored": r2(g[f"fp_for_{pos}"], g["win"]),
                     "allowed": r2(g[f"fp_against_{pos}"], g["win"])})
    return pd.DataFrame(rows)
