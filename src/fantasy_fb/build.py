"""Build the team-game table: one row per team per game."""
import pandas as pd

from . import ingest
from .paths import PROCESSED
from .scoring import POSITIONS, fantasy_points

# One ID per franchise across relocations
FRANCHISE = {"STL": "LA", "LAR": "LA", "SD": "LAC", "OAK": "LV"}


def franchise(s: pd.Series) -> pd.Series:
    return s.replace(FRANCHISE)


def build_team_games(player_stats=None, schedules=None, include_future=False) -> pd.DataFrame:
    ps = ingest.load_player_stats() if player_stats is None else player_stats
    sc = ingest.load_schedules() if schedules is None else schedules

    ps = ps[(ps["season_type"] == "REG") & ps["position"].isin(POSITIONS)].copy()
    ps["team"] = franchise(ps["team"])
    for fmt in ("std", "half", "ppr"):
        ps[f"fp_{fmt}"] = fantasy_points(ps, fmt)

    # Points scored per team per game, in total and by position
    keys = ["season", "week", "game_id", "team"]
    total = ps.groupby(keys)[["fp_std", "fp_half", "fp_ppr"]].sum()
    by_pos = ps.pivot_table(index=keys, columns="position", values="fp_half", aggfunc="sum", fill_value=0)
    by_pos = by_pos.reindex(columns=POSITIONS, fill_value=0).add_prefix("fp_for_")
    fp = total.join(by_pos).reset_index()

    # Schedule: expand to home/away rows, keep played regular-season games
    sc = sc[sc["game_type"] == "REG"]
    if not include_future:
        sc = sc[sc["home_score"].notna()]
    sc = sc.copy()
    sc["home_team"] = franchise(sc["home_team"])
    sc["away_team"] = franchise(sc["away_team"])
    home = pd.DataFrame({
        "game_id": sc["game_id"], "season": sc["season"], "week": sc["week"], "gameday": sc["gameday"],
        "team": sc["home_team"], "opponent": sc["away_team"], "is_home": 1,
        "points_for": sc["home_score"], "points_against": sc["away_score"],
        "rest": sc["home_rest"], "opp_rest": sc["away_rest"], "div_game": sc["div_game"],
        "spread": sc["spread_line"], "total_line": sc["total_line"],
    })
    away = pd.DataFrame({
        "game_id": sc["game_id"], "season": sc["season"], "week": sc["week"], "gameday": sc["gameday"],
        "team": sc["away_team"], "opponent": sc["home_team"], "is_home": 0,
        "points_for": sc["away_score"], "points_against": sc["home_score"],
        "rest": sc["away_rest"], "opp_rest": sc["home_rest"], "div_game": sc["div_game"],
        "spread": -sc["spread_line"], "total_line": sc["total_line"],  # spread from this team's view: + means favored
    })
    games = pd.concat([home, away], ignore_index=True)
    games["win"] = (games["points_for"] > games["points_against"]).astype(float)
    games.loc[games["points_for"] == games["points_against"], "win"] = 0.5
    games.loc[games["points_for"].isna(), "win"] = float("nan")

    tg = games.merge(fp.drop(columns=["season", "week"]), on=["game_id", "team"], how="left")

    # Fantasy points allowed = opponent's fantasy points scored in the same game
    opp = tg[["game_id", "team", "fp_std", "fp_half", "fp_ppr"] + [f"fp_for_{p}" for p in POSITIONS]].copy()
    opp.columns = ["game_id", "opponent"] + [
        c.replace("fp_for_", "fp_against_").replace("fp_", "fp_against_", 1) if c.startswith("fp_for_") is False
        else c.replace("fp_for_", "fp_against_")
        for c in opp.columns[2:]
    ]
    tg = tg.merge(opp, on=["game_id", "opponent"], how="left")
    for fmt in ("std", "half", "ppr"):
        tg[f"fp_diff_{fmt}"] = tg[f"fp_{fmt}"] - tg[f"fp_against_{fmt}"]
    return tg.sort_values(["season", "week", "game_id", "is_home"]).reset_index(drop=True)


if __name__ == "__main__":
    tg = build_team_games()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tg.to_parquet(PROCESSED / "team_games.parquet")
    print(tg.shape)
    print(tg.head(4).T)
