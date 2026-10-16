"""Team fantasy points rebuilt from play-by-play, with and without garbage time."""
import nflreadpy as nfl
import pandas as pd

from . import ingest
from .build import franchise
from .paths import PROCESSED

COLS = ["season", "week", "game_id", "season_type", "posteam", "defteam", "play_type", "wp", "passing_yards",
        "rushing_yards", "receiving_yards", "pass_touchdown", "rush_touchdown", "interception",
        "fumble_lost", "complete_pass", "epa", "qb_kneel", "qb_spike"]
GARBAGE = (0.10, 0.90)  # offense win probability outside this band = garbage time


def _fp(d: pd.DataFrame, rec=0.5) -> pd.Series:
    # Team-level fantasy points: passing yards count for the passer (0.04) and the receiver (0.1);
    # pass TDs count 4 + 6; rushing 0.1/yd + 6/TD
    z = lambda c: d[c].fillna(0)
    return (0.04 * z("passing_yards") + 0.1 * z("receiving_yards") + 0.1 * z("rushing_yards")
            + 10 * z("pass_touchdown") + 6 * z("rush_touchdown") - 2 * z("interception")
            - 2 * z("fumble_lost") + rec * z("complete_pass"))


def build_pbp_team_games() -> pd.DataFrame:
    seasons = list(range(ingest.FIRST_SEASON, ingest.current_season() + 1))
    p = nfl.load_pbp(seasons=seasons).select(COLS).to_pandas()
    p = p[(p.season_type == "REG") & p.posteam.notna() & p.play_type.isin(["pass", "run"]) & ~(p.qb_kneel == 1) & ~(p.qb_spike == 1)]
    p["fp"] = _fp(p)
    p["clean"] = p.wp.between(*GARBAGE)
    p["posteam"], p["defteam"] = franchise(p.posteam), franchise(p.defteam)
    g = p.groupby(["game_id", "posteam"])
    out = pd.DataFrame({
        "fp_pbp": g["fp"].sum(),
        "fp_pbp_clean": p[p.clean].groupby(["game_id", "posteam"])["fp"].sum(),
        "epa_total": g["epa"].sum(),
        "epa_clean": p[p.clean].groupby(["game_id", "posteam"])["epa"].sum(),
    }).fillna(0).reset_index().rename(columns={"posteam": "team"})
    return out


def build_qb_games() -> pd.DataFrame:
    """Per-QB per-game dropbacks and total EPA (passes, sacks, scrambles)."""
    seasons = list(range(ingest.FIRST_SEASON, ingest.current_season() + 1))
    p = nfl.load_pbp(seasons=seasons).select(["game_id", "season_type", "qb_dropback", "passer_player_id", "epa"]).to_pandas()
    p = p[(p.season_type == "REG") & (p.qb_dropback == 1) & p.passer_player_id.notna() & p.epa.notna()]
    return (p.groupby(["game_id", "passer_player_id"]).agg(dropbacks=("epa", "size"), epa_sum=("epa", "sum"))
            .reset_index().rename(columns={"passer_player_id": "qb_id"}))


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = build_pbp_team_games()
    df.to_parquet(PROCESSED / "pbp_team_games.parquet")
    print(df.shape); print(df.describe().round(2))
    qb = build_qb_games()
    qb.to_parquet(PROCESSED / "qb_games.parquet")
    print(qb.shape)
