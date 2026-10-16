"""Download nflverse data and cache it as parquet."""
import sys

import nflreadpy as nfl
import pandas as pd

from .paths import RAW

FIRST_SEASON = 2010


def _cached(name, loader, refresh):
    path = RAW / f"{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    RAW.mkdir(parents=True, exist_ok=True)
    df = loader().to_pandas()
    df.to_parquet(path)
    return df


def current_season():
    return int(nfl.get_current_season())


def load_player_stats(refresh=False):
    seasons = list(range(FIRST_SEASON, current_season() + 1))
    return _cached("player_stats", lambda: nfl.load_player_stats(seasons=seasons), refresh)


def load_schedules(refresh=False):
    seasons = list(range(FIRST_SEASON, current_season() + 1))
    return _cached("schedules", lambda: nfl.load_schedules(seasons=seasons), refresh)


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    print("player_stats", load_player_stats(refresh).shape)
    print("schedules", load_schedules(refresh).shape)
