"""Fantasy scoring computed from raw stats (offense: QB/RB/WR/TE only)."""
import pandas as pd

POSITIONS = ["QB", "RB", "WR", "TE"]
RECEPTION_VALUE = {"std": 0.0, "half": 0.5, "ppr": 1.0}


def fantasy_points(df: pd.DataFrame, fmt: str = "std") -> pd.Series:
    f = lambda c: df[c].fillna(0)
    return (
        0.04 * f("passing_yards") + 4 * f("passing_tds") - 2 * f("passing_interceptions")
        + 0.1 * f("rushing_yards") + 6 * f("rushing_tds")
        + 0.1 * f("receiving_yards") + 6 * f("receiving_tds")
        + 2 * (f("passing_2pt_conversions") + f("rushing_2pt_conversions") + f("receiving_2pt_conversions"))
        - 2 * (f("sack_fumbles_lost") + f("rushing_fumbles_lost") + f("receiving_fumbles_lost"))
        + 6 * f("special_teams_tds")
        + RECEPTION_VALUE[fmt] * f("receptions")
    )
