import pandas as pd
import pytest

from fantasy_fb.build import build_team_games
from fantasy_fb.scoring import POSITIONS, fantasy_points
from fantasy_fb import ingest


@pytest.fixture(scope="module")
def tg():
    return build_team_games()


def test_two_rows_per_game(tg):
    assert (tg.groupby("game_id").size() == 2).all()


def test_no_missing_fantasy_points(tg):
    cols = ["fp_std", "fp_half", "fp_ppr", "fp_against_std", "fp_diff_half"]
    assert tg[cols].notna().all().all()


def test_32_teams_and_game_counts(tg):
    full = tg[tg["season"] < tg["season"].max()]
    assert (full.groupby("season")["team"].nunique() == 32).all()
    n = full.groupby(["season", "team"]).size()
    expected = n.index.get_level_values("season").map(lambda s: 17 if s >= 2021 else 16)
    exp = pd.Series(expected.values, index=n.index)
    exp[(2022, "BUF")] = exp[(2022, "CIN")] = 16  # Week 17 game cancelled (Hamlin)
    assert (n == exp).all()


def test_against_equals_opponent_for(tg):
    m = tg.merge(tg, left_on=["game_id", "opponent"], right_on=["game_id", "team"], suffixes=("", "_o"))
    assert (m["fp_against_half"] - m["fp_half_o"]).abs().max() < 1e-9


def test_scoring_matches_nflverse():
    ps = ingest.load_player_stats()
    ps = ps[ps["position"].isin(POSITIONS) & (ps["season_type"] == "REG")]
    assert (fantasy_points(ps, "std") - ps["fantasy_points"]).abs().max() < 0.05
    assert (fantasy_points(ps, "ppr") - ps["fantasy_points_ppr"]).abs().max() < 0.05
