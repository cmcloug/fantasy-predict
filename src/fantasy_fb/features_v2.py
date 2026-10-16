"""v2 features: v1 blends + EPA + opponent-adjusted ratings + Elo + starting-QB quality."""
import pandas as pd

from . import ingest
from .build import build_team_games
from .experiments import with_pbp
from .features import FEATURES, build_game_features
from .paths import PROCESSED
from .ratings import elo_ratings, qb_ratings, ridge_ratings

V2_FEATURES = ["d_epa_blend", "d_ridge_epa", "d_ridge_fp", "d_elo_epa", "d_elo_fp", "d_qb"]


def build_v2(include_future: bool = False, alpha: float = 10.0) -> pd.DataFrame:
    tg = with_pbp(build_team_games(include_future=include_future)).reset_index(drop=True)
    tg["epa_diff"] = tg.epa_total - tg.epa_total_opp
    tg["fpc_diff"] = tg.fp_pbp_clean - tg.fp_pbp_clean_opp

    # v1 blended features (fantasy diff, point diff) with an EPA-based version added
    gf = build_game_features(include_future=include_future)
    epa_gf = build_game_features(include_future=include_future, tg=tg.assign(fp_diff_half=tg.epa_diff))
    assert (gf.game_id.values == epa_gf.game_id.values).all()
    gf["d_epa_blend"] = epa_gf["d_fp_diff_blend"].values  # same row order: both sorted by season/week/game_id

    tg["ridge_epa"] = ridge_ratings(tg, "epa_total", alpha)
    tg["ridge_fp"] = ridge_ratings(tg, "fp_pbp_clean", alpha)
    qb = qb_ratings(tg, ingest.load_schedules(), pd.read_parquet(PROCESSED / "qb_games.parquet"))
    tg["qb_rating"] = qb["qb_rating"].values

    home, away = tg[tg.is_home == 1].set_index("game_id"), tg[tg.is_home == 0].set_index("game_id")
    extra = pd.DataFrame(index=home.index)
    for c, name in [("ridge_epa", "d_ridge_epa"), ("ridge_fp", "d_ridge_fp"), ("qb_rating", "d_qb")]:
        extra[name] = home[c] - away[c]
    extra["qb_home"], extra["qb_away"] = home["qb_rating"], away["qb_rating"]
    for actual, name in [("epa_diff", "d_elo_epa"), ("fpc_diff", "d_elo_fp")]:
        e = elo_ratings(tg, actual, k=0.1 if actual == "epa_diff" else 0.2).set_index("game_id")
        extra[name] = e.iloc[:, 0] - e.iloc[:, 1]
    gf = gf.merge(extra.reset_index(), on="game_id", how="left")
    return gf


if __name__ == "__main__":
    gf = build_v2(include_future=True)
    print(gf.shape); print(gf[V2_FEATURES].describe().round(3))
