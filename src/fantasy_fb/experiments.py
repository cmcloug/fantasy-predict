"""Does filtering garbage time (or using EPA) make fantasy-style differentials better predictors?"""
import pandas as pd

from . import evaluate
from .build import build_team_games
from .features import build_game_features
from .paths import PROCESSED
from .season_view import r2

VARIANTS = {"box-score fantasy": "fp_diff_half", "play-by-play fantasy": "fp_diff_pbp",
            "pbp fantasy, no garbage time": "fp_diff_clean", "EPA, no garbage time": "epa_diff_clean",
            "EPA, all plays": "epa_diff_all"}


def with_pbp(tg=None) -> pd.DataFrame:
    tg = build_team_games() if tg is None else tg
    pbp = pd.read_parquet(PROCESSED / "pbp_team_games.parquet")
    tg = tg.merge(pbp, on=["game_id", "team"], how="left")
    opp = pbp.rename(columns={"team": "opponent", **{c: c + "_opp" for c in pbp.columns if c not in ("game_id", "team")}})
    tg = tg.merge(opp, on=["game_id", "opponent"], how="left")
    tg["fp_diff_pbp"] = tg.fp_pbp - tg.fp_pbp_opp
    tg["fp_diff_clean"] = tg.fp_pbp_clean - tg.fp_pbp_clean_opp
    tg["epa_diff_clean"] = tg.epa_clean - tg.epa_clean_opp
    tg["epa_diff_all"] = tg.epa_total - tg.epa_total_opp
    return tg


if __name__ == "__main__":
    tg = with_pbp()
    done = tg[tg.season < tg.season.max()]
    ts = done.groupby(["season", "team"]).agg(win=("win", "mean"), **{k: (v, "mean") for k, v in VARIANTS.items()}, pd=("points_for", "mean"))
    print("== Same-season R² vs win % ==")
    for k in VARIANTS: print(f"{k:32s} {r2(ts[k], ts['win']):.3f}")
    ts = ts.reset_index().sort_values(["team", "season"]); ts["next"] = ts.groupby("team")["win"].shift(-1)
    d = ts.dropna(subset=["next"])
    print("\n== Next-season win % R² ==")
    for k in VARIANTS: print(f"{k:32s} {r2(d[k], d['next']):.3f}")

    print("\n== Walk-forward 2018-2025, logistic on home-minus-away blend (fantasy-type feature only) ==")
    rows = []
    for k, col in VARIANTS.items():
        gf = build_game_features(tg=tg.assign(fp_diff_half=tg[col]))
        evaluate.MODELS = {k: ["d_fp_diff_blend"]}
        pr = evaluate.backtest(gf); pr = pr[(pr.season <= 2025) & (pr.model == k)]
        rows.append(evaluate.summarize(pr))
    print(pd.concat(rows).round(4).to_string(index=False))
