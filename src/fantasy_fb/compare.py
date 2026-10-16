"""Walk-forward comparison of v1 vs v2 feature sets (and gradient boosting) against Vegas."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import summarize
from .features_v2 import build_v2
from .paths import PROCESSED

V1 = ["d_fp_diff_blend", "d_pd_blend"]
MODELS = {
    "v1: fantasy + point diff": (V1, "lr"),
    "+ EPA": (V1 + ["d_epa_blend"], "lr"),
    "+ ridge ratings": (V1 + ["d_epa_blend", "d_ridge_epa", "d_ridge_fp"], "lr"),
    "+ Elo": (V1 + ["d_epa_blend", "d_elo_epa", "d_elo_fp"], "lr"),
    "+ QB": (V1 + ["d_epa_blend", "d_qb"], "lr"),
    "all v2 features": (V1 + ["d_epa_blend", "d_ridge_epa", "d_ridge_fp", "d_elo_epa", "d_elo_fp", "d_qb"], "lr"),
    "all v2, gradient boosting": (V1 + ["d_epa_blend", "d_ridge_epa", "d_ridge_fp", "d_elo_epa", "d_elo_fp", "d_qb"], "gbm"),
    "Vegas spread": (["spread"], "lr"),
    "Vegas + all v2": (["spread"] + V1 + ["d_epa_blend", "d_ridge_epa", "d_ridge_fp", "d_elo_epa", "d_elo_fp", "d_qb"], "lr"),
}


def make(kind):
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=150, min_samples_leaf=40, l2_regularization=5.0, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))


def backtest(gf: pd.DataFrame, models=MODELS, seasons=range(2018, 2027)) -> pd.DataFrame:
    gf = gf[gf["home_win"].notna() & (gf["home_win"] != 0.5)].copy()
    gf["spread"] = gf["spread"].fillna(0)
    out = []
    for s in seasons:
        train, test = gf[gf.season < s], gf[gf.season == s]
        for name, (cols, kind) in models.items():
            m = make(kind).fit(train[cols].fillna(0), train.home_win.round())
            p = np.clip(m.predict_proba(test[cols].fillna(0))[:, 1], 0.02, 0.98)
            out.append(pd.DataFrame({"season": s, "model": name, "game_id": test.game_id.values, "p_home": p, "home_win": test.home_win.values}))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    gf = build_v2()
    gf.to_parquet(PROCESSED / "game_features_v2.parquet")
    pr = backtest(gf)
    pr.to_parquet(PROCESSED / "backtest_v2_preds.parquet")
    pd.set_option("display.width", 200)
    for title, d in [("2018-2025", pr[pr.season <= 2025]), ("2025 only", pr[pr.season == 2025]), ("2026 so far (live)", pr[pr.season == 2026])]:
        print(f"\n== {title} ==")
        print(summarize(d).round(4).to_string(index=False))
