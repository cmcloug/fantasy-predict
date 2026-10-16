"""Walk-forward backtest: for each test season, train only on earlier seasons."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .features import build_game_features
from .paths import PROCESSED

TEST_SEASONS = list(range(2018, 2027))  # 2026 is the live, in-progress season

MODELS = {
    "fantasy diff": ["d_fp_diff_blend"],
    "fantasy diff + recent form": ["d_fp_diff_blend", "d_fp_diff_l4"],
    "actual point diff": ["d_pd_blend"],
    "fantasy + point diff": ["d_fp_diff_blend", "d_pd_blend"],
    "fantasy + rest": ["d_fp_diff_blend", "rest_diff"],
    "Vegas spread": ["spread"],
    "Vegas spread + fantasy diff": ["spread", "d_fp_diff_blend"],
}


def _fit_predict(train, test, cols):
    m = LogisticRegression(C=1.0, max_iter=1000)  # intercept captures home-field edge
    m.fit(train[cols], train["home_win"].round())
    return m.predict_proba(test[cols])[:, 1]


def backtest(gf: pd.DataFrame) -> pd.DataFrame:
    gf = gf[gf["home_win"] != 0.5].copy()  # drop ties
    gf["spread_home"] = gf["spread"]  # home spread: + means home favored
    out = []
    for s in TEST_SEASONS:
        train, test = gf[gf["season"] < s], gf[gf["season"] == s].copy()
        preds = {
            "always home team": np.full(len(test), 0.55),
            "better record so far": 0.5 + 0.1 * np.sign(test["d_win_sea"]) + 0.05 * (test["d_win_sea"] == 0),
        }
        for name, cols in MODELS.items():
            if "spread" in cols:
                train = train.assign(spread=train["spread_home"]); test = test.assign(spread=test["spread_home"])
            preds[name] = _fit_predict(train.dropna(subset=cols), test.fillna({"spread": 0}), cols)
        for name, p in preds.items():
            out.append(pd.DataFrame({"season": s, "model": name, "game_id": test["game_id"].values,
                                     "p_home": np.clip(np.asarray(p, float), 0.02, 0.98),
                                     "home_win": test["home_win"].values}))
    return pd.concat(out, ignore_index=True)


def summarize(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model), d in preds.groupby("model"):
        y = d["home_win"].round()
        rows.append({"model": model, "games": len(d),
                     "accuracy": ((d["p_home"] > 0.5) == (y == 1)).mean(),
                     "brier": brier_score_loss(y, d["p_home"]),
                     "log_loss": log_loss(y, d["p_home"])})
    return pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)


if __name__ == "__main__":
    gf = build_game_features()
    preds = backtest(gf)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(PROCESSED / "backtest_preds.parquet")
    hist = preds[preds["season"] <= 2025]
    pd.set_option("display.width", 200)
    print("== Backtest 2018-2025 ==")
    print(summarize(hist).round(4).to_string(index=False))
    print("\n== 2025 only (final exam) ==")
    print(summarize(preds[preds["season"] == 2025]).round(4).to_string(index=False))
    print("\n== 2026 so far (live) ==")
    print(summarize(preds[preds["season"] == 2026]).round(4).to_string(index=False))
