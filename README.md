# fantasy-fb

Do NFL teams' fantasy points predict wins? Visualizes fantasy production vs. wins and tests whether it predicts real games.

## Run it
```
pip install -r requirements.txt
PYTHONPATH=src python -m fantasy_fb.ingest --refresh   # download nflverse data
PYTHONPATH=src python -m fantasy_fb.build              # data/processed/team_games.parquet
PYTHONPATH=src python -m fantasy_fb.evaluate           # walk-forward backtest
PYTHONPATH=src python -m fantasy_fb.predict            # refresh data, pick next week, log picks (run before kickoff)
streamlit run app/streamlit_app.py
pytest
```

## Method
- One row per team per game (2010+, regular season, QB/RB/WR/TE), with fantasy points scored **and allowed**.
- Pre-game features use only earlier games (`shift(1)`); early in a season they blend in last year's numbers.
- Models are logistic regressions evaluated walk-forward: each season is predicted from earlier seasons only.

## Findings (team-seasons 2010-2025, half-PPR unless noted)
| Predictor of win % (same season) | R² |
|---|---|
| Fantasy points scored | 0.35 |
| Fantasy points allowed | 0.15 |
| Fantasy point differential | 0.59 |
| Actual point differential (reference) | 0.83 |

- Standard scoring tracks wins better than PPR (differential R² 0.65 / 0.59 / 0.53 for std / half / PPR).
- Next-season win % R²: last year's win % 0.13, fantasy differential 0.12, point differential 0.16.

Backtest 2018-2025 (accuracy / Brier, lower Brier is better): Vegas spread 66.4% / 0.2105; fantasy + point diff 63.3% / 0.2247;
fantasy diff alone 61.2% / 0.2299; better record so far 61.1% / 0.2369; always home 54.0% / 0.2485.
Adding fantasy differential to the Vegas spread improves Brier only trivially (0.2105 → 0.2100).

**Takeaway:** fantasy differential is a decent predictor, clearly better than home field or record, but actual point differential
is at least as good and the betting line is better than either. Fantasy points add little the market doesn't already know.

## Garbage-time and EPA experiment (`python -m fantasy_fb.experiments`, needs `python -m fantasy_fb.pbp` first)
Fantasy points rebuilt from play-by-play, dropping plays where the offense's win probability was outside 10-90%.

| Differential used | Same-season R² vs win % | Next-season R² | Game backtest Brier (2018-25) | Accuracy |
|---|---|---|---|---|
| Box-score fantasy (half-PPR) | 0.59 | 0.118 | 0.2299 | 61.2% |
| Play-by-play fantasy | 0.60 | 0.120 | 0.2290 | 61.7% |
| Play-by-play fantasy, no garbage time | 0.68 | 0.123 | 0.2290 | 62.1% |
| EPA, no garbage time | 0.82 | 0.140 | 0.2251 | 62.7% |
| EPA, all plays | 0.77 | 0.145 | 0.2243 | 63.7% |

Garbage-time filtering lifts the same-season fit a lot (0.60 to 0.68) but barely helps prediction, so much of that gain is
description, not forecasting. EPA beats every fantasy variant at predicting games, which is the best case for adding EPA features.

## v2 features (`python -m fantasy_fb.compare`)
Added: EPA differential, opponent-adjusted ridge ratings (EPA and fantasy), margin-based Elo, and the starting QB's shrunk EPA/dropback
(starter is known at kickoff; for future games the team's last starter is assumed). Walk-forward 2018-2025:

| Model | Accuracy | Brier |
|---|---|---|
| Vegas spread | 66.4% | 0.2105 |
| Vegas + all v2 | 66.0% | 0.2111 |
| All v2 features (logistic) | 63.6% | 0.2210 |
| v1 + EPA + starting QB (used for live picks) | 64.1% | 0.2210 |
| All v2, gradient boosting | 63.9% | 0.2220 |
| v1 (fantasy + point diff) | 63.3% | 0.2247 |

The QB feature drives most of the gain. On the 2025 hold-out alone v2 does not beat v1 (Brier 0.2259 vs 0.2215; 271 games is noisy),
and gradient boosting does not beat logistic regression. Nothing here beats the betting line.
