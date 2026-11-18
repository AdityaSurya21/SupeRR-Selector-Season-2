# Beyond the Scoreboard: Clutch Lab

**Rajasthan Royals · SupeRR Selector Season 2 · Round 1**
**Aditya Surya · Indian Institute of Technology Roorkee**

### ▶ [Open the interactive dashboard](https://YOUR-USERNAME.github.io/superr-selector-clutch-lab/)

This project identifies the T20 batters and bowlers who perform most reliably under pressure. It uses IPL and Full-Member T20I ball-by-ball data from January 2022 to November 2026.

---

## Final rankings

| # | Batters | Bowlers |
|---|---|---|
| 1 | Suryakumar Yadav | Sunil Narine |
| 2 | Vaibhav Sooryavanshi | Jasprit Bumrah |
| 3 | Abhishek Sharma | Varun Chakravarthy |
| 4 | Tim David | Matheesha Pathirana |
| 5 | Heinrich Klaasen | Rashid Khan |
| 6 | Phil Salt | Mujeeb Ur Rahman |
| 7 | Nicholas Pooran | Josh Hazlewood |
| 8 | Jos Buttler | Krunal Pandya |
| 9 | Rinku Singh | Mitchell Santner |
| 10 | Shivam Dube | Noor Ahmad |

## Method in brief

1. **Pressure is leverage, not the clock.** A win-probability model covers both innings. The Dynamic Pressure Index (DPI) is the expected swing in win probability a ball can cause. A pressure ball is in the top 20% of DPI (DPI ≥ 1.27). Tested against match results, a run above expectation on these balls is worth **2.4×** an ordinary run. On death-overs balls it is worth only 0.91×.
2. **Performance is runs above expectation.** Each ball is compared with what an average player would do in the same situation. Wickets are priced in runs using a Duckworth-Lewis-style resource curve.
3. **Ratings come from a Pressure-Adjusted Player Model.** It is an opponent-adjusted random-effects model, with shrinkage chosen by cross-validation.
4. **Uncertainty comes from a 500-draw match bootstrap.** The Clutch Rating, which sets the ranking, is the cautious 20th-percentile estimate.
5. **The rankings were stress-tested** across 27 alternative model specifications.

## What's in this repository

| Path | Contents |
|---|---|
| `index.html` | The interactive dashboard. It is a single file that also works offline. |
| `report/SupeRR_Selector_Final_Submission.pdf` | The full submission (50 pages): the main report plus Appendix B, Model Validation and Robustness. |
| `report/SupeRR_Selector_Final_Report.pdf` | The main methodology report on its own. |
| `report/Model_Validation_and_Robustness.pdf` | The technical validation document on its own. |
| `dashboard/` | A one-page static summary dashboard (PDF and PNG). |
| `results/Top_10_Batters.csv`, `results/Top_10_Bowlers.csv` | The final lists, with metrics, uncertainty and an explanation for each player. |
| `results/Complete_Player_Ratings.csv` | Ratings for every batter and bowler. |
| `results/Final_Unstop_Submission_List.txt` | The exact names and order entered on Unstop. |
| `analysis/SupeRR_Selector_Analysis.ipynb` | The full pipeline as an executed notebook, from raw data to rankings. |
| `analysis/src/` | The same pipeline as Python modules. |

## Reproducing the analysis

The raw competition data is not included in this repository. To reproduce the results:

1. Place the six supplied CSV files in `analysis/data/`.
2. Run:

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib pyarrow nbconvert ipykernel
cd analysis
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 SupeRR_Selector_Analysis.ipynb
```

The run takes about 15 minutes. The seed is fixed (20260516), so the results are fully reproducible.
