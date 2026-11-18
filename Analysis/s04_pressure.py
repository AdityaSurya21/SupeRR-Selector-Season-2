"""Step 4 - Defining pressure, and testing the definition against alternatives.

Candidate definitions of a 'pressure ball' are compared on a criterion that does not
depend on the leverage model itself:

    Result-importance test. For every team innings we add up the batting side's
    wicket-adjusted runs above expectation (RAE) separately on 'pressure' balls and
    on all other balls, then regress the match result on the two sums:
        logit P(win) = a + b_P * RAE_pressure + b_N * RAE_other
    b_P / b_N says how many times more a run above expectation is worth when it is
    produced on a pressure ball. A definition that really isolates the moments that
    decide T20 matches should give a large multiplier; one that simply marks 'the
    last five overs' will not.

The candidate that wins this test, with the broadest coverage of both innings, is
used as the Dynamic Pressure Index (DPI).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from common import PROC, save_json


def definitions(b):
    li = b.LI
    q80 = li.quantile(0.8)
    inn2 = b.innings == 2
    d = {}
    d["DPI: leverage top 20%"] = li >= q80
    d["DPI: leverage top 10%"] = li >= li.quantile(0.9)
    d["DPI: leverage top 30%"] = li >= li.quantile(0.7)
    grp = b.innings.astype(str) + b.phase
    d["Phase-relative leverage top 20%"] = li >= b.groupby(grp).LI.transform(lambda s: s.quantile(0.8))
    d["Death overs (16-20)"] = b.over >= 16
    d["Last 5 overs of a chase"] = inn2 & (b.over >= 16)
    d["Chase RRR >= 10"] = inn2 & (b.rrr >= 10)
    d["Close game (WP 25-75%) after over 12"] = b.wp.between(0.25, 0.75) & (b.over >= 13)
    d["Rule-based composite"] = (inn2 & (b.rrr >= 10)) | ((b.wkts_before >= 4) & (b.over <= 10)) | (b.over >= 17)
    # a typical hand-weighted index: z-scores of required-rate gap, wickets down and over number
    gap = np.where(inn2, (b.rrr.clip(upper=36) - b.crr.fillna(b.env_rpo)), 0)
    z = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
    idx = 0.4 * z(gap) + 0.3 * z(b.wkts_before.values) + 0.3 * z(b.over.values)
    d["Hand-weighted index (0.4/0.3/0.3) top 20%"] = pd.Series(idx >= np.nanquantile(idx, 0.8), index=b.index)
    return d


def result_importance(b, flag):
    s = b[b.faced & b.bat_won.notna() & (b.result_class == "decided")].copy()
    s["P"] = flag.loc[s.index].astype(bool)
    g = s.groupby(["match_id", "innings"])
    t = pd.DataFrame({"won": g.bat_won.first(),
                      "rae_p": s[s.P].groupby(["match_id", "innings"]).bat_rae.sum(),
                      "rae_n": s[~s.P].groupby(["match_id", "innings"]).bat_rae.sum()}).fillna(0)
    X = sm.add_constant(t[["rae_p", "rae_n"]] / 10)
    fit = sm.Logit(t.won, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": t.index.get_level_values(0)})
    base = sm.Logit(t.won, sm.add_constant((t.rae_p + t.rae_n) / 10)).fit(disp=0)
    bp, bn = fit.params["rae_p"], fit.params["rae_n"]
    # delta-method SE for the ratio
    cov = fit.cov_params().loc[["rae_p", "rae_n"], ["rae_p", "rae_n"]].values
    grad = np.array([1 / bn, -bp / bn ** 2])
    se = float(np.sqrt(grad @ cov @ grad))
    return {"b_pressure_per10": float(bp), "b_other_per10": float(bn), "multiplier": float(bp / bn),
            "multiplier_se": se, "llf_gain_vs_unsplit": float(fit.llf - base.llf), "n_innings": int(len(t))}


def coverage(b, flag):
    f = flag.astype(bool)
    return {"share_balls": float(f.mean()),
            "share_in_innings1": float(f[b.innings == 1].mean()),
            "share_of_pressure_in_innings1": float((f & (b.innings == 1)).sum() / max(f.sum(), 1)),
            "share_of_pressure_powerplay": float((f & (b.phase == "Powerplay")).sum() / max(f.sum(), 1)),
            "share_of_pressure_middle": float((f & (b.phase == "Middle")).sum() / max(f.sum(), 1)),
            "share_of_pressure_death": float((f & (b.phase == "Death")).sum() / max(f.sum(), 1)),
            "mean_LI": float(b.LI[f].mean()),
            "share_of_total_leverage": float(b.li_raw[f].sum() / b.li_raw.sum())}


def behaviour_by_pressure(b):
    s = b[b.faced].copy()
    s["LI_band"] = pd.qcut(s.LI, 10, labels=[f"D{i}" for i in range(1, 11)])
    t = s.groupby("LI_band", observed=True).agg(
        LI_mean=("LI", "mean"), balls=("LI", "size"), strike_rate=("bat_runs", lambda x: 100 * x.mean()),
        dismissal_pct=("striker_out", lambda x: 100 * x.mean()), boundary_pct=("is_boundary", lambda x: 100 * x.mean()),
        dot_pct=("is_dot", lambda x: 100 * x.mean()), share_inn2=("innings", lambda x: (x == 2).mean()),
        share_death=("phase", lambda x: (x == "Death").mean()), wicket_share_of_leverage=("wicket_share_of_leverage", "mean"),
        mean_wp=("wp", "mean"))
    return t.reset_index()


def main():
    b = pd.read_parquet(PROC / "balls_full.parquet")
    defs = definitions(b)
    rows = []
    for name, flag in defs.items():
        r = {"definition": name, **coverage(b, flag), **result_importance(b, flag)}
        rows.append(r)
        print(f"{name:45s} cover={r['share_balls']:.3f} inn1={r['share_of_pressure_in_innings1']:.2f} mult={r['multiplier']:.2f}±{r['multiplier_se']:.2f}")
    comp = pd.DataFrame(rows)
    comp.to_csv(PROC / "pressure_definition_comparison.csv", index=False)
    beh = behaviour_by_pressure(b)
    beh.to_csv(PROC / "behaviour_by_pressure_decile.csv", index=False)

    # Chosen definition: continuous DPI = LI; a 'pressure ball' is DPI in the top 20%.
    q80, q95 = b.LI.quantile(0.8), b.LI.quantile(0.95)
    b["DPI"] = b.LI
    b["pressure"] = b.LI >= q80
    b["extreme"] = b.LI >= q95
    b["pressure_tier"] = np.select([b.LI >= q95, b.LI >= q80, b.LI >= b.LI.quantile(0.4)], ["Extreme", "High", "Normal"], "Low")
    # pressure 'mode' - what kind of pressure: survival (wicket-dominated) or scoring (run-dominated)
    # Thresholds are the quartiles of the wicket share of leverage on pressure balls: in a
    # 'Survival' ball the swing from a wicket is the dominant risk, in a 'Scoring' ball it is the runs.
    ws = b.wicket_share_of_leverage
    lo, hi = ws[b.pressure].quantile(0.25), ws[b.pressure].quantile(0.75)
    b["pressure_mode"] = np.select([ws >= hi, ws <= lo], ["Survival", "Scoring"], "Balanced")
    for name, flag in defs.items():
        b["def_" + "".join(ch for ch in name if ch.isalnum())[:30]] = flag.astype(bool).values
    b.to_parquet(PROC / "balls_final.parquet", index=False)
    save_json({"q80_LI": float(q80), "q95_LI": float(q95), "definitions": list(defs.keys()),
               "def_columns": {name: "def_" + "".join(ch for ch in name if ch.isalnum())[:30] for name in defs}},
              "pressure_meta.json")
    print(beh)


if __name__ == "__main__":
    main()
