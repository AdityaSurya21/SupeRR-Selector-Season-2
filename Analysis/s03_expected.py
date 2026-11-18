"""Step 3 - Expected performance: what would an average player have done on this ball?

For every delivery we estimate, from the match situation alone (no player identity):
  * batting:  expected runs off the bat, P(striker dismissed), P(boundary), P(dot)
  * bowling:  expected runs conceded, P(bowler-credited wicket)

The situation includes the pressure state itself (balls left, wickets, required rate,
batting position of the striker, scoring environment). That is deliberate: a residual
then means "better or worse than an average player *in this exact situation*", which
is the comparison the brief asks for ("performance relative to equivalent match
situations"), and it stops death-overs hitting from looking like clutch skill simply
because everyone scores faster at the death.

All predictions are out-of-fold by match (5-fold GroupKFold).

Single currency - Wicket-adjusted Runs Above Expectation (RAE):
    batting  RAE = (runs - E[runs]) - c(s) * (out - P(out))
    bowling  RAE = (E[conceded] - conceded) + c(s) * (wicket - P(wicket))
c(s) is the run value of a wicket in state s: the drop in expected remaining runs when
one wicket is lost with the same balls left (the logic behind DLS resources). In a
chase c(s) is capped at the runs still needed, because a wicket cannot cost more runs
than the side still requires.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from common import PROC, SEED, save_json

ENV_REF = 8.5


def ctx(b, side):
    X = pd.DataFrame({
        "innings": b.innings, "balls_left": b.balls_left, "balls_total": b.balls_total,
        "wkts_in_hand": b.wkts_in_hand, "over": b.over,
        "rrr_f": np.where(b.innings == 2, b.rrr.clip(upper=36), -1),
        "rn_f": np.where(b.innings == 2, b.runs_needed.clip(upper=300), -1),
        "crr_f": b.crr.fillna(-1), "env_rpo": b.env_rpo,
        "bat_pos": b.batting_order.clip(upper=11),
        "spin": (b.pace_or_spin == "Spin").astype(int),
        # competition environment: IPL conditions (impact-player era, venues) differ from T20Is
        # by more than the lagged run-rate captures, so the expectation is set within competition-season
        "is_ipl": (b.comp_group == "IPL").astype(int), "season": b.season,
    })
    if side == "bowl":
        X = X.drop(columns=["spin"])     # spin/pace is the bowler's own attribute
    return X


def wicket_cost_table(b):
    """Resource table in the Duckworth-Lewis functional form, fitted to these data.

    Expected remaining runs with u balls left and w wickets in hand (environment-relative):
        R(u, w) = Z(w) * (1 - exp(-lambda * u / Z(w)))
    The shared lambda (the scoring rate when resources are unlimited) is the DL constraint
    that keeps the curves for different wicket counts consistent with each other. A free
    boosted table was tried first and produced implausible costs in sparse cells (e.g. the
    first wicket at the halfway mark 'costing' 0.15 runs), so it was rejected."""
    from scipy.optimize import least_squares
    ok = b.bat_won.notna() & ~b.is_dls & (b.innings == 1)
    f = b[ok].copy()
    f["y"] = (f.groupby("match_id").total_runs.transform("sum") - f.runs_before) * ENV_REF / f.env_rpo
    g = f[f.wkts_in_hand >= 1].groupby(["wkts_in_hand", "balls_left"]).y.agg(["mean", "size"]).reset_index()
    w_, u_, m_, n_ = g.wkts_in_hand.values, g.balls_left.values, g["mean"].values, g["size"].values

    def pred(t, w, u):
        lam, Z = np.exp(t[0]), np.exp(t[1:])[w - 1]
        return Z * (1 - np.exp(-lam * u / Z))
    r = least_squares(lambda t: (pred(t, w_, u_) - m_) * np.sqrt(n_), np.r_[np.log(1.6), np.log(np.linspace(15, 300, 10))])
    lam, Z = float(np.exp(r.x[0])), np.exp(r.x[1:])
    u = np.arange(121)
    tab = np.zeros((121, 11))
    for w in range(1, 11):
        tab[:, w] = Z[w - 1] * (1 - np.exp(-lam * u / Z[w - 1]))
    p = pred(r.x, w_, u_)
    r2 = 1 - np.sum(n_ * (m_ - p) ** 2) / np.sum(n_ * (m_ - np.average(m_, weights=n_)) ** 2)
    return tab, {"lambda": lam, "Z": Z.tolist(), "weighted_r2": float(r2)}


def main():
    b = pd.read_parquet(PROC / "balls_wp.parquet")
    tab, dl = wicket_cost_table(b)
    bl = b.balls_left.clip(0, 120).astype(int).values
    w = b.wkts_in_hand.clip(0, 10).astype(int).values
    cost = (tab[bl, w] - tab[bl, np.maximum(w - 1, 0)]) * b.env_rpo.values / ENV_REF
    cost = np.where(b.innings == 2, np.minimum(cost, b.runs_needed.clip(lower=0).values), cost)
    b["wicket_cost"] = np.maximum(cost, 0)

    fb = b.faced.values           # balls faced by the striker
    gkf = GroupKFold(n_splits=5)
    kw = dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=400,
              l2_regularization=1.0, early_stopping=True, n_iter_no_change=30, random_state=SEED)
    cols = {"x_bat_runs": np.full(len(b), np.nan), "x_out": np.full(len(b), np.nan),
            "x_boundary": np.full(len(b), np.nan), "x_dot": np.full(len(b), np.nan),
            "x_bowl_runs": np.full(len(b), np.nan), "x_bowl_wkt": np.full(len(b), np.nan)}
    Xbat, Xbowl = ctx(b, "bat"), ctx(b, "bowl")
    diag = {}
    for k, (tr, te) in enumerate(gkf.split(b, groups=b.match_id)):
        trb = tr[fb[tr]]
        teb = te[fb[te]]
        for name, y, model in [
            ("x_bat_runs", b.bat_runs.values, HistGradientBoostingRegressor(loss="poisson", **kw)),
            ("x_out", b.striker_out.values.astype(int), HistGradientBoostingClassifier(**kw)),
            ("x_boundary", b.is_boundary.values.astype(int), HistGradientBoostingClassifier(**kw)),
            ("x_dot", b.is_dot.values.astype(int), HistGradientBoostingClassifier(**kw)),
        ]:
            model.fit(Xbat.iloc[trb], y[trb])
            p = model.predict(Xbat.iloc[teb]) if name == "x_bat_runs" else model.predict_proba(Xbat.iloc[teb])[:, 1]
            cols[name][teb] = p
        for name, y, model in [
            ("x_bowl_runs", b.bowler_runs.values, HistGradientBoostingRegressor(loss="poisson", **kw)),
            ("x_bowl_wkt", b.bowler_wicket.values.astype(int), HistGradientBoostingClassifier(**kw)),
        ]:
            model.fit(Xbowl.iloc[tr], y[tr])
            p = model.predict(Xbowl.iloc[te]) if name == "x_bowl_runs" else model.predict_proba(Xbowl.iloc[te])[:, 1]
            cols[name][te] = p
        print("fold", k)
    for c, v in cols.items():
        b[c] = v

    # residual currencies
    b["bat_rae"] = np.where(fb, (b.bat_runs - b.x_bat_runs) - b.wicket_cost * (b.striker_out - b.x_out), np.nan)
    b["bowl_rae"] = (b.x_bowl_runs - b.bowler_runs) + b.wicket_cost * (b.bowler_wicket - b.x_bowl_wkt)
    b["bat_wpa"] = np.where(fb, b.dwp, np.nan)
    b["bowl_wpa"] = -b.dwp

    # diagnostics: does context beat a constant?
    s = fb
    def ll(y, p):
        return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    diag["batting_out"] = {"model": ll(b.striker_out[s], b.x_out[s]), "constant": ll(b.striker_out[s], np.full(s.sum(), b.striker_out[s].mean()))}
    diag["bowling_wicket"] = {"model": ll(b.bowler_wicket, b.x_bowl_wkt), "constant": ll(b.bowler_wicket, np.full(len(b), b.bowler_wicket.mean()))}
    def r2(y, p):
        return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    diag["batting_runs_r2"] = r2(b.bat_runs[s].values, b.x_bat_runs[s].values)
    diag["bowling_runs_r2"] = r2(b.bowler_runs.values, b.x_bowl_runs.values)
    diag["calibration_bat_runs"] = {"actual": float(b.bat_runs[s].mean()), "expected": float(b.x_bat_runs[s].mean())}
    diag["calibration_out"] = {"actual": float(b.striker_out[s].mean()), "expected": float(b.x_out[s].mean())}
    diag["calibration_bowl_wkt"] = {"actual": float(b.bowler_wicket.mean()), "expected": float(b.x_bowl_wkt.mean())}
    diag["mean_wicket_cost_by_phase"] = b.groupby("phase").wicket_cost.mean().to_dict()
    diag["dl_resource_fit"] = dl
    diag["wicket_cost_examples"] = {f"{bl}b_{w}w": float(tab[bl, w] - tab[bl, w - 1]) for bl in (120, 60, 30, 12) for w in (10, 6, 3)}
    save_json(diag, "expected_model_diagnostics.json")
    np.save(PROC / "wicket_cost_table.npy", tab)
    b.to_parquet(PROC / "balls_full.parquet", index=False)
    print(diag)


if __name__ == "__main__":
    main()
