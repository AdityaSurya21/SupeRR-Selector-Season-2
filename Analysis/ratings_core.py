"""Core of the player-rating model, reused by the main pipeline and the robustness suite.

Pressure-Adjusted Player Model (PAPM)
-------------------------------------
For every ball i, with RAE_i the wicket-adjusted runs above expectation from the batting
side's point of view (sign flipped for bowlers), and P_i = 1 on a pressure ball:

    RAE_i = mu + tau * P_i
            + a_j   + d_j * P_i          (the player being rated: normal level + pressure shift)
            + o_k   + e_k * P_i          (the opponent on that ball: bowler for batters, batter for bowlers)
            + noise

a_j, d_j, o_k, e_k are random effects with Normal(0, sigma^2) priors. Their posterior
mode is a ridge regression; the prior variances (penalties) are chosen by 5-fold
cross-validation over *matches*, i.e. by how well the model predicts deliveries in
matches it has not seen. This is empirical Bayes: the data decide how much a small
sample is trusted. The opponent terms mean a run against Jasprit Bumrah in the 19th over
counts for more than a run against a part-timer.

Pressure performance of player j = a_j + d_j (runs above expectation per pressure ball).
Normal performance = a_j. Pressure shift = d_j.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr


def build(df, actor, opp, pflag):
    a = pd.Categorical(df[actor].values)
    o = pd.Categorical(df[opp].values)
    n = len(df)
    P = df[pflag].values.astype(float)
    i = np.arange(n)
    na, no = len(a.categories), len(o.categories)
    blocks = dict(
        A=sparse.csr_matrix((np.ones(n), (i, a.codes)), shape=(n, na)),
        D=sparse.csr_matrix((P, (i, a.codes)), shape=(n, na)),
        O=sparse.csr_matrix((np.ones(n), (i, o.codes)), shape=(n, no)),
        E=sparse.csr_matrix((P, (i, o.codes)), shape=(n, no)),
    )
    fixed = sparse.csr_matrix(np.c_[np.ones(n), P])
    return {"blocks": blocks, "fixed": fixed, "actors": np.asarray(a.categories), "opps": np.asarray(o.categories),
            "na": na, "no": no, "P": P, "acodes": a.codes, "ocodes": o.codes}


def fit(des, y, lam_base, lam_delta, lam_opp=None, lam_opp_delta=None, w=None):
    lam_opp = lam_base if lam_opp is None else lam_opp
    lam_opp_delta = lam_delta if lam_opp_delta is None else lam_opp_delta
    bl = des["blocks"]
    sc = [1 / np.sqrt(lam_base), 1 / np.sqrt(lam_delta), 1 / np.sqrt(lam_opp), 1 / np.sqrt(lam_opp_delta)]
    X = sparse.hstack([des["fixed"] * 100.0, bl["A"] * sc[0], bl["D"] * sc[1], bl["O"] * sc[2], bl["E"] * sc[3]]).tocsr()
    yy = np.asarray(y, float)
    if w is not None:
        sw = np.sqrt(w)
        X = sparse.diags(sw) @ X
        yy = yy * sw
    sol = lsqr(X, yy, damp=1.0, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    na, no = des["na"], des["no"]
    k = 2
    out = {"mu": sol[0] * 100, "tau": sol[1] * 100}
    out["a"] = sol[k:k + na] * sc[0]; k += na
    out["d"] = sol[k:k + na] * sc[1]; k += na
    out["o"] = sol[k:k + no] * sc[2]; k += no
    out["e"] = sol[k:k + no] * sc[3]
    return out


def predict(des, coef, rows=None):
    bl = des["blocks"]
    pred = (coef["mu"] + coef["tau"] * des["P"] + bl["A"] @ coef["a"] + bl["D"] @ coef["d"]
            + bl["O"] @ coef["o"] + bl["E"] @ coef["e"])
    return pred if rows is None else pred[rows]


def cv_select(df, y, actor, opp, pflag, grid_base, grid_delta, folds=5, seed=0):
    """Choose prior variances by match-grouped cross-validation (held-out MSE)."""
    rng = np.random.default_rng(seed)
    mids = df.match_id.unique()
    fold_of = dict(zip(mids, rng.integers(0, folds, len(mids))))
    f = df.match_id.map(fold_of).values
    y = np.asarray(y, float)
    des = build(df, actor, opp, pflag)
    res = []
    null_mse = []
    for lb in grid_base:
        for ld in grid_delta:
            se = 0.0
            for k in range(folds):
                tr = f != k
                sub = {**des, "blocks": {n: m[tr] for n, m in des["blocks"].items()}, "fixed": des["fixed"][tr],
                       "P": des["P"][tr]}
                c = fit(sub, y[tr], lb, ld)
                te = ~tr
                p = (c["mu"] + c["tau"] * des["P"][te] + des["blocks"]["A"][te] @ c["a"] + des["blocks"]["D"][te] @ c["d"]
                     + des["blocks"]["O"][te] @ c["o"] + des["blocks"]["E"][te] @ c["e"])
                se += ((y[te] - p) ** 2).sum()
            res.append({"lam_base": lb, "lam_delta": ld, "mse": se / len(y)})
    for k in range(folds):
        tr = f != k
        te = ~tr
        P = des["P"]
        m0 = np.array([y[tr & (P == v)].mean() for v in (0, 1)])
        null_mse.append(((y[te] - m0[P[te].astype(int)]) ** 2).sum())
    t = pd.DataFrame(res)
    t["null_mse"] = sum(null_mse) / len(y)
    t["improvement_vs_null_pct"] = 100 * (t.null_mse - t.mse) / t.null_mse
    best = t.loc[t.mse.idxmin()]
    return t, float(best.lam_base), float(best.lam_delta)


def bootstrap(df, y, actor, opp, pflag, lam_base, lam_delta, B=500, seed=1):
    """Match-level (cluster) bootstrap: resample whole matches with replacement and refit.
    Returns arrays (B x n_actors) of normal level a and pressure level a + d."""
    rng = np.random.default_rng(seed)
    des = build(df, actor, opp, pflag)
    codes, uniq = pd.factorize(df.match_id)
    nm = len(uniq)
    y = np.asarray(y, float)
    A = np.empty((B, des["na"]))
    PRS = np.empty((B, des["na"]))
    for bi in range(B):
        cnt = np.bincount(rng.integers(0, nm, nm), minlength=nm)
        w = cnt[codes].astype(float)
        keep = w > 0
        sub = {**des, "blocks": {n: m[keep] for n, m in des["blocks"].items()}, "fixed": des["fixed"][keep], "P": des["P"][keep]}
        c = fit(sub, y[keep], lam_base, lam_delta, w=w[keep])
        # players absent from a resample get their prior mean (0), exactly as the model implies
        A[bi] = c["a"]
        PRS[bi] = c["a"] + c["d"]
    return des["actors"], A, PRS


# ---------------------------------------------------------------------------
# Two-component PAPM
# ---------------------------------------------------------------------------
# Runs and dismissals carry very different amounts of repeatable signal (split-half
# reliability of the run residual is ~0.54, of the combined RAE only ~0.21, because
# aggressive players score more *and* get out more). Fitting one ridge to the combined
# currency forces a single shrinkage level on both. The two-component model shrinks
# each part by its own cross-validated amount and recombines them at the player's own
# wicket price:
#     RAE_j(pressure) = runs_j(pressure) - cbar_j(pressure) * outs_j(pressure)     (batting)
#     RAE_j(pressure) = runs_saved_j(pressure) + cbar_j(pressure) * wkts_j(pressure)   (bowling)

def components(df, side):
    if side == "bat":
        return (df.bat_runs - df.x_bat_runs).values, -(df.striker_out.astype(float) - df.x_out).values
    return (df.x_bowl_runs - df.bowler_runs).values, (df.bowler_wicket.astype(float) - df.x_bowl_wkt).values


def wicket_price(df, actor, actors):
    p = df[df.pressure].groupby(actor).wicket_cost.mean()
    n = df[~df.pressure].groupby(actor).wicket_cost.mean()
    return p.reindex(actors).fillna(df[df.pressure].wicket_cost.mean()).values, n.reindex(actors).fillna(df[~df.pressure].wicket_cost.mean()).values


def two_component(df, side, lam_run, lam_wkt, w=None, des=None):
    actor, opp = ("striker_id", "bowler_id") if side == "bat" else ("bowler_id", "striker_id")
    des = build(df, actor, opp, "pressure") if des is None else des
    yr, yk = components(df, side)
    cr = fit(des, yr, *lam_run, w=w)
    ck = fit(des, yk, *lam_wkt, w=w)
    return des, cr, ck


def combine(cr, ck, cp, cn):
    normal = cr["a"] + cn * ck["a"]
    pressure = (cr["a"] + cr["d"]) + cp * (ck["a"] + ck["d"])
    return normal, pressure


def bootstrap_two_component(df, side, lam_run, lam_wkt, B=500, seed=1):
    actor, opp = ("striker_id", "bowler_id") if side == "bat" else ("bowler_id", "striker_id")
    rng = np.random.default_rng(seed)
    des = build(df, actor, opp, "pressure")
    cp, cn = wicket_price(df, actor, des["actors"])
    yr, yk = components(df, side)
    codes, uniq = pd.factorize(df.match_id)
    nm = len(uniq)
    NOR = np.empty((B, des["na"])); PRS = np.empty((B, des["na"]))
    RUN_P = np.empty((B, des["na"])); WK_P = np.empty((B, des["na"]))
    SHIFT = np.empty((B, des["na"]))
    for bi in range(B):
        cnt = np.bincount(rng.integers(0, nm, nm), minlength=nm)
        wt = cnt[codes].astype(float)
        keep = wt > 0
        sub = {**des, "blocks": {n: m[keep] for n, m in des["blocks"].items()}, "fixed": des["fixed"][keep], "P": des["P"][keep]}
        cr = fit(sub, yr[keep], *lam_run, w=wt[keep])
        ck = fit(sub, yk[keep], *lam_wkt, w=wt[keep])
        NOR[bi], PRS[bi] = combine(cr, ck, cp, cn)
        RUN_P[bi] = cr["a"] + cr["d"]
        WK_P[bi] = ck["a"] + ck["d"]
        SHIFT[bi] = cr["d"] + cp * ck["d"]          # behavioural shift at a fixed (pressure) wicket price
    return des["actors"], NOR, PRS, RUN_P, WK_P, SHIFT
