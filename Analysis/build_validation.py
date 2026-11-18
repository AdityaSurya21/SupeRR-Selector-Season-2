"""Builds Model_Validation_and_Robustness.pdf - the technical companion to the main report."""
import pandas as pd
from common import PROC, OUT
from report_common import DEL, fig, j, f1, render_pdf

D, WP, EX, H, PV, RO = (j(n) for n in ["data_summary.json", "wp_validation.json", "expected_model_diagnostics.json",
                                       "rating_hyperparameters.json", "predictive_validation.json", "robustness_overview.json"])
FB = pd.read_parquet(PROC / "final_bat.parquet")
FW = pd.read_parquet(PROC / "final_bowl.parquet")


def tbl(df, cols, heads, fmts=None, cls="compact"):
    fmts = fmts or {}
    h = "".join(f"<th{' class=r' if i else ''}>{x}</th>" for i, x in enumerate(heads))
    rows = []
    for _, r in df.iterrows():
        cells = []
        for i, c in enumerate(cols):
            v = r[c]
            s = fmts[c](v) if c in fmts else (f"{v:.3f}" if isinstance(v, float) else str(v))
            cells.append(f"<td{' class=r' if i else ''}>{s}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='{cls}'><tr>{h}</tr>{''.join(rows)}</table>"


def spec_table():
    rows = []
    for k, v in WP["specifications_selection_2024"].items():
        rows.append({"spec": k, "all": v["all"]["brier"], "i1": v["innings_1"]["brier"], "i2": v["innings_2"]["brier"], "auc": v["all"]["auc"]})
    d = pd.DataFrame(rows).sort_values("all")
    return tbl(d, ["spec", "all", "i1", "i2", "auc"], ["Specification (validation season 2024)", "Brier all", "Brier inn. 1", "Brier inn. 2", "AUC"])


def test_table():
    t = WP["selected_test_2025_26"]
    l = WP["logistic_handcrafted"]
    rows = [("Selected model", t["all"], t["innings_1"], t["innings_2"]),
            ("Logistic, hand-built cricket features", l["all"], l["innings_1"], l["innings_2"])]
    h = "<tr><th>Model (test 2025-26)</th><th class=r>Brier</th><th class=r>Log-loss</th><th class=r>AUC</th><th class=r>Brier inn. 1</th><th class=r>Brier inn. 2</th></tr>"
    body = "".join(f"<tr><td>{n}</td><td class=r>{a['brier']:.4f}</td><td class=r>{a['log_loss']:.4f}</td><td class=r>{a['auc']:.3f}</td><td class=r>{b['brier']:.4f}</td><td class=r>{c['brier']:.4f}</td></tr>" for n, a, b, c in rows)
    body += f"<tr><td>Direct boosted classifier, first innings</td><td class=r>-</td><td class=r>-</td><td class=r>-</td><td class=r>{WP['direct_gbm_first_innings']['innings_1']['brier']:.4f}</td><td class=r>-</td></tr>"
    body += f"<tr><td>No information (0.5)</td><td class=r>0.2500</td><td class=r>0.6931</td><td class=r>0.500</td><td class=r>0.2500</td><td class=r>0.2500</td></tr>"
    o = WP["out_of_fold"]
    body += f"<tr><td>Selected model, out-of-fold all seasons (used for ratings)</td><td class=r>{o['all']['brier']:.4f}</td><td class=r>{o['all']['log_loss']:.4f}</td><td class=r>{o['all']['auc']:.3f}</td><td class=r>{o['innings_1']['brier']:.4f}</td><td class=r>{o['innings_2']['brier']:.4f}</td></tr>"
    return f"<table class='compact'>{h}{body}</table>"


def calib_table():
    d = pd.DataFrame(WP["selected_calibration"])
    return tbl(d, ["bin", "pred", "obs", "n"], ["Predicted bin", "Mean predicted", "Observed", "Balls"], {"n": lambda v: f1(v, 0)})


def cv_tables():
    out = []
    for f, lab in [("bat_runs", "Batting - run component"), ("bat_wickets", "Batting - dismissal component"),
                   ("bowl_runs", "Bowling - run component"), ("bowl_wickets", "Bowling - wicket component")]:
        t = pd.read_csv(OUT / f"cv_lambda_{f}.csv")
        p = t.pivot(index="lam_base", columns="lam_delta", values="improvement_vs_null_pct")
        h = "<tr><th>λ normal \\ λ shift</th>" + "".join(f"<th class=r>{'10⁸' if c >= 1e7 else int(c)}</th>" for c in p.columns) + "</tr>"
        best = p.max().max()
        rows = "".join("<tr><td>" + str(int(i)) + "</td>" + "".join(
            f"<td class=r>{'<b>' if abs(v - best) < 1e-12 else ''}{v:.4f}{'</b>' if abs(v - best) < 1e-12 else ''}</td>" for v in r) + "</tr>"
            for i, r in p.iterrows())
        out.append(f"<h4>{lab}</h4><table class='compact'>{h}{rows}</table>")
    return "".join(out)


def pv_table(side):
    p = PV[side]
    keys = [k for k in p["split_half_repeatability"] if not k.startswith("_")]
    rows = []
    for k in keys:
        rows.append({"m": k, "sh": p["split_half_repeatability"][k], "lo": p["leave_one_season_out_pooled"][k]["weighted_r"],
                     "los": p["leave_one_season_out_pooled"][k]["spearman"], "fw": p["forward_2022_24_to_2025_26"][k]["weighted_r"],
                     "fws": p["forward_2022_24_to_2025_26"][k]["spearman"]})
    d = pd.DataFrame(rows)
    fm = {c: (lambda v: f"{v:+.3f}") for c in ["sh", "lo", "los", "fw", "fws"]}
    return tbl(d, ["m", "sh", "lo", "los", "fw", "fws"], ["Measure", "Split-half r", "LOSO weighted r", "LOSO Spearman", "Forward weighted r", "Forward Spearman"], fm)


def rank_interval_table(df):
    d = df.head(15)
    return tbl(d, ["rank", "player_name", "clutch_rating", "pressure_rae100", "pressure_rae100_sd", "rank_p10", "rank_median", "rank_p90", "p_top10", "p_top3", "core_top10_share", "top10_share_all_specs"],
               ["#", "Player", "Clutch Rating", "Pressure level", "SD", "Rank P10", "Rank median", "Rank P90", "P(top 10)", "P(top 3)", "Core specs top-10", "All specs top-10"],
               {"rank": str, "clutch_rating": lambda v: f"{v:.1f}", "pressure_rae100": lambda v: f"{v:+.1f}", "pressure_rae100_sd": lambda v: f"{v:.1f}",
                "rank_p10": lambda v: f"{v:.0f}", "rank_median": lambda v: f"{v:.0f}", "rank_p90": lambda v: f"{v:.0f}",
                "p_top10": lambda v: f"{100*v:.0f}%", "p_top3": lambda v: f"{100*v:.0f}%", "core_top10_share": lambda v: f"{100*v:.0f}%",
                "top10_share_all_specs": lambda v: f"{100*v:.0f}%"})


def full_robust(side):
    S = pd.read_csv(OUT / f"robustness_summary_{side}.csv")
    return tbl(S, ["specification", "top10_overlap", "spearman_vs_baseline", "n_eligible", "baseline_top10_eligible"],
               ["Specification", "Top-10 overlap", "Spearman ρ", "Eligible players", "Baseline top-10 still eligible"],
               {"top10_overlap": lambda v: f"{v}/10", "spearman_vs_baseline": lambda v: "-" if pd.isna(v) else f"{v:.3f}"})


def body():
    ex = D["extras_calibration"]
    pdef = pd.read_csv(PROC / "pressure_definition_comparison.csv")
    hb, hw = H["bat"], H["bowl"]
    parts = []
    A = parts.append
    A(f"""<div class="kicker">Companion document</div><h1>Model Validation and Robustness</h1>
<p>This document collects the evidence behind every modelling choice in <i>Beyond the Scoreboard</i>: how well each model predicts data it has not seen,
how alternatives compare, how shrinkage was tuned, which pressure measures are repeatable, how stable the rankings are, and what we did to prevent leakage.
All values are produced by the pipeline and read programmatically when this document is built.</p>
<div class="statrow">
<div class="stat"><div class="v">{WP['selected_test_2025_26']['all']['brier']:.3f}</div><div class="l">Brier score of the win-probability model on unseen 2025-26 matches (logistic baseline {WP['logistic_handcrafted']['all']['brier']:.3f})</div></div>
<div class="stat pink"><div class="v">{pdef.set_index('definition').loc['DPI: leverage top 20%'].multiplier:.2f}×</div><div class="l">result-importance multiplier of DPI pressure balls; best scoreboard definition {pdef[~pdef.definition.str.contains('everage|DPI')].multiplier.max():.2f}×</div></div>
<div class="stat"><div class="v">{RO['bat']['mean_spearman']:.2f} / {RO['bowl']['mean_spearman']:.2f}</div><div class="l">mean rank correlation with the baseline across 27 specifications (batters / bowlers)</div></div>
<div class="stat pink"><div class="v">{PV['bat']['split_half_repeatability']['PAPM pressure level']:.2f} / {PV['bowl']['split_half_repeatability']['PAPM pressure level']:.2f}</div><div class="l">split-half repeatability of the final pressure measure (raw clutch delta: {PV['bat']['split_half_repeatability']['Raw clutch delta (pressure - normal RAE/100)']:+.2f} / {PV['bowl']['split_half_repeatability']['Raw clutch delta (pressure - normal RAE/100)']:+.2f})</div></div></div>

<h2><span class="num">1</span>Data reconstruction</h2>
<p>Wides, byes and leg-byes carry no run count in ball_by_ball.csv. Their average value was estimated by Huber regression of the official 'won by X runs' margin on the
difference in extras counts between innings ({ex['n_matches']} clean matches):</p>
<table class="compact"><tr><th>Extra</th><th class=r>Runs per event</th><th class=r>SE</th></tr>
{''.join(f"<tr><td>{k}</td><td class=r>{v:.3f}</td><td class=r>{ex['se'][k]:.3f}</td></tr>" for k, v in ex['estimates'].items())}</table>
<p>Residual margin error: median {ex['median_abs_margin_error']:.2f} runs; {100*ex['share_within_2_runs']:.0f}% within 2 runs, {100*ex['share_within_5_runs']:.0f}% within 5. Two
outliers (90 and 40 runs) are DLS results with no revised target recorded - not data errors - and are excluded from win-probability training. Chase targets:
{', '.join(f'{v} {k}' for k, v in D['target_methods'].items())}. Legal-ball check: {100*D['share_overs_with_six_legal']:.1f}% of overs have exactly six legal deliveries.</p>

<h2><span class="num">2</span>Win-probability model</h2>
<h4>Specification selection (train 2022-23, validate 2024)</h4>{spec_table()}
<p>'_rel' specifications express runs needed and remaining runs relative to the scoring environment (runs × 8.5 / environment run rate). They were the
decisive improvement: the environment-free models could not extrapolate to the 2024-26 scoring rise. Pre-match Elo improves chase forecasts once runs are environment-relative.</p>
<h4>Final test (train 2022-24, test 2025-26)</h4>{test_table()}
<h4>Calibration on the test seasons</h4>{calib_table()}
<p>Additional checks: median gap between the end-of-first-innings WP and the start-of-chase WP {WP['innings_break_continuity_median_abs_gap']:.4f}; mean WP change on
live balls {WP['mean_dwp_live_balls']:+.5f} (a calibrated forecast should drift by zero on average). Outcome model log-loss on 2025-26
{WP['outcome_model']['log_loss_model']:.4f} versus {WP['outcome_model']['log_loss_base_rates']:.4f} for training base rates.</p>
<div class="callout pink"><b>A failure we fixed.</b> The first version used a leave-one-out seasonal run rate and an unrounded Elo gap as features, and random
row-level early stopping. Because every ball of a match shares its label, the boosted model learned to recognise matches: training Brier 0.005, test 0.31 -
worse than a coin flip. All models now use match-grouped validation, coarse environment features and monotone constraints.</div>

<h2><span class="num">3</span>Pressure definition test</h2>
{tbl(pdef, ['definition', 'share_balls', 'share_of_pressure_in_innings1', 'share_of_pressure_death', 'share_of_total_leverage', 'multiplier', 'multiplier_se', 'llf_gain_vs_unsplit'],
     ['Definition', 'Balls', 'From inn. 1', 'In death overs', 'Leverage share', 'Multiplier', 'SE (delta method)', 'Log-lik gain'],
     {c: (lambda v: f"{100*v:.0f}%") for c in ['share_balls', 'share_of_pressure_in_innings1', 'share_of_pressure_death', 'share_of_total_leverage']} |
     {'multiplier': lambda v: f"{v:.2f}", 'multiplier_se': lambda v: f"{v:.2f}", 'llf_gain_vs_unsplit': lambda v: f"{v:.1f}"})}
<p>Logit of team result on batting runs above expectation (per 10 runs) on pressure and other balls, {int(pdef.n_innings.iloc[0])} team innings from decided matches;
standard errors clustered by match. The test uses actual results and runs, not the leverage model's outputs; win probabilities themselves are out-of-fold.</p>

<h2><span class="num">4</span>Expected-performance models</h2>
<table class="compact"><tr><th>Model</th><th class=r>Model</th><th class=r>Constant</th><th>Note</th></tr>
<tr><td>Striker dismissal (log-loss)</td><td class=r>{EX['batting_out']['model']:.4f}</td><td class=r>{EX['batting_out']['constant']:.4f}</td><td>out-of-fold by match</td></tr>
<tr><td>Bowler wicket (log-loss)</td><td class=r>{EX['bowling_wicket']['model']:.4f}</td><td class=r>{EX['bowling_wicket']['constant']:.4f}</td><td></td></tr>
<tr><td>Runs off the bat (R²)</td><td class=r>{EX['batting_runs_r2']:.4f}</td><td class=r>0</td><td>single-ball noise dominates</td></tr>
<tr><td>Runs conceded (R²)</td><td class=r>{EX['bowling_runs_r2']:.4f}</td><td class=r>0</td><td></td></tr>
<tr><td>Calibration: runs per ball</td><td class=r>{EX['calibration_bat_runs']['expected']:.4f}</td><td class=r>{EX['calibration_bat_runs']['actual']:.4f}</td><td>expected vs actual</td></tr>
<tr><td>Calibration: dismissal rate</td><td class=r>{EX['calibration_out']['expected']:.4f}</td><td class=r>{EX['calibration_out']['actual']:.4f}</td><td>expected vs actual</td></tr>
<tr><td>Calibration: bowler wicket rate</td><td class=r>{EX['calibration_bowl_wkt']['expected']:.4f}</td><td class=r>{EX['calibration_bowl_wkt']['actual']:.4f}</td><td>expected vs actual</td></tr></table>
<p>Wicket value: Duckworth-Lewis form fitted to first innings, λ = {EX['dl_resource_fit']['lambda']:.3f}, weighted R² = {EX['dl_resource_fit']['weighted_r2']:.4f}.
Mean wicket cost by phase: powerplay {EX['mean_wicket_cost_by_phase']['Powerplay']:.1f}, middle {EX['mean_wicket_cost_by_phase']['Middle']:.1f}, death {EX['mean_wicket_cost_by_phase']['Death']:.1f} runs.</p>

<h2><span class="num">5</span>Shrinkage: cross-validation surfaces</h2>
<p>Percentage improvement in held-out mean squared error over a model with no player effects (5-fold, folds are matches). Best cell in bold.
For run components the optimum is at an effectively infinite shift penalty (no player-specific pressure effect in run-scoring); for dismissal and wicket
components it is interior.</p>{cv_tables()}
<p>Implied spreads (per 100 balls): batting ability {hb['implied_ability_sd_runs_per100']:.1f} runs, batting dismissal ability {hb['implied_ability_sd_wkts_per100']:.2f},
batting dismissal shift {hb['implied_shift_sd_wkts_per100']:.2f}; bowling ability {hw['implied_ability_sd_runs_per100']:.1f} runs, wicket ability {hw['implied_ability_sd_wkts_per100']:.2f},
wicket shift {hw['implied_shift_sd_wkts_per100']:.2f}.</p>

<h2><span class="num">6</span>Predictive validation of pressure measures</h2>
<h4>Batters</h4>{pv_table('bat')}
<h4>Bowlers</h4>{pv_table('bowl')}
<p>Targets are raw pressure-ball RAE per 100 balls in the held-out period; correlations are weighted by held-out pressure balls. Minimum samples: 60 (batters) / 90 (bowlers)
pressure balls in the estimation period, 40 / 60 in the target period. Split-half uses odd versus even match IDs with 40 / 60 pressure balls in each half.
The low batting correlations reflect an unreliable target: the raw pressure RAE of a batter over 40-plus balls has near-zero split-half reliability itself, which caps
any predictor. Incremental regression (forward design): adding match consistency and WPA to the shrunk pressure level does not significantly improve prediction for batters
(n = {PV['bat']['forward_2022_24_to_2025_26']['_incremental_regression']['n_players']}) or bowlers (n = {PV['bowl']['forward_2022_24_to_2025_26']['_incremental_regression']['n_players']}).</p>
{fig('F05_repeatability', 'Split-half repeatability of pressure measures.')}

<h2><span class="num">7</span>Ranking robustness</h2>
<h4>Batters</h4>{full_robust('bat')}
<h4>Bowlers</h4>{full_robust('bowl')}
<p>For subset specifications (single competition, single innings, leave-one-season-out) the sample thresholds are scaled by 0.6; for alternative pressure definitions they are scaled by
the share of balls the definition flags. 'First innings only' leaves too few eligible bowlers for a meaningful comparison. The <b>core</b> specifications used for the tie band exclude the
three scoreboard definitions, the close-game definition, the unshrunk and conventional rankings and the single-competition or single-innings subsets.</p>
{fig('F14_robustness_summary', 'Top-10 overlap and rank correlation for every specification.')}
{fig('F09_rank_stability_bat', 'Batters: rank in selected specifications.')}
{fig('F09_rank_stability_bowl', 'Bowlers: rank in selected specifications.')}

<h2><span class="num">8</span>Bootstrap rank uncertainty</h2>
<p>500 match-level bootstrap resamples; in each, players are ranked by the resampled pressure level among the eligible pool. Rank P10-P90 is the 80% interval.</p>
<h4>Batters</h4>{rank_interval_table(FB)}
<h4>Bowlers</h4>{rank_interval_table(FW)}
{fig('F11_top10_probability', 'Probability of a top-10 place: bootstrap (bars) and specifications (ticks).')}

<h2><span class="num">9</span>Leakage controls</h2>
<table class="compact"><tr><th>Risk</th><th>Control</th></tr>
<tr><td>A ball scored by a model that saw its outcome</td><td>All win probabilities, leverages and expectations are 5-fold out-of-fold with folds = matches.</td></tr>
<tr><td>Features that identify matches</td><td>Environment = prior 40 matches only, rounded to 0.1; Elo chronological and rounded to 25; match-grouped early stopping.</td></tr>
<tr><td>Future information in team strength</td><td>Elo updated strictly after each match.</td></tr>
<tr><td>Optimistic forecast accuracy</td><td>WP specification chosen on 2024; the 2025-26 test was run once.</td></tr>
<tr><td>Residual leakage (disclosed)</td><td>Expectation models include later seasons in their training folds (no player identity is used); shrinkage penalties used in the
validation of Section 6 were chosen on all data; the bootstrap holds penalties fixed. Each makes reported uncertainty slightly optimistic.</td></tr></table>

<h2><span class="num">10</span>Critical review: three judges</h2>
<p>Before finalising we reviewed the work from three perspectives and changed the analysis where a criticism held. What follows records the criticisms, what we did, and what remains.</p>
<h4>Judge 1 - professional cricket analyst</h4>
<p><i>"Your top bowler hasn't played an international since 2019 and your #2 batter is 15."</i> Both are what the supplied data say; Narine's rating is entirely IPL, and
Sooryavanshi's sample is the smallest in the list, so we rank him on a conservative bound and flag him explicitly. <i>"A batter with a negative pressure record shouldn't be in a pressure top ten."</i>
Agreed - this led to the evidence gate, which removed Romario Shepherd. <i>"Is a ball with 28 needed off 5 really low pressure?"</i> It is low <i>leverage</i>, and we
show the case explicitly (Case 3) rather than hide it. <i>"Chases dominate."</i> True and intended, but we tested a phase-relative index: rank correlation 0.99 for batters.
<b>Open:</b> no T20 World Cup data; captaincy decisions about who bowls the 19th over are not modelled.</p>
<h4>Judge 2 - statistician</h4>
<p><i>"The first WP model was badly overfit."</i> Found and fixed (Section 2). <i>"Grid optimum at the edge."</i> Grids were widened until optima were interior or at the
'no effect' limit. <i>"One shrinkage level for runs and wickets is wrong."</i> Replaced by the two-component model, which raised repeatability. <i>"Your shift mixes wicket price
with behaviour."</i> The reported shift now holds the wicket price fixed. <i>"Post-hoc rules."</i> The tie band and evidence gate were added after seeing robustness results; the
main report shows the rating-only order and exactly which players each rule moved. <b>Open:</b> batting forecasts are weak for every measure; intervals are conditional on penalties.</p>
<h4>Judge 3 - competition evaluator</h4>
<p><i>"Does it stand out?"</i> The definition test (conventional pressure definitions fail against results) and the finding that run-scoring clutch is indistinguishable
from zero while dismissal-risk clutch is real are, as far as we know, not standard in cricket analytics. <i>"Is it readable?"</i> Every statistic is expressed in runs per 100 balls, with
plain-language explanations per player and three verified case studies. <b>Open:</b> some marginal selections (batting #10, bowling #10) rest on small rating differences; the
alternatives are named.</p>"""
    )
    return "".join(parts)


def cover():
    return f"""<div class="cover"><div class="band" style="height:110mm"><div class="gold"></div><div class="stripe"></div>
<div class="inner"><div class="pre">Beyond the Scoreboard · Technical companion</div>
<div class="title" style="font-size:27pt">MODEL VALIDATION<br>& ROBUSTNESS</div>
<div class="sub">SupeRR Selector Season 2 – Round 1</div></div></div>
<div class="below" style="top:122mm"><div class="lead">Out-of-sample tests of every model, the cross-validation that set the amount of shrinkage, repeatability of
competing pressure measures, 27 alternative ranking specifications, bootstrap rank intervals, leakage controls, and a critical three-judge review.</div></div>
<div class="meta"><span>Companion to SupeRR_Selector_Final_Report.pdf</span><span>November 2026</span></div></div>"""


def main():
    out = DEL / "Model_Validation_and_Robustness.pdf"
    render_pdf(cover(), body(), out, "Model Validation and Robustness - SupeRR Selector S2 Round 1",
               "Model Validation and Robustness", "Technical companion to Beyond the Scoreboard")
    print("written", out)


if __name__ == "__main__":
    main()
