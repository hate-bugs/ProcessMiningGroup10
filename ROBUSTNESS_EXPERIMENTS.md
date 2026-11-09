# Common robustness experiments

The completed runs, verification results, interpretation and figure links are in
[the v2 results bundle](reports/robustness_v2/RESULTS.md).

Both interventions use the existing notebook simulation, 40 initial miners, 780
days, eight independent seeds **29101–29108**, and 50% adviser adoption with
complementary assignments. An all-control market is also run for every seed and
scenario. Weight 6, discount 0, and all other original stress-scenario parameters
are retained. There is no new tuning or agent action. The descriptive baseline
and the original feedback experiment remain unchanged.

## Interventions

**Permanent shutdown:** at the start of day 360, close 0%, 5%, 10%, or 20%–100%
in ten-point steps of miners *still active*. Existing liquidation, cancellation
refunds, permanent inactivity and upward rounding are unchanged. A separate
random ranking produces nested target sets across severities without consuming
simulator randomness. Selection is not stratified by adviser assignment. Actual
headcounts and IDs are exported. The model has no entrants; fixed identities are
checked at the end of each run. Forced closure records are excluded from adviser
training. This models closure with asset recovery, not hardware destruction.

**Temporary cost shock:** compare a 1× baseline with a 2× multiplier applied to
electricity prices on **days 360–539 inclusive**. At the start of **day 540**, the
original daily energy-price calculation resumes. Each day first draws its usual
grid prices, including existing random energy shocks, then applies the multiplier
once. It does not compound and is not clipped back to the original price ceiling.
The multiplier therefore changes actual energy expenditure and the existing
profitability, cash-reserve and investment calculations. It introduces no fixed
charge or cost on paused/exited miners. Existing actions and guidance are unchanged.
Agents observe current prices through the existing state; their decision code
does not inspect the intervention schedule or receive advance knowledge of its end.

Both studies have matched initial conditions and identical dynamics before day
360. Equal percentages in a mixed and an all-control market can imply different
headcounts because their active populations may differ.

## Run and reproduce

Run from the repository root with the existing environment:

```powershell
# Pipeline checks use separate smoke seeds and reduced durations.
.venv/Scripts/python.exe scripts/run_disruption_experiment.py --experiment shutdown --smoke
.venv/Scripts/python.exe scripts/run_disruption_experiment.py --experiment cost-shock --smoke

# Full configurations: 288 shutdown markets and 48 cost-shock markets.
.venv/Scripts/python.exe scripts/run_disruption_experiment.py --experiment shutdown --workers 2
.venv/Scripts/python.exe scripts/run_disruption_experiment.py --experiment cost-shock --workers 2

# Optional configuration; retain 1.0 as the paired baseline.
.venv/Scripts/python.exe scripts/run_disruption_experiment.py --experiment cost-shock --days 780 --intervention-day 360 --shock-end-day 540 --cost-multipliers 1.0 2.0

# Recompute analysis into a new version without rerunning an existing v2 study.
.venv/Scripts/python.exe scripts/run_disruption_experiment.py --analyze-only output_mas/cost_shock_v2/full_validation

.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Duration is configured with `--days`; shock duration is `--shock-end-day` minus
`--intervention-day`. Additional multipliers can be supplied, but the default is
only 1× and 2×. A pre-intervention day and a recovery phase are required. The
shutdown severity list and seed list are retained in the protocol functions.

By default, each invocation creates a unique run directory beneath
`output_mas/disruption_v2/` or `output_mas/cost_shock_v2/`. `--output` may name an
empty or new directory. Nonempty destinations are rejected; historical outputs
are never silently replaced. Original results in `output_mas/disruption_sweep/`
and `reports/disruption/` remain available. The completed v2 runs for this task
use `full_validation` within each experiment's v2 directory.
The final cost-shock analysis, with an explicit paired recovery comparison, is in
`output_mas/cost_shock_v2/analysis_validation/`. Its provenance records
`simulation_rerun: false` and embeds the original simulation manifest, which records
the full rerun. Analysis-only mode also preserves its input directory and always
writes a separate version.

## Common estimands

For every scenario, seed and cohort, first average complementary assignments
within seed. The cohort contains every originally assigned miner, including exits
and forced closures. The existing cumulative return R(t) is the mean individual
net-worth change divided by that miner's initial net worth, expressed in percent.

- **Post-intervention return:** P(t) = R(t) − R(359). It is an increment in percentage
  points of initial wealth, not a percentage return on day-359 wealth.
- **Post-intervention guided advantage:** A(t) = P_guided(t) − P_control(t).
- **Primary paired effect:** D(t) = A_disruption(t) − A_baseline(t), retaining the
  seed pairing. This removes advantages accumulated before intervention and
  compares the disruption with its matched baseline.
- **Secondary reference-wealth return:** the mean of each original miner's
  `(wealth(t) − wealth(359)) / wealth(359)`. This has a different denominator.
  It is unavailable if any required reference wealth is nonpositive; invalid
  members or complementary assignments are not silently dropped.

All subtractions occur at seed level before statistical aggregation. Means, 95%
percentile intervals, seed counts and positive-seed counts use **5,000 bootstrap
resamples of independent seeds**, with bootstrap seed 19431. Miners, days and
complementary assignments are not independent bootstrap units. Comparisons are
exploratory, correlated and unadjusted for multiplicity. Curve intervals are
pointwise, not simultaneous bands. Use paired-effect intervals rather than the
overlap of marginal cohort intervals to compare scenarios.

## Adaptation, recovery and competition

The cost-shock phases are days 1–359, 360–539, and 540–780. Phase returns and
per-day increments are exported separately. The explicit recovery measure is:

`[R_shock(780) − R_baseline(780)] − [R_shock(539) − R_baseline(539)]`.

A positive value means the cohort's position relative to baseline improved after
prices returned. It closes a deficit only if the day-539 gap was negative. It is
not a time-to-recovery estimate, and higher final wealth alone does not establish
faster recovery. Recovery under this definition does not require individual
exited miners to re-enter; exits remain permanent under the original model.
The guided-minus-control difference in this recovery-gap change is also computed
within seed and bootstrapped as a paired effect. Separate cohort confidence
intervals are not used to infer which group recovered better.

Daily records track active miners, actual mining participants, operating costs,
rewards, net mining profit, wealth, and the model's actual actions: `mine`, `pause`
(the counterpart of Skip), `join_pool`, `leave_pool`, `repair`, `exit`, and capital
decisions `invest`, `sell_hardware`, `hold_hardware`. Mining participation is
measured using the existing mining-day counter, which increments with a positive
daily energy charge; both configured studies have positive energy prices and cost
factors. Action rates use the original assigned cohort denominator. Raw decision
counts are also available for conditional analyses.

Behavioral plots average days within phase-separated 30-day bins, then average
complementary assignments within seed. Bins never cross an intervention boundary.
They are plotted at the midpoint of their actual observed date interval. Seed-level
inputs and bootstrap intervals are retained even where plots show means only.

Network exports track active counts, mining participation, hash power, rewards,
and reward opportunities per actual participant/hash unit. Ratios with no
participants are undefined and retained as missing, not replaced with zero.
These help identify changes in competition and composition. They do not fully
isolate the causal contribution of adaptation: a frozen-adviser or no-learning
ablation would be needed for that claim. The 100% shutdown endpoint is a boundary
check with no subsequent production, not a learning or recovery result.

## Files and outputs

Implementation is shared rather than maintaining a second simulator:

- `disruption_experiment.py`: intervention plans, model extension, common runner,
  daily measurement, historical shutdown exports and versioned provenance.
- `robustness_analysis.py`: post-reference estimands, paired statistics, phase and
  recovery measures, behavioral/network summaries, figures and reports.
- `scripts/run_disruption_experiment.py`: both experiments and configurable inputs.
- `scripts/plot_disruption_cohorts.py`: retained separate-cohort plots and final
  comparison CSVs, now invoked by the shutdown export pipeline.
- `tests/test_disruption_experiment.py` and `tests/test_robustness_analysis.py`:
  permanent-closure invariants, neutral interventions, exact shock boundaries,
  repeatability, pairing, reference subtraction, recovery and output protection.

Each new run includes `ROBUSTNESS_REPORT.md` and:

| Output | Meaning |
|---|---|
| `snapshots.csv`, `agents.csv`, `daily.csv` | Existing checkpoint, final-agent and network exports, with additional columns |
| `cohort_daily.csv` | Daily cohort actions, participation, energy costs, profits and wealth |
| `cohort_return_seed_values.csv`, `cumulative_return_effects.csv` | Cumulative returns and intervals |
| `post_intervention_seed_values.csv`, `post_intervention_effects.csv` | P(t), paired cohort changes, and reference-wealth returns |
| `paired_advantage_seed_values.csv`, `paired_advantage_effects.csv` | A(t) and primary D(t), including positive-seed counts |
| `behavior_seed_values.csv`, `behavior_effects.csv` | Phase-separated behavioral trajectories |
| `network_seed_values.csv`, `network_effects.csv` | Participation, rewards and competition diagnostics |
| `phase_*`, `recovery_*` | Cost-shock phase, explicit recovery, and paired guided-minus-control recovery statistics |
| `fits.csv`, `shutdown_audit.json` or `intervention_audit.json` | Fitting chronology and intervention configuration/targets |
| `experiment_config.json` | Configuration, seeds, parameters, versions, hashes, aggregation, bootstrap and rerun status |

PNG and PDF figures include cumulative returns, post-intervention returns, paired
advantages, network opportunities, and cohort adaptation trajectories. Cost-shock
plots mark both day 360 and day 540. Shutdown plots include the primary paired
effect against severity. All prior shutdown `effects.csv`, `seed_effects.csv`,
`shutdown_sweep`, separate-cohort figures and final-comparison filenames are also
generated in new shutdown runs.

## Comparability and limitations

The original 41.86-point feedback result uses seeds 19101–19108; these robustness
studies use the already established disruption seeds 29101–29108. Their matched
no-disruption baseline is therefore 72.15 points cumulatively. Comparisons must
use the baseline belonging to these studies rather than subtracting the original
paper's mean. No statistical conclusion should be chosen to favor guided miners.

Economic parameters remain synthetic. Fixed expected block issuance, liquidation
with asset recovery, permanent endogenous exits, no entry, and no shutdown-induced
network-security or coin-price feedback limit real-world interpretation. More
returns after partial shutdown may reflect lower operating costs and competition.
The new analysis measures these factors alongside guidance outcomes rather than
assuming all changes are improvements in learning.
