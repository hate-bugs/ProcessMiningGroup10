# ProcessMiningGroup10

`notebook.ipynb` models mining operations and uses process mining to feed learned
process knowledge back into adaptive agents. Section 13 is the current automated
online feedback experiment. Open the notebook from the repository root, or run it
headlessly with the commands below.

## Current experiment

The research question is whether PM agents outperform concurrent non-PM agents
at **50% adoption**. This is different from improving total market welfare.
Earlier population-wide feedback did not improve performance; those results remain
in `output_mas/daily_feedback/`, and a source snapshot is retained locally in
`output_mas/previous_notebook.ipynb`. Development probes also did not establish a
system-wide gain from the online method. The new question and that negative evidence
are disclosed in the notebook.

- **40 agents, 780 days, 12 feedback updates.** The same agents retain their wealth,
  hardware, identities and local action values throughout. The first 60 days are
  unguided warm-up; every 60 days thereafter a new adviser uses only observed logs.
- **Randomized complementary assignments.** Half of the agents receive advice.
  Each scenario is repeated with the assignments reversed. Both cohorts keep all
  original decision rules, local/social learning and capital policies. An independent
  all-control market measures spillovers and total-market changes.
- **A discovered process model.** `process_policy.py` mines an outcome-annotated
  transition system from ordered decision cases: observed context plus previous
  activity, action, next process state, frequency and realized operating reward.
  It plans over the empirical graph and produces confidence-weighted scores.
  Discount zero reduces this to immediate edge-reward scoring, as selected in
  the recorded validation run; multistep planning was evaluated but not selected.
  Activity labels are opaque; there are no hand-written mining recommendations.
  Missing next events are censored; actual exits are absorbing.
- **Automatic adaptation.** Every update mines the past 180 days of both cohorts'
  decision logs. Five observations and two agent cases are required for advice.
  The adviser only affects subsequent decisions, never the observations used to fit it.
  Operating advice is learned; the existing capital policy is identical across cohorts.
- **Actual parameter optimization.** Weight 2/6 and discount 0/0.9 are evaluated
  on three validation seeds with both assignments. PM and a no-sequence ablation
  receive the same search budget. The baseline learning rate 0.18 and exploration
  0.09 were selected in the earlier baseline search and are shared by both cohorts.
- **Eight new test seeds.** Configuration is frozen before final evaluation on
  seeds 19101-19108. Online adaptation within each test market is part of the
  algorithm; it uses only that market's past, never future events or other test runs.
  Smoke-test and development seeds are separate.

The primary endpoint is PM minus concurrent-control **mean normalized economic
return at day 780**. Complementary assignments are averaged within each seed;
5,000 bootstrap resamples use seeds, not dependent agents or assignments.
Other metrics, intermediate rounds, sequence ablation and welfare comparisons are
secondary, with unadjusted descriptive intervals. Eight scenarios provide limited
precision. A positive average relative advantage is not a guarantee for every
miner or evidence that full adoption improves system-wide outcomes. Sequence history
must be judged against the no-sequence ablation, not inferred from PM versus control.

## Model accounting

One cycle is one day with a Poisson number of block opportunities averaging 144.
Energy is billed once daily. Hardware reviews occur every 30 days, with at least
90 days between trades and a 14-day delivery delay. Lots age and are valued using
a declining resale curve; economic return includes cash, hardware and prepaid orders.
Daily actions learn from operating outcomes; capital actions receive delayed credit.

The descriptive analysis in sections 7-12 retains 150 agents over 365 days. The
feedback study uses the existing stress scenario: energy cost factor 6.2, shock
probability 0.14, shock multipliers 2.5-4.5. Daily issuance scales by 40/150, avoiding
an artificial increase in expected initial revenue per miner. This is not exact
population-size equivalence. Parameters are synthetic, not real Bitcoin calibration.

## Run and reproduce

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-experiment.txt
.venv/Scripts/python.exe scripts/run_feedback_experiment.py --write-notebook
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The runner executes the actual model definitions and every section-13 code cell.
`--write-notebook` embeds real outputs. Omit it to export only, or use `--smoke`
for a short pipeline check in a separate directory. Old offline validation caches
cannot be used by the new online protocol. Sections 8-12 additionally need PM4Py
and Graphviz; they are not required for the feedback experiment.

Exports in `output_mas/online_feedback/` (gitignored):

- `parameter_search.csv`: every validation candidate, seed and assignment.
- `experiment_config.json`: design, parameters, selections, seeds, versions and hashes.
- `learning_curve.csv`, `seed_effects.csv`, `held_out_results.csv`: complete learning
  curves and paired effects, including any negative results.
- `agent_outcomes.csv`, `run_audit.csv`, `fit_audit.csv`: identities, outcomes and
  which observations were available before each feedback update.
- `welfare_comparison.csv`, `sequence_effect.csv`: spillovers, total-market effects
  and the independently tuned no-sequence comparison.
- `evidence/*`: compressed decision traces, discovered transition probabilities,
  support counts, graph values and advice scores for every test-market update.
- `online_learning_curve.png` and `.pdf`: figures generated from the exported data.
- `period_learning_curve.csv`, `period_seed_effects.csv` and the corresponding
  PNG/PDF: incremental gains per feedback period, avoiding a purely cumulative
  interpretation. A frozen-adviser comparison would be needed to isolate the
  incremental benefit of refitting from initial guidance.

`process_feedback.py` and `feedback_experiment.py` retain the earlier offline method.
`process_policy.py` and `online_feedback_experiment.py` implement the current loop.
Tests cover accounting, zero-weight equivalence, intact baseline scores, discovery
and planning, action-label invariance, reward-dependent advice, censoring, randomized
assignment, continuing agent cases, and strict chronological feedback.
