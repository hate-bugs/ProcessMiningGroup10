# ProcessMiningGroup10

The notebook simulates adaptive cryptocurrency-mining agents and uses process
mining to guide subsequent simulation runs. It retains the bandit-style learner;
Q-learning is a possible later extension.

Open `notebook.ipynb` from this repository so that `process_feedback.py` is
importable. Run sections 2–7 for the model, then section 13 for the feedback
experiment. Sections 8–12 provide the existing descriptive process-mining
analyses and are not required to run the experiment.

The feedback experiment:

- Corrects learning rewards to use change in net asset value, including cash,
  working hardware at resale value, and damaged hardware net of repair cost.
  Every decision is settled once, including pauses and exits. The existing
  capitulation thresholds still use mining operating losses.
- Mines context-specific `previous action → next action` transitions from
  complete decision histories. Five-cycle subsequent returns annotate those
  edges. Rare edges receive no advice; supported scores are bounded and shrunk.
- Trains on four seeds, then freezes the adviser before evaluating six different
  seeds. Each evaluation seed runs a baseline, a process adviser, and an adviser
  without sequence history. External market/shock randomness is paired.
- Reports economic returns, drawdown, bankruptcy, survival, advice coverage and
  paired seed bootstrap intervals. The direct process-versus-no-sequence
  comparison tests the added contribution of sequence history.

The default experiment uses **150 agents and 120 cycles** per run. Its results
are a pilot, not evidence that the same effects hold at the main notebook's
1,000 agents and 365 cycles. Change `EXPERIMENT_PARAMS` in section 13 to run at
that scale; keep training/evaluation configurations matched. Tune adviser
settings on separate validation seeds, not the final evaluation seeds.

Run the same experiment without Jupyter or Graphviz:

```powershell
.venv/Scripts/python.exe scripts/run_feedback_experiment.py
```

The runner executes the notebook's actual model and experiment cells. It saves
tables, configuration and a plot under `output_mas/feedback/` (gitignored).
`process_adviser_edges.csv` exposes the mined evidence; `paired_effects.csv`
compares each adviser with the baseline; `sequence_effects.csv` compares the
process adviser directly with the adviser without history.

Run the accounting and experiment-integrity checks:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The checks execute the model definitions directly from the notebook. They cover
investment/repair/sale accounting, minimum-hash limits, pause/exit settlement,
reward reconciliation, complete outcome windows, adviser support, legal actions,
zero-weight equivalence, and shared external scenarios.

The reward uses a stated resale valuation, not estimated future earning power.
The adviser learns observational associations under the training policy, not
causal effects. Sequence frequency and process conformance do not establish that
a policy is profitable. A later Q-learning extension needs explicit decisions
about state sufficiency, reward timing, delayed credit, terminal states and the
effects of other agents learning simultaneously.
