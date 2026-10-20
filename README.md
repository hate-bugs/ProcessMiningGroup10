# ProcessMiningGroup10

The notebook models daily mining operations and slower hardware decisions, using
process mining to guide subsequent runs. It retains bandit-style learners;
Q-learning is a possible later extension.

Open `notebook.ipynb` from this repository so that `process_feedback.py` and
`mining_hardware.py` are importable. Run sections 2–7 for the model, then section 13 for the feedback
experiment. Sections 8–12 provide the existing descriptive process-mining
analyses and are not required to run the experiment.

The feedback experiment:

- Uses **one cycle per day** and a Poisson number of block opportunities averaging
  144. Winners are assigned in one batch; each agent makes one operating decision
  and pays energy once per day. Hash shares remain fixed within the day.
- Reviews hardware every 30 days, staggered by agent, with at least 90 days
  between purchases/sales. Reviews can keep equipment unchanged. Purchases
  require cash, capacity and an estimated marginal payback, and take 14 days to
  install. They do not replace daily operating decisions.
- Tracks hardware in lots. Initial equipment is 180 days old. Resale value falls
  from 90% of acquisition cost when new to 10% at the 1,095-day useful-life limit.
  Equipment then retires and is salvaged. Repairs retain its age.
- Measures economic return as change in cash plus age-valued hardware, damaged
  hardware net of repair cost, and prepaid orders. Orders stay at cost until
  installation. Exit liquidates equipment and cancels orders at a 90% refund.
- Learns daily operating values from operating profit plus repair/exit asset
  effects, and capital values from
  observed 180-day wealth changes. Incomplete capital targets at the time limit
  are not learned. Overlapping capital decisions share subsequent outcomes.
- Mines separate operating histories (30-day outcomes) and hardware-review
  histories (180-day outcomes). Capital edges link reviews while financial
  windows still include intervening days. Rare edges receive no advice.
- Trains on four one-year seeds, then freezes both advisers before evaluating
  six different seeds. Each seed runs a baseline, process advice, and advice
  without sequence history. Market/shock/block randomness is paired.
- Reports economic returns, drawdown, bankruptcy, survival, advice coverage and
  paired seed bootstrap intervals. The direct process-versus-no-sequence
  comparison tests the added contribution of sequence history.

The main scenario uses **150 agents and 365 days**. Evaluation runs last **1,095
days**, with snapshots at days 365 and 1,095. The first-year snapshot cannot use
future rewards. All arms use the same 150-agent population. The longer evaluation
tests generalization beyond the one-year training horizon.

The agents represent the entire **scaled model network**, not small miners
receiving all actual Bitcoin rewards. Expected issuance is one normalized reward
unit per day, divided across 144 expected blocks. At the initial price of 1,000,
expected total revenue is 1,000 currency units/day, not 144,000. Changing block
resolution does not multiply expected revenue. There is no external hash network,
real difficulty retargeting, transaction-fee market or halving schedule.

All parameters are illustrative scenario assumptions, not a live BTC/USD or ASIC
calibration. Purchase cost is 80 per abstract hash unit (adjusted by efficiency);
daily energy uses a factor of 0.50 times grid price and hash, divided by efficiency.
Neutral log-price drift replaces the earlier assumed annual price doubling.
The hardware intervals and valuation curve are configurable, not universal
industry schedules. Tune on separate validation seeds, not evaluation seeds.

Run the same experiment without Jupyter or Graphviz:

```powershell
.venv/Scripts/python.exe scripts/run_feedback_experiment.py
```

The runner executes the notebook's actual model and experiment cells. It saves
tables, configuration and a plot under `output_mas/daily_feedback/` (gitignored).
`operating_adviser_edges.csv` and `capital_adviser_edges.csv` expose the evidence; `paired_effects.csv`
compares each adviser with the baseline; `sequence_effects.csv` compares the
process advisers directly with advice without history. Earlier results in
`output_mas/feedback/` belong to the previous one-block model.

Run the accounting and experiment-integrity checks:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The checks execute the model definitions directly from the notebook. They cover
reward conservation, daily billing, hardware ageing and delivery, review/cooldown
schedules, delayed credit, wealth reconciliation, complete outcome windows,
adviser support, legal actions, zero-weight equivalence and shared scenarios.

Economic return uses an explicit age-based asset valuation, not a profitability forecast.
The adviser learns observational associations under the training policy, not
causal effects. Sequence frequency and process conformance do not establish that
a policy is profitable. A later Q-learning extension needs explicit decisions
about state sufficiency, reward timing, delayed credit, terminal states and the
effects of other agents learning simultaneously.
