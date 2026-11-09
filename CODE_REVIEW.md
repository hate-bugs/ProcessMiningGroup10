# Evaluation of the code supporting the report

Reviewed on 2026-11-09 against **Process Mining and Feedback in Adaptive Proof-of-Work Mining**, supplied as `Applying_Process_Mining_to_Agent_Based_Simulation_of_the_Bitcoin_Mining_Ecosystem (1).pdf`.

The current report uses the simulation and descriptive analyses in `notebook.ipynb`, the hardware accounting in `mining_hardware.py`, and the online experiment in `online_feedback_experiment.py` and `process_policy.py`. The older offline experiment modules are not the source of the report's headline feedback result.

**Assessment:** The full validation search and held-out feedback experiment reproduce the reported findings. Three errors in the baseline descriptive analysis require correction: the Gini coefficient, event-frequency counts, and within-day timing of capital trajectories. These errors do not enter the primary feedback-return calculation.

## Findings requiring correction

### 1. Incorrect Gini coefficient in Section 6.2

**Location:** `notebook.ipynb`, helper `gini`, JSON source line 236.

The function unconditionally executes `arr = arr - arr.min()`. Subtracting the minimum changes inequality relative to total capital. For positive capital it calculates the Gini of capital above the poorest agent's capital, rather than the Gini of capital itself.

A fresh run of the baseline reproduces the reported inputs and gives:

| Quantity | Recomputed value |
|---|---:|
| Surviving miners | 150 / 150 |
| Mean final capital | 3,590.972147 |
| Median final capital | 3,393.702835 |
| Minimum final capital | 1,566.218011 |
| Current function's Gini | 0.335498869 |
| Standard final-capital Gini | **0.189169588** |

For a simple check, `gini([100, 200])` returns 0.5; the standard result is 1/6. The report's **0.34 should be approximately 0.19**, and the interpretation of dispersion should be reconsidered.

**Correction:** Preserve nonnegative observations without translating them. Explicitly define or reject negative inputs. Recompute the Gini time series and correct the report. This helper is used for model metrics, not the feedback treatment-return calculation.

### 2. Figure 2b and Section 6.2 confuse filtered graph counts with event frequencies

**Location:** `notebook.ipynb`, Section 12.9, JSON source lines 4091–4097 and 4128.

The code computes activity counts, filters the DFG, discards the returned activity-count dictionary, and renders the filtered graph without explicit activity counts. The installed PM4Py renderer then estimates node frequencies from retained incoming edges plus start counts. Removing edges makes these estimates smaller than the actual frequency of the retained activity.

Every activity number quoted in Section 6.2 matches the sum of incoming edges in the exported filtered DFG. They are not event frequencies, even when restricting attention to the displayed activity labels.

| Activity | Report: top 10% | Actual: top 10% | Report: bottom 10% | Actual: bottom 10% |
|---|---:|---:|---:|---:|
| Win reward | 3,908 | **4,086** | 2,540 | **2,624** |
| Continue mining | 2,636 | **2,802** | 2,517 | **2,738** |
| Join pool | 901 | **939** | 883 | **962** |
| Leave pool | 871 | **928** | 897 | **955** |
| Pause mining | 696 | **737** | 691 | **737** |
| Losing streak | 258 | **270** | 891 | **943** |

These actual counts agree between the saved baseline CSV and a fresh baseline run. The general observation that operating-action frequencies are similar remains plausible, but the numbers and bars require correction.

**Correction:** Export activity frequencies directly from each cohort's event log. Keep DFG edge frequencies as a separate output. Pass explicit activity counts when rendering node labels, and use the direct event counts for Figure 2b.

### 3. Figure 2a does not consistently use end-of-day capital

**Locations:** `notebook.ipynb`, `run_mining_round` around JSON source lines 1194–1201; Section 12.11 around line 4299.

The model logs `Win reward` before `agent.update_after_profit(...)` applies the reward and energy cost. Section 12.11 then uses the last event per agent/day as its capital snapshot. If a miner has no later event, this remains pre-settlement capital. Miners with later imitation events can instead have post-settlement capital. The plotted cohorts therefore mix different within-day observation times.

A fresh baseline run instrumented with actual end-of-day agent capital found:

- 42,129 of 54,750 agent-day observations differed from the last logged capital.
- Mean absolute agent-day discrepancy: 10.56 capital units.
- Largest absolute agent-day discrepancy: 192.04 units.
- Largest daily cohort-mean discrepancy: 46.50 units for the top decile and 16.62 for the bottom decile.

The final-day cohort means remain correct because finalization adds terminal events after settlement. The report's final means of 6,151 and 1,938 are reproduced. This is a trajectory-measurement error, not evidence that the simulation's cash accounting or primary feedback return is incorrect.

**Correction:** Record a dedicated per-agent snapshot after daily settlement and compute Figure 2a from those snapshots. Do not infer daily financial state from whichever behavioral event happened last.

## Experimental design assessment

The online evaluation contains several appropriate safeguards:

- Assignment is randomized, balanced, and reversed in a complementary run without consuming simulator randomness.
- Controls retain the same baseline decision rules, learning, imitation, and capital policy; only access to adviser scores changes.
- Price, hardware-failure and block-lottery streams are separated from endogenous action randomness.
- Advice fitted after day `d` affects decisions from day `d + 1`; the fit uses only observed history.
- Agents continue across updates with their wealth and learned values intact.
- Validation and test seeds are distinct; each adviser variant has the same tuning budget.
- Complementary assignments are averaged within seed before bootstrap resampling. Interacting agents are not counted as independent replications.
- Return includes cash, hardware and prepaid orders; exited agents remain in cohort means.
- The report correctly distinguishes relative advantage from market-wide improvement, and does not claim a demonstrated benefit from sequence history or periodic refitting.

The selected discount of zero means the successful policy uses context-conditioned immediate observed rewards. Multistep transition planning is implemented and tested, but the reported result does not establish its benefit. Likewise, the separately tuned no-sequence comparison is not a matched-weight estimate of the isolated effect of sequence history. The report explicitly acknowledges these limits.

## Validation and reproducibility

`python -m unittest discover -s tests -v` completed successfully: **51 tests passed**. These cover accounting, assignment fairness, zero-weight equivalence, discovered policy behavior, censoring, and chronological fitting. Passing them does not detect the three descriptive-analysis issues above.

The baseline was rerun from the actual notebook definitions, using the repository runner's `load_model()`. Capital summary statistics reproduce the report. A separate instrumented run checked event-log capital against settled capital.

The complete online experiment was rerun from scratch: **48 validation runs and 40 held-out test markets**, taking approximately 245 seconds. The validation scores agree with every saved three-decimal candidate mean, and selection reproduces weight 6 / discount 0 for the full adviser and weight 2 / discount 0 for the no-sequence adviser. No saved selection cache was used.

| Feedback comparison | Fresh mean difference (percentage points) | Fresh 95% bootstrap interval |
|---|---:|---:|
| Full adviser minus concurrent controls: mean return | 41.855060 | 29.814651 to 54.387535 |
| Full adviser minus concurrent controls: median return | 12.090931 | 1.812507 to 27.490739 |
| Full adviser minus concurrent controls: survival | 11.250000 | 7.500000 to 15.312500 |
| Full adviser minus concurrent controls: maximum drawdown | -0.242203 | -0.460704 to 0.015387 |
| No-sequence adviser minus its controls: mean return | 46.226066 | 30.781444 to 63.918045 |
| Full minus no-sequence relative advantage | -4.371006 | -12.838855 to 3.292776 |
| Guided miners minus unguided market: mean return | 16.871700 | 10.559485 to 24.065797 |
| Concurrent controls minus unguided market: mean return | -24.983359 | -34.256020 to -16.800830 |
| Mixed market minus unguided market: mean return | -4.055830 | -8.890719 to 0.605779 |

The full adviser's seed-level advantage is positive on all eight seeds, ranging from 16.124171 to 71.401048 points. Mean returns are 90.718611% for guided miners, 48.863552% for concurrent controls, and 73.846911% for the unguided market. These reproduce the report's rounded values. All 12 successive post-warm-up average increments are positive, as stated in the report. All 384 feedback fits satisfy the chronological audit, and all 1,600 final agent records are retained.

At the start of review, `output_mas/online_feedback/` was absent in this checkout. The notebook contained saved feedback outputs, but the original detailed online result files and configuration manifest were unavailable. Fresh full-experiment results, source hashes, environment versions, and decision evidence are saved separately in `output_mas/code_review_feedback/`, preserving existing analysis outputs.

The installed environment differs from `requirements-experiment.txt`: Mesa 3.3.1, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1, Matplotlib 3.11.0 and PM4Py 2.7.22.4 were used for this review. The descriptive PM4Py dependency is not pinned in the installation cell. Archive the exact environment and original result tables with the paper; the entire output directory is currently gitignored.

No simulation or analysis source was modified during this review. Corrections above are recommendations, not applied changes.
