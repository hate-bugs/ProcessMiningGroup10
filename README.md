# Process Mining and Feedback in Adaptive Proof-of-Work Mining

## Install

Use Python 3.11 and run the commands from the repository root. Create an environment with `python -m venv .venv`, then activate it:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS / Linux: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.venv/Scripts/python.exe` in place of `python` after creating the environment. For process model exports, install Graphviz and make sure `dot` is on `PATH`.

## Run

Open [notebook.ipynb](notebook.ipynb), select the `.venv` kernel and run all cells for the descriptive analysis and feedback experiment. To run the analyses separately from the terminal:

```bash
python -m scripts.run_baseline
python -m scripts.run_descriptive_analysis --output output_mas/current/descriptive
python -m scripts.run_feedback_experiment
```

Add `--smoke` to the feedback command for a short check. It writes to `output_mas/checks/feedback_smoke/` by default.

## Outputs

| Run | Default output folder | Main files |
|---|---|---|
| Baseline script | `output_mas/current/baseline/` | `summary.json`, `capital_trajectory.csv`, `paper_activity_counts.csv`, `activity_counts.csv`, `baseline_figure2.png` and `.pdf` |
| Descriptive analysis | `output_mas/current/descriptive/` | `data/` for event logs and snapshots, `figures/` for PNG and PDF plots and CSV data, `process_models/` for PNG, PDF, DOT, JSON, PNML and BPMN exports |
| Feedback experiment | `output_mas/current/feedback/` | `parameter_search.csv`, `held_out_results.csv`, `learning_curve.csv`, `seed_effects.csv`, `online_learning_curve.png` and `.pdf`, and `evidence/` for decision records and discovered transitions |

Git ignores `output_mas/`. Give repeat terminal runs a new or empty directory with `--output`, for example `--output output_mas/runs/feedback_02`. Rerunning notebook cells can overwrite files at their default paths.

## Supplementary experiments

```bash
python -m scripts.run_disruption_experiment --experiment shutdown --workers 2
python -m scripts.run_disruption_experiment --experiment cost-shock --workers 2
```

Each run writes CSVs, figures and `ROBUSTNESS_REPORT.md` to a new folder under `output_mas/supplementary/`. Adding `--smoke` directs the shorter run to `output_mas/checks/`.
