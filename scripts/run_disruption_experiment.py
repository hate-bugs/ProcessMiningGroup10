"""Run versioned shutdown/cost-shock studies without retuning the adviser."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from uuid import uuid4
import shutil
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scripts.disruption_experiment import (cost_shock_protocol, disruption_protocol, export_results,
    manifest, run_job, scenario_values, validate_disruption_protocol)


def prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Refusing to overwrite historical results in {path}; choose a new --output directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experiment", choices=["shutdown", "cost-shock"], default="shutdown")
    parser.add_argument("--days", type=int)
    parser.add_argument("--intervention-day", type=int)
    parser.add_argument("--shock-end-day", type=int)
    parser.add_argument("--cost-multipliers", type=float, nargs="+", help="Include 1.0 as the matched baseline")
    parser.add_argument("--analyze-only", type=Path, help="Recompute analysis from a v2 run into a new version, without rerunning markets")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.analyze_only:
        if args.smoke or any(value is not None for value in [args.days, args.intervention_day, args.shock_end_day, args.cost_multipliers]):
            parser.error("Analysis-only mode uses the source run's configuration unchanged")
        source = args.analyze_only
        source_config_path = source / "experiment_config.json"
        source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
        if not source_config.get("completed") or source_config.get("schema_version") != 2:
            parser.error("Analysis-only mode requires a completed v2 simulation export")
        protocol = source_config["protocol"]
        try:
            output = prepare_output(args.output or source.parent / f"analysis_{uuid4().hex[:10]}")
        except ValueError as error:
            parser.error(str(error))
        for item in source.iterdir():
            if item.is_file():
                shutil.copy2(item, output / item.name)
        provenance = manifest(protocol)
        provenance.update(simulation_rerun=False, analysis_recomputed=True,
                          source_run=str(source.resolve()), source_simulation_manifest=source_config,
                          source_manifest_sha256=hashlib.sha256(source_config_path.read_bytes()).hexdigest())
        config = output / "experiment_config.json"
        config.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        import pandas as pd
        from scripts.robustness_analysis import export_robustness_analysis
        frames = {key: pd.read_csv(output / f"{key}.csv") for key in ["snapshots", "cohort_daily", "daily"]}
        export_robustness_analysis(frames, protocol, output)
        provenance["completed"] = True
        config.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"Recomputed analysis without rerunning simulations: {output / 'ROBUSTNESS_REPORT.md'}", flush=True)
        return
    protocol = cost_shock_protocol(args.smoke) if args.experiment == "cost-shock" else disruption_protocol(args.smoke)
    if args.experiment == "shutdown" and (args.shock_end_day is not None or args.cost_multipliers is not None):
        parser.error("Cost-shock options require --experiment cost-shock")
    for argument, key in [(args.days, "days"), (args.intervention_day, "shutdown_day"),
                          (args.shock_end_day, "shock_end_day"), (args.cost_multipliers, "cost_multipliers")]:
        if argument is not None:
            protocol[key] = argument
    try:
        validate_disruption_protocol(protocol)
        mode = "smoke" if args.smoke else "full"
        category = "checks" if args.smoke else "supplementary"
        study = "cost_shock" if args.experiment == "cost-shock" else "shutdown"
        default = Path("output_mas") / category / study / f"{mode}_{uuid4().hex[:10]}"
        output = prepare_output(args.output or default)
    except ValueError as error:
        parser.error(str(error))
    config = output / "experiment_config.json"
    config.write_text(json.dumps(manifest(protocol), indent=2), encoding="utf-8")
    jobs = [(protocol, seed, pct, family, swap) for pct in scenario_values(protocol)
            for seed in protocol["seeds"]
            for family, swap in [("all_control", False), ("process", False), ("process", True)]]
    results = []
    if args.workers == 1:
        iterator = map(run_job, jobs)
        for i, result in enumerate(iterator, 1):
            results.append(result)
            print(f"{i}/{len(jobs)} complete: {result['audit']['run_id']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, result in enumerate(pool.map(run_job, jobs), 1):
                results.append(result)
                print(f"{i}/{len(jobs)} complete: {result['audit']['run_id']}", flush=True)
    export_results(results, protocol, output)
    config.write_text(json.dumps(manifest(protocol, completed=True), indent=2), encoding="utf-8")
    print(f"Saved fresh results and report: {output / 'ROBUSTNESS_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
