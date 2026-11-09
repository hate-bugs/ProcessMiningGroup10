"""Run the supplementary permanent-shutdown sweep without rerunning tuning."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from disruption_experiment import disruption_protocol, export_results, manifest, run_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    protocol = disruption_protocol(args.smoke)
    output = args.output or Path("output_mas/disruption_smoke" if args.smoke else "output_mas/disruption_sweep")
    output.mkdir(parents=True, exist_ok=True)
    config = output / "experiment_config.json"
    config.write_text(json.dumps(manifest(protocol), indent=2), encoding="utf-8")
    jobs = [(protocol, seed, pct, family, swap) for pct in protocol["severities"]
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
    print(f"Saved fresh results and report: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
