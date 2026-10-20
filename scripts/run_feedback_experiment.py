"""Run notebook section 13 headlessly, without the Graphviz analysis cells.

From the repository root:
    .venv/Scripts/python.exe scripts/run_feedback_experiment.py
"""
import ast
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    notebook = json.loads((ROOT / "notebook.ipynb").read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    scope = types.ModuleType("mining_feedback_experiment")
    sys.modules[scope.__name__] = scope
    scope.display = lambda value: print(value.to_string() if hasattr(value, "to_string") else value)
    for text in code:
        if text.startswith((
            "from __future__ import annotations", "def gini(",
            "@dataclass\nclass MiningParams", "class AdaptiveMiningAgent(",
            "class AdaptiveMiningModel(",
        )):
            exec(compile(text, "<notebook definitions>", "exec"), scope.__dict__)
    # Execute only the params assignment from section 7, not the large baseline.
    configuration = next(text for text in code if text.startswith("# Configure simulation parameters"))
    tree = ast.parse(configuration)
    assignment = next(
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "params" for target in node.targets)
    )
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "<notebook params>", "exec"), scope.__dict__)
    output = ROOT / "output_mas" / "daily_feedback"
    output.mkdir(parents=True, exist_ok=True)

    def save_plot():
        plt.gcf().savefig(output / "held_out_comparison.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    plt.show = save_plot
    for prefix in ("from dataclasses import replace\nfrom pathlib import Path", "def feedback_metrics("):
        text = next(text for text in code if text.startswith(prefix))
        exec(compile(text, "<notebook feedback experiment>", "exec"), scope.__dict__)


if __name__ == "__main__":
    main()
