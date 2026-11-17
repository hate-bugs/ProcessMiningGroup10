"""Run section 13 without Jupyter/Graphviz; --write-notebook embeds real outputs."""
import argparse
import ast
import base64
import contextlib
import io
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


def load_model():
    notebook = json.loads((ROOT / "notebook.ipynb").read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    scope = types.ModuleType("mining_feedback_experiment")
    sys.modules[scope.__name__] = scope
    for text in code:
        if text.startswith((
            "from __future__ import annotations", "def gini(",
            "@dataclass\nclass MiningParams", "class AdaptiveMiningAgent(",
            "class AdaptiveMiningModel(",
        )):
            exec(compile(text, "<notebook definitions>", "exec"), scope.__dict__)
    # Read the parameter assignment from section 7 without running the baseline.
    configuration = next(text for text in code if "params = MiningParams(" in text)
    tree = ast.parse(configuration)
    assignment = next(
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "params" for target in node.targets)
    )
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "<notebook params>", "exec"), scope.__dict__)
    return notebook, scope


class Tee(io.StringIO):
    def write(self, value):
        sys.__stdout__.write(value)
        sys.__stdout__.flush()
        return super().write(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Small pipeline check; not paper results")
    parser.add_argument("--output", type=Path, help="New or empty export directory; existing results are preserved")
    parser.add_argument("--write-notebook", action="store_true", help="Embed executed section-13 outputs")
    args = parser.parse_args()
    if args.smoke and args.write_notebook:
        parser.error("Do not embed smoke-test outcomes as full notebook results")
    output = args.output or ROOT / "output_mas" / ("checks/feedback_smoke" if args.smoke else "current/feedback")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; use --output with a new directory")
    notebook, scope = load_model()
    scope.SMOKE_TEST = args.smoke
    scope.EXPERIMENT_OUTPUT = output
    outputs = []

    def display(value):
        text = value.to_string() if hasattr(value, "to_string") else str(value)
        data = {"text/plain": text}
        if hasattr(value, "to_html"):
            data["text/html"] = value.to_html()
        outputs.append({"output_type": "display_data", "data": data, "metadata": {}})
        sys.__stdout__.write(text + "\n")

    def show():
        for number in plt.get_fignums():
            buffer = io.BytesIO()
            plt.figure(number).savefig(buffer, format="png", dpi=140, bbox_inches="tight")
            outputs.append({"output_type": "display_data", "metadata": {}, "data": {
                "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "text/plain": "Feedback learning curve",
            }})
        plt.close("all")

    scope.display = display
    plt.show = show
    section_started = False
    counter = 1
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and source.startswith("## 13."):
            section_started = True
        if not section_started or cell["cell_type"] != "code":
            continue
        outputs = []
        capture = Tee()
        with contextlib.redirect_stdout(capture):
            exec(compile(source, "<notebook section 13>", "exec"), scope.__dict__)
        if capture.getvalue():
            outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": capture.getvalue()})
        cell["outputs"] = outputs
        cell["execution_count"] = counter
        counter += 1
    if args.write_notebook:
        (ROOT / "notebook.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("Saved actual experiment outputs in notebook.ipynb")


if __name__ == "__main__":
    main()
