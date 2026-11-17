"""Execute notebook sections 2–12 headlessly in a new working directory.

Requires requirements.txt and Graphviz's dot executable on PATH.
Original notebook outputs and previous generated files are left intact.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil

from scripts.run_feedback_experiment import ROOT
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New or empty working directory")
    args = parser.parse_args()
    if not shutil.which("dot"):
        parser.error("Install Graphviz and put its dot executable on PATH")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    notebook = json.loads((ROOT / "notebook.ipynb").read_text(encoding="utf-8"))
    os.chdir(output)
    # Notebook cells export every figure/model before displaying it.
    plt.show = lambda: plt.close("all")
    scope = {"__name__": "__main__", "display": print}
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"])
        # Route the notebook's canonical export root into this isolated run.
        source = source.replace("output_mas/current/descriptive", ".")
        if cell["cell_type"] == "markdown" and source.startswith("## 13."):
            break
        if cell["cell_type"] != "code" or "%pip" in source:
            continue
        print(f"Executing descriptive cell {index}", flush=True)
        tree = ast.parse(source)
        # Display the final expression as Jupyter does, without launching a UI.
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, f"<notebook cell {index}>", "exec"), scope)
        if last is not None:
            value = eval(compile(ast.Expression(last.value), f"<notebook cell {index}>", "eval"), scope)
            if value is not None:
                print(value)
        plt.close("all")
    print(f"Completed all descriptive cells. Exports: {output}")


if __name__ == "__main__":
    main()
