"""Persistent figure and process-model exports used by the notebook."""
import json
from collections import Counter
from pathlib import Path


def incoming_edge_counts(dfg):
    """Counts used in the paper's activity panel, summed over retained incoming edges."""
    counts = Counter()
    for (_, target), frequency in dfg.items():
        counts[target] += int(frequency)
    return dict(counts)


def save_plot(figure, destination):
    """Save the displayed Matplotlib figure as PNG and vector PDF."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(destination.with_suffix(f".{extension}"), dpi=180, bbox_inches="tight")


def _save_graph(graph, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".dot").write_text(graph.source, encoding="utf-8")
    for extension in ("png", "pdf"):
        destination.with_suffix(f".{extension}").write_bytes(graph.pipe(format=extension))


def save_dfg(dfg, starts, ends, destination, activities_count=None):
    """Export a DFG image, vector PDF, Graphviz source and edge/count data."""
    from pm4py.visualization.dfg import visualizer

    graph = visualizer.apply(dfg, activities_count=activities_count, parameters={
        "start_activities": starts, "end_activities": ends,
    })
    _save_graph(graph, destination)
    data = {
        "edges": [{"source": source, "target": target, "frequency": int(count)}
                  for (source, target), count in sorted(dfg.items())],
        "start_activities": {key: int(value) for key, value in starts.items()},
        "end_activities": {key: int(value) for key, value in ends.items()},
        "activity_counts": (None if activities_count is None else
                            {key: int(value) for key, value in activities_count.items()}),
    }
    Path(destination).with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def save_petri_net(net, initial_marking, final_marking, destination):
    """Export visualizations and an editable PNML process model."""
    import pm4py
    from pm4py.visualization.petri_net import visualizer

    _save_graph(visualizer.apply(net, initial_marking, final_marking), destination)
    pm4py.write_pnml(net, initial_marking, final_marking, str(Path(destination).with_suffix(".pnml")))


def save_bpmn(model, destination):
    """Export visualizations and an editable BPMN process model."""
    import pm4py
    from pm4py.visualization.bpmn import visualizer

    _save_graph(visualizer.apply(model), destination)
    pm4py.write_bpmn(model, str(Path(destination).with_suffix(".bpmn")))
