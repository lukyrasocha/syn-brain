import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Config
RESULTS_FOLDER = "data/metrics"
FIGURE_FOLDER = "figures/metrics"
FIGURE_PATH = os.path.join(FIGURE_FOLDER, "metrics.png")

os.makedirs(FIGURE_FOLDER, exist_ok=True)

# load json
metrics_per_run = {}
for file in os.listdir(RESULTS_FOLDER):
    if file.endswith(".json") and file.startswith("metrics_"):
        path = os.path.join(RESULTS_FOLDER, file)
        with open(path, 'r') as f:
            data = json.load(f)
        run_name = os.path.splitext(file)[0].replace("metrics_", "")
        metrics = {}
        for metric in ["fid", "kid", "clip"]:
            value = data.get(metric)
            if isinstance(value, (int, float)):
                metrics[metric] = value
        metrics_per_run[run_name] = metrics

# Organize by metrics
runs = sorted(metrics_per_run.keys())
metrics = ["fid", "kid", "clip"]
metric_values = {m: [] for m in metrics}

for m in metrics:
    for run in runs:
        value = metrics_per_run[run].get(m)
        metric_values[m].append(value if value is not None else np.nan)


colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
run_colors = {run: colors[i] for i, run in enumerate(runs)}

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(14, 7), sharex=True)
fig.subplots_adjust(bottom=0.18)
bars_for_legend = []

for idx, m in enumerate(metrics):
    ax = axs[idx]
    y = metric_values[m]
    bars = []
    for i, run in enumerate(runs):
        bar = ax.bar(i, y[i], color=run_colors[run])
        if idx == 0:
            bars.append(bar[0])
    if idx == 0:
        bars_for_legend = bars
    ax.set_ylabel(m.upper())
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax.set_xticks([])

# Add the legend using bar handles
fig.legend(bars_for_legend, runs, loc='lower center', ncol=len(runs))#, bbox_to_anchor=(0.5, 0.05))

plt.tight_layout()
plt.savefig(FIGURE_PATH)
plt.close()

print(f"Subplots saved to: {FIGURE_PATH}")