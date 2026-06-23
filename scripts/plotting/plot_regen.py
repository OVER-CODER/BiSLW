#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from user - multiple methods comparison
timesteps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

methods = {
    'LaWa': [0.99, 0.99, 0.97, 0.96, 0.95, 0.94, 0.92, 0.92, 0.90, 0.89],
    'HiDDeN': [0.98, 0.95, 0.89, 0.85, 0.81, 0.74, 0.70, 0.66, 0.62, 0.58],
    'Stable Sig': [0.99, 0.98, 0.96, 0.93, 0.92, 0.91, 0.90, 0.89, 0.87, 0.86],
    'BiSLW (Ours)': [0.99, 0.99, 0.98, 0.96, 0.96, 0.95, 0.94, 0.93, 0.92, 0.92],
}

# Colors and styles for each method
styles = {
    'LaWa': {'color': 'purple', 'marker': 's', 'linestyle': '--'},
    'HiDDeN': {'color': 'red', 'marker': '^', 'linestyle': '-.'},
    'Stable Sig': {'color': 'green', 'marker': 'd', 'linestyle': ':'},
    'BiSLW (Ours)': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
}

# Formal style for research paper
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (6, 4),
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

fig, ax = plt.subplots(figsize=(6, 4))

# Plot each method
for method, accuracies in methods.items():
    accuracies_pct = [a * 100 for a in accuracies]
    style = styles[method]
    ax.plot(timesteps, accuracies_pct, 
            color=style['color'], 
            marker=style['marker'], 
            linestyle=style['linestyle'],
            label=method,
            markerfacecolor='white' if 'Ours' in method else style['color'],
            markeredgecolor=style['color'],
            markeredgewidth=1.2)

ax.set_xlabel('Regeneration Timestep')
ax.set_ylabel('Bit Accuracy (%)')
ax.set_xlim([25, 525])
ax.set_ylim([55, 102])
ax.set_xticks([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
ax.set_yticks([60, 70, 80, 90, 100])

ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)

output_dir = Path(__file__).parent.parent.parent / 'results' / 'regeneration_analysis'
output_dir.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(output_dir / 'regeneration_robustness.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'regeneration_robustness.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved plot to {output_dir}/regeneration_robustness.png")
