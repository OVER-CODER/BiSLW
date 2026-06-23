#!/usr/bin/env python3
"""
Plot Mask Radius vs Accuracy.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from mask radius study
mask_radius = [0.15, 0.20, 0.25, 0.30, 0.35]
acc = [0.89, 0.91, 0.93, 0.92, 0.90]
acc_pct = [a * 100 for a in acc]

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
    'lines.markersize': 8,
})

fig, ax = plt.subplots(figsize=(6, 4))

# Plot line with markers
ax.plot(mask_radius, acc_pct, 
        color='blue', 
        marker='o', 
        linestyle='-',
        markerfacecolor='white',
        markeredgecolor='blue',
        markeredgewidth=1.5,
        linewidth=2,
        markersize=8)

# Add data point labels
for x, y in zip(mask_radius, acc_pct):
    ax.annotate(f'{y:.0f}%', (x, y), textcoords="offset points", 
                xytext=(0, 8), ha='center', fontsize=9)

ax.set_xlabel('Mask Radius')
ax.set_ylabel('Bit Accuracy (%)')
ax.set_title('Watermark Accuracy vs Mask Radius')

ax.set_xlim([0.12, 0.38])
ax.set_ylim([87, 95])
ax.set_xticks(mask_radius)

ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Output directory
output_dir = Path(__file__).parent.parent.parent / 'results' / 'ablation_analysis'
output_dir.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(output_dir / 'mask_radius_accuracy.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'mask_radius_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved plot to {output_dir}/mask_radius_accuracy.png")
