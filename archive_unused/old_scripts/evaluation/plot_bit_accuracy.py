#!/usr/bin/env python3
"""
Plot Bit Length vs Accuracy trade-off.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from bit length study
bits = [32, 48, 64, 96, 128]
acc = [0.98, 0.94, 0.93, 0.91, 0.87]
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
ax.plot(bits, acc_pct, 
        color='blue', 
        marker='o', 
        linestyle='-',
        markerfacecolor='white',
        markeredgecolor='blue',
        markeredgewidth=1.5,
        linewidth=2,
        markersize=8)

# Add data point labels
for x, y in zip(bits, acc_pct):
    ax.annotate(f'{y:.0f}%', (x, y), textcoords="offset points", 
                xytext=(0, 8), ha='center', fontsize=9)

ax.set_xlabel('Bit Length')
ax.set_ylabel('Bit Accuracy (%)')
ax.set_title('Watermark Accuracy vs Capacity')

ax.set_xlim([20, 140])
ax.set_ylim([84, 100])
ax.set_xticks(bits)

ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Output directory
output_dir = Path(__file__).parent.parent.parent / 'results' / 'bit_length_analysis'
output_dir.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(output_dir / 'bit_accuracy_tradeoff.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'bit_accuracy_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved plot to {output_dir}/bit_accuracy_tradeoff.png")
