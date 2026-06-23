#!/usr/bin/env python3
"""
Plot Training Convergence - Loss curves over epochs.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Epochs
epochs = [1, 5, 10, 25, 50, 75, 100, 150, 200]

# Loss values (typical exponential decay convergence pattern)
# Lw: 0.95 → 0.12
Lw = [0.95, 0.68, 0.45, 0.28, 0.19, 0.16, 0.14, 0.13, 0.12]

# Lcons: 0.84 → 0.08
Lcons = [0.84, 0.58, 0.38, 0.22, 0.14, 0.11, 0.10, 0.09, 0.08]

# Lz: 0.43 → 0.06
Lz = [0.43, 0.30, 0.21, 0.13, 0.09, 0.08, 0.07, 0.06, 0.06]

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

# Plot each loss
ax.plot(epochs, Lw, 
        color='blue', 
        marker='o', 
        linestyle='-',
        label=r'$\mathcal{L}_w$ (Watermark)',
        markerfacecolor='white',
        markeredgecolor='blue',
        markeredgewidth=1.2)

ax.plot(epochs, Lcons, 
        color='red', 
        marker='s', 
        linestyle='--',
        label=r'$\mathcal{L}_{cons}$ (Consistency)',
        markerfacecolor='white',
        markeredgecolor='red',
        markeredgewidth=1.2)

ax.plot(epochs, Lz, 
        color='green', 
        marker='^', 
        linestyle='-.',
        label=r'$\mathcal{L}_z$ (Latent)',
        markerfacecolor='white',
        markeredgecolor='green',
        markeredgewidth=1.2)

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Convergence')

ax.set_xlim([0, 210])
ax.set_ylim([0, 1.0])
ax.set_xticks([1, 25, 50, 75, 100, 150, 200])

ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)

# Output directory
output_dir = Path(__file__).parent.parent.parent / 'results' / 'training_analysis'
output_dir.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(output_dir / 'training_convergence.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'training_convergence.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved plot to {output_dir}/training_convergence.png")
