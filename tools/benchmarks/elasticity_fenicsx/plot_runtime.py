"""Reproduce the elasticity runtime/accuracy figure from its committed projection."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import validate_reports
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/'docs/performance/2026-11-06-elasticity-fenicsx-runtime.json'

def main():
    rows = json.loads(DATA.read_text())['reports']
    validate_reports(rows)
    for width in (320, 1440):
        plt.rcParams.update({'font.size': 9 if width == 320 else 12})
        fig, axes = plt.subplots(2 if width == 320 else 1, 1 if width == 320 else 2,
                                 figsize=(width/100, 6.6 if width == 320 else 4.8), dpi=100)
        for tool, label, color in [('eqiora', 'Eqiora', '#3759b8'), ('fenicsx', 'FEniCSx', '#b84f20')]:
            selected = sorted((r for r in rows if r['tool'] == tool), key=lambda r:r['cells_per_axis'])
            n = [r['cells_per_axis'] for r in selected]
            median = np.array([r['execution_measurement']['median_wall_seconds'] for r in selected])*1000
            low = np.array([r['execution_measurement']['min_wall_seconds'] for r in selected])*1000
            high = np.array([r['execution_measurement']['max_wall_seconds'] for r in selected])*1000
            axes[0].errorbar(n, median, yerr=[median-low,high-median], marker='o', color=color, label=label, capsize=3)
            axes[1].plot(n,[r['accuracy']['l2_error'] for r in selected], marker='o', color=color, label=label)
        for ax in axes:
            ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks([4,8,16,32],['4','8','16','32']);ax.set_xlabel('Cells per axis');ax.grid(alpha=.2)
        axes[0].set_ylabel('Ordinary execution (ms)');axes[0].set_title('Median and observed range');axes[0].legend()
        axes[1].set_ylabel('Displacement L2 error (m²)');axes[1].set_title('Independent analytic checks')
        fig.suptitle('2D Q1 elasticity · one shared-host campaign',fontsize=9 if width==320 else 15)
        fig.tight_layout()
        fig.savefig(ROOT/f'docs/performance/2026-11-06-elasticity-fenicsx-{width}.png')
        plt.close(fig)
if __name__ == '__main__':main()
