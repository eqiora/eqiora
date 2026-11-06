"""Plot the recorded 3D campaign; recheck its fields against the analytic oracle."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from accuracy import check_accuracy

ROOT = Path(__file__).resolve().parents[3]
RECORD = ROOT/'docs/performance/2026-11-06-poisson-3d-fenicsx-runtime.json'
OUTPUT = RECORD.with_suffix('.png')


def main():
    record = json.loads(RECORD.read_text())
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 6.8), layout='constrained')
    for tool, color, marker in (('eqiora', '#2563eb', 'o'), ('fenicsx', '#c2410c', 's')):
        rows = sorted((r for r in record['reports'] if r['tool'] == tool), key=lambda r:r['cells_per_axis'])
        cells = [r['cells_per_axis'] for r in rows]
        errors = [check_accuracy(r['coordinates'], r['values'], r['cells_per_axis'], 3)['continuous_l2'] for r in rows]
        times = [r['execution_measurement'] for r in rows]
        med = [r['median_wall_seconds']*1000 for r in times]
        lo = [(r['median_wall_seconds']-r['min_wall_seconds'])*1000 for r in times]
        hi = [(r['max_wall_seconds']-r['median_wall_seconds'])*1000 for r in times]
        label = 'Eqiora dev wheel' if tool == 'eqiora' else 'FEniCSx 0.11'
        axes[0].errorbar(cells, med, yerr=[lo, hi], color=color, marker=marker, capsize=4, label=label)
        axes[1].plot(cells, errors, color=color, marker=marker, fillstyle='none' if tool == 'fenicsx' else 'full', markersize=8, label=label)
    for ax in axes:
        ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xticks([4,8,16], ['4','8','16'])
        ax.set_xlabel('Cells per axis'); ax.grid(True, which='both', alpha=.2)
    axes[0].set_ylabel('Execution median (ms)'); axes[0].set_title('Seven calls per tool/mesh\nBars: observed range', fontsize=10); axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Continuous L2 error (m³ᐟ²)'); axes[1].set_title('Fields pass independent\nanalytic checks', fontsize=10)
    fig.suptitle('3D Q1 Poisson\nOne busy shared-host campaign\nWorker SMT sibling 100% busy', fontsize=10)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    print(OUTPUT)


if __name__ == '__main__':
    main()
