"""Reproduce the accuracy figure from the retained validated observations."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from accuracy import LEVELS, check_accuracy, exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, default=Path(__file__).with_name('results')/'correctness.json')
    parser.add_argument('--output-directory', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    inventory = json.loads(args.record.with_name('raw-sha256.json').read_text())
    for row in record['rows']:
        relative = row['committed_candidate_field']
        path = args.record.parent/record['committed_raw_directory']/relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != inventory[relative]['sha256']:
            raise ValueError('committed original candidate does not match its original digest')
        raw = json.loads(path.read_text())
        coordinates = np.array([[float.fromhex(x) for x in point] for point in raw['coordinates_float64_hex']])
        values = np.array([[float.fromhex(x) for x in vector] for vector in raw['values_float64_hex']])
        checked = check_accuracy(coordinates, values, row['cells_per_axis'])
        if checked != row['accuracy']:
            raise ValueError('recorded accuracy differs from independent reconstruction')
    references = []
    for n in LEVELS:
        points = np.array([(i/n, j/n) for i in range(n+1) for j in range(n+1)])
        references.append(check_accuracy(points, exact(points)[0], n))
    for mobile in (False, True):
        fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                                 figsize=(3.2, 6.4) if mobile else (8.8, 4.1), layout='constrained')
        for ax, key, title in zip(axes.flat, ('relative_l2_error', 'relative_h1_seminorm_error'),
                                 ('Displacement L2 error', 'Gradient H1 error'), strict=True):
            for tool, label, color, marker in (('eqiora', 'Eqiora wheel', '#2563eb', 's'),
                                                ('fenicsx', 'FEniCSx 0.11', '#c2410c', 'o')):
                rows = sorted((row for row in record['rows'] if row['tool'] == tool), key=lambda row: row['cells_per_axis'])
                ax.loglog(LEVELS, [row['accuracy'][key] for row in rows], color=color,
                          marker=marker, ms=7 if tool == 'eqiora' else 10, mfc=color if tool == 'eqiora' else 'none',
                          lw=1.3 if tool == 'eqiora' else 0, label=label)
            ax.loglog(LEVELS, [row[key] for row in references], ':', color='#4b5563', lw=1.2,
                      label='Exact-field Q1 interpolant')
            ax.set_xticks(LEVELS, labels=[str(n) for n in LEVELS]); ax.set_xlabel('Cells per axis')
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_ylabel('Relative continuous error'); ax.set_title(title, fontsize=12)
            ax.grid(True, which='both', alpha=.2)
        axes.flat[0].legend(fontsize=8 if mobile else 9, loc='lower left')
        fig.suptitle('2D elasticity: matched Q1 accuracy\nCorrectness only; no runtime claim', fontsize=12)
        path = args.output_directory/('accuracy-mobile.png' if mobile else 'accuracy.png')
        fig.savefig(path, dpi=160); plt.close(fig)
        print(path)


if __name__ == '__main__':
    main()
