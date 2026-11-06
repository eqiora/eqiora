"""Render actual cold fields only after the complete committed reference gate passes."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.tri as mtri
import numpy as np
from recheck_saved import recheck


def sampled_velocity(data, vertex_values, bubbles, subdivisions=4):
    lattice = [(i, j) for i in range(subdivisions+1) for j in range(subdivisions+1-i)]
    index = {point: i for i, point in enumerate(lattice)}
    bary = np.array([(1-(i+j)/subdivisions, i/subdivisions, j/subdivisions) for i, j in lattice])
    small_triangles = []
    for i in range(subdivisions):
        for j in range(subdivisions-i):
            small_triangles.append((index[i, j], index[i+1, j], index[i, j+1]))
            if i+j <= subdivisions-2:
                small_triangles.append((index[i+1, j], index[i+1, j+1], index[i, j+1]))
    cells, coordinates = np.asarray(data['cells']), np.asarray(data['coordinates'])
    points = np.einsum('qv,cva->cqa', bary, coordinates[cells])
    velocity = np.einsum('qv,cva->cqa', bary, vertex_values[cells])
    velocity += (27*np.prod(bary, axis=1))[None, :, None]*bubbles[:, None, :]
    triangles = (np.array(small_triangles)[None, :, :]
                 +len(lattice)*np.arange(len(cells))[:, None, None]).reshape(-1, 3)
    return points.reshape(-1, 2), velocity.reshape(-1, 2), triangles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, default=Path(__file__).with_name('results')/'correctness.json')
    parser.add_argument('--output-directory', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    record, reference, coordinates, velocity, associated, accuracy = recheck(args.record)
    triangles = np.asarray(reference.data['cells'])
    pressure = associated['pressure_vertex']['values']
    sample_points, sample_values, sample_triangles = sampled_velocity(
        reference.data, velocity, associated['velocity_cell_bubble']['values'])
    pressure_mesh = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1], triangles)
    velocity_mesh = mtri.Triangulation(sample_points[:, 0], sample_points[:, 1], sample_triangles)
    edges = coordinates[np.asarray(reference.data['boundaries']['cylinder'])]
    names = ('velocity', 'pressure', 'force_x', 'force_y')
    ratios = [accuracy['agreement'][name]/reference.manifest['metrics'][name]['tolerance'] for name in names]
    for mobile in (False, True):
        fig, axes = plt.subplots(3, 1, figsize=(3.2, 7.9) if mobile else (9.2, 6.8),
                                 gridspec_kw={'height_ratios': [1, 1, .8]}, layout='constrained')
        for ax, mesh, values, label, palette in (
                (axes[0], pressure_mesh, pressure, 'Pressure [Pa]', 'viridis'),
                (axes[1], velocity_mesh, np.linalg.norm(sample_values, axis=1), 'Speed [m/s]', 'magma')):
            colors = ax.tripcolor(mesh, values, shading='gouraud', cmap=palette)
            ax.add_collection(LineCollection(edges, colors='#172033', linewidths=.6))
            ax.set_aspect('equal'); ax.set_ylim(0, .41)
            ax.set_xlim((0, .55) if mobile else (0, 2.2))
            ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
            ax.set_title(label, fontsize=11)
            fig.colorbar(colors, ax=ax, pad=.02, fraction=.05)
        ax = axes[2]
        ax.bar(range(4), ratios, color='#0f766e', width=.55)
        ax.set_yscale('log'); ax.set_ylim(1e-8, 3)
        ax.axhline(1., color='#475569', ls='--', lw=1., label='Frozen acceptance gate')
        ax.set_xticks(range(4), labels=(['u\nRMS', 'p\nRMS', 'x\nforce', 'y\nforce'] if mobile else ['u RMS', 'p RMS', 'x force', 'y force']))
        ax.set_ylabel('Difference / allowance', fontsize=10)
        ax.grid(True, axis='y', which='major', alpha=.2); ax.legend(fontsize=8, loc='upper right')
        for i, ratio in enumerate(ratios):
            ax.text(i, ratio*1.6, f'{ratio:.1e}', ha='center', fontsize=8)
        title = ('Steady Stokes cylinder\nEqiora cold fields — near cylinder\nCorrectness only\nNo continuum/runtime claim' if mobile else
                 '2D steady Stokes: Eqiora cold fields and FEniCSx agreement\nOne shared affine mesh; no continuum or runtime claim')
        fig.suptitle(title, fontsize=10.5 if mobile else 14)
        output = args.output_directory/('single-mesh-mobile.png' if mobile else 'single-mesh.png')
        fig.savefig(output, dpi=160); plt.close(fig)
        print(output)


if __name__ == '__main__':
    main()
