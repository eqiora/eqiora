#!/usr/bin/env python3
"""Render the rechecked original native field; no Eqiora import or solve."""
import argparse
import itertools
from pathlib import Path
import recheck_saved as saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    saved.main()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    values,energy,traction = saved.protocol.decode((saved.HERE/'evidence/native-recovery-1d010/primary.json').read_text())
    reference = saved.reference
    bounds = reference.budgets()
    points = np.array([[float(v) for v in node] for node in reference.NODES])
    displacement = np.array([float(v) for v in values]).reshape(27,3)
    index = {tuple(point):n for n,point in enumerate(points)}
    edges = set()
    for anchor in itertools.product((0.,1.),(0.,1.5),(0.,2.)):
        corners = [tuple(anchor[a]+(1.,1.5,2.)[a]*((corner>>a)&1) for a in range(3)) for corner in range(8)]
        for corner in range(8):
            for axis in range(3):
                edges.add(tuple(sorted((index[corners[corner]],index[corners[corner^(1<<axis)]]))))
    fig = plt.figure(figsize=(11.5,5.5),layout='constrained')
    field = fig.add_subplot(1,2,1,projection='3d')
    for i,j in sorted(edges):
        field.plot(*points[[i,j]].T,color='#aeb7be',linewidth=.7,alpha=.55)
        field.plot(*(points+displacement)[[i,j]].T,color='#087f8c',linewidth=1.1)
    field.quiver(*points.T,*displacement.T,color='#087f8c',linewidth=.65,arrow_length_ratio=.16)
    field.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='Saved native displacement, scale 1\n8 retained Hex cells · 27 vector nodes')
    field.set_box_aspect((2.5,3.8,4.7));field.view_init(elev=22,azim=-57)
    check = fig.add_subplot(1,2,2)
    displacement_error = max(abs(a-b) for a,b in zip(values,reference.U))
    residual = max(map(abs,reference.residual(bounds['K'],bounds['b'],values)))
    ratios = [float(displacement_error/bounds['E']),float(residual/bounds['tau']),
        float(abs(energy-reference.Q(261,32))/bounds['energy_analytic']),
        max(float(abs(v-target)/budget) for v,target,budget in zip(traction,(-reference.Q(27,2),-reference.Q(9,2),-reference.Q(9,4)),bounds['traction_analytic']))]
    check.barh(['Displacement','Free residual','Strain energy','Left stress integral'],ratios,color='#087f8c')
    check.axvline(1,color='#aa4444',linestyle='--',linewidth=1)
    check.set_xscale('log');check.set_xlim(1e-10,2)
    check.set(xlabel='Observed error / unchanged independent ceiling',title='Original field and public functionals\nEach ratio must stay below 1')
    check.grid(axis='x',alpha=.2);check.invert_yaxis()
    fig.suptitle('One affine Cartesian Hex Q1 elasticity patch',fontsize=15)
    check.set_ylim(4.5,-.6)
    check.text(.025,.035,f'Energy {float(energy):.12g} J · analytic 261/32 J\nLeft stress integral (−13.5, −4.5, −2.25) N\nInitial PDE runs: 2 · read-only recovery new runs: 0',fontsize=9,transform=check.transAxes)
    args.output.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output/'cartesian-hex-elasticity.png',dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
