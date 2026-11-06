"""Full-vector mathematical comparison against frozen P2 reference levels.

This module has no Eqiora imports or artifact semantics. A future producer must
separately authenticate its actual mesh, entities and effective quadrature.
"""
from fractions import Fraction as Q
from exact import mesh
from refinement import assemble,enclosure_sqrt


def compare(reference, records):
    d=reference['dimension'];n=reference['N']
    nodes=mesh(d,n)[3]
    if len(records)!=len(nodes):raise ValueError('incomplete vertex/edge coefficient inventory')
    coordinate_index={tuple(x):i for i,x in enumerate(nodes)}
    candidate=[None]*len(nodes)
    for row in records:
        coordinates=tuple(map(Q,row['coordinates_m']))
        if coordinates not in coordinate_index:raise ValueError('unknown physical interpolation point')
        i=coordinate_index[coordinates]
        if candidate[i] is not None:raise ValueError('duplicate physical interpolation point')
        text=row['value_binary64_hex'];value=float.fromhex(text)
        if value.hex()!=text:raise ValueError('noncanonical binary64 value')
        candidate[i]=Q.from_float(value)
    K,M,_,_,essential,*_=assemble(d,n)
    if any(candidate[i] for i in essential):raise ValueError('nonzero essential trace')
    expected=[Q.from_float(float.fromhex(x)) for x in reference['coefficients_binary64_hex']]
    delta=[a-b for a,b in zip(candidate,expected)]
    output={}
    for name,matrix in [('L2',M),('H1_seminorm',K)]:
        squared=sum(delta[i]*sum(a*delta[j] for j,a in row.items()) for i,row in enumerate(matrix))
        _,upper=enclosure_sqrt(squared)
        r=reference[name]
        # Triangle inequality: exact candidate-to-retained-reference distance,
        # certified retained-reference-to-Galerkin distance, and sqrt enclosure.
        reference_uncertainty=Q(r['approximation_uncertainty_upper'])
        lower=Q(r['lower']); allowance=lower/Q(1000)
        combined=upper+reference_uncertainty
        if lower<=0 or combined>allowance:raise ValueError(f'{name} combined numeric allocation exceeded')
        output[name]={'combined_uncertainty_upper':str(combined),'allowance':str(allowance)}
    return output
