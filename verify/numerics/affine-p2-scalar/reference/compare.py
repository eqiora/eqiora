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
    K,M,F,U,essential,detB,metric,_=assemble(d,n)
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
        reference_norm_width=Q(r['upper'])-lower-2*reference_uncertainty
        def quadratic(vector):
            return sum(candidate[i]*sum(a*candidate[j] for j,a in row.items()) for i,row in enumerate(vector))
        exact=detB/Q(30**d) if name=='L2' else detB*sum(metric[i][i] for i in range(d))/Q(3*30**(d-1))
        cross=U if name=='L2' else F
        error_squared=exact+quadratic(matrix)-2*sum(a*b for a,b in zip(candidate,cross))
        error_lower,error_upper=enclosure_sqrt(error_squared)
        combined=upper+reference_uncertainty+reference_norm_width+(error_upper-error_lower)
        if lower<=0 or combined>allowance:raise ValueError(f'{name} combined numeric allocation exceeded')
        if error_lower<lower-allowance or error_upper>Q(r['upper'])+allowance:
            raise ValueError(f'{name} finite-level error interval exceeded')
        output[name]={'combined_uncertainty_upper':str(combined),'allowance':str(allowance),
                      'error_lower':str(error_lower),'error_upper':str(error_upper)}
    return output


def finite_rate_bounds(coarse,fine):
    """Bounds on 2**rate; logs are unnecessary for the acceptance decision."""
    a,b=Q(coarse['lower']),Q(coarse['upper'])
    c,e=Q(fine['lower']),Q(fine['upper'])
    t=a/Q(1000);s=c/Q(1000)
    if a<=0 or c<=0:raise ValueError('nonpositive frozen error lower bound')
    return (a-t)/(e+s),(b+t)/(c-s)


def check_rate_interval(coarse_candidate,fine_candidate,coarse_reference,fine_reference):
    low,high=finite_rate_bounds(coarse_reference,fine_reference)
    a,b=Q(coarse_candidate['error_lower']),Q(coarse_candidate['error_upper'])
    c,e=Q(fine_candidate['error_lower']),Q(fine_candidate['error_upper'])
    if min(a,c)<=0 or a/e<low or b/c>high:raise ValueError('finite-level rate interval exceeded')
    return {'two_to_rate_lower':str(a/e),'two_to_rate_upper':str(b/c),
            'frozen_two_to_rate_lower':str(low),'frozen_two_to_rate_upper':str(high)}


def compare_levels(references,records_by_n):
    if sorted(records_by_n)!=[2,4,8] or sorted(r['N'] for r in references)!=[2,4,8]:
        raise ValueError('complete N2/N4/N8 level set required')
    if len({r['dimension'] for r in references})!=1:raise ValueError('mixed dimension level set')
    refs={r['N']:r for r in references}
    levels={n:compare(refs[n],records_by_n[n]) for n in (2,4,8)}
    rates={}
    for n in (2,4):
        rates[n]={name:check_rate_interval(levels[n][name],levels[2*n][name],refs[n][name],refs[2*n][name]) for name in ('L2','H1_seminorm')}
    return {'levels':levels,'rates':rates}
