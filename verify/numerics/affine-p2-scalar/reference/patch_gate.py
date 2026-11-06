"""Exact independent patch acceptance; no Eqiora artifact semantics."""
from fractions import Fraction as Q
from exact import patch_value


def check(packet, coefficients, constrained_actions):
    values = list(map(Q,coefficients)); actions = list(map(Q,constrained_actions))
    nodes = packet['nodes_m']; essential = packet['essential']; free = packet['free']
    if len(values)!=len(nodes) or len(actions)!=len(essential):
        raise ValueError('incomplete coefficient or constrained-action inventory')
    expected = list(map(Q,packet['exact_coefficients']))
    if any(values[i]!=expected[i] for i in essential): raise ValueError('essential trace mismatch')
    if any(abs(a-b)>Q(packet['coefficient_allowance']) for a,b in zip(values,expected)):
        raise ValueError('coefficient allowance exceeded')
    residual = [sum(Q(a)*v for a,v in zip(row,values))-Q(f)-Q(g) for row,f,g in zip(packet['stiffness'],packet['volume_load'],packet['natural_load'])]
    if sum(residual[i]**2 for i in free)>Q(packet['epsilon'])**2: raise ValueError('free residual allowance exceeded')
    rho=Q(packet['rho'])
    if any(abs(a-residual[i])>rho for i,a in zip(essential,actions)): raise ValueError('constrained action allowance exceeded')
    exact_total=sum(map(Q,packet['exact_constrained_actions']))
    if abs(sum(actions)-exact_total)>Q(packet['reaction_sum_allowance']): raise ValueError('total constrained action allowance exceeded')
    return True
