"""Declared hierarchical ordering, essential vertices and physical input scaling."""
import numpy as np
from accuracy import U, H, P
from topology import mesh_arrays


def layout(data):
    coordinates, cells, _ = mesh_arrays(data)
    nv, nc = len(coordinates), len(cells)
    size = 3*nv+2*nc
    essential = np.unique(np.concatenate([np.asarray(data['boundaries'][name]).reshape(-1)
                                          for name in ('inlet', 'walls', 'cylinder')]))
    constrained = (2*essential[:, None]+np.arange(2)).reshape(-1)
    free = np.setdiff1d(np.arange(size), constrained)
    scales = np.concatenate((np.full(2*(nv+nc), U), np.full(nv, P)))[free]
    prescribed = np.zeros(size)
    inlet = np.unique(np.asarray(data['boundaries']['inlet']).reshape(-1))
    y = coordinates[inlet, 1]
    prescribed[2*inlet] = 4*U*y*(H-y)/(H*H)
    zero = np.unique(np.concatenate([np.asarray(data['boundaries'][name]).reshape(-1)
                                     for name in ('walls', 'cylinder')]))
    prescribed[2*zero] = 0
    return {'size': size, 'essential': essential, 'constrained': constrained, 'free': free,
            'scales': scales, 'prescribed': prescribed}


def canonical_fields(fields):
    return np.concatenate((np.asarray(fields['velocity_vertex']).reshape(-1),
                           np.asarray(fields['velocity_cell_bubble']).reshape(-1),
                           np.asarray(fields['pressure_vertex']).reshape(-1)))
