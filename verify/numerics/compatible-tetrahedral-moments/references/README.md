# Independent derivation

Use vertices 0=(0,0,0), 1=(2,0,0), 2=(0,3,0), 3=(0,0,4), and optionally
4=(2,3,4). Cells 0123 and 1234 have volumes 4 and 8 by determinants. Canonical edges
point from smaller to larger vertex; canonical faces use ascending vertex order.

For r=(-y,x,0), integrating along an edge a→b gives
`-(a_y+b_y)(b_x-a_x)/2 + (a_x+b_x)(b_y-a_y)/2`.
Its curl is (0,0,2), so curl energy is 4 times total volume: 16 or 48.
For f=(x,y,z), face abc has oriented area vector `(b-a)×(c-a)/2`;
its flux is the dot product of that vector with `(a+b+c)/3`.
Divergence is 3, so integrated divergence is 12 and 24, and energy is 36 or 108.
The outward incidences of shared face 123 are +1 and -1. Tangents (-2,3,0) and
(-2,0,4), and normal area vector (6,4,3), independently specify its traces.
Multiplication by 1+2i multiplies moments and derivatives by that factor and energies by 5.

The vertex-to-edge incidence G has -1 at the initial vertex and +1 at the terminal vertex.
Omit vertex 0 to remove constant potentials. On one tetrahedron its Gram matrix is
`4I-J` (size 3); the inverse is `(I+J)/4`. Thus `G^T r=(-6,6,0)` yields
potential `q=(-3/2,3/2,0)`. On two tetrahedra the graph is K5 without edge 04:

```
G^T G = [ 4 -1 -1 -1 ]       G^T r = (-12,12,0,0)
        [-1  4 -1 -1 ]       q = (-12/5,12/5,0,0)
        [-1 -1  4 -1 ]
        [-1 -1 -1  3 ]
```

The representative `r-Gq` satisfies `G^T(r-Gq)=0`. Since curl grad=0,
its curl–curl action is unchanged. This is an explicit Euclidean cochain representative,
not automatic gauge selection. Signed boundary-of-boundary cancellation proves CG=0 and
DC=0 over integers, without a numerical rank threshold.

Floating action/reconstruction comparisons use a fixed 4096 machine-epsilon relative bound
(with unit absolute floor), allowing 27 positive quadrature points and bounded affine
contractions. Iterative projection uses the existing 1e-9 relative solution bound and 1e-9
absolute constraint/action bound; requested solver residual tolerances are 1e-13 relative and
1e-14 absolute. These bounds were fixed before running this case, not fitted to output.

## Full polynomial action

For the first cell, barycentric gradients are
(-1/2,-1/3,-1/4), (1/2,0,0), (0,1/3,0), (0,0,1/4).
For the second cell, in vertex order 1,2,3,4 they are
(1/4,-1/6,-1/8), (-1/4,1/6,-1/8), (-1/4,-1/6,1/8), (1/4,1/6,1/8).
These follow by solving affine vertex interpolation, independently of element tabulation.
An oriented edge ij contributes `4*volume*(grad(lambda_i) cross grad(lambda_j))_z`
to the curl–curl action of r. Thus the complete canonical vectors are
(8/3,-8/3,0,8/3,0,0) and
(8/3,-8/3,0,8/3,-8/3,8/3,8/3,-8/3,0).

Each unit face flux basis has integrated divergence equal to its outward incidence.
Since div(f)=3, its div–div action entry is three times the sum of those incidences:
(-3,3,-3,3) or (-3,3,-3,0,3,-3,3). The shared-face action cancels exactly.
Complex action vectors multiply by 1+2i. These full vectors prevent an incorrect
operator with coincidentally correct scalar energy from passing the oracle.
