# Compatible tetrahedral moments

This serial-host case first executes the ordinary source Model → resolved Plan → Run → Result
path and Plan/Result replay for real and complex lowest-order tetrahedral edge and face moments.
A positive mass term gives that constant-field solve a unique solution. The case then assembles
pure curl–curl or div–div forms on one and two non-unit tetrahedra through the same source
compiler, mapped basis, signed region map and COO/CSR owners.

Independent polynomial line/area integrals check coefficient meaning, curl/divergence energy,
physical reconstruction and shared tangential/normal traces under positive cell permutations.
Plan gradient modes are consumed by the existing linear solver to impose an explicitly chosen
Euclidean coefficient gauge. Exact integer cochain identities are reported separately from
floating-point quadrature and solver checks.

See [derivation](references/README.md), [expected projections](expected/README.md) and
[models](models/README.md). The manifest selects the executable test; no source hash is an oracle.

This proves the stated affine lowest-order profile only. It does not launch Gmsh, select an
automatic PDE gauge, prove all topological nullspaces, solve Maxwell spectra, prescribe nonzero
moment boundary data, or establish convergence, general conservation, positivity or stability.
The coefficient gauge is not an L2/Coulomb gauge.

## Derivative and trace boundary

Continuum regularity and a numerical basis are separate contracts. For this static linear
binding the admitted products are:

| Space | Global weak products used by assembly | Conforming interface trace |
| --- | --- | --- |
| Lowest tetrahedral edge | value and curl, including mass and curl–curl weak pairings | tangential component |
| Lowest tetrahedral face | value and divergence, including mass and div–div weak pairings | normal component |

A physical basis tabulation supplies element-local first gradients, curl and divergence.
Those local polynomials do not grant a globally H1 field, arbitrary classical second
derivatives, or interchangeable weak and pointwise products. Full-gradient/symmetric-gradient
pairings on moment spaces reject at binding. The current moment path admits homogeneous
natural laws only; essential/full traces and nonzero natural data reject. Transient and
nonlinear moment assembly remain outside this profile.

For the coordinate-derivative work tracked in #985/#995 and the smooth representation in
#1005, the common request is a Field on an exact support with dimensioned differential
meaning. A future smooth trial representation may admit classical coordinate derivatives
without a Mesh; this case instead admits the stated weak products on a retained Mesh.
Neither representation grants the other's derivative capability, numerical basis or error
bound. This case supplies no evidence for execution of the smooth representation.
