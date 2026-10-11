# Reference authority and derivation

The primary authority is RFC 0071, sections “Three-generator Cartesian 3D
profile” and “Three-generator profile obligations.” Existing owners remain
authoritative for each pair, the current Model and Transaction artifact
domains, and `CartesianMeshEnvelopeV1`. The oracle does not read an
implementation table or a producer artifact.

Starting only from the six replayed boundary-physical Port identities and the
three replayed mesh-axis arrays, the test derives:

- positive periods from exact parent bounds and axes from boundary geometry;
- base entity indices from free-axis families and last-axis-fastest anchors;
- quotient anchors by reducing only fixed upper anchors modulo `(2, 3, 4)`;
- complete box orbits from independent fixed-cut toggles;
- ordered quotient closures from tensor-product local bits;
- face and cell incidence in `Z/2 x Z/3 x Z/4`;
- one positive packet per cell and axis; and
- seam geometry by lifting the lower neighbor through the exact parent period.

For `C = 2 * 3 * 4`, the RFC identities yield `8C = 192` quotient entities,
`27C = 648` closure-vertex references, `3C = 72` positive packets, and seam
counts `(C/2, C/3, C/4) = (12, 8, 6)`. Summing complete orbit membership gives
the independently generated box inventory
`(2*2+1)(2*3+1)(2*4+1) = 315`. These are exact structural integers, not fitted
values or producer output.

[Taylor--Green science](../../../../docs/verification/benchmark-roadmap.md#incompressible-and-low-mach-flow-candidates) and [numerical execution](../../../../docs/verification/gallery/taylor-green-3d.md#dependencies) remain
downstream non-authorities for this case. No published CFD result, equation,
solver, tolerance, or implementation scratch was consulted.
