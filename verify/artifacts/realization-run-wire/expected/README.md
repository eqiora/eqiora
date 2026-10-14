# Current Realization wire specimen

`realization.json` is independently authored canonical JSON followed by one
text newline. Model/Domain/Field IDs are 1/2/3; the external Model digest is an
opaque repeated-byte reference. One physical-coordinate P1 Field on two uniform
cells uses two-point Gauss quadrature, an SPD system, CG with relative tolerance
1/2, absolute tolerance zero, eight iterations, and one host worker.

The test checks exact decode/re-encode bytes. The separate ordinary Poisson
path proves actual Model linkage. This specimen asserts no historical decode,
external Model availability, solver convergence, or physical result.
