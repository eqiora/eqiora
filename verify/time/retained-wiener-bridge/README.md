# Retained serial Wiener bridge

This case exercises the ordinary retained path used by the finite-time Run.
Its scientific reference is the Gaussian conditioning identity, derived without
using generated sample values: for unit Brownian motion, `X=W(1)` has mean zero
and variance one; `Y=W(1/2)-X/2` has mean zero, variance `1/4`, and zero covariance
with `X`. The test fixes 4096 sample identities before drawing. The five expected
moments are `[0, 1, 0, 1/4, 0]` for `[X, X², Y, Y², XY]`. Bounds are six standard
errors, using `Var(X²)=2`, `Var(Y²)=1/8`, and `Var(XY)=1/4` under this reference.
These finite-sample checks can expose variance, conditioning, or channel-addressing
errors; they do not prove independence of every hashed stream.

The retained arithmetic is exact rational addition of binary64 normal draws.
For every tested split, `(W(b)-W(a))+(W(c)-W(b))=W(c)-W(a)` exactly. Reverse-order
queries, discarded interval queries, and a finer dyadic grid preserve previously
admitted endpoints. Sample and channel identities select streams; occurrence
lineage does not change draws, and an explicit coupling group selects shared draws.
Invalid horizons, variances, and grid depths reject; zero covariance stays zero.

The sampler uses versioned SHA-256 addressing and native binary64 Box-Muller
transcendentals. Separately rounded child increments need not sum exactly.
Cross-platform libm identity, arbitrary partitions, strong/weak solver convergence,
distributed paths, and installed Python execution are outside this case.
