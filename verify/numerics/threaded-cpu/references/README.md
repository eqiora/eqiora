# Reference provenance

The Poisson evidence compares against the one-worker Eqiora reference CG path. Rayon
1.12.0 supplies the isolated bounded host thread pool; Eqiora owns row
partitionability, capability admission, solver policy, true-residual
acceptance, and execution evidence.

The MINRES/Jacobi values are independently derived from literal equations,
without using a serial solve as their numerical oracle. For
`A=[[4,1],[1,1]]`, direct multiplication gives `A*[1,1]=[5,2]`; leading
minors 4 and 3 establish SPD. Its inverse is `[[1,-1],[-1,4]]/3`, with
Frobenius norm `sqrt(19)/3<1.5`. For `A=[[4,3],[3,1]]`, multiplication gives
`A*[1,-2]=[-2,1]`; determinant -5 establishes indefinite inertia despite the
positive diagonal. Its inverse is `[[1,-3],[-3,4]]/-5`, with Frobenius norm
`sqrt(35)/5<1.2`. Thus the original absolute residual target `2^-40` bounds each
component error by `2^-39` in both systems through `x-x_exact=-A^-1*r`.
These targets and falsifiers are fixed before execution; no observed solver
output supplies an expected value or tolerance.

Both systems use `D=diag(4,1)` and `W=diag(1/2,1)`. Their symmetric congruences
are `[[1,1/2],[1/2,1]]` and `[[1,3/2],[3/2,1]]`; the latter has eigenvalues
`5/2,-1/2`. The test independently reapplies the original literal matrix, keeps
the original two-iteration ceiling and checks exact serial/threaded numerical
parity only as an additional reproducibility property. Normal and transposed
actions retain the requested orientation on one- and four-worker pools.
Separate nonsingular symmetric matrices `[[d,3],[3,1]]`, for `d=0,-1`, have
determinants -9,-10; their zero RHS cannot bypass the strictly positive Jacobi
diagonal requirement through the pool adapter.

- <https://docs.rs/rayon/1.12.0/rayon/struct.ThreadPoolBuilder.html>
- <https://docs.rs/rayon/1.12.0/rayon/struct.ThreadPool.html>
