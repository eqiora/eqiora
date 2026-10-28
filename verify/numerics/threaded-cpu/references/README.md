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

## Complex solver and ordinary-Run witnesses

Every advertised complex Reference tuple uses an independently specified
property witness, with Identity/Jacobi selected only where advertised. For
`x=[1+i,2-i]`, the HPD matrix `[[4,i],[-i,1]]` gives
`b=[5+6i,3-2i]`; its determinant is 3 and inverse Frobenius norm is
`sqrt(19)/3<1.5`. Replacing off-diagonal `i` by `3i` gives a Hermitian
indefinite matrix with determinant -5, `b=[7+10i,5-4i]`, and inverse norm
`sqrt(35)/5<1.2`. Both equal their conjugate transpose.

For `x=[1+2i,-2+i]`, the General matrix `[[2+i,1-2i],[3+i,4-i]]`
gives `A*x=[10i,-6+13i]` and `A^H*x=[-1+8i,-12+6i]`. Its determinant
is `4+7i` and inverse Frobenius norm squared is `37/65<1`.
The complex-symmetric, non-Hermitian diagonal matrix `diag(2+i,4-i)` gives
`A*x=[5i,-7+6i]` and `A^H*x=[4+3i,-9+2i]`, with inverse norm squared
`1/5+1/17<1`. Conjugate transpose preserves each inverse Frobenius norm.
Thus the fixed absolute residual target `2^-40` bounds component error by
`2^-39` for every normal/adjoint tuple; the fixed iteration cap is 16.
No expectation comes from a serial or threaded solver output.

The ordinary Model `(1+2i)*z=(3+i)*p` has `z=p-ip` and observable
`|z|²+p=2p²+p`, hence `z=3-3i` and observable 21 at `p=3`.
The real Model `2*z=2*p` has `z=p` and observable `z²+4*p`, hence
`z=3` and observable 21 at `p=3`. Both Models retain relative/absolute
linear tolerances `1e-13`/`1e-15`, eight iterations, Field error below `2^-39`
and Observable error below `1e-10`. Exact/manual and all three automatic
objectives use the same independently fixed equations and expectations with
one/four workers. The positive facade advertises only the admitted
BiCGSTAB/Identity/Reproducible tuple, so selecting that exact tuple follows
from capability admission, not observed output. These tests check ordinary
execution and exact Plan/Result reopening with no reopening solver calls;
invalid typed adapters reject before numerical work. Threaded costs are
explicitly unknown because this case derives neither threaded workspace
ownership nor execution-overhead work bounds.
A separate real `2*z=6` control admits a large serial BiCGSTAB numeric-payload
cap before the same cap is rejected as unproved for Rayon; this reaches the
execution-specific resource boundary rather than an unknown complex bound.
