# Independent spectrum and evolution

Let X=[[0,1],[1,0]] and D=diag(3,0,-3). The two models use H=2X J
and H=(2X tensor I3 + I2 tensor D) J. X has eigenvectors (1,±1)/sqrt(2)
and eigenvalues ±1. Tensor factors retain right-factor-fastest order.
Thus the spectra are {-2,2} J and {-5,-2,-1,1,2,5} J.
The explicit 6×6 matrix has D on both diagonal blocks and 2I3 on both
cross blocks. Its initial first coordinate lies entirely in the D=3 subspace;
only coordinates 0 and 3 can become nonzero. In the two-level model the active
coordinates are 0 and 1 and the center energy is zero.

For an energy eigenvalue lambda and i hbar psi'=H psi, one midpoint step
multiplies the mode by (1-i h lambda/(2 hbar))/(1+i h lambda/(2 hbar)).
This has unit modulus and phase -2 atan(h lambda/(2 hbar)). With h=0.01 s,
T=0.2 s, N=20 and hbar in {1,2} J s, define theta±=2N atan(h(c±2)/(2 hbar)),
where c is 0 or 3 J. The final first amplitude is
(exp(-i theta+)+exp(-i theta-))/2 and the active other amplitude is
(exp(-i theta+)-exp(-i theta-))/2. All other amplitudes vanish. Their squared
magnitudes sum to one. Energy expectation stays c because midpoint is a rational
function of this Hermitian H and preserves its modal weights. This is the discrete
Cayley solution, not a claim that finite-step phase equals the continuous phase.

The fixed 1e-11 absolute margin for eigenvalues, amplitudes and these Observables
is much larger than binary64 roundoff accumulated across six components and
20 steps with Newton tolerances 1e-12 relative and 1e-14 absolute, but far below
the nonzero continuous-vs-Cayley phase error at these steps. It was chosen from
that separation, not fitted to implementation output. The selected norm contract
uses 1e-10 absolute tolerance and never rescales State values.

For the independent mass-matrix probe, M=[[2,1],[1,2]], F=[[1,-2],[2,-1]]
gives M^-1 F=[[0,-1],[1,0]] by direct multiplication. A term x^3-x vanishes
on all unit coordinate probes, yet must fail the structural affine proof.
An upper Hamiltonian coefficient 1e-300 J with zero lower conjugate is not
Hermitian, although division by hbar=1e100 J s erases its generator coefficient
in binary64. Raw complete-matrix checking must reject it before that scaling.
