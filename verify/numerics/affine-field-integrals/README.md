# Affine Field integrals

The ordinary compiled Model, authenticated Mesh, resolved P1 Plan and accepted Result
evaluate typed volume, boundary-trace and parent-outward flux Observables. The triangle
has vertices (0,0), (4,0), (0,3) metres. Three cells meet at the free interior vertex
(4/3,1); only exterior vertices receive essential boundary data. The field
T=300+2*x-3*y K is harmonic and belongs exactly to P1, so the independent continuum
solution also fixes the discrete solution without discretization error.

With conductivity 2 W/(m K), the heat flux is (-4,6) W/m². The base, diagonal and
left edges have lengths 4,5,3 m and parent-outward unit normals (0,-1), (3/5,4/5),
(-1,0). Their integrated fluxes are respectively -24,12,12 W/m. Their sum is zero.
The slanted edge tests both vector components and its physical length: replacing its
normal by an axis direction or using reference length cannot preserve the expected value.

On the diagonal the endpoint temperatures are 308 and 291 K; its trace integral is
5*(308+291)/2 = 1497.5 K m. The triangle area is 6 m² and its three vertex
temperatures sum to 899 K, giving volume integral 6*899/3 = 1798 K m².
The absolute tolerance is 1e-9 in each stated SI unit, fixed before execution for
binary64 reference solution, interpolation and summation. Relative tolerance is zero.
Expected values are never obtained from the assembled operator or reconstructed field.

Two-point-per-axis simplex Duffy quadrature integrates these affine fields exactly.
The test supplies explicit rules and compares the shared automatic rule selection to
them. A hypercube rule on the same one-dimensional edge must reject at reference-cell
admission. Exact output units, Result occurrence and retained quadrature are asserted.
Cyclic permutations of each cell's vertices must preserve the values. Result replay
repeats the observations; querying leaves the accepted bytes unchanged. Replay against
a different interior vertex position, with the same cell count, must reject.

Provider bytes are an authenticated bounded fixture; this case does not execute Gmsh.
Three-dimensional, curved, partial-facet, complex, transient and derivative observations
are outside the claim. Installed Python additionally compares the admitted P1
two-region partition and Q1 single-region paths over a 4-by-3 rectangle in
`bindings/python/tests/test_spatial_integrals.py`.

Run `cargo run -p eqiora-verify -- run --case numerics.affine-field-integrals`.
