# Semantic expectations

For canonical edge order 01,02,03,12,13,23, rotation moments are (0,0,0,6,0,0).
For two cells, order 01,02,03,12,13,14,23,24,34 gives (0,0,0,6,0,6,0,-6,0).
The gauge representatives are respectively
(3/2,-3/2,0,3,-3/2,3/2) and
(12/5,-12/5,0,6/5,-12/5,18/5,12/5,-18/5,0).

For canonical face order 012,013,023,123, radial fluxes are (0,0,0,12).
With faces 124,134,234 appended they are (0,0,0,12,12,-12,12).
The [derivation](../references/README.md) supplies action energies, derivatives and bounds.

The ordinary positive solve uses constant u=(2,3,4), whose line integrals are
(4,9,16,5,12,7) and face integrals (12,-12,12,36). Complex runs multiply these by
1+2i. Physical Field shape remains three while each entity owns one scalar moment.
Result coefficient dimensions are physical Field dimensions times length or area.
Same-shaped nodal reinterpretations, stale Space bindings and foreign Mesh execution reject.
Exact integer cochain identities have zero tolerance; floating checks use the stated bounds.
