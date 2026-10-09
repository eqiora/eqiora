# Independent derivative and orientation references

In a right-handed Cartesian frame let `F=(y²z,z²x,x²y)`, with coordinates measured in
metres and dimensionless unit basis vectors. Direct differentiation gives

- `curl(F)=(x²−2xz,y²−2xy,z²−2yz)` in m²;
- `curl(curl(F))=(−2z,−2x,−2y)` in m;
- `div(curl(F))=0`, `curl(grad(xyz))=0`; and
- the first component of `div(grad(F))` is `2z` in m.

The case samples three asymmetric integer points. These small integer polynomial
operations are exact in binary64; assertions require equality, not a fitted tolerance.
The functions are polynomials and therefore have commuting continuous mixed partials.
These identities do not assert anything about discrete differential complexes.

Binding the first basis Parameter to `(2,0,0)` changes `Fx` to `2y²z`.
At `(2,3,5)`, direct derivatives give curl components `cy=6 m²` and `cz=−35 m²`.
This distinguishes actual bindings from the Parameter's retained default value.
The ordinary Result observation at that point must give `cx=−16 m²`.

For the same constant vector `U=(2,3,5) m³`, the lower and upper x-face normals
are `−ex` and `+ex`. Thus the y-component of `n×U` is respectively `+5 m³`
and `−5 m³`. Boundary points bind every parent axis, including the exact normal
coordinate. Wrong face position, foreign coordinate factor and missing axes reject.
Replacing the explicit polynomial with an unknown vector Field reaches the missing
reconstruction rejection; it must never silently differentiate that Field as a constant.

