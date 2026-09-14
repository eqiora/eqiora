# Conservation storage correspondence

A fixed-volume physical Law retains storage, accumulation, outward flux, and source identities.
This case checks thermal storage in three dimensions through source compilation, semantic
admission, and exact Model artifact replay. It does not claim a transient spatial solve.

The independent identity is d(c T)/dt = c dT/dt for a fixed Parameter c. Thermal storage
has dimension kg/(m s²), and accumulation and source have dimension kg/(m s³).
The retained physical flux -k grad(T) has dimension kg/s³. No numerical tolerance or
solution observation supplies these expectations.

A second consumer uses dimensionless q² storage. Its derivative is 2 q dq/dt.
The native falsifier substitutes q for q² while preserving that accumulation, the exact
balance roots, and all term dimensions. Local Relation construction still succeeds;
the independent semantic derivative check must reject the mismatched stored quantity.
This distinguishes storage correspondence from balance construction and dimensional typing.
A canceled-storage native falsifier checks d(q-q)/dt = 0 first, then changes the
stored Field from a continuous state to a variable. The zero derivative must not hide
the ineligible storage dependency.

The closed polynomial checker does not prove arbitrary constitutive functions, moving-domain
transport, weak-form equivalence, numerical stability, or solver suitability. Ordinary product
tests separately cover source/Python authoring and steady outward-flux boundary conversion.

Run `mise run pr -- --case language.conservation-storage-correspondence`.
