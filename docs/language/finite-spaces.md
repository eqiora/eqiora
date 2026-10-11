# Finite component spaces and maps

This page describes how finite-dimensional spaces, coordinates, and linear maps are written in
Eqiora source. It is part of the [language specification](core.md); the
[two-state example](finite-state.md) uses more of it than is implemented today.

The compiler currently supports single orthonormal spaces, integer, real, and complex
coordinates, real and complex maps, and `apply`, `compose`, `pair`, `transpose`, and `adjoint`
in finite affine models. Products of exactly two single spaces also support `tensor_product` and
`permute_factors(value, [1, 0])` (or the identity `[0, 1]`); on product maps, the permutation
applies to both the source and the target. Nested products, products mixing a space and a dual,
coordinate selectors, `identity`, and the rest of the two-state example are specified but not yet
implemented. The [capability matrix](../capability-matrix.md) has the current status.

## Declarations and types

```eqiora
space Levels = orthonormal(ground, excited);
space Channels = orthonormal(command, feedback);
space Joint = product(Levels, Channels);

variable amplitude: coordinates<complex<1>, Levels>;
variable channels: coordinates<V, Channels>;
parameter coupling: map<complex<J>, Levels, Levels> =
  linear_map(Levels, Levels, [[0 [J], 2 [J]], [2 [J], 0 [J]]]);
```

`space name = orthonormal(label, ...);` declares a finite, nonempty, ordered orthonormal basis.
The labels must be distinct, and the order in which they are written is the coordinate order.
A space is identified by its declaration: another space with the same labels is a different
space.

`space name = product(space, space, ...);` names the ordered tensor product of two or more
spaces. The name is shorthand for the product, not a new basis. Factor order and nesting are
kept, and changing them requires an explicit operation. The right factor varies fastest, so
`Joint` orders its basis as `(ground, command)`, `(ground, feedback)`, `(excited, command)`,
`(excited, feedback)`. A space can appear more than once as a factor. The size of a product is
checked before it is expanded.

| Type | Meaning |
|---|---|
| `coordinates<S, B>` | Coordinates with scalar type `S` in space `B` |
| `coordinates<S, dual<B>>` | Linear dual coordinates, not implicit conjugation |
| `map<S, A, B>` | Linear map from `A` to `B`, whose matrix coefficients have scalar type `S` |

For coordinates, `S` may also be `integer`: signed whole numbers without a physical dimension.
Maps always have real or complex coefficients; integer maps are not supported. For real and
complex values, `S` is a dimension or `complex<dimension>`. Applying a map multiplies the input
dimension by the map's coefficient dimension, so `map<J, Levels, Levels>` applied to
dimensionless coordinates gives coordinates in joules. Real values are promoted to complex where
needed; an imaginary part is never discarded.

An orthonormal declaration fixes the inner product of the space. Non-orthonormal metrics are not
supported yet. Finite-element bases are numerical choices made in the realization and are not
declared as spaces.

## Values and operations

`coordinates(B, [values])` explicitly constructs coordinates in `B`; `linear_map(A, B, rows)`
constructs a map from `A` to `B`. Map rows follow the target basis and columns the source basis.
The literals must have exactly the declared sizes. A nested array literal is only the way to
write the values; arrays never convert to coordinates implicitly.

For a single labeled space, `value.label` selects one coordinate. For a product space, use
`coordinate(value, [factor indexes])` with one zero-based index per factor. Both keep the
dimension and scalar domain of the value. An index out of range or with the wrong number of
entries is an error.

| Operation | Type and convention |
|---|---|
| `apply(M, x)` | Requires `x` in the exact source space of `M`; result is in its target |
| `compose(M, N)` | Apply `N` first, then `M`; the middle spaces must match exactly |
| `identity(B)` | Dimensionless real identity map on `B` |
| `inner(x, y)` | Same space; sum of `conj(x_j) * y_j` in the declared orthonormal basis |
| `pair(d, x)` | Linear evaluation of a dual coordinate on its exact primal space, without conjugation |
| `transpose(M)` | Map from the target dual to the source dual, without conjugation |
| `adjoint(M)` | Map from target to source using their admitted inner products; conjugate transpose here |
| `tensor_product(M, N)` | Map between the ordered products of the source and target factors |
| `permute_factors(x, permutation)` | Explicit coordinate permutation, retaining the reordered product type |

For coordinates, `tensor_product(x, y)` gives coordinates in the product space with components
`x_i * y_j`, ordered with the right factor varying fastest. Neither operand is conjugated.
`transpose` maps between dual spaces; it is not the same as the adjoint, even though both swap
rows and columns of the matrix.

Multiplication multiplies dimensions, and addition needs equal dimensions and the same space.
A spatial vector or plain array of the right length is never accepted in place of coordinates.
All sizes are fixed at compile time.

## Exact counts and signed changes

`counts<Species>` holds a nonnegative whole-number count for each label of a single space.
`counts(Species, [values])` checks the number of entries and the integer range; arrays and
floating-point values are never converted to counts. The space has no chemical formula, charge,
or conservation law attached to it.

```eqiora
space Species = orthonormal(A, B);

model Transfer() {
  clock tick = periodic(1[s], phase = 0[s]);
  state population: counts<Species> at tick;
  parameter change: coordinates<integer, Species> = coordinates(Species, [-1, 1]);
  initial {
    population = counts(Species, [2, 9007199254740993]);
  }
  relation transfer at tick {
    next(population) = pre(population) + change;
  }
}
```

Counts can only be added to integer coordinates in the same space. Every component is checked
before the state changes. In the example, the first two ticks give `[1, 9007199254740994]` and
`[0, 9007199254740995]`; the third would make a count negative, so it is rejected and the state
from the second tick is kept. A space with the same labels or size is still a different space, and reordering the labels
changes what each coordinate means.

This represents one signed stoichiometric column. Full map algebra on counts, conservation
metadata for reactions, stochastic sampling, and conversion from counts to amounts or
concentrations are not supported.

## Compact grammar

```text
space-decl = "space" name [notation] "=" space-definition ";"
space-definition = "orthonormal" "(" name {"," name} ")"
                 | "product" "(" space-ref "," space-ref {"," space-ref} ")"
space-ref = qualified-name | "dual" "<" space-ref ">"
coordinate-type = "coordinates" "<" scalar-type "," space-ref ">"
map-type = "map" "<" scalar-type "," space-ref "," space-ref ">"
```

A space declaration has no `on` or `at`: support and activation belong to the values that use
the space. Space declarations follow the usual rules for visibility, forward references,
duplicate names, and cycles, and product spaces are subject to the same size limits as arrays
and families.
