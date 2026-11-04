# Component expression arguments

A component can borrow a typed expression without introducing a Field or
turning the expression into a compile-time Parameter:

```eqiora
component Poisson(
  support body: volume(ambient_dimension = 2),
  variable temperature: K on body,
  expression source: K / m ^ 2 on body
) {
  relation balance on body {
    -div(grad(temperature)) = source;
  }
}
```

Bind the argument in the caller with the other named arguments:

```eqiora
instance equation: Poisson(
  body = plate,
  temperature = temperature,
  source = -6 * amplitude * coordinate(0) / length ^ 3
);
```

Here `amplitude` has units K and `length` has units m. The expression is checked
on the exact support bound to `body`. An implicit `coordinate(0)` uses that
support. A named coordinate keeps its caller support; a coordinate from another
volume is rejected. The expression keeps its operation order and original
Parameter identities, including when forwarded through another component.
Names in the component body cannot capture names in the caller's argument.

The contract requires an explicit value type, units and support, with continuous
or static dependencies. Scalars, vectors and tensors use the existing type and
frame checks. Defaults,
clocked or event-local expression arguments, and expression requirements in
Model signatures are rejected. Dimension inference and units derived from
another signature argument remain separate work. Ordinary solver admission
still decides whether the resulting equations can execute with a chosen method.

Expression names occupy the shared signature/body namespace. Every required
expression must be bound once. A declaration has no initializer, Field role,
spatial regularity, or activation override.

## Compiler decision

The signature has a distinct expression category. It shares the existing
value-head syntax representation used by input/output declarations, but never
allocates a Field, Parameter, relation, or synthetic entity identity. The common
typed alias owner checks the declared type, exact support and dependency
activation. Expansion binds supports first, freezes the caller expression with
resolved lexical names and coordinates, and stores it through the existing
runtime scoped-value owner. Local aliases use that same runtime storage.

Canonical source identity assigns expression requirements their own category;
the existing Parameter and Field signature records are unchanged. Source package
releases retain source text and canonical declaration bytes. Expanded Models and
transactions contain only the existing expression nodes and original entities.
The source-category extension does not change an existing record or the Model
wire schema. Identity tests distinguish the new category and contract units while preserving
formatting equivalence. Artifact tests reopen compiled package and Model
artifacts through the ordinary readers.

The execution acceptance problem uses the existing 2D Cartesian Q1 method:
`temperature = amplitude * (x / length)^3`, with a zero left value, right outward
normal gradient `3 * amplitude / length`, and zero top/bottom normal gradients.
Its source is `-6 * amplitude * x / length^3` and its amplitude derivative is
`(x / length)^3`. Imported and hand-written equations are checked separately
against these analytic expressions and the existing Q1 convergence criterion.
