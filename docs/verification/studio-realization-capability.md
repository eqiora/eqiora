# Studio spatial capability status

Studio has no spatial execution or result-visualization workflow at present. Its earlier
scalar-elliptic Realization, exact-cylinder flow, structural, unstructured-field, and
application-specific Plan/Run adapters were removed when all clients moved to the common
Model-first lifecycle.

Studio currently presents a fixed example with semantic inspection, local Parameter
edit previews, revision navigation, and a CAD projection. Source compile/check,
CAD authoring, and packaged model execution need a runtime connection.

A future spatial workflow in Studio must use the same path as every other client: caller-owned
Geometry, typed Mesh and numerical settings, the root `resolve(model, ...)`, and the common Plan,
State, Run, Result, and typed observations. It must not bring back application-specific intents,
specialized resolvers, a second source of Geometry, or a Studio-only result transport. Such a
workflow needs a new verification case once a real consumer of the common output exists.

Current Studio verification is listed under the remaining Studio cases in the
[capability matrix](../capability-matrix.md) and the generated evidence catalog; the removed
field-view cases are not evidence for the current code.
