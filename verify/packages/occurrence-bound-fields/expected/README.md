# Acceptance contract

The executable integration test owns the exact assertions. Acceptance
requires:

- exactly two flattened Fields, both owned by the root Model;
- zero Field-slot nodes, edges, display aliases, or wire payloads;
- both nested Relations referring to the exact bound Field IDs;
- exact volume support on the target Fields and Relations;
- complete support- and Field-binding occurrence provenance;
- canonical equality under declaration, binding, file, and dependency-alias
  permutations;
- target-sensitive source identity and Relation references; and
- typed rejection of missing, duplicate, unknown, wrong-kind, dimension,
  shape, frame, ambient-dimension, and exact-support failures before graph
  mutation, plus parser rejection of any non-`continuum` slot family.

Source identity uses the current typed-signature and named-binding namespace returned by
`LocalSourceIdentity::namespace`; its retired epochs are not accepted as current identities.
Changing an exact Field target changes that identity; historical encodings
are not retained as compatibility alternatives.
