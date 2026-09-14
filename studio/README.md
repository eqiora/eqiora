# Eqiora Studio

Eqiora Studio is an accessible browser projection for exploring the shape and interaction model of canonical Eqiora models. It contains fixed, runtime-validated example projections; it is not a second model implementation.

The current surface demonstrates:

- source, semantic outline, relation, inspector, and source-linked diagnostic presentation;
- local `Parameter` value-edit preview and bounded revision navigation over the fixed example;
- workspace-only graph layout and keyboard-accessible commands; and
- one fixed CAD projection with Domain selection.

The browser does not compile or execute Eqiora models, construct canonical CAD identity, render generated Python, save files, or run solver and package workflows. Those operations require a future runtime that can use the public Eqiora facade without inheriting the retired GTK3 desktop dependency.

## Boundaries

```text
fixed example projection
      ↓ runtime-validated Studio DTO
React presentation and local interaction state
```

The bridge schema remains `eqiora.studio.bridge/v5`; it is independent of the canonical Model wire. Browser examples carry preview identities and never represent scientific execution or canonical artifact identity.

`src/application-registry.ts` owns the closed presentation registry. Applicability is derived from typed accepted state, never source-text or component-tree inspection. Layout and local value-edit history remain presentation state.

## Develop and verify

Use the repository-owned Node version and commands:

```bash
npm ci
npm run check
npm test
npm run build
npm run test:e2e
```
