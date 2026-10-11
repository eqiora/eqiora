# Eqiora Studio

Eqiora Studio is an accessible browser interface for exploring how Eqiora models are structured and how people will work with them. It displays fixed example data, validated at runtime, and does not implement models itself.

It currently shows:

- source, semantic outline, relation, inspector, and source-linked diagnostic presentation;
- local `Parameter` value-edit preview and bounded revision navigation over the fixed example;
- workspace-only graph layout and keyboard-accessible commands; and
- one fixed CAD projection with Domain selection.

The browser does not compile or execute Eqiora models, construct canonical CAD identity, render generated Python, save files, or run solver and package workflows. Those operations require a future runtime that can use the public Eqiora facade without inheriting the retired GTK3 desktop dependency.

## Architecture

```text
fixed example projection
      ↓ runtime-validated Studio DTO
React presentation and local interaction state
```

The bridge schema is `eqiora.studio.bridge/v5`, separate from the Model wire format. Example identities exist only for the preview and never stand for a real execution or artifact.

`src/application-registry.ts` holds the fixed list of views. Which views apply is decided from typed state, never by inspecting source text or the component tree. Layout and the local value-edit history are presentation state only.

## Develop and verify

Use the repository-owned Node version and commands:

```bash
npm ci
npm run check
npm test
npm run build
npm run test:e2e
```
