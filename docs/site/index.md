# Model meaning once. Realize it many ways.

Eqiora is an open-source, meaning-first foundation for scientific modeling,
simulation, differentiation, and execution.

Its central boundary is simple:

> A model states typed mathematical relations. A realization chooses how those
> relations are discretized, solved, and executed.

That separation lets block diagrams, acausal physical networks, PDE fields,
hybrid dynamics, and reusable components share one canonical meaning without
making a numerical method or hardware backend part of the model.

[Get started](get-started.md){ .md-button .md-button--primary }
[See verified capabilities](capabilities.md){ .md-button }

## Explore Eqiora

<div class="eqiora-grid" markdown>

<a class="eqiora-card" href="concepts/">
  <strong>Understand the model</strong>
  <span>Relations, activations, connections, realization, and evidence.</span>
</a>

<a class="eqiora-card" href="python/">
  <strong>Author from Python</strong>
  <span>Use a typed client of the same canonical Rust implementation.</span>
</a>

<a class="eqiora-card" href="examples/">
  <strong>Follow a small path</strong>
  <span>Start with readable examples, then inspect falsifying evidence.</span>
</a>

<a class="eqiora-card" href="architecture/">
  <strong>Study the boundaries</strong>
  <span>See how semantics, lowering, realization, adapters, and evidence fit.</span>
</a>

</div>

!!! note "Release 0.1.2"
    Eqiora is pre-1.0 research software under active development. The
    [capability matrix](capabilities.md) and generated
    [verification guide](evidence/index.md) bound what is currently supported;
    this site does not widen those claims.

## One source of truth

This website is a curated projection, not a parallel specification. Detailed
contracts remain in the repository's
[architecture](https://github.com/nkiyohara/eqiora/blob/main/docs/architecture.md),
[RFCs](https://github.com/nkiyohara/eqiora/tree/main/rfcs),
[capability matrix](https://github.com/nkiyohara/eqiora/blob/main/docs/capability-matrix.md),
and validated [`verify/` manifests](https://github.com/nkiyohara/eqiora/tree/main/verify).
