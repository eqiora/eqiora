# AI-authored platform strategy

- Status: Accepted
- Revised: 2026-08-26
- Related: [high-risk and parallel development](vertical-slice-development.md),
  [local verification](local-verification.md), and [roadmap](../roadmap.md)

## Decision

Eqiora optimizes for **truthful product lead time**: the time from a concrete accepted outcome to
a merged, focused-tested capability whose verification status is reported independently and
honestly. It does not optimize agent utilization, document volume, open lanes, contract count,
test count, or CI activity in isolation.

AI makes implementation cheap. It does not make context, integration, compatibility, scientific
evidence, or human attention free. Process and abstraction therefore need the same deletion
pressure as production code.

The default development path is one primary agent, one local change, the narrowest useful
check, one self-review, and stop. The same primary agent owns implementation through pull
request integration. Durable scientific, compatibility, artifact, security, release, CI,
governance, and architecture risk keeps its applicable existing evidence and focused review,
without requiring a second actor. Nothing else inherits that ceremony by analogy.

## Product position

Eqiora aims to make computational engineering capabilities machine-checkable end to end, with
narrow claims and independently truthful verification status. Performance claims remain
workload-specific and bind
hardware, mesh, ordering, solver, preconditioner, tolerances, transfer accounting, warmup, and
repetitions.

Machine-checkability does not mean every implementation receives a new schema, proof framework,
registry, or exact-byte oracle. A focused test at an existing typed boundary is the normal
evidence for ordinary behavior. Independent scientific derivation and exact-artifact oracles
remain exceptional because their failure cost is durable.

Generated or hand-written physics must not create a bespoke proof system per module. Prefer
existing compiler, realization, and evidence paths when they genuinely reduce repeated
invariant-bearing formulas. Do not build a universal form, plugin, or intermediate
representation before a current capability and a second real consumer show the need.

## Claim-local evidence development

Evidence development is active under [RFC 0089](../../rfcs/0089-resume-claim-local-evidence-development.md).
Add or change registered evidence when a durable scientific, semantic, artifact, security, release,
or CI-trust claim needs a stronger falsifier than an ordinary focused product test. Derive its
expected values, relations, tolerances, and failure probes independently of implementation output.

Evidence binds the smallest semantic projection that can falsify its claim. Do not use unrelated
whole source files, package roots, generated trees, or broad inventories as proxies for a narrow
surface. Exact bytes remain appropriate when those bytes are the actual public or persisted
contract. Investigate a mismatch; never copy observed output into an expectation merely to pass.

## Context scales by navigation

Large repositories stay usable when agents can find the right context without loading all
context at once.

- Keep root `AGENTS.md` short and repository-wide.
- Put path-specific instructions only near code that repeatedly needs different rules.
- Treat documentation as a map to source, tests, manifests, and decisions, not as a copy of
  them.
- State each instruction once in its authoritative owner and link to it.
- Start with the goal, domain context, hard constraints, approval boundaries, and success
  criteria. Let the agent discover implementation details from the repository.
- Expose or invoke only tools relevant to the current task; unused tools and instructions
  consume attention.
- Record what was not checked. Confidence is not evidence.

Add a skill only for a repeatable workflow with reusable instructions or assets. Add an MCP or
other protocol only for a real external system boundary. A one-off repository task needs
neither.

Subagents spend additional context and tokens. Prefer them for independent, read-heavy, or
noisy exploration whose outputs can be summarized. Keep ordinary writable work with one agent;
parallel writes require disjoint ownership and an integration benefit greater than their merge
cost.

## Architecture discipline

Preserve the general mathematical surface when choosing a first executable profile. A
convenient algorithm or fixture size must not become a permanent dimension, model-family,
or composition restriction. Keep mathematical admissibility separate from execution
resources, numerical conditioning, and provider availability. When a concrete consumer
exposes such a restriction, replace the limiting algorithm at its existing owner and
verify that consumer; do not merely increase a hard-coded size or imply untested generality.

Choose the smallest design that closes the current claim:

- use an existing type before adding a wrapper or translation layer;
- use a private helper before a public abstraction;
- require a current invariant owner and real consumer;
- keep semantic identity, execution provenance, and presentation state separate;
- do not move branching, configuration, or validation into every consumer;
- split oversized implementation files instead of raising ceilings; and
- delete speculative pilots when their second consumer or measured benefit does not arrive.

An architecture check must protect a plausible defect in a current supported surface.
Private source layout is not a product artifact. Line, token, complexity, or public-surface
ratchets may prevent unreviewable code, but each ceiling is debt: it needs a current reason,
must not rise through ordinary work, and should disappear when a simpler invariant replaces it.

## Pre-1.0 API convergence

Eqiora's current Rust and Python authoring APIs are design candidates, not backward-
compatibility obligations. Until an explicit stable compatibility promise is accepted, choose
the most coherent final surface even when that breaks a current caller. Migrate every
repository-owned consumer, example, ordinary test, and document in the same change, and delete
the displaced name and path. Do not retain a legacy overload, alias, wrapper, deprecation shim,
dual lifecycle, or transition period solely for backward compatibility.

Published release artifacts remain immutable historical objects, but current readers and authoring
surfaces need not preserve an obsolete pre-1.0 API, schema, or wire epoch without an explicit stable
or external interoperability promise. A breaking correction ships in a new release and is stated
plainly. Migrate repository-owned producers, consumers, examples, docs, tests, and evidence
atomically, then delete displaced schemas, codecs, aliases, overloads, shims, and parallel paths.
Security and data-integrity validation moves to the new boundary; it is not weakened.

## Evidence discipline

Evidence should be cheaper than the mistake it prevents.

- Treat an accepted case as support only for its explicit claim and non-claims.
- Use focused tests for ordinary behavior; register evidence only for a durable claim that needs
  independent scientific, artifact, security, release, or CI-trust authority.
- Derive evidence independently and change it only with an explicit rationale for the changed
  claim, never to follow observed implementation output.
- Do not generalize a result across environments, caches, hosts, or aborted runs.
- Fix nondeterministic product or build output rather than teaching an oracle to accept it.
- Replace broad exact witnesses with claim-local semantic checks unless exact identity is the
  contract being tested.

## Process deletion rules

Before retaining a document, gate, review, receipt, template, or coordination step, name:

1. the current decision it can change or plausible defect it can expose;
2. the authority that does not already carry the same fact; and
3. the condition under which it will be deleted.

Delete or consolidate it when those answers are absent. Historical RFCs may preserve decisions,
but active instructions must not route ordinary work through superseded process. Do not create
a replacement document merely to narrate the deletion.

The following are regressions unless they replace more durable complexity than they add:

- a giant integration pull request used as a coordination mechanism;
- a risk-classifier framework for rules that fit in a short list;
- prose copies of paths, hashes, manifests, evidence indices, or GitHub relations;
- permanent actor identities or receipts without an adversarial trust boundary;
- a new orchestration layer to work around build, cache, sync, or editor limits; and
- mandatory broad verification after a focused check already answers the integration question.

## Evaluation and amendment

Use real delivery failures to amend the strategy. Useful observations include capability lead
time, first-pass success, correction cycles, integration conflicts, escaped defects, and
repeated verification work. They are diagnostic, not standing reporting requirements.

Change one rule only when an observed failure can distinguish the new rule from the current
one. Remove the superseded rule in the same change. A failed experiment is deleted rather than
kept as optional architecture.

## Research basis

The 2026-08-22 revision applies current official guidance for
[Codex AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md),
[Codex subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents), and
[lean GPT-5.6 prompting](https://developers.openai.com/api/docs/guides/latest-model).
These sources inform the context and orchestration rules; repository evidence remains the
authority for Eqiora-specific decisions.
