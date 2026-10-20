# High-risk and parallel capability development

This guide is an exception path. Ordinary localized work follows the one-pass rule in
[`AGENTS.md`](../../AGENTS.md) and does not create a contract cell, lane, independent oracle,
or separate integration role.

Read this document only when the actual delta is high risk under `AGENTS.md`, or when the user
explicitly requests parallel writable work.

## Close a capability, not a process

A change that adds or extends an executable or user-visible capability closes one narrow path:

```text
bounded claim
  -> existing invariant owner
  -> ordinary positive execution path
  -> focused product tests
  -> capability-matrix update
```

Introduce a new typed contract only when the changed invariant has no correct existing owner.
Do not create a contract artifact merely because work is called a capability. A private
behavior change can remain private; an adapter does not need a new central type for local
convenience.

The claim states the narrow supported boundary and important non-claims. One fixture does not
imply a general dimension, field shape, backend, package, hardware, or performance promise.
Code presence is neither verification nor maturity.

Evidence is claim-local and independently derived. Use focused product tests for ordinary behavior;
add or change registered evidence when a durable scientific, artifact, security, release, or
CI-trust claim needs it. Never pin unrelated whole files or inventories as a proxy for a narrower
surface, and never tune an expectation to observed implementation output.

## Minimal outcome contract

When high risk or parallelism requires a frozen boundary, the Issue or implementation prompt is
enough. It states only:

- the observable result and ordinary positive path;
- important non-claims and named failure conditions;
- relevant public, persisted, security, or scientific authority;
- input and resource bounds that belong to the claim;
- writable paths and the shared seam owner for parallel writes; and
- the condition that stops work and returns a missing requirement.

Do not freeze implementation details such as allocator behavior, worklist lifetime, process
layout, sandbox technology, or scheduler design unless the public claim is about that mechanism.
Do not duplicate machine-derivable paths, hashes, manifests, or registries in prose.

No separate contract file, schema, receipt, or sealed handoff is required by default. Add one
only when a concrete trust boundary needs a durable artifact and no existing authority carries
it.

## Evidence authority

Evidence remains independently runnable and authoritative only for its stated claim. A failure may
expose a product defect, build nondeterminism, an unsupported claim, or an obsolete witness. Change
evidence only from an independent derivation with an explicit risk-focused rationale. Prefer a
claim-local semantic projection; exact bytes belong only to an exact-byte contract.

## Parallel writable work

Parallel writes are useful only when they shorten the critical path after coordination cost.
Use them when the user requests them or when all of these are true:

- tasks have independent outputs or consume one accepted shared seam;
- writable paths are disjoint;
- each result can be integrated independently; and
- failure of one task does not invalidate the others.

One writer owns an invariant-bearing shared seam until its reference path is accepted.
Consumers then start from that exact revision and do not widen the seam for local convenience.
Use a separate worktree per mergeable writable lane. Sibling lanes do not merge one another.

The integrator alone reconciles genuinely shared registration points such as workspace
manifests and lockfiles, crate-root public inventories, the capability matrix, shared workflow
registries, and artifact version registrars. A lane returns the smallest proposed registration
delta rather than editing an overlapping root.

Read-only research, navigation, and adversarial review are safer to parallelize. Do not spawn
agents to restate the same context, fill capacity, or manufacture an independent role for
low-risk work.

Coordination state stays disposable and outside the repository. GitHub Issues own closable
work, the roadmap owns durable dependency order, and `verify/` owns executable claims. Create
no activity ledger or second planning graph.

## Abstraction and API budget

Before adding a crate, public type, trait, enum variant, wire field, registry, or generic
framework, answer:

1. Which current invariant gains one clear owner?
2. Which real consumer needs it now?
3. Why is a private implementation or existing type insufficient?
4. What existing branch, type, or repeated reasoning becomes simpler or disappears?
5. What compatibility promise and deletion cost does it create?

Use the smallest owner that satisfies the current responsibility. A public or durable abstraction
needs a current contract or concrete consumer that a private implementation cannot serve.
Follow the [platform strategy](ai-authored-platform-strategy.md#architecture-discipline): do not
wait for a second consumer when a shared invariant already requires one owner, and complete
necessary abstraction with the capability that exposes it. Similar code and anticipated reuse
alone do not establish a shared responsibility. A small crate or schema is still permanent complexity.

When a ceiling blocks work, split or simplify the implementation. Raising a ceiling or adding
architecture debt is itself a reviewed high-risk change with a reason and deletion condition.

## Integration

Prefer one pull request per independently releasable outcome. Do not use a release pull request
as a queue for unrelated features, documentation migrations, generated trees, and process
experiments. If a pull request exceeds the review or provider envelope, reduce or split its
scope; do not add machinery solely to carry a giant envelope.

Before integration:

1. inspect the complete diff and remove unrelated changes;
2. confirm the narrow claim, non-claims, and independently derived evidence, if any;
3. run the narrowest repository-owned checks required by
   [local verification](local-verification.md);
4. perform and record a risk-focused review of the actual high-risk delta; and
5. wait for relevant hosted checks when durable risk requires them.

A localized correction reruns the exposing check and receives only the focused review its risk
requires. A moved head does not by itself require whole-delta rereview or a broad gate rerun.

Open an Issue when alternatives need coordination, the work has durable dependencies, or it
cannot close in one pull request. Reuse an existing Issue when it already owns the outcome.
GitHub parent/sub-issue and blocking relations are the coordination graph; prose checklists do
not create a second one.

## Stop and simplify

Stop the affected work, not the whole repository, when:

- the accepted seam cannot express a discovered requirement;
- evidence disagrees with the product or cannot reach its ordinary positive path;
- parallel lanes begin editing the same invariant;
- a new framework has only hypothetical consumers;
- the process artifact becomes larger than the product delta; or
- a check or review cannot name a decision it may change.

Return the missing requirement or narrow the claim. Do not relax a gate, widen an abstraction,
add another registry, or create successor ceremony to preserve sunk process work.
