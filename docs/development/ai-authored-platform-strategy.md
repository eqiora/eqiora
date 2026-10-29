# AI-authored platform strategy

- Status: Accepted
- Revised: 2026-10-29
- Related: [AGENTS.md](../../AGENTS.md), [roadmap direction](../roadmap.md#direction),
  [high-risk and parallel development](vertical-slice-development.md), and
  [local verification](local-verification.md)

This document explains why Eqiora is developed the way `AGENTS.md` describes. `AGENTS.md`
holds the rules; this file holds the reasons and the evidence behind them.

## Decision

Eqiora optimizes for **useful product lead time**: the time from a user outcome in the
[roadmap direction](../roadmap.md#direction) to a merged, tested capability whose status is
reported honestly. It does not optimize agent utilization, document volume, open lanes, test
count, or CI activity on their own.

AI makes implementation cheap. It does not make context, integration, compatibility, scientific
evidence, or human attention free. Process, documents, and abstractions get the same deletion
pressure as production code.

## Product position

The product promise is: write the textbook equation, choose the method, run it fast on CPU
threads, MPI or GPU, and differentiate the result. Well-known equations and boundary conditions
come from standard packages; any equation can still be written by hand.

Verification serves that promise. Narrow, honest claims keep users from trusting results the
code does not support, but they are not the product. When a check, ratchet, or document does
not protect a user outcome or a plausible defect, delete it.

Generality is part of the product. Build the general mechanism at its existing owner rather
than a special case, and ship the API features users expect from comparable tools. Do not add
speculative frameworks for consumers that do not exist yet.

## Architecture discipline

Decide who owns meaning, validation, and change before choosing an implementation. Physical
units, input identity, mathematical domains, diagnostics, and persisted data need one owner
across Rust, Python, and execution backends.

Necessary abstraction is part of finishing a capability. Establish or extend one owner in the
same change when:

- fixing one rule would otherwise need coordinated edits in several implementations;
- two entry points disagree on the meaning of the same operation or data; or
- a new consumer would otherwise copy an invariant or work around its owner's restriction.

Migrate consumers and delete the displaced paths together. Similar code alone is not evidence
of a shared responsibility: numerical evaluation, exact algebra, independent test oracles, and
different discretization methods stay separate when they answer to different contracts.

A first executable profile must not become a permanent limit on dimension, model family, or
composition. Keep mathematical admissibility separate from execution resources, conditioning,
and provider availability. Every architecture ceiling is debt: it needs a current reason, must
not rise through ordinary work, and goes away when a simpler invariant replaces it.

## Pre-1.0 API convergence

Current Rust and Python APIs are design candidates, not compatibility obligations. Choose the
most coherent final surface even when it breaks a current caller, migrate every repository
consumer in the same change, and delete the old name or path. Published releases stay immutable
history; a breaking change ships in a new release and says so plainly. Security and
data-integrity validation moves to the new boundary and is never weakened.

## Evidence

Evidence should cost less than the mistake it prevents. Focused tests at an existing typed
boundary cover ordinary behavior. Registered evidence under
[RFC 0089](../../rfcs/0089-resume-claim-local-evidence-development.md) is for a durable
scientific, artifact, security, release, or CI-trust claim that needs an independent falsifier.

Expected values come from analytic or manufactured solutions, published references, or running
an established open-source tool, never from Eqiora's own output. Bind the smallest check that
can falsify the claim. Investigate a mismatch; never copy it into an expectation. A number from
one environment says nothing about another.

## Lessons from agent-built projects

Large projects built mostly by coding agents in 2026 converged on a few practices. Eqiora
adopts these:

- **A short router, details in docs.** OpenAI's Codex team replaced one large rules file with a
  ~100-line `AGENTS.md` that points into `docs/`. Decisions that agents cannot see in the
  repository do not exist for them.
- **Each rule names what enforces it.** SolidCraft's `AGENTS.md` lists every hard rule with
  the check or skill behind it, then a "working on X, read Y" table.
- **One core, many clients.** SolidCraft and ArtCraft route the GUI, CLI, scripts, and agent
  tools through one command engine. Eqiora's equivalent is one compile → resolve → run path
  for Python, Rust, CLI, Studio, and the language server.
- **Trusted oracles.** Anthropic's parallel-Claude C compiler used GCC as a reference;
  SolidCraft checks parts against analytic and FreeCAD-computed values. A verifier must be
  nearly perfect or agents solve the wrong problem.
- **Measure depth, not presence.** ArtCraft's menu-coverage numbers were criticized as presence
  rather than working depth. Eqiora reports what is verified to work, not what is registered.
- **Constraints beat reminders.** Cursor found "no TODOs, no partial implementations" worked
  better than asking agents to finish. Where a rule keeps failing, add a check.
- **Clear roles, few gates.** Cursor's planners own scope and write no code; workers finish one
  task and return a handoff. A single overloaded agent and a central integrator both became
  bottlenecks. Specialized maintenance roles (deduplication, docs, performance) ran well in the
  C compiler project and in OpenAI's background clean-up tasks.
- **Context is the scarce resource.** Test output should be terse and grep-able, with details in
  files. Agents need their own worktree and build directory to avoid lock contention.

## Process deletion rules

Before keeping a document, gate, review, receipt, template, or coordination step, name:

1. the decision it can change or the plausible defect it can expose;
2. the authority that does not already carry the same fact; and
3. the condition under which it will be deleted.

Delete or consolidate it when those answers are missing. Historical RFCs may keep old decisions,
but active instructions must not route ordinary work through superseded process.

These are regressions unless they replace more complexity than they add:

- a giant integration pull request used for coordination;
- a risk-classifier framework for rules that fit in a short list;
- prose copies of paths, hashes, manifests, evidence indices, or GitHub relations;
- permanent actor identities or receipts without an adversarial trust boundary;
- a new orchestration layer to work around build, cache, sync, or editor limits; and
- mandatory broad verification after a focused check already answered the question.

## Evaluation and amendment

Amend this strategy from real delivery failures: lead time, first-pass success, correction
cycles, integration conflicts, escaped defects, and repeated verification work. Change one rule
only when an observed failure distinguishes the new rule from the old one, and remove the
superseded rule in the same change.

## Sources

- OpenAI, [Harness engineering](https://openai.com/index/harness-engineering/), 2026-02-11.
- Cursor, [Towards self-driving codebases](https://cursor.com/blog/self-driving-codebases),
  2026-02-05, and [Scaling long-running autonomous coding](https://cursor.com/blog/scaling-agents),
  2026-01-14.
- Anthropic, [Building a C compiler with a team of parallel Claudes](https://www.anthropic.com/engineering/building-c-compiler).
- SolidCraft repository `AGENTS.md`, `CONTRIBUTING.md`, and `ROADMAP.md`
  ([tangency-labs/solidcraft](https://github.com/tangency-labs/solidcraft)), October 2026.
- ArtCraft development write-up, [syskb.com](https://syskb.com/artcraft-claude-code-rust/), 2026.
