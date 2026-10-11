# Governance

Eqiora is an independent, community-governed open-source project. This file
describes its bootstrap governance until an elected Technical Steering
Committee (TSC) replaces it.

## Principles

- Technical decisions, accepted design records, verification contracts, and
  release claims are public.
- No contributor receives architectural authority by virtue of employment,
  funding, vendor affiliation, or tooling.
- Semantic-kernel and conformance changes use the public RFC process.
- Verification evidence is derived independently of the implementation's output and
  limited to the claim it can disprove, as set out in
  [RFC 0089](rfcs/0089-resume-claim-local-evidence-development.md).
- Official functionality is not withheld for a closed edition.

## Roles

**Contributors** participate through issues, RFCs, reviews, documentation,
tests, code, models, or verification evidence.

**Maintainers** review and merge changes in an area, uphold compatibility and
quality gates, and disclose conflicts of interest. Maintainer status is earned
through sustained, constructive contribution and may be removed for prolonged
inactivity or conduct violations.

**Bootstrap maintainer.** Until a TSC is elected, the repository owner listed
in `CODEOWNERS` administers merges, releases, and security reports. This
operational role does not waive RFC, compatibility, or evidence requirements.
Where a single-maintainer repository cannot provide independent approval, the
exception is explicit and must not be represented as independent review.

## Decisions

1. Reversible implementation details are decided in pull-request review.
2. Public API, semantics, persisted formats, architecture, governance, and
   compatibility changes use the RFC process.
3. Maintainers seek rough consensus and record material objections.
4. A provisional decision records its rationale and the conditions that would
   overturn it. It is revisited when contrary evidence or repeated integration
   problems appear, not on a fixed schedule.

## TSC transition

A governance RFC may establish a TSC after the project has at least three
active maintainers from at least two independent affiliations. It will define
elections, terms, appeals, conflict disclosure, and public decision records.

## Working groups

Working-group paths in `CODEOWNERS` are architectural placeholders until a
public charter and actual maintainers are recorded. A placeholder confers no
authority.
