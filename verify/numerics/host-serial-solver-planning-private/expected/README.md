# Expected-value authority

This private companion maintains no expected-value file. All exact decisions,
rejection rerankings, reason traces, controls, and ledgers are frozen once in
the primary case's
[`policy-v9.json`](../../host-serial-solver-planning/expected/policy-v9.json).
That authority identifies the exact owned canonical operator, freezes the
direct apply/diagonal self-control and reset, records two total applications
and zero diagonal calls for successful true-residual acceptance, and zero of
both for a selected failure or preflight rejection.
The same authority freezes the full objective traces for a candidate whose
evidence and provider descriptor are simultaneously stale, distinguishing
evidence-before-provider validation.
