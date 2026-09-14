# Frozen expected evidence

- `analytic.json` is the read-only exact-rational Oracle A.
- `symbolic.json` is the read-only independently constructed Oracle B.

`run_case.py --check` verifies both scientific files and their derivation sources against the
precommitted hashes, reruns Oracle B deterministically, and proves exact agreement. Provider
state and invalidation behavior are covered by focused product tests rather than a second copy
of the implementation state machine.
