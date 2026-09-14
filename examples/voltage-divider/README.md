# Grounded voltage divider

With Eqiora installed, run `python examples/voltage-divider/run.py`.
The [single maintained source](src/main.eqi) composes the bundled
`Eqiora.Electrical.Basic@0.1.0` source, resistors and explicit ground through
ordinary imports and Connections. The runner creates an exact local package
lock, selects Faer SparseLu explicitly, and executes Plan → State → Run → Result.
It prints typed current, midpoint voltage and into-component powers through
Result-owned Observables. No time integration, registry fetch or provider
installation is required.

Ohm and Kirchhoff independently give:

```text
I = 12 / (1000 + 2000) = 0.004 A
V_midpoint = 2000 I = 8 V
P_upper = 1000 I² = 0.016 W
P_lower = 2000 I² = 0.032 W
P_source = −12 I = −0.048 W
P_upper + P_lower + P_source = 0
```

Ground prescribes zero voltage, not zero current. Removing it leaves an
unreferenced uniform potential shift and resolution rejects. Observables add
no unknowns or equations. Doubling the supply gives 8 mA and 16 V; it changes
the exact Plan and cannot reuse the original Result artifact.

The installed product test checks these independent values, State/Plan/Result
replay, the same equations compiled directly from the maintained Basic source,
and a moved vendor-only project after deleting its original store. The direct
and packaged Models have equal structural fingerprints, while their exact
Model/Result ownership and package provenance remain distinguishable.

This is a finite linear ideal circuit. It does not add nonlinear or dynamic
circuit execution, or a new bundled curated divider Component. The mathematical
source is shared with native tests and the site reference; no duplicate circuit
source is maintained there.
