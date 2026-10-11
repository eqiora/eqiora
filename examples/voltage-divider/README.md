# Grounded voltage divider

With Eqiora installed, run `python examples/voltage-divider/run.py`.
The [model source](src/main.eqi) builds the circuit from the bundled
`Eqiora.Electrical.Basic@0.1.0` source, resistors, and ground using ordinary
imports and Connections. The runner creates a local package lock, selects Faer
SparseLu explicitly, and runs Plan → State → Run → Result. It prints the current,
the midpoint voltage, and the power absorbed by each component as Result
Observables. No time integration, registry fetch, or provider installation is
needed.

Ohm's and Kirchhoff's laws give, by hand:

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
no unknowns or equations. Doubling the supply gives 8 mA and 16 V; that is a
different Plan, so the original Result cannot be reused.

The installed product test checks these independent values, State/Plan/Result
replay, the same equations compiled directly from the maintained Basic source,
and a moved vendor-only project after deleting its original store. The direct
and packaged Models have equal structural fingerprints, while their Model and
Result identities and package provenance stay distinct.

This is a linear circuit of ideal components: there is no nonlinear or dynamic
circuit execution and no bundled divider Component. The native tests and the
website use this same source.
