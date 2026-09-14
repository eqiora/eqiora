# Specimen: a grounded resistor divider

The [maintained package](../../examples/voltage-divider/README.md) composes
`Eqiora.Electrical.Basic` through ordinary imports. Its
[single mathematical source](../../examples/voltage-divider/src/main.eqi) is
shared by native tests, installed Python and the package reference.

The [runner](../../examples/voltage-divider/run.py) creates an exact lock from the
bundled dependency, compiles the Model and selects an explicit Faer SparseLu
Plan. It initializes State, executes Run and observes current, voltage and power
from the common Result. No clock or hidden initialization equation is introduced.

## Independent electrical meaning

The supply fixes 12 V across a 1 kΩ resistor in series with a 2 kΩ resistor.
Ohm's law gives I = 12/3000 = 4 mA and a midpoint of 2000 I = 8 V.
Current is positive into each component. Therefore the resistor powers are
0.016 W and 0.032 W, while the source absorbs −0.048 W. Their sum is zero.

Ground fixes voltage only. It does not prescribe zero terminal current.
Each Connection equates across values and sums signed through currents to zero.
Seven ports carry fourteen scalar values; seven component equations and seven
connection equations close this example. This count is not a general solvability
proof. Removing ground leaves a uniform-potential freedom and rejects resolution.

The typed Observables add no unknowns. Result access rejects foreign observable
handles, and replay rejects a Result under another exact Plan. Doubling the
supply produces 8 mA and 16 V instead of reusing the old result.

## Packages and replay

The package name does not grant compiler privileges. Direct compilation of the
maintained Basic declarations with this same composition has the same structural
fingerprint and independent predictions. The package route additionally retains
its exact release/source provenance. Installed tests cover Plan/State/Result
replay and a moved vendored project after deleting the original store.

Run `python examples/voltage-divider/run.py` with the current package installed.
This bounded steady linear circuit adds no nonlinear, transient or distributed
circuit capability. The divider is a maintained example; it is not yet shipped
as a curated composite package.
