# Example: a grounded resistor divider

The [voltage-divider package](../../examples/voltage-divider/README.md) builds a circuit from
`Eqiora.Electrical.Basic` with ordinary imports. Its
[source](../../examples/voltage-divider/src/main.eqi) is used by the Rust tests, the Python
tests, and the package reference.

The [runner](../../examples/voltage-divider/run.py) locks the bundled dependency, compiles the
Model, and resolves a Plan that uses Faer's sparse LU solver. It then creates the initial State,
runs it, and reads the current, voltages, and powers from the Result. There is no clock and no
initial equation.

## The circuit

A 12 V supply drives a 1 kΩ resistor in series with a 2 kΩ resistor. Ohm's law gives
I = 12/3000 = 4 mA and a midpoint voltage of 2000 I = 8 V. With current counted as positive into
each component, the resistors absorb 0.016 W and 0.032 W and the source absorbs −0.048 W, so the
powers sum to zero.

Ground fixes a voltage reference only; it does not force any terminal current to zero. Each
connection makes the voltages equal and the signed currents sum to zero. The seven ports carry
fourteen scalar unknowns, and seven component equations plus seven connection equations
determine them. Matching counts do not prove solvability in general, but here removing ground
leaves the overall potential undetermined and resolution fails.

Observables add no unknowns. A Result only accepts observable handles from its own Model, and
it cannot be replayed against a different Plan. Doubling the supply voltage gives 8 mA and 16 V
from a new run rather than reusing the old Result.

## Packages and replay

Importing a package gives it no special treatment by the compiler: compiling the same
component declarations directly produces the same structural fingerprint and the same values.
The package route also records which release and source the components came from. The Python
tests replay Plan, State, and Result, and run a vendored copy of the project after deleting the
original package store.

Run `python examples/voltage-divider/run.py` with the current package installed. This is a
steady linear circuit; nonlinear, transient, and distributed circuits are not covered by this
example. The divider is an example, not a packaged composite component.
