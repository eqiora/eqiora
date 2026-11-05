# Example: steady and transient heated bodies

The [heated-body package](../../examples/heated-body/README.md) runs from the installed Python
development build. Its [source](../../examples/heated-body/src/main.eqi) imports
`SteadyHeat2d` and `FixedTemperature2d` from bundled `Eqiora.Scalar@0.2.0` for a
unit-square cross-section held at 300 K. The [hand-written alternative](../../examples/heated-body/handwritten.eqi)
retains the same physical law and an explicit weak form whose tests vanish on the boundary.
The [Python runner](../../examples/heated-body/run.py)
supplies the geometry and chooses Q1 elements and a solver.

On a mesh of four Q1 cells, a hand calculation gives 300 K at the eight boundary nodes and
300 + 3/32 K at the centre. The Python tests check the heat balance at the free node, run the
model both directly and as a package, replay the Plan and Result, and confirm that the lock is
unchanged after moving the vendored project offline.

Temperature is an absolute field in kelvin, not an offset from the boundary value. This is a
steady discrete test problem; it does not give the exact continuous temperature profile.

The same source also defines `TransientHeatedBody`, with a constant positive volumetric heat
capacity and an initial temperature of 300 K. It keeps the same boundary condition and is
advanced with Q1 elements and backward Euler through the usual Plan, State, Run, and Result
APIs. On the same four-cell mesh, the interior mass is 1/9, the stiffness 8/3, and the heating
load 1/4, in coherent SI units per unit depth. With unit capacity and a step of 1/24 s, the
centre value after n steps is `300 + (3/32)*(1 - 2**(-n))` K. The tests check the heat balance at
every step, restarting, replay, and running the moved project offline.

The transient model is assembled from the compiled region equations; there is no separately
written transient weak form to compare against. Variable capacities, material interfaces,
three-dimensional heat conduction, and bundled transient thermal components are not covered here.
