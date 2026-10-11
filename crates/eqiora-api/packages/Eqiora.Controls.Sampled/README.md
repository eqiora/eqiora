# Sampled controls

`Eqiora.Controls.Sampled@0.1.0` provides two real-scalar components for
clocked systems:

| Component | Inputs | Outputs at each tick |
| --- | --- | --- |
| `UnitDelay` | periodic `tick`, `initial_value: 1`, `u: 1` | `y = pre(memory)` |
| `DiscreteIntegrator` | periodic `tick`, `initial_value: 1`, `rate: 1/s` | `before = pre(memory)`, `after = next(memory)` |

Each instance holds one state, `memory`, starting at `initial_value`. At each
tick the delay stores `u`, and the integrator stores
`pre(memory) + period(tick) * rate`. All equations at a tick hold
simultaneously, so their order in the source does not determine what is
output or updated first. To choose the integrator's timing, connect either
`before` or `after`: at the first tick these are the initial value and the
first updated value. Outputs exist only at ticks; the components do not hold
a value between them.

The caller supplies the clock, including its phase, and input samples on the
same clock. Two clocks with equal periods are still different clocks. To feed
a continuous signal in, `sample` it explicitly; to observe an output
continuously, `hold` it explicitly in the caller. Connecting signals on
different clocks is an error; there is no automatic rate conversion.

## Physical quantities

The components work on dimensionless real scalars. To process a physical
quantity, divide it by a fixed nonzero scale of the same dimension, pass the
normalized signal (or the rate divided by the scale) to the component, and
multiply the output by the scale. Normalize the initial value the same way.
For example, use a scale in V and a rate in V/s for a voltage, or a scale in m
and a rate in m/s for a displacement. A rate with the wrong dimension fails to
compile.

The repository example `examples/standard-sampled-components` does this for
both cases, with quarter- and half-second clocks, and runs from Python
through an offline locked package.
