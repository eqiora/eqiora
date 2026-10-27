# Example: harmonic response of an RC circuit

A harmonic form turns a linear time-domain model into equations for complex amplitudes at one
angular frequency. This example applies it to a resistor-capacitor circuit, reusing the same
equations as the time-domain model. Forms written in `.eqi` source and forms built from Python
compile to the same representation and run through the usual complex Plan and Result. The
[verification case](../../verify/numerics/harmonic-response/README.md) checks RC and RLC
amplitudes, a one-dimensional scalar wave, reconstruction, sign conventions, and a comparison
with the RC transient. Ordered frequency sweeps use independent ordinary Runs as described below.

## The model and its harmonic form

An ideal voltage source drives a series resistor and a capacitor connected to ground. The
equations below are written for the grounded circuit directly: `source` and `voltage` are
potentials relative to ground, and current is positive from the source through the resistor
into the capacitor.

```eqiora
model RC(
  parameter resistance: Ohm = 1000 [Ohm],
  parameter capacitance: F = 1e-6 [F],
  parameter initial_voltage: V = 0 [V],
  parameter omega: 1 / s = 1000 [1 / s],
  parameter drive: complex<V> = math.complex(1 [V], 0 [V]),
  input source: V
) {
  state voltage: V;
  variable current: A;

  initial {
    voltage = initial_voltage;
  }
  relation network {
    source - voltage = resistance * current;
    current = capacitance * derivative(voltage);
  }

  form harmonic_response for network {
    harmonic(angular_frequency = omega, convention = negative_exponential, normalization = peak);
    excitation source = drive;
    amplitude voltage_hat: complex<V> for voltage;
    amplitude current_hat: complex<A> for current;
  }
}
```

`resistance` and `capacitance` are positive, real constants. The circuit is lumped: no geometry,
mesh, or clock. The time-domain model needs an initial voltage and a source waveform; the
harmonic form supplies the waveform through its excitation instead.

A harmonic form contains three kinds of entries:

- `harmonic(...)` sets the angular frequency, the `exp(-i*omega*t)` sign convention, and peak
  normalization. All three arguments are required, and the angular frequency must be positive
  and real.
- `excitation original_input = complex_expression;` gives the peak complex amplitude of a real
  input. Every time-dependent input of the selected relations needs one.
- `amplitude name: complex<dimension> for original_value;` names the complex amplitude of a real
  unknown. It has the same dimension, shape, frame, and support as the original, and it does not
  add an unknown to the original Model.

Notation for an amplitude goes after its name, as in other declarations. The amplitude inherits
`on` from the original value (you may repeat it before `for`); `at` is not allowed. Every unknown
in the selected relations needs exactly one amplitude. Duplicate amplitudes, amplitudes for
values from another model, wrong types, clocked values, and unknowns without an amplitude are
errors. The form covers every equation in the selected relations, not only the one containing
the time derivative.

The compiler checks that the selected equations are linear with constant coefficients on a
fixed domain and that every source and boundary term is accounted for. Writing a form does not
by itself make these assumptions true: nonlinear, time-varying, and history-dependent equations
are rejected before solving. The amplitude problem and its reconstruction keep a record of the
model they came from.

## Derived amplitude equations

With this convention, each time derivative becomes multiplication by `-i*omega`. The resulting
equations are:

```text
drive - voltage_hat = resistance * current_hat
current_hat = -i * omega * capacitance * voltage_hat
voltage_hat = drive / (1 - i * omega * resistance * capacitance)
```

The first equation is in volts and the second in amperes, and `omega*R*C` is dimensionless.
Because these equations are derived from the time-domain model, there is no separate AC model to
keep in sync. Solver, backend, and factorization choices are made when solving, not in the form.

With the written inputs, `omega*R*C = 1`. Rationalizing `1/(1-i)` gives:

```text
voltage_hat = (0.5 + 0.5*i) V
current_hat = (0.0005 - 0.0005*i) A
```

Check both the real and imaginary parts: magnitudes alone would not reveal a flipped sign
convention. The frequency in hertz is `1000/(2*pi) Hz`, not 1000 Hz. Convert a frequency in hertz
with `omega = 2*math.pi*f`; both have dimension 1/s, so the compiler cannot tell them apart.

The time signal is the real part of the amplitude times `exp(-i*omega*t)`, with `t` measured from
the model's time origin. The source is therefore `cos(omega*t) V` and the steady capacitor voltage
is `(0.5*cos(omega*t) + 0.5*sin(omega*t)) V`. The voltage lags the source by `pi/4`, even though
its complex amplitude has a positive phase in this convention.

## Initialization and the static limit

The harmonic voltage at `t = 0` is 0.5 V, but the initial condition is zero. The full solution of
the initial-value problem therefore adds a transient `-0.5*exp(-t/(R*C)) V` to the harmonic
response, decaying with `R*C = 0.001 s`. The harmonic form describes only the steady periodic
response, not the initial-value problem.

The original initial condition is kept with the model; it is neither applied to the amplitudes
nor deleted. Changing the initial voltage changes the transient but not the steady harmonic
response.

At zero frequency, the DC problem gives a capacitor voltage equal to the source and zero current.
The harmonic form rejects zero frequency rather than dividing by it or applying sinusoidal power
normalization to a constant signal. The amplitude equations do approach the DC answer as the
frequency goes to zero, but DC is still solved as a separate problem.

## Power and rejection checks

For peak phasors at nonzero frequency, mean absorbed real power is
`real(V_hat * conj(I_hat))/2` under the passive current orientation. The resistor absorbs
0.00025 W, the ideal capacitor absorbs zero mean real power, and the source absorbs
-0.00025 W. These sum to zero. Omitting the peak factor doubles the result; using RMS
phasors instead requires dividing both amplitudes by `sqrt(2)` and omitting the half factor.

If complex power is defined as `V_hat * conj(I_hat)/2`, the capacitor's imaginary part is
positive with this negative-exponential convention. A conventional positive-exponential
reactive-power sign cannot be copied without an explicit conversion.

A missing excitation, an amplitude with the wrong dimension, an amplitude for a value from
another model, and nonlinear or time-varying equations are all errors. The comparison with the
decaying transient needs strictly positive R and C: with other values a harmonic solution may
still exist, but the transient does not decay. Accepting a harmonic form does not prove that the
system is stable. A version of this circuit built from connectors still needs an explicit ground,
and a source or boundary term left out of the form is an error rather than being treated as zero.

## Python authoring and reconstruction

`Module.model(...)` and `Module.component(...)` expose `harmonic_form(...)`. Given the
original Relation, input, Parameter and Field handles corresponding to the source above:

```python
rc.harmonic_form(
    "harmonic_response", [network], angular_frequency=omega,
    convention="negative_exponential", normalization="peak",
    excitations=[(source, drive)],
    amplitudes=[
        ("voltage_hat", voltage, eqiora.ValueType.complex(volts)),
        ("current_hat", current, eqiora.ValueType.complex(amps)),
    ],
)
```

Here `volts` and `amps` are the `Dimension` values used in the original model. Each amplitude
entry gives the amplitude's name, the original Field, and the complex `ValueType`; the spatial
support comes from the Field. Amplitudes belong to the form and do not add unknowns to the
original Model. A Component or Model can have one named form, and weak-form test functions are
not part of a harmonic form.

Resolve with the usual linear solve settings and a provider that supports complex values.
`plan.model` is the amplitude Model, and `plan.harmonic_original_model` is the original real
Model, including its initial conditions. `plan.harmonic_amplitudes` is a tuple of
`(name, original_field, amplitude_field)` entries, and `plan.harmonic_angular_frequency` is in
inverse seconds.

For a finite Result, `result.reconstruct_harmonic_fields(time_seconds=t)` returns
`(original_field, real_value, original_value_type)` tuples in coherent SI units. For a scalar
spatial Result, `result.reconstruct_harmonic_field_block(original_field, time_seconds=t, block=0)`
returns real coefficients in the layout of the amplitude block. Both give the steady harmonic
response, not the solution of the initial-value problem. Pass the original Field, not the
amplitude Field. Derivatives of the reconstructed response are not supported and raise an error.

## Ordered frequency Runs

Use `EvaluationMapPlan.from_parameters` with the original Model's `omega` Parameter and one
complete value per selected Parameter in each point. Values use the Parameter's declared coherent
SI units; angular frequency is in inverse seconds. Convert cyclic frequency explicitly with
`omega = 2 * math.pi * frequency_hz`.

```python
import eqiora as q

# Keep the original compiled Model handle used to resolve plan.
omega = model.parameter("omega")
sweep = q.EvaluationMapPlan.from_parameters(
    plan, [omega], [(500.0,), (1000.0,), (500.0,)], workers=1,
)
collection = sweep.execute_recorded()
collection.write("rc-sweep.eqcollection")
for member in collection[0]:
    if member.status == "accepted":
        for name, original, amplitude in member.plan.harmonic_amplitudes:
            print(name, member.result.output(amplitude).values("global"))
    else:
        print(member.status, member.diagnostics)
```

Source aliases belong to the original compiled Model handle. The artifact-backed
`plan.harmonic_original_model` accepts exact Parameter IDs, not those source aliases.

Every point starts from the same original Model and repeats harmonic reduction with the selected
frequency. Each accepted occurrence retains its own exact Plan and Result. Select amplitude Fields
from `member.plan`, since a changed frequency changes the reduced Model identity. Finite amplitudes
use the `global` association and have no Mesh; spatial amplitudes retain their spatial associations.
Order and duplicate occurrences survive collection reopening. Zero and negative angular frequencies
are recorded as failed occurrences; accepted siblings remain available.

This route also handles ordinary stationary scalar-wave Plans through the same map and collection
owners. Each occurrence executes an independent Run: the map does not reuse a factorization across
frequencies or provide harmonic derivatives. Read saved results with `EvaluationCollection.read`
without executing the solver; retry selected unsuccessful points as a new map and retain both groups
with `EvaluationCollection.combine`.
