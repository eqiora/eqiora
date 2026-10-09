# Specimen: harmonic RC response

This specimen fixes an implemented, bounded harmonic Formulation. It uses the same
resistor/capacitor mathematics as the time-domain model. Source and Python Module requests
share the compiler's AST and the common complex Plan/Result owners. The
[registered case](../../verify/numerics/harmonic-response/README.md) independently verifies
RC/RLC amplitudes, a one-dimensional scalar wave, reconstruction, conventions and a positive-RC
transient comparison. It does not establish arbitrary harmonic or frequency-sweep workflows.

## Original model and reduction request

The circuit is an ideal voltage source, series resistor, and capacitor to an explicit zero
reference. The following equations expose the already-grounded network directly. `source`
and `voltage` are potentials relative to that reference; current is positive from source
through resistor into the capacitor.

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

`resistance` and `capacitance` are positive, real, fixed parameters. This is a lumped circuit
with no spatial boundary, mesh, moving support, or periodic clock. The fresh transient model
requires its voltage initial condition and a supplied continuous source waveform. The harmonic
candidate instead prescribes that waveform through the excitation's reconstruction.

The three child kinds in this form are closed mathematical syntax:

- `harmonic(...)` selects the angular-frequency role, `exp(-i*omega*t)` convention, and peak
  normalization. All three named entries are required. The initial profile admits positive
  real angular frequency only.
- `excitation original_input = complex_expression;` binds an original real input to its
  peak complex amplitude. Every external time-dependent input of the selected system must
  have an explicit admitted excitation or a separately justified decomposition.
- `amplitude name: complex<dimension> for original_value;` maps an original real unknown to
  its complex amplitude with the same dimension, shape, frame, and support. It does not
  allocate another unknown in the original Model or attach a temporal clock to frequency.

Amplitude notation, when present, follows its name as in other declarations. Its `on` support
is inherited from the original value and may be explicitly asserted before `for`; `at` is
inapplicable. Every transformed unknown has one amplitude mapping. Duplicate, foreign,
wrong-type, clocked, or unmapped dependencies reject. The form retains all selected relations
and exact original occurrences, not just the equation containing a time derivative.

Admission checks fixed-domain linear time-invariant coefficients and compatible source/boundary
accounting. A user-written form is a request, not proof of those hypotheses. Nonlinear,
time-varying, history-dependent, or otherwise unsupported expressions reject before solving.
The effective reduced problem and reconstruction are retained with their original lineage.

## Derived amplitude equations

Substituting the declared ansatz differentiates each amplitude by `-i*omega`. The resulting
mathematics, shown explicitly for inspection, is:

```text
drive - voltage_hat = resistance * current_hat
current_hat = -i * omega * capacitance * voltage_hat
voltage_hat = drive / (1 - i * omega * resistance * capacitance)
```

The first equation is in volts and the second in amperes; `omega*R*C` is dimensionless.
There is no independently maintained AC law to drift from these equations. A standard RC
component may expose the same original network for this Formulation; source-level `solve`,
frequency sweeps, backend names, and numerical factorization choices do not enter the form.

With the written inputs, `omega*R*C = 1`. Rationalizing `1/(1-i)` gives:

```text
voltage_hat = (0.5 + 0.5*i) V
current_hat = (0.0005 - 0.0005*i) A
```

Both complex components matter. Magnitudes alone cannot expose a flipped Fourier sign.
Cyclic frequency for this example is `1000/(2*pi) Hz`, not 1000 Hz. A caller supplying cyclic
frequency must convert it explicitly using `omega = 2*math.pi*f`; equal physical dimensions
do not establish the same frequency role.

Reconstruction uses the real part of the amplitude times `exp(-i*omega*t)`, measured relative
to the declared continuous timeline origin. Thus the source is `cos(omega*t) V` and the
settled capacitor voltage is `(0.5*cos(omega*t) + 0.5*sin(omega*t)) V`. This voltage lags
the real cosine input by `pi/4` despite the positive complex amplitude phase in this convention.

## Initialization and the static limit

The harmonic voltage at the time origin is 0.5 V, whereas the original initial condition is
zero. For this positive R/C model, the complete initial-value solution therefore includes
`-0.5*exp(-t/(R*C)) V` in addition to the reconstructed harmonic response. The transient decays
with `R*C = 0.001 s`. A reduction that claims exact initial-value equivalence would be false.

The form retains the original initial condition in its lineage while explicitly restricting
the response class. It does not apply that initial equation to amplitudes or silently delete
it from the original Model. Changing the initial voltage changes the transient model even
when its settled harmonic response stays the same.

At zero angular frequency, the separate real DC problem gives capacitor voltage equal to
the constant source and zero current. This form's positive-frequency profile rejects zero;
it must not divide by frequency or apply sinusoidal power normalization to a DC signal.
The amplitude equations have the expected continuous limit, but admission and observation
conventions still distinguish DC from a nonzero-frequency periodic response.

## Power and rejection checks

For peak phasors at nonzero frequency, mean absorbed real power is
`real(V_hat * conj(I_hat))/2` under the passive current orientation. The resistor absorbs
0.00025 W, the ideal capacitor absorbs zero mean real power, and the source absorbs
-0.00025 W. These sum to zero. Omitting the peak factor doubles the result; using RMS
phasors instead requires dividing both amplitudes by `sqrt(2)` and omitting the half factor.

If complex power is defined as `V_hat * conj(I_hat)/2`, the capacitor's imaginary part is
positive with this negative-exponential convention. A conventional positive-exponential
reactive-power sign cannot be copied without an explicit conversion.

Reject missing excitation, wrong amplitude dimensions, a foreign original-value mapping,
and an unsupported nonlinear or time-varying replacement. The transient-decay comparison
requires strictly positive R and C; nonpositive values invalidate that decay claim even
when a formal harmonic response exists. Harmonic admission alone does not certify stability.
A floating connector-network version must still supply a ground;
the grounded scalar equations here do not authorize implicit grounding elsewhere. A source
or boundary omitted by a reduction must fail correspondence rather than become zero by default.

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

Here `volts` and `amps` are the original `Dimension` values. Each amplitude triple names
the amplitude, its original Field, and its full complex `ValueType`; spatial support is
inherited from that Field. Amplitudes are local to the Formulation and are not additional
unknowns in the original Model. Each Component/Model owns one named form; weak tests are
declared before their owning weak form and do not belong to a harmonic request.

Resolve through the ordinary linear solve policy with a complex-capable provider.
`plan.model` is the distinct amplitude Model, while `plan.harmonic_original_model` retains
the real Model including its initial conditions. `plan.harmonic_amplitudes` is an ordered
tuple of `(name, original_field, amplitude_field)` exact references;
`plan.harmonic_angular_frequency` is in inverse seconds.

For a finite Result, `result.reconstruct_harmonic_fields(time_seconds=t)` returns
`(original_field, real_value, original_value_type)` tuples in coherent SI units. For a
scalar spatial Result, `result.reconstruct_harmonic_field_block(original_field,
time_seconds=t, block=0)` returns real coefficients with the mapped amplitude block's
association and layout. These are restricted harmonic responses, not initial-value
solutions. A Field reference from the amplitude Model cannot substitute for its original
Field in reconstruction. Unsupported response derivatives reject explicitly.
