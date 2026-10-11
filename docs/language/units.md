# Units and dimensions

This page lists the units available in Eqiora source and in Python's
`eqiora.units`. Both use the same catalog, which the compiler maintains; there are
no package-specific unit aliases. A declaration's dimension, the unit you write a
value in, and the unit used to display a result are three separate things.

## Base and derived units

The base symbols are `kg`, `m`, `s`, `A`, `K`, `mol`, and `cd`; `1` is dimensionless.
The derived symbols below are coherent: each can be used as a dimension name and as
an input unit with scale one.

| Symbol | Definition |
|---|---|
| `Hz` | `1/s` |
| `N` | `kg*m/s^2` |
| `Pa` | `N/m^2` |
| `J` | `N*m` |
| `W` | `J/s` |
| `C` | `A*s` |
| `V` | `W/A` |
| `Ohm` | `V/A` |
| `S` | `A/V` |
| `F` | `C/V` |
| `H` | `V*s/A` |
| `Wb` | `V*s` |
| `T` | `Wb/m^2` |

Symbols are case-sensitive: `S` is conductance, `s` is time, and `T` is magnetic flux
density. Ohm is spelled `Ohm`; there is no Unicode alternative. `g` is an input unit
equal to 1/1000 kg, not a separate base dimension. There is no angle unit, so degrees
are never converted implicitly: trigonometric functions take dimensionless radians,
and converting between cyclic and angular frequency must be written out.

Celsius and Fahrenheit are not available. They need an offset as well as a scale, and
the language does not yet distinguish absolute temperatures from temperature
differences.

## Declaration initializers

A bare number used to initialize a declaration with an explicit dimension takes that
dimension's coherent unit. These two declarations have the same value:

```eqiora
parameter density: kg / m ^ 3 = 1;
parameter explicit_density: kg / m ^ 3 = 1[kg / m ^ 3];
```

To convert, write the input unit: `parameter density: kg / m ^ 3 = 1[g / cm ^ 3];`
is 1000 kg/m³. An incompatible input unit is an error. This rule applies only to
numeric declaration initializers, not to other expressions or instance arguments,
and does not change the existing rule for a contextual zero.

## Prefixes and conversion

Similarly, `parameter pressure: Pa = 2;` and `parameter pressure: Pa = 2[Pa];` are the
same, while `parameter pressure: Pa = 2[kPa];` is 2000 Pa. Unknowns have no
declaration initializer; give an initial condition instead, such as
`state pressure: Pa; initial { pressure = 2[kPa]; }`. A nonzero initial value needs
explicit units.

The available prefixes are:

| Prefix | Scale |
|---|---|
| `n` | 1/1,000,000,000 |
| `u` | 1/1,000,000 |
| `m` | 1/1,000 |
| `c` | 1/100 |
| `k` | 1,000 |
| `M` | 1,000,000 |
| `G` | 1,000,000,000 |

One prefix may precede `m`, `s`, `A`, `K`, `mol`, `cd`, `g`, or any derived unit
above. A prefix cannot precede `kg`, a dimension alias, or a unit that already has a
prefix. A symbol is first looked up as written; if it is not a known unit, one prefix
is split off, and the result must be an unambiguous known unit. Stacked prefixes,
other capitalizations, and the micro sign are not accepted. So `ms`, `kOhm`, `GPa`,
and `uF` are valid, and `mkg`, `kkOhm`, and `KOhm` are not.

Prefixed symbols are input units, not dimensions. `parameter resistance: Ohm = 1 [kOhm];`
is valid, but `kOhm` cannot be used as a type, because a dimension carries no scale.
Products, quotients, and powers of units combine dimensions and scales exactly before
the final rounding to a floating-point number. A rational power whose root is not
exact is an error.

In Python, `eqiora.lang.quantity(1, units.Ohm.prefixed("k"))` produces the same
source; Python does not evaluate unit scales itself. A quantity keeps its decimal
coefficient and exponent exactly until the compiler applies the unit's power of ten,
and the result is rounded once to binary64 (round to nearest, ties to even). Neither
the number nor the scale has to be representable on its own: `1e-400[km ^ 100]` is
`1e-100[m ^ 100]`, and `0.1[nm]` equals `1e-10[m]` with no intermediate rounding.
Infinite results and nonzero values that round to zero are errors; subnormal values
are kept, and zero becomes positive zero.

A decimal literal may be at most 256 bytes long. Its written and normalized decimal
exponents must fit in a signed 64-bit integer, and composed unit scale powers in a
signed 32-bit integer. A model's source identity keeps the exact decimal value and
unit expression, while the Model itself stores the binary64 value. Floating-point
values passed from Python are written as their shortest scientific decimal; pass a
Python `Decimal` to keep an exact decimal value. Expressions without units are
evaluated numerically as usual.

## Clock periods

Clock periods and phases are kept as exact rational numbers of seconds rather than
recovered from a rounded literal: `periodic(10[ms], phase = 0[s])` has a period of
exactly 1/100 s, and the phase may be omitted. Literal arithmetic with `+`, `-`,
`*`, and `/` allows periods such as `periodic(1[s] / 3)`. Every intermediate value
must have a reduced numerator and a positive denominator that fit in a `u64`. The
period must be positive and the phase nonnegative, and both must have the dimension
of time. These expressions cannot refer to values or use numerical operators, and
they are never converted to binary64. Exact periods do not imply any multi-clock
scheduling. Names of ordinary values do not affect unit lookup.

For reference: 10 ms is 1/100 s, 1 kOhm is 1000 Ohm, 210 GPa is
210,000,000,000 Pa, and 1 uF is 1/1,000,000 F.

## Exact dimension powers

Dimension exponents are reduced rational numbers. Write an integer directly or a
fraction in parentheses: `m ^ -1`, `m ^ (-1 / 2)`, or `Hz ^ (-1 / 2)`. Equivalent
fractions such as `m ^ (2 / 4)` and `m ^ (1 / 2)` are the same dimension. Numerators
and denominators, including the unreduced source values, must not exceed
2,147,483,647 in magnitude, and denominators must be positive.

For example, `(m ^ (-1 / 2)) ^ 2 * m` is dimensionless, which is what a real wave
amplitude on an interval needs, and `Hz ^ (-1 / 2)` equals `s ^ (1 / 2)`.
`math.sqrt(area)` has dimension length when `area` has dimension `m ^ 2`; evaluating
it requires a nonnegative value, and its derivative at zero is an error. Dimension
arithmetic does not decide which branch a numerical power takes.

## Structural aliases

`dimension name = dimension-expression;` defines a dimension alias. Top-level aliases
may be `public`, and notation follows the name. An alias cannot have `on`, `at`, a
scale, an offset, or a nominal quantity tag. All aliases in a scope are collected
before their dependencies are resolved, so order does not matter; duplicate names,
cycles, and excessive resource use are errors.

Aliases live in type expressions and do not clash with value names. Input units come
only from the catalog above, so an alias cannot introduce a hidden conversion: with
`dimension Energy = J;`, you can use `Energy` as a type but must still write the
literal as `2 [J]`, not `2 [Energy]`. An alias does not remove the nominal identity
of a connector, species, basis, or support.

Python uses the same dimension values and module resolution:

```python
from eqiora import Dimension, Module, ValueType

units = Module("units", package="org.example.mechanics")
units.dimension("Speed", Dimension(length=1, time=-1))
consumer = Module("main")
mechanics = consumer.import_module("mechanics", units)
speed_type = ValueType.real(mechanics.dimension("Speed"))
```

`Module.dimension` exports a public alias, and `ModuleRef.dimension` looks up a public
alias in the imported module, including modules parsed from `.eqi` files. The result
is an ordinary `Dimension` that you can use to build a `ValueType`. Generated source
keeps the alias declaration and the module import, although value types may be
written in their expanded SI form. Private or unknown aliases and dependency cycles
are errors.

## Presentation

A plot or table can display a value in any compatible unit without changing the value
stored in the Model. A display label cannot redefine a quantity, for example by
turning Hz into angular frequency. New units are added to the shared catalog, where
source, Python, and type checking all see them.
