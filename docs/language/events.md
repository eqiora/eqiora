# Authored crossing events

An event owns a real scalar zero-crossing guard and an explicit direction:

```eqiora
event impact = crossing(height, direction = falling);
let reflected_velocity at impact = -restitution * pre(velocity);
relation reset at impact {
  next(height) = 0[m];
  next(velocity) = reflected_velocity;
}
```

The guard keeps its physical dimension: it is a quantity that crosses zero, not a Boolean test
or an equation. `direction` must be given and must be `rising`, `falling`, or `any`. Events are
private to their Model or Component, and each Component instance has its own events even when
names and guards are identical. All Relations that name the same event belong to that one event.
Fields, ports, and borrowed clocks keep their periodic-clock behaviour.

In a reset, `pre(x)` is the value of a continuous state just before the event and `next(x)` its
value just after. Reset equations are solved together, and states that a reset does not target
keep their values. The interpreter solves the whole reset, followed by the continuous
consistency solve, before committing anything; if either fails, no part of the new state is kept.

An alias declared `at` an event can only be used by that event's resets. Using it, directly or
through another alias, at a different event or in an ordinary continuous Relation is an error.
An `at event` assertion must depend on a reset of that event or on one of its aliases; it cannot
turn a constant or an ordinary state read into an event dependency. Aliases add no unknowns or
equations.

The interpreter arms a crossing once the guard has an accepted value outside the tolerance band
on the side the direction requires. The event stays armed while the guard approaches zero
through the band. The crossing is then located at zero; the tolerance band does not shift it. If
a reset puts the guard exactly on zero, the guard must move away again before it can fire.
Localization and solver tolerances are execution settings; the guard, direction, and reset
equations are part of the Model.

## Explicit reset priority

`crossing` accepts an optional last argument `priority = SIGNED_INTEGER_LITERAL`, a signed 64-bit
integer that defaults to zero. When several events fire at the same instant, they are processed
from highest to lowest priority. An event whose reset targets any state already reset by a
higher-priority event is skipped entirely, including its updates to other states. Events of equal
priority that both survive must reset different states; they are then applied together, reading
the same pre-event values. Declaration order never breaks a tie.

For example, with A (priority 2, resets x), B (priority 1, resets x and y), and C (priority 0,
resets y), A and C fire. B is skipped as a whole, so it does not block C. If a winning reset
fails, the whole step is rolled back; a skipped event is not tried instead. A skipped event's
crossing is consumed, stays consumed after a checkpoint restart, and can fire again once its
guard has moved away and returned. Activation records list only the events that fired. Priority
is saved with the Model and is part of its identity. The Diffsol event path does not support
nonzero priorities.

## Coincident activations

A periodic clock ticks at exact times. If an armed guard evaluates to zero at a tick time, the
event fires together with the tick. This is a numerical test at the tick time; it does not prove
that two roots coincide mathematically. If a guard is nonzero but its root bracket overlaps a
tick, the run stops with an error naming the event and the clock. Two events whose brackets
overlap without a common zero are also an error; the localization tolerance never decides which
comes first.

The first microstep solves the Relations of all events and ticks firing at that instant
together. Every `pre` and every `sample` at that instant reads the same values from before the
step. Different events and ticks may update different states. An event and a tick that both
set the same `next` state are always an error, even if another event would otherwise skip the
first one. Conflicts between events follow the priority rules above; two surviving events of
equal priority that set the same state cause an error naming both. Splitting one event's
equations across several Relations still makes one event.

After the resets and the consistency solve, any guards that the reset itself crossed fire in
further event-only microsteps, each reading the state left by the previous one. The tick and its
input samples happen only once. State, clock progress, input positions, and outputs are
committed only when no further events fire. A conflicting reset, a failed solve, or too many
zero-time iterations leaves the previous state in place, and the error identifies the events
involved. Runs cannot continue through Zeno behaviour.

For a concrete affine example, start `x = y = z = 0` with slopes `1`, `2` and `0` in
consistent units. At a period-1 tick at time 1, sample `x + y` into clocked memory while
guards `x - 1` and `y - 2` reset `x = 10` and `y = 20`. Memory becomes 3. A guard
`x - 5` crossed by that reset can set `z = 7` in the next microstep. The stabilized state
is `(10, 20, 7)` with memory 3; at time 1.25 it is `(10.25, 20.5, 7)` with memory 3.
These values follow directly from the linear equations with quarter-second steps.

`ExecutionSession` steps through the same event boundaries as ordinary runs. Its activation
sequence lists the events and ticks that fired in each microstep of the most recent boundary.
In-memory checkpoints keep the compiled program, the request, the state, event arming, clock
progress, input positions, and which outputs are present. Resuming from a checkpoint does not
repeat initialization or a tick that was already processed. Checkpoints cannot yet be saved to
disk.

A thermostat that starts at 20 K with a slope of +1 K/s can reverse its slope at 22 K and 18 K
with `next(rate) + pre(rate) = 0[K/s]`. Worked by hand, the switches occur at 2, 6, and
10 seconds, and at 12 seconds the temperature is 20 K with slope -1 K/s. The hysteresis comes
from the state and its resets; a [conditional value](conditionals.md) on its own creates neither
hysteresis nor an event.

## Finite mode-controlled components

A component can switch between modes using an enum state, an explicit initial mode, and ordinary
events. Switching changes values inside a fixed set of equations: a `case` selects a value, but it
never adds or removes equations, changes the topology, or changes which variables are solved for.
For example:

```eqiora
enum Mode { Heating, Cooling }
model Thermostat() {
  state mode: Mode;
  state temperature: K;
  initial { mode = Mode.Heating; temperature = 20[K]; }
  relation flow {
    derivative(temperature) = case mode {
      Mode.Heating => 1[K/s], Mode.Cooling => -1[K/s]
    };
  }
  event upper = crossing(temperature - 22[K], direction = rising);
  event lower = crossing(temperature - 18[K], direction = falling);
  relation cool at upper { next(mode) = Mode.Cooling; }
  relation heat at lower { next(mode) = Mode.Heating; }
}
```

An enum value is identified by its declaration and has no numeric encoding or derivative. The
initial mode must be assigned. Every `case` needs exactly one arm per enum member, and the order
of the arms has no effect. Use `execution_session` to inspect mode values and the events that
fired.

All `pre(mode)` and other `pre` reads in one microstep see the values from before the step, even
if the reset changes the mode. If two surviving events of equal priority write the same state,
the whole step is rejected, and reordering the declarations does not change that. A failed reset
or consistency solve leaves the old mode and the rest of the state unchanged.

Holding or enabling a value is written as equations. A reset leaves states it does not target
unchanged, and a mode-dependent reset can use `pre(command)` to keep the current command. A real
command memory needs its own evolution equation, such as `derivative(command) = 0[K/s^2]`. A
Fault mode can set the plant rate to zero while keeping the command, and the recovery event can
then reset the command explicitly. Entering Fault does not disable events or freeze the
component's outputs; signal outputs keep following their clocks.

The [mode tests](../../crates/eqiora/tests/finite_mode_control.rs) run this thermostat and a
thermal controller with a fault mode. The thermostat switches at 2, 6, and 10 seconds. In the
fault case, the fault at 2.5 seconds keeps the temperature at 21.5 K and the command at
-1 K/s; recovery at 4 seconds changes the command to +1 K/s. Later switches occur at 4.5 and
8.5 seconds, and the temperature at 10 seconds is 19.5 K. All of these follow from integrating
the piecewise constant rates. Checkpoints taken after each switch keep the mode, command, reset
memory, and event arming, and resuming does not repeat a switch. These are product tests; the
separate thermostat and fault-plant benchmark cases are not affected by them.

Statechart features are not supported: hierarchical states (which need ancestry and entry/exit
ordering), parallel regions (which need activation and write-conflict rules), and history states
(which need to restore a saved configuration). Switching between different sets of equations
would additionally need balance and regularity checks for each mode and a consistent
reinitialization. Dynamic topology and saving event checkpoints to disk are also not supported.

In Python, `Component.event(name, guard, *, direction=..., priority=0)` returns a `q.Event`
handle. Pass it as `at=event` to `relation(...)` or `let_alias(...)`; a Clock or an Event from
another component cannot be used instead. Python generates the same source as hand-written
`.eqi` and has no separate event evaluator. The Python helpers do not yet cover a complete ODE
model, so crossing dynamics are run from compiled source.

Resting contact, several roots of one guard within a single numerical step, saving event
checkpoints to disk, events in the Diffsol ODE path, and general hybrid systems are not
supported.
