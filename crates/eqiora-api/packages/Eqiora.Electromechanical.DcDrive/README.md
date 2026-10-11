# Eqiora.Electromechanical.DcDrive

A small package of electromechanical components, built on
`Eqiora.Electrical.Basic`, for the sampled DC-drive verification case. It adds
a rotational connector and the components to build a drive:

- a signal-controlled ideal voltage source;
- a permanent-magnet DC motor with resistance, inductance, back EMF, and
  torque coupling;
- a viscously damped inertial load;
- an ideal speed sensor that draws no power;
- the `RotationalFlange` conserving connector.

The compiler treats it like any other package. Its components mix causal
signals, dynamic fields, and conserving ports, which makes them useful test
cases for general mixed network execution. They are not a motor library:
switching electronics, saturation, and detailed component behavior are out
of scope.

