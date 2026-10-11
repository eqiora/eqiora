# org.example.dc_motor_control

An example package for a linear permanent-magnet DC drive under periodic
proportional control. It depends directly on `Eqiora.Electrical.Basic`, for the
electrical ground, and on `Eqiora.Electromechanical.DcDrive`, for the
actuator, motor, load, sensor, and rotational connector.

`SampledPController` is defined in this package. Its 10 ms clock is part of the
model; the numerical step size and execution scheduling are chosen separately
when the model is realized and run.

The `drive` dependency in `package.json` records the semantic digest of
`Eqiora.Electromechanical.DcDrive@0.1.0`. Resolution is offline, and a changed
package, digest, or dependency edge is rejected before compilation.
