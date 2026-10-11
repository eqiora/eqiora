# Electric motor multiphysics

Status: planned; this page does not advance any electromagnetics, rotating-machinery, thermal,
cooling-flow, or motor-system capability. Publish it first as an **Unverified product
example**. Promote it to an accepted result only after its candidate, independent oracle,
falsifiers, and acceptance requirements pass under RFC 0089; evidence work does not hold back the
product example.

## What it shows

One electric motor whose electromagnetic, circuit, rotational, thermal, and coolant-flow parts
exchange power through explicit, conserving interfaces. The video shows torque production, loss
generation, temperature rise, and heat removal, one primary field at a time.

No single benchmark covers the whole system. COMPUMAG TEAM Problem 30 serves as the
electromagnetic rotating-machine precursor; it does not validate the final cooled motor. Each
subsystem has its own verification or comparison evidence, and the assembled system claims only
the cross-domain balances and operating range it actually demonstrates.

Unless the accepted case includes them explicitly, the entry does not claim industrial motor
fidelity, saturation, hysteresis, three-dimensional end effects, manufacturing tolerances,
general sliding interfaces, general circuit or control libraries, conjugate heat transfer, or
experimental validation.

## Storyboard

| Presentation time | Content |
|---|---|
| 0--2 s | Motor section, phases/circuit, prescribed operating point, rotor, materials, thermal contacts, and coolant path |
| 2--6 s | Magnetic flux density is the sole primary field; rotor angle/speed and electromagnetic torque remain visible |
| 6--10 s | Accepted loss density replaces the field and transfers power into the thermal state |
| 10--14 s | Temperature replaces loss; a short coolant-flow projection shows the accepted heat-removal path |
| 14--18 s | Torque, electrical input, losses, stored/removed heat, and balance defect settle before a labelled neutral diagram reset |

Fields are never overlaid just to look complex. Rotor motion, field phase, and thermal time run on
different physical time scales, so their mapping to playback is stated explicitly.

## Evidence needed for acceptance

The prerequisite cases cover, separately:

- magnetostatic and eddy-current analytic or manufactured verification;
- gauge, vector-space, interface, and torque/loss projection checks;
- TEAM Problem 30 torque, induced voltage, and loss comparisons under prescribed motion;
- circuit and rotational-network verification;
- heat generation, thermal conduction and contact, and coolant-flow verification;
- cross-domain power and energy balance for the assembled system.

The key system observables are torque and speed, electrical input power, electromagnetic and
ohmic losses, stored thermal energy, coolant heat removal, and the closed power/energy defect.
The TEAM 30 precursor is rejected if any retained spatial harmonic fails its source-derived slip,
torque, voltage, or loss relation; synchronous fundamental speed does not imply zero total torque
while other winding harmonics remain. The full system is rejected if its cross-domain balance
exceeds the budget fixed in advance, whatever the fields look like.

Analytic limits and balance identities use two independent oracles. TEAM comparison values and
any later experimental motor data are extracted and normalised independently of the
implementation.

## Dependencies

- low-frequency electromagnetic fields with an explicit gauge and conducting motion;
- prescribed, then coupled, rotor motion, with torque, voltage, current, and loss projections;
- circuit/drive, thermal material and contact, and coolant-flow components;
- explicit power-transfer identities between every participating domain;
- multirate accepted trajectories, and a video projection that never suggests one shared
  physical time scale.

TEAM 30 is implemented and accepted before the full system is specified. The full motor comes
last because it combines the largest number of otherwise independent pieces.

## Accessibility and promotion

The reduced-motion alternative is an ordered four-panel still: magnetic field, loss,
temperature and coolant path, and power balance. The text alternative names the transfer sequence
and operating point; hot or high-field regions are not indicated by colour alone.

Promotion requires accepted evidence for every displayed subsystem, the TEAM 30 precursor,
explicit cross-domain balance, one accepted multirate result lineage, and the common publication
check. Experimental validation is a separate decision.

## Primary source

K. Davey,
[“Induction Motor Analyses — International T.E.A.M. Workshop Problem 30”](https://www.compumag.org/jsite/images/stories/TEAM/problem30a.pdf),
with the companion
[“Analytic Analysis of Single and Three Phase Induction Motors”](https://www.compumag.org/jsite/images/stories/TEAM/problem30b.pdf).
These cover the electromagnetic precursor only, not the full multiphysics motor.
