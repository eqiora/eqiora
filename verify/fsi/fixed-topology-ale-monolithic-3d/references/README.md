# Independent references

The checked-in direct Model is the semantic oracle. It contains ordinary
conservative incompressible Navier--Stokes, small-strain elastodynamic, and
conserving mechanical-interface Relations. ALE motion is absent from the Model
and enters only through the Realization selected by RFC 0070.

The exact package fixture uses immutable releases
`Eqiora.Mechanics.Interfaces@0.3.0`,
`Eqiora.Fluid@0.5.0`, and
`Eqiora.Solid.LinearElasticity@0.7.0`. Those releases own physical laws and
interfaces only; mesh, quadrature, spaces, time method, coupling, nonlinear
iteration, solver, target, and schedule remain Realization concerns.

The numerical oracles are independent of accepted solver iterates:

- every analytic Jacobian column is reconstructed and compared with centered
  complete-residual reassembly, including geometry dependence; only columns
  with disjoint structurally proven row support share a perturbation, and
  sealed harmonic mesh-motion drivers conservatively remain singleton colors;
- current coordinates and consecutive-state mesh velocity replay from the
  immutable reference topology and accepted displacement;
- `dJ/dt - J div(w)` and the complete cubic determinant path are recomputed per
  tetrahedron;
- fluid and solid interface actions are recovered from separate uneliminated
  residuals; and
- backward-Euler order is measured from `h`, `h/2`, and `h/4` solutions in one
  consistent tetrahedral P1 mass norm.

The MINI pair follows Arnold, Brezzi, and Fortin,
<https://doi.org/10.1007/BF02576171>. The endpoint differential ALE/GCL
formulation follows Förster, Wall, and Ramm,
<https://doi.org/10.1002/fld.1093>, and Fehn et al.,
<https://arxiv.org/abs/2003.07166>. These references motivate the formulation;
only the executable Eqiora checks admit this registered evidence.
