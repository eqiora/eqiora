# Eqiora.Solid

`PlaneStrainLinearElasticity2d` and `PlaneStressLinearElasticity2d` are
ready-made small-strain isotropic models. Both take Young's modulus and
Poisson's ratio; the name says which two-dimensional assumption applies.

`YoungModulus` and `PoissonRatio` are reusable property contracts.
`PlaneStrainMaterial2d` and `PlaneStressMaterial2d` accept a typed material
composition built from those contracts.

The same package exposes `IsotropicBalance2d` and
`DisplacementTractionInterface2d` for projects that work directly with Lamé
parameters or want to replace one part. Its boundary components set fixed,
free, or field-driven displacement or traction.
