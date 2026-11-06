# Analytic reference

The [existing MINI Stokes oracle](../../../fluid/simplicial-mini-stokes-2d/README.md)
owns the manufactured polynomial, independent error quadrature and convergence
budgets. The [case explanation](../README.md) derives the physical similarity
forcing by differentiation. The exact pressure has zero mean; integrating the
momentum equation gives reaction plus body-force balance.

The integration test evaluates P1 barycentric functions and the normalized
bubble `27 lambda_0 lambda_1 lambda_2` directly from their definitions, including
physical derivatives. It reads only ordinary accepted Result coefficient blocks.
None of the analytic fields or acceptance thresholds come from Eqiora output.
