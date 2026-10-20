# Continuous expression chain rule

The source compiler differentiates admitted real scalar polynomial, elementary-function and quotient expressions through the
shared formal partial transform. Independent Parameters remain fixed in physical time.
`time()` is the enclosing continuous coordinate in seconds; periodic and event contexts do not
supply that coordinate. Canonical expressions retain State rate symbols and typed operator
applications. Ordinary numerical lowering projects those applications with the existing scalar
owner while exact Model artifacts retain their definitions.

The executable owner is `crates/eqiora/tests/time_expression_derivatives.rs`. It checks nonzero
multi-State rates, operator and alias composition, typed constant zero, explicit time dependence,
and the nonlinear stored quantity through the common implicit initialization and time lifecycle.
Artifact replay must admit the same lowering proof. Algebraic and discrete dependencies,
nonsmooth value rules and ambient bare `time` reject.

At `q=3`, `y=5`, with rates `q̇=2/s`, `ẏ=3/s`, the elementary probes use
`sin(q-3)`, `cos(q-3)`, `exp(q-3)`, `log(q-2)` and `sqrt(y-1)`.
Their rates are respectively `2/s`, `0/s`, `2/s`, `2/s` and `3/(4s)`.
The quotient rate of `q/(y-1)` is `(2*4-3*3)/16=-1/(16s)`.
For the nested `sin((derivative(q*q)-12/s)*1s)` expression, additionally set
`q̈=3/s²`. Its argument is zero and its rate is
`(2*q̇²+2*q*q̈)*1s=(8+18)/s=26/s`. This checks expansion of a compound
inner time derivative through an elementary call. These independent values are exactly
representable in binary64.

This case does not claim second-order reduction, the general smooth-function catalog,
material/ALE derivatives, or derivatives through resets.
