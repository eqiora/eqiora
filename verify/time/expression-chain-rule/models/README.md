# Executable source models

`crates/eqiora/tests/time_expression_derivatives.rs` owns the exact source fixtures. Its
multi-State model declares `stored(x,y)=x*x*y`, States q/y and a fixed Parameter; its storage
model compares `derivative(q*q)=2[1/s]` with `2*q*derivative(q)=2[1/s]`. Both declare q(0)=1
for the time solve. The shared source factory varies only the authored left-hand expression.
