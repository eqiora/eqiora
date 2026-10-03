# Expected results

The executable assertions live in
[`tensor_contractions.rs`](../../../../crates/eqiora/tests/tensor_contractions.rs).
The source fixtures retain explicit independent Relation sides, including the
expanded stiffness terms. The test checks complete value types and each ordered
real/imaginary component through ordinary Model evaluation. Expectations are
not regenerated from evaluator output.
