# Model inputs

The source fixture and Python Module construction live in
[`python_authored_scalar_form.rs`](../../../../crates/eqiora-python/tests/python_authored_scalar_form.rs).
They retain the exact Domain, Field, four essential boundaries, held coefficient
and load Parameters, test dimension, Observable, and authored Formulation.

Only those semantic inputs and the selected solution coefficients and energy
value enter this claim. No whole-file, package-root, or generated-tree digest is
an oracle for the mathematics.
