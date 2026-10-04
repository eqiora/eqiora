# Model inputs

The source fixture and Python Module construction live in
[`python_authored_scalar_form.rs`](../../../../crates/eqiora-python/tests/python_authored_scalar_form.rs).
They retain the exact Domain, Field, held coefficient and load Parameters, test
dimension, Observable, and authored Formulation. One fixture has four essential
sides; the other has an essential left side and exact zero-flux Laws on the other
three sides. Its direction vanishes only on the left.

Only those semantic inputs and the selected solution coefficients and energy
value enter this claim. No whole-file, package-root, or generated-tree digest is
an oracle for the mathematics.
