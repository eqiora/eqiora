# Native polynomial patch evidence

These are the original field, Model, Plan, mesh, authored source and comparison
bytes from the accepted second native attempt at the revision in `inventory.json`.
The producer performed one ordinary solve in each dimension. Each dimension
retained all 25 or 125 nodal coefficients and all 5 or 25 upper-face physical
reaction coordinates. The independent rational comparator accepted its unchanged
coefficient, essential-data, free-residual and reaction gates.

The first attempt failed model compilation before any solve because its authored
array was not a spatial tensor. The corrected source uses the same analytic
outward flux as scalar data on each Cartesian face. Both dimensions passed a
compile/resolve-only preflight before the second attempt. There was no tolerance
change. The original failed attempt and machine-local process receipts remain
external; this inventory binds the successful original receipts by their hashes.
It does not present their contents as clone-available process evidence.

To repeat the mathematical checks without Eqiora, from this case's directory run:

```sh
python3 reference/polynomial.py evidence/polynomial-5f8130/d2/polynomial-input.json --output "$scratch/d2.json"
python3 reference/polynomial.py evidence/polynomial-5f8130/d3/polynomial-input.json --output "$scratch/d3.json"
```

This rechecks the complete fields and physical reactions with the existing
independent rational oracle. It does not rerun the PDE or reopen native artifacts.
The inventory records the SHA-256 of every original retained scientific file.
This polynomial patch is a reproduction and boundary test; it does not establish
smooth-solution convergence, distributed execution or performance.
