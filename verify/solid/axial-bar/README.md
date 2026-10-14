# Canonical linear axial bar

This is the first benchmark-roadmap case executed from a canonical Eqiora
model. The source declares a Cartesian interval and its oriented boundary
Domains, a continuum scalar displacement Field, a strong equilibrium
Relation, and explicit essential and natural boundary Relations.

The built-in default realization lowers the method-neutral expression
`-div(E A grad(u)) = 0` to continuous P1 finite elements on an affine line
mesh. Essential values are eliminated by `AssemblyMap`; the natural end load
is a boundary-local contribution; the clamp reaction is recovered from the
uneliminated equilibrium residual.

Run the case directly:

```bash
cargo test -p eqiora-numerics --test canonical_axial_bar
```

Run every machine-readable verification contract:

```bash
cargo run -p eqiora-verify -- run --jobs 1
```

The verified claim is deliberately one-dimensional and scalar. The canonical
operator and domain contracts carry runtime spatial dimension, but built-in
multidimensional meshes and tensor elasticity remain separate roadmap work.
