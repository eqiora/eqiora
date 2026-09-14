"""Run the maintained divider using the installed package and common finite Plan."""
from pathlib import Path
import tempfile

import eqiora


def resolve(model):
    return eqiora.resolve(model, solve=eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
        relative_tolerance=1e-12, absolute_tolerance=1e-14,
        maximum_iterations=100,
    ))


def main():
    project = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="eqiora-divider-") as scratch:
        store = Path(scratch) / "store"
        store.mkdir()
        lock = eqiora.add_bundled_dependency(
            project, store, "Eqiora.Electrical.Basic", version="0.1.0")
        model = eqiora.compile_package(store, lock, entry="VoltageDivider")
        plan = resolve(model)
        result = eqiora.run(plan, state=eqiora.State.initial(plan))
        outputs = (
            ("current", "A"), ("midpoint", "V"), ("upper_power", "W"),
            ("lower_power", "W"), ("source_power", "W"),
        )
        for name, unit in outputs:
            print(f"{name}: {result.observe(model.observable(name)).value} {unit}")


if __name__ == "__main__":
    main()
