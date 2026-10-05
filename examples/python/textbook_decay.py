"""Run the decay model shared by Get started and the ODE lesson."""

from __future__ import annotations

import argparse
from pathlib import Path

import eqiora


OUTPUT_TIMES_S = (0.25, 0.5, 1.0)


def solve(source_path: Path) -> tuple[tuple[float, float], ...]:
    """Return the requested (time in seconds, dimensionless value) samples."""

    model = eqiora.compile(path=source_path)
    field = model.field(model.field_ids[0])
    plan = eqiora.resolve(
        model,
        temporal=eqiora.time.Tsitouras45(
            initial_step_s=0.01,
            relative_tolerance=1.0e-9,
            absolute_tolerances={(field, 0): 1.0e-11},
        ),
    )
    result = eqiora.run(
        plan,
        state=eqiora.State.initial(plan),
        until_s=OUTPUT_TIMES_S[-1],
        output_times_s=OUTPUT_TIMES_S,
    )
    series = result.series(field)
    times = series.time.numpy()
    values = series.values.numpy()
    return tuple(
        (float(time_s), float(value))
        for time_s, value in zip(times, values, strict=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=Path("decay.eqi"))
    args = parser.parse_args()
    for time_s, value in solve(args.model):
        print(f"t={time_s:.2f}, x={value:.10f}")


if __name__ == "__main__":
    main()
