#!/usr/bin/env python3
"""Render the current mixed-boundary example without changing its mesh or solve."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def produce(output: Path) -> None:
    examples = Path(__file__).resolve().parents[2] / "examples/python"
    sys.path.insert(0, str(examples))
    try:
        from mixed_boundary_elasticity import solve

        plan, result = solve()
    finally:
        sys.path.pop(0)
    import eqiora.matplotlib as eqplot

    figure = eqplot.plot_deformed_field(result, field=plan.fields[0], scale=1)
    figure.set_size_inches(8, 5.2)
    axes = figure.axes[0]
    axes.set_title("Mixed-boundary displacement")
    axes.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.12)
    figure.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    produce(parser.parse_args().output)
