"""Reproduce the accuracy-only figure from validated public observations."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

from accuracy import LEVELS, ROOT, check_accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=ROOT / "docs/performance/2026-11-05-poisson-fenicsx-correctness.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/performance/2026-11-05-poisson-fenicsx-accuracy.png")
    args = parser.parse_args()
    data = json.loads(args.receipt.read_text())
    rows = {tool: sorted((row for row in data["rows"] if row["tool"] == tool),
                         key=lambda row: row["cells_per_axis"]) for tool in ("eqiora", "fenicsx")}
    references = {}
    for group in rows.values():
        if tuple(row["cells_per_axis"] for row in group) != LEVELS:
            raise ValueError("figure requires the four declared mesh levels for both tools")
        for row in group:
            n = row["cells_per_axis"]
            coordinates = np.array([(i/n, j/n) for i in range(n+1) for j in range(n+1)])
            values = np.array([float.fromhex(value) for value in row["nodal_values_float64_hex"]])
            checked = check_accuracy(coordinates, values, n)
            references[n] = checked["analytic_discrete_l2"]
            observed = row["accuracy"]["continuous_l2"]
            if not math.isfinite(observed) or abs(observed-checked["continuous_l2"]) > checked["comparison_allowance"]:
                raise ValueError("figure's reported error differs from its observed field")

    fig, ax = plt.subplots(figsize=(8.4, 5), layout="constrained")
    ax.loglog(LEVELS, [references[n] for n in LEVELS],
              color="#374151", ls=":", lw=2.3, label="Analytic discrete reference", zorder=1)
    ax.loglog(LEVELS, [row["accuracy"]["continuous_l2"] for row in rows["eqiora"]],
              color="#2563eb", marker="s", ms=7, lw=1.4, label="Eqiora development wheel", zorder=2)
    ax.loglog(LEVELS, [row["accuracy"]["continuous_l2"] for row in rows["fenicsx"]],
              color="#c2410c", marker="o", mfc="none", mew=1.7, ms=11, lw=0, label="FEniCSx 0.11", zorder=3)
    ax.set_xticks(LEVELS, labels=[str(n) for n in LEVELS])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Cells per axis")
    ax.set_ylabel(r"Continuous $L^2$ field error [m]")
    ax.set_title("2D Poisson: matched Q1 accuracy")
    ax.grid(True, which="both", alpha=.2)
    ax.legend(frameon=False)
    ax.text(.03, .04, "Observed orders 1.992–1.999\nCorrectness smoke; no performance results",
            transform=ax.transAxes, fontsize=9, color="#374151")
    fig.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
