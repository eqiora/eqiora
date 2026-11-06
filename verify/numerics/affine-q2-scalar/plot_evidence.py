#!/usr/bin/env python3
"""Plot rechecked original errors; no Eqiora import or candidate solve."""
import argparse
import json
import math
from pathlib import Path

import recheck_saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recheck_saved.main()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    root = Path(__file__).resolve().parent / "evidence/sine-5f8130"
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
    args.output.mkdir(parents=True, exist_ok=True)
    for mobile in (False, True):
        fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                                 figsize=(4, 7) if mobile else (12, 4.8), layout="constrained")
        for dimension, color, marker in ((2, "#176a9b", "o"), (3, "#ad4b20", "s")):
            lengths = (2, 3, 4)[:dimension]
            norm0 = math.sqrt(math.prod(lengths) / 2**dimension)
            norms = (norm0, math.pi * math.sqrt(sum(1 / length**2 for length in lengths)) * norm0)
            errors = [json.loads((root / f"d{dimension}-n{n}/comparison.json").read_text())["continuous_error_intervals"] for n in (2, 4, 8)]
            for k, ax in enumerate(axes):
                values = [(float(e[k][0]) + float(e[k][1])) / (2 * norms[k]) for e in errors]
                ax.loglog((2, 4, 8), values, marker=marker, color=color, linewidth=1.8, label=f"{dimension}D original full field")
        for ax, title in zip(axes, ("Relative L2 error", "Relative H1 seminorm error")):
            ax.set_title(title)
            ax.set_xlabel("Cells per axis, N")
            ax.set_ylabel("Dimensionless error ratio")
            ax.set_xticks((2, 4, 8), labels=("2", "4", "8"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="both", alpha=0.2)
            ax.legend(fontsize=10)
        name = "sine-errors-320.png" if mobile else "sine-errors.svg"
        fig.savefig(args.output / name, dpi=80 if mobile else 120)
        if not mobile:
            fig.savefig(args.output / "sine-errors-1440.png", dpi=120)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
