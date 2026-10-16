# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Independent tensor sine-mode solution; no Eqiora import or native output."""
import argparse
from pathlib import Path

import mpmath as mp

mp.mp.dps = 80


def moments(nodes, values):
    """Exact segment integrals of p² and p sin(pi x)."""
    mass = mp.mpf(0)
    cross = mp.mpf(0)
    for x, y, a, b in zip(nodes, nodes[1:], values, values[1:]):
        length = y - x
        mass += length * (a*a + a*b + b*b) / 3
        cross += (a * mp.cos(mp.pi*x) - b * mp.cos(mp.pi*y)) / mp.pi
        cross += (b-a) * (mp.sin(mp.pi*y) - mp.sin(mp.pi*x)) / (length * mp.pi**2)
    return mass, cross


def row(n, d):
    """Continuous errors from the FEM and TPFA discrete sine amplitudes."""
    h = mp.mpf(1) / n
    theta = mp.pi * h
    stiffness = 4 * mp.sin(theta/2)**2 / h
    mass_eigenvalue = h * (2 + mp.cos(theta)) / 3
    nodes = [(1 - 1/mp.sqrt(3))/2, (1 + 1/mp.sqrt(3))/2]
    load = h * sum((1-t) * mp.cos(theta*t) for t in nodes)
    fem_amplitude = mp.pi**2 * load**d / (stiffness * mass_eigenvalue**(d-1))
    fvm_amplitude = mp.pi**2 * h*h / (4 * mp.sin(theta/2)**2)
    fem_nodes = [i*h for i in range(n+1)]
    fvm_nodes = [mp.mpf(0), *((i + mp.mpf("0.5"))*h for i in range(n)), mp.mpf(1)]
    errors = []
    for amplitude, locations in [(fem_amplitude, fem_nodes), (fvm_amplitude, fvm_nodes)]:
        mass, cross = moments(locations, [mp.sin(mp.pi*x) for x in locations])
        squared_error = amplitude**2 * mass**d - 2*amplitude * cross**d + mp.mpf("0.5")**d
        errors.append(mp.sqrt(squared_error))
    return h, *errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    numerics = Path(__file__).resolve().parents[2]
    for dimension, name, levels in [
        (2, "cartesian-poisson-fem-fvm", [4, 8, 16, 32]),
        (3, "cartesian-poisson-3d-fem-fvm", [4, 8, 16]),
    ]:
        lines = ["cells_per_axis,h_max,fem_l2,fvm_l2"]
        lines.extend(
            str(n) + "," + ",".join(mp.nstr(value, 25) for value in row(n, dimension))
            for n in levels
        )
        expected = "\n".join(lines) + "\n"
        path = numerics / name / "expected/convergence.csv"
        if args.check:
            if path.read_text() != expected:
                raise SystemExit(f"independent sine-mode reference differs: {path}")
        else:
            path.write_text(expected)
        print(f"{'checked' if args.check else 'wrote'} {path}")


if __name__ == "__main__":
    main()
