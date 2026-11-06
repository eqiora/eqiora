"""Poisson-specific worker command line and declared mesh levels."""
from pathlib import Path
from accuracy import LEVELS, declared_levels

def worker_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=2)
    parser.add_argument("--cells", type=int, choices=LEVELS, required=True)
    parser.add_argument("--mode", choices=("correctness-only", "benchmark"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--build-profile")
    args = parser.parse_args()
    if args.cells not in declared_levels(args.dimension):
        parser.error("mesh level has no declared reference for this dimension")
    return args
