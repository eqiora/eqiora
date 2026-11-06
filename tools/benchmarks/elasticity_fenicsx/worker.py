"""Problem-local command-line bounds; execution belongs to shared_measurement."""
import argparse
from pathlib import Path
from accuracy import LEVELS


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', type=int, choices=(2,), default=2)
    parser.add_argument('--cells', type=int, choices=LEVELS, required=True)
    parser.add_argument('--mode', choices=('correctness-only', 'benchmark'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--revision')
    parser.add_argument('--build-profile')
    return parser.parse_args()
