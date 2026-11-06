"""Shared cylinder worker inputs; the process collector owns resource controls."""
import argparse
from pathlib import Path


def arguments(tool):
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-manifest', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--case-label')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('correctness-only', 'benchmark'), required=True)
    if tool == 'eqiora':
        parser.add_argument('--wheel', type=Path, required=True)
        parser.add_argument('--revision', required=True)
        parser.add_argument('--build-profile', choices=('release',), required=True)
    return parser.parse_args()
