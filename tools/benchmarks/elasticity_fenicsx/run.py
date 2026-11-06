"""Run the declared elasticity sequence through the shared fresh-child collector."""
import argparse
import json
import os
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_campaign import ComparisonCase, collect_reports
from shared_measurement import ComparisonError
from accuracy import LEVELS
from validate import validate_reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eqiora-python', type=Path, required=True)
    parser.add_argument('--fenicsx-python', type=Path, required=True)
    parser.add_argument('--eqiora-wheel', type=Path, required=True)
    parser.add_argument('--eqiora-revision', required=True)
    parser.add_argument('--eqiora-build-profile', choices=('release',), required=True)
    parser.add_argument('--mode', choices=('correctness-only', 'benchmark'), required=True)
    parser.add_argument('--cpu', type=int, required=True)
    args = parser.parse_args()
    args.dimension = 2
    if not re.fullmatch('[0-9a-f]{40}', args.eqiora_revision):
        raise ComparisonError('actual producer revision must be a full lowercase Git object ID')
    if args.cpu not in os.sched_getaffinity(0):
        raise ComparisonError('allocated CPU is outside caller affinity')
    if args.output.exists():
        raise ComparisonError('fresh comparison output directory must not already exist')
    for path in (args.eqiora_python, args.fenicsx_python, args.eqiora_wheel):
        if not path.is_file():
            raise ComparisonError(f'required installed artifact/interpreter is missing: {path}')
    cases = [ComparisonCase(f"n{cells}", ("--dimension", str(args.dimension), "--cells", str(cells)))
             for cells in LEVELS]
    reports = collect_reports(args, cases, problem_root=Path(__file__).resolve().parent)
    if reports is None:
        raise ComparisonError('worker failed; retained logs are authoritative, not a completed comparison')
    summary = {'mode': args.mode, 'dimension': 2, 'levels': list(LEVELS),
               'stable_timing_claim': False, 'validated': validate_reports(reports)}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print('PASS elasticity analytic accuracy, true residual, nonzero equilibrium and complete convergence')


if __name__ == '__main__':
    try:
        main()
    except ComparisonError as error:
        print(json.dumps({'error': str(error), 'complete_comparison': False}))
        raise SystemExit(2)
