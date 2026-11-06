"""Recheck saved vector fields and complete manufactured convergence sequences."""
import argparse
import json
from pathlib import Path
from accuracy import ComparisonError, LEVELS, check_accuracy, check_convergence, check_equilibrium


def validate_reports(reports):
    rows = {tool: {} for tool in ('eqiora', 'fenicsx')}
    for report in reports:
        tool, cells = report['tool'], report['cells_per_axis']
        if tool not in rows or cells not in LEVELS or cells in rows[tool]:
            raise ComparisonError('unexpected or duplicate tool/mesh report')
        # Do not trust a serialized PASS flag or serialized accuracy numbers.
        rows[tool][cells] = check_accuracy(report['coordinates'], report['values'], cells)
        probe = report['equilibrium_probe']
        check_equilibrium(probe['reaction'], probe['body_force'])
        for solve in (report['solve'], probe['solve']):
            if (solve['algorithm'] != 'conjugate-gradient' or solve['preconditioner'] != 'identity'
                    or (solve['relative_tolerance'], solve['absolute_tolerance'], solve['maximum_iterations']) != (1e-12, 1e-14, 10000)):
                raise ComparisonError('saved solve policy differs from protocol')
            initial, residual = solve['initial_residual_norm'], solve['true_residual_norm']
            import math
            if not math.isfinite(initial) or not math.isfinite(residual) or initial < 0 or residual < 0:
                raise ComparisonError('invalid saved residual norms')
            target = max(1e-14, 1e-12*initial)
            if solve['residual_target'] != target or residual > target:
                raise ComparisonError('saved true residual fails analytic target')
    return {tool: {'accuracy': rows[tool], 'observed_orders': check_convergence(rows[tool])} for tool in rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_reports([json.loads(path.read_text()) for path in args.reports]), indent=2, allow_nan=False))
