"""Collect the input-declared cylinder cases through the shared process owner."""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import subprocess
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mesh_levels import LEVELS, mesh_level
from reference_reader import load_reference
from shared_measurement import ROOT, ComparisonError, digest
from qualification import validate_qualification


@dataclass(frozen=True)
class FrozenCase:
    label: str
    manifest: Path
    sha256: str
    reference: object

    def worker_arguments(self):
        return ('--case-label', self.label, '--reference-manifest', str(self.manifest),
                '--reference-sha256', self.sha256)


def load_cases(path):
    payload = json.loads(path.read_text())
    if set(payload) != {'cases'} or set(payload['cases']) != set(LEVELS):
        raise ComparisonError('campaign must contain exactly the three input-declared mesh levels')
    cases = []
    for label in LEVELS:
        entry = payload['cases'][label]
        if set(entry) != {'reference_manifest', 'sha256'} or not re.fullmatch('[0-9a-f]{64}', entry['sha256']):
            raise ComparisonError('each case requires its independently frozen manifest path and full SHA256')
        manifest = (path.parent/entry['reference_manifest']).resolve()
        reference = load_reference(manifest, entry['sha256'])
        if (reference.data.get('mesh_level') != label
                or reference.data.get('mesh_controls') != mesh_level(label).controls()):
            raise ComparisonError('case label/input controls differ from its independently calibrated mesh')
        cases.append(FrozenCase(label, manifest, entry['sha256'], reference))
    if len({case.reference.data['geometry_digest'] for case in cases}) != 1:
        raise ComparisonError('all mesh levels must retain identical authored geometry')
    if len({case.reference.data['model_digest'] for case in cases}) != 1:
        raise ComparisonError('all mesh levels must retain the same compiled physical model')
    if len({case.reference.data['mesh_digest'] for case in cases}) != len(cases):
        raise ComparisonError('three levels must identify distinct mesh realizations')
    return cases


def validate_reports(reports, cases, mode):
    by_label = {case.label: case for case in cases}
    seen, summary = set(), {}
    for report in reports:
        key = (report.get('case_label'), report.get('tool'))
        if key[0] not in by_label or key[1] not in ('eqiora', 'fenicsx') or key in seen or report['mode'] != mode:
            raise ComparisonError('unexpected/duplicate case-tool report or execution mode')
        seen.add(key); case = by_label[key[0]]
        if report['mesh']['digest'] != case.reference.data['mesh_digest']:
            raise ComparisonError('worker mesh differs from the independently frozen case')
        extra = {name: {'association': field['association'], 'values': np.asarray(field['values'])}
                 for name, field in report['associated_fields'].items()}
        checked = case.reference.candidate(report['coordinates'], report['values'], extra)
        if mode == 'benchmark':
            measurement = report['execution_measurement']
            if measurement['warmups'] != 2 or len(measurement['samples']) != 7 or not report['profile']['required_phases_available']:
                raise ComparisonError('ordinary timing protocol or required native phases are incomplete')
        summary.setdefault(key[0], {})[key[1]] = {'accuracy': checked,
            'raw_report': f'{key[0]}/{key[1]}/result.json',
            'execution_measurement': report.get('execution_measurement')}
    if seen != {(label, tool) for label in LEVELS for tool in ('eqiora', 'fenicsx')}:
        raise ComparisonError('complete three-case/two-tool comparison is required')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, help='required successful correctness receipts before benchmark mode')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eqiora-python', type=Path, required=True)
    parser.add_argument('--fenicsx-python', type=Path, required=True)
    parser.add_argument('--eqiora-wheel', type=Path, required=True)
    parser.add_argument('--eqiora-revision', required=True)
    parser.add_argument('--eqiora-build-profile', choices=('release',), required=True)
    parser.add_argument('--mode', choices=('correctness-only', 'benchmark'), required=True)
    parser.add_argument('--cpu', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.cpu not in os.sched_getaffinity(0):
        raise ComparisonError('campaign requires a fresh output path and an available allocated CPU')
    if not re.fullmatch('[0-9a-f]{40}', args.eqiora_revision):
        raise ComparisonError('declare the actual full optimized producer source')
    for path in (args.references, args.eqiora_python, args.fenicsx_python, args.eqiora_wheel):
        if not path.is_file(): raise ComparisonError(f'required artifact/interpreter is missing: {path}')
    if args.mode == 'benchmark' and (args.qualification is None or not args.qualification.is_file()):
        raise ComparisonError('benchmark requires reviewed correctness/process/assembly-audit qualification for all cases')
    try:
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError as error:
        raise ComparisonError('campaign requires a clean committed source checkout') from error
    if dirty: raise ComparisonError('freeze the committed harness before a scientific campaign')
    cases = load_cases(args.references)
    qualification = (validate_qualification(args.qualification, cases, {
        'sha256': digest(args.eqiora_wheel), 'source_revision': args.eqiora_revision,
        'build_profile': args.eqiora_build_profile}) if args.mode == 'benchmark' else None)
    # The primary owns this general process interface; no private launch loop.
    from shared_campaign import ComparisonCase, collect_reports
    reports = collect_reports(args, [ComparisonCase(case.label, case.worker_arguments()) for case in cases],
                              problem_root=Path(__file__).resolve().parent)
    if reports is None: raise ComparisonError('worker failed; original logs retained, no completed comparison')
    summary = {'mode': args.mode, 'stable_timing_claim': False, 'continuum_convergence_claim': False,
               'qualification': qualification, 'cases': validate_reports(reports, cases, args.mode)}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print('PASS three declared cylinder meshes; all frozen correctness and ordinary-lifecycle gates')


if __name__ == '__main__':
    try:
        main()
    except ComparisonError as error:
        print(json.dumps({'error': str(error), 'complete_comparison': False}))
        raise SystemExit(2)
