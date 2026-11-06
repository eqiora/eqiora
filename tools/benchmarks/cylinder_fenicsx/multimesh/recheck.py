"""Clone-only replay of retained scientific observations; no native PDE execution."""
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys
import tarfile
import tempfile
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]
from field_artifact import read_candidate
from qualification import check_process, check_cold_pair, check_audit_observations
from run import load_cases, validate_reports
from shared_measurement import ComparisonError, digest

HERE = Path(__file__).resolve().parent


def check_inventory_entry(name, entry, actual_sha256):
    """Only explicitly listed metadata may be projected; all bytes stay bound."""
    allowed = []
    if name in {f'{folder}/{level}/fenicsx/result.json' for folder in ('correctness', 'benchmark')
                for level in ('coarse', 'standard', 'fine')}:
        allowed = ['environment.mpi_library']
    elif name in {f'reference-controls/{level}-process-receipt.json' for level in ('coarse', 'standard', 'fine')}:
        allowed = ['environment_controls.PATH']
    elif name == 'reference-controls/launch-plan.json':
        allowed = [f'entries.{index}.environment_controls.PATH' for index in range(3)]
    if entry['omitted_fields'] != allowed or entry['public_sha256'] != actual_sha256:
        raise ComparisonError('public projection fields or artifact bytes differ from the enumerated inventory')
    if (not allowed and (entry['scope'] != 'original bytes' or entry['original_sha256'] != actual_sha256)
            or allowed and (entry['scope'] != 'explicit public metadata projection'
                            or entry['original_sha256'] == actual_sha256)):
        raise ComparisonError('public original/projection identity is inconsistent')


@contextmanager
def replay():
    record = json.loads((HERE/'record.json').read_text())
    inventory = json.loads((HERE/'artifact-inventory.json').read_text())
    archive_path = HERE/'observations.tar.xz'
    if digest(archive_path) != record['archive_sha256']:
        raise ComparisonError('public observation archive digest mismatch')
    scratch = Path.home()/'.cache/eqiora/cylinder-public-replay'; scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        root = Path(directory)
        with tarfile.open(archive_path) as archive:
            members = archive.getmembers()
            if len(members) != len(inventory) or {m.name for m in members} != set(inventory):
                raise ComparisonError('public archive file inventory mismatch')
            for member in members:
                relative = Path(member.name)
                if not member.isfile() or relative.is_absolute() or '..' in relative.parts:
                    raise ComparisonError('public archive contains a non-file or escaping path')
                target = root/relative; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
                check_inventory_entry(member.name, inventory[member.name], digest(target))
        descriptor = root/'local-references.json'
        descriptor.write_text(json.dumps({'cases': {level: {
            'reference_manifest': str(root/'references'/level/'reference-manifest.json'),
            'sha256': record['reference_manifests'][level]} for level in record['reference_manifests']}}))
        cases = load_cases(descriptor); reports = {}; summary = {}
        for mode, folder in (('correctness-only', 'correctness'), ('benchmark', 'benchmark')):
            reports[folder] = []; summary[folder] = {}
            for case in cases:
                summary[folder][case.label] = {}
                for backend in ('eqiora', 'fenicsx'):
                    relative = f'{folder}/{case.label}/{backend}/result.json'; directory = root/Path(relative).parent
                    report = json.loads((root/relative).read_text()); receipt = json.loads((directory/'process-receipt.json').read_text())
                    check_process(receipt, label=case.label, backend=backend, expected_artifact=record['producer'])
                    if (receipt['actual_harness_revision'] != record['campaign_harness_revision']
                            or receipt['result_sha256'] != inventory[relative]['original_sha256']):
                        raise ComparisonError('original process receipt does not bind original report identity')
                    omitted = inventory[relative]['omitted_fields']
                    if omitted != (['environment.mpi_library'] if backend == 'fenicsx' else []):
                        raise ComparisonError('public report projection omitted unexpected scientific or control fields')
                    candidate = directory/'candidate-field.json'; candidate_sha = digest(candidate)
                    coordinates, primary, associated = read_candidate(candidate, candidate_sha)
                    check_cold_pair(report, coordinates, primary, associated)
                    accuracy = case.reference.candidate(coordinates, primary, associated)
                    expected = {'cold_candidate': 1, 'separate_profile': 1, 'warmups': 2 if folder == 'benchmark' else 0,
                                'timed_samples': 7 if folder == 'benchmark' else 0}
                    if (not report['profile']['required_phases_available'] or report['expected_execution_calls'] != expected
                            or report['diagnostic_execution']['additional_calls_per_tool'] != 1
                            or report['environment']['affinity'] != [20]
                            or set(report['environment']['thread_environment'].values()) != {'1'}):
                        raise ComparisonError('recorded lifecycle or execution control differs from declared campaign')
                    if folder == 'correctness' and backend == 'eqiora':
                        audit = json.loads((root/folder/case.label/'assembly-audit.json').read_text())
                        if (audit['pde_solves'] != 0 or audit['candidate_field_sha256'] != candidate_sha
                                or audit['candidate_worker_receipt_sha256'] != digest(directory/'process-receipt.json')
                                or audit['frozen_reference_manifest_sha256'] != case.sha256
                                or audit['candidate_worker_actual_harness_revision'] != record['campaign_harness_revision']
                                or any(audit['candidate_worker_artifact'].get(k) != v for k, v in record['producer'].items())):
                            raise ComparisonError('stored assembly-only audit is bound to different cold inputs')
                        check_audit_observations(case.reference, audit, primary, associated, accuracy)
                    reports[folder].append(report)
                    entry = {'agreement': accuracy['agreement'], 'scaled_residual_inf': accuracy['scaled_residual_inf']}
                    if folder == 'benchmark':
                        measurement = report['execution_measurement']; samples = [s['wall_seconds'] for s in measurement['samples']]
                        if (statistics.median(samples) != measurement['median_wall_seconds']
                                or min(samples) != measurement['min_wall_seconds'] or max(samples) != measurement['max_wall_seconds']):
                            raise ComparisonError('stored timing summary differs from original samples')
                        entry.update(median_seconds=statistics.median(samples), range_seconds=[min(samples), max(samples)])
                    summary[folder][case.label][backend] = entry
            validate_reports(reports[folder], cases, mode)
        yield record, cases, reports, summary


if __name__ == '__main__':
    with replay() as (_, _, _, summary):
        print(json.dumps({'state': 'complete scientific and stored-audit replay passed',
                          'scope': 'original scientific bytes and labelled public metadata projections; no original-source requalification or PDE solve',
                          'cases': summary}, indent=2, allow_nan=False))
