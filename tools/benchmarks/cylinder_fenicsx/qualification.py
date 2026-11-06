"""Validate existing correctness/process/audit receipts before any timing child starts."""
import json
from pathlib import Path
import re
import numpy as np
from shared_measurement import ROOT, ComparisonError, digest, shared_sources
from mesh_levels import LEVELS
from field_artifact import read_candidate
from budget import compare

PROBLEM_ROOT = Path(__file__).resolve().parent


def verified_descriptor(base, descriptor):
    if set(descriptor) != {'path', 'sha256'} or not re.fullmatch('[0-9a-f]{64}', descriptor['sha256']):
        raise ComparisonError('qualification artifacts require explicit paths and full SHA256 values')
    path = (base/descriptor['path']).resolve()
    if not path.is_file() or digest(path) != descriptor['sha256']:
        raise ComparisonError('correctness qualification artifact hash mismatch')
    return path


def check_process(receipt, *, label, backend, expected_artifact):
    if (receipt.get('case') != label or receipt.get('backend') != backend
            or type(receipt.get('terminal_exit_code')) is not int or receipt['terminal_exit_code'] != 0
            or receipt.get('harness_dirty') is not False
            or not re.fullmatch('[0-9a-f]{40}', receipt.get('actual_harness_revision', ''))
            or not receipt.get('finished_at') or receipt.get('launch_error')):
        raise ComparisonError('correctness requires an actual successful clean-source process receipt')
    artifact = receipt.get('artifact')
    if backend == 'eqiora' and (not isinstance(artifact, dict) or any(artifact.get(key) != value for key, value in expected_artifact.items())):
        raise ComparisonError('correctness producer/wheel/profile differs from the proposed timing artifact')


def check_harness(report):
    sources = report['environment']['harness_sha256']
    current = {path.name: digest(path) for path in PROBLEM_ROOT.glob('*.py')}
    current.update({str(path.relative_to(ROOT)): digest(path) for path in shared_sources()})
    if sources != current:
        raise ComparisonError('timing code differs from the complete correctness-qualified harness')


def check_report_reference(report, expected_sha256):
    if report.get('environment', {}).get('frozen_reference_manifest_sha256') != expected_sha256:
        raise ComparisonError('correctness report frozen reference differs from the declared case')


def check_cold_pair(report, coordinates, primary, associated):
    if (not np.array_equal(np.asarray(report['coordinates']), coordinates)
            or not np.array_equal(np.asarray(report['values']), primary)
            or set(report['associated_fields']) != set(associated)):
        raise ComparisonError('retained candidate does not match the process cold arrays')
    for name, field in associated.items():
        recorded = report['associated_fields'][name]
        if (recorded['association'] != field['association']
                or not np.array_equal(np.asarray(recorded['values']), field['values'])):
            raise ComparisonError('retained candidate does not match the process cold associated fields')


def validate_qualification(path, cases, expected_artifact):
    payload = json.loads(path.read_text())
    if set(payload) != {'cases'} or set(payload['cases']) != set(LEVELS):
        raise ComparisonError('benchmark qualification must cover all three declared meshes')
    qualified = {}
    for case in cases:
        entry = payload['cases'][case.label]
        if set(entry) != {'eqiora_process_receipt', 'fenicsx_process_receipt', 'assembly_audit'}:
            raise ComparisonError('each qualification case requires both actual process receipts and its independent audit')
        receipts, reports = {}, {}
        for backend in ('eqiora', 'fenicsx'):
            receipt_path = verified_descriptor(path.parent, entry[f'{backend}_process_receipt'])
            receipt = json.loads(receipt_path.read_text())
            check_process(receipt, label=case.label, backend=backend, expected_artifact=expected_artifact)
            report_path = receipt_path.with_name('result.json')
            if not report_path.is_file() or digest(report_path) != receipt.get('result_sha256'):
                raise ComparisonError('correctness report does not match the actual process result digest')
            report = json.loads(report_path.read_text())
            if (report.get('mode') != 'correctness-only' or report.get('case_label') != case.label
                    or report.get('tool') != backend or report['mesh']['digest'] != case.reference.data['mesh_digest']
                    or not report['profile']['required_phases_available']):
                raise ComparisonError('case, mesh, correctness mode or required phase qualification is incomplete')
            check_report_reference(report, case.sha256)
            check_harness(report)
            extra = {name: {'association': field['association'], 'values': np.asarray(field['values'])}
                     for name, field in report['associated_fields'].items()}
            case.reference.candidate(report['coordinates'], report['values'], extra)
            cold_path = receipt_path.with_name('candidate-field.json')
            cold_coordinates, cold_values, cold_associated = read_candidate(cold_path, digest(cold_path))
            check_cold_pair(report, cold_coordinates, cold_values, cold_associated)
            receipts[backend] = (receipt_path, receipt); reports[backend] = report
        audit_path = verified_descriptor(path.parent, entry['assembly_audit'])
        audit = json.loads(audit_path.read_text())
        candidate_path = receipts['eqiora'][0].with_name('candidate-field.json')
        candidate_digest = digest(candidate_path)
        if (audit.get('state') != 'assembly-only independent reconstruction passed' or audit.get('pde_solves') != 0
                or audit.get('candidate_field_sha256') != candidate_digest
                or audit.get('candidate_worker_receipt_sha256') != digest(receipts['eqiora'][0])
                or audit.get('candidate_worker_actual_harness_revision') != receipts['eqiora'][1]['actual_harness_revision']
                or audit.get('frozen_reference_manifest_sha256') != case.sha256
                or any(audit.get('candidate_worker_artifact', {}).get(key) != value for key, value in expected_artifact.items())):
            raise ComparisonError('independent audit does not bind the successful candidate/reference/producer')
        coordinates, primary, extra = read_candidate(candidate_path, candidate_digest)
        check_cold_pair(reports['eqiora'], coordinates, primary, extra)
        accuracy = case.reference.candidate(coordinates, primary, extra)
        fields = {'velocity_vertex': primary, 'velocity_cell_bubble': extra['velocity_cell_bubble']['values'],
                  'pressure_vertex': extra['pressure_vertex']['values']}
        compare(case.reference.data, {'fields': fields, 'force_on_domain': audit['reconstructed_force_on_domain'],
                                     'force_on_selection': audit['reconstructed_force_on_selection'],
                                     'scaled_residual_inf': accuracy['scaled_residual_inf']},
                case.reference.reference, case.reference.manifest)
        allowance = case.reference.reference['global_reaction_allowance']
        reaction = np.asarray(audit['reconstructed_global_essential_reaction_on_domain'])
        if audit['frozen_global_reaction_allowance'] != allowance or reaction.shape != (2,) or not np.isfinite(reaction).all() or np.any(np.abs(reaction) > allowance):
            raise ComparisonError('qualification audit global reaction fails the unchanged frozen bound')
        for side in ('inlet', 'outlet'):
            flux = audit['ufl_boundary_fluxes'][side]
            if not np.isfinite(flux) or abs(flux-accuracy['analytic_checks'][f'{side}_flux']) > accuracy['analytic_checks']['flux_arithmetic_allowance']:
                raise ComparisonError('qualification audit flux fails the unchanged input-derived bound')
        qualified[case.label] = {'reference_manifest_sha256': case.sha256,
                                 'mesh_digest': case.reference.data['mesh_digest'],
                                 'candidate_field_sha256': candidate_digest, 'assembly_audit_sha256': digest(audit_path),
                                 'actual_harness_revision': receipts['eqiora'][1]['actual_harness_revision'],
                                 'producer': expected_artifact}
    return qualified
