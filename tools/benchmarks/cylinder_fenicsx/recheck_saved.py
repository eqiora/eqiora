"""Recheck the committed original cold fields with the complete frozen reference gate."""
import argparse
import json
import math
import numpy as np
from pathlib import Path
from field_artifact import read_candidate
from budget import compare
from reference_reader import load_reference
from shared_measurement import ComparisonError, digest


def recheck(record_path):
    record = json.loads(record_path.read_text())
    inventory = json.loads(record_path.with_name('raw-sha256.json').read_text())
    raw = record_path.parent/record['committed_raw_directory']
    for relative, item in inventory.items():
        if item.get('committed_original_bytes') or item.get('committed_projection'):
            path = raw/relative
            if not path.is_file() or digest(path) != item['sha256']:
                raise ComparisonError(f'committed observation artifact mismatch: {relative}')
    reference = load_reference(record_path.parent/record['committed_reference_manifest'],
                               record['frozen_reference_manifest_sha256'])
    relative = record['committed_candidate_field']
    candidate = record_path.parent/relative
    coordinates, primary, associated = read_candidate(candidate, inventory[str(candidate.relative_to(raw))]['sha256'])
    accuracy = reference.candidate(coordinates, primary, associated)
    audit = json.loads((record_path.parent/record['committed_independent_audit']).read_text())
    candidate_digest = inventory[str(candidate.relative_to(raw))]['sha256']
    if (audit['pde_solves'] != 0 or audit['frozen_reference_manifest_sha256'] != reference.manifest_sha256
            or audit['candidate_field_sha256'] != candidate_digest
            or audit['candidate_worker_actual_harness_revision'] != record['candidate_and_audit_harness_revision']):
        raise ComparisonError('independent audit scope/reference/cold field identity does not match the recorded case')
    native_receipt_digest = inventory['native-process-receipt.json']['sha256']
    projection = json.loads((raw/'native-process-summary.json').read_text())
    if (projection['original_receipt_sha256'] != native_receipt_digest
            or audit['candidate_worker_receipt_sha256'] != native_receipt_digest
            or audit['candidate_worker_artifact']['actual_producer_source'] != record['actual_optimized_producer_source']
            or audit['candidate_worker_artifact']['wheel']['sha256'] != record['wheel']['sha256']):
        raise ComparisonError('independent audit/process projection does not bind the original producer receipt')
    # Recheck stored, integrity-bound UFL observations against unchanged gates.
    # This does not reassemble UFL or perform an additional native/PDE solve.
    fields = {'velocity_vertex': primary, 'velocity_cell_bubble': associated['velocity_cell_bubble']['values'],
              'pressure_vertex': associated['pressure_vertex']['values']}
    compare(reference.data, {'fields': fields,
                             'force_on_domain': audit['reconstructed_force_on_domain'],
                             'force_on_selection': audit['reconstructed_force_on_selection'],
                             'scaled_residual_inf': accuracy['scaled_residual_inf']}, reference.reference, reference.manifest)
    global_reaction = np.asarray(audit['reconstructed_global_essential_reaction_on_domain'])
    allowance = reference.reference['global_reaction_allowance']
    if (audit['frozen_global_reaction_allowance'] != allowance or global_reaction.shape != (2,)
            or not np.isfinite(global_reaction).all() or np.any(np.abs(global_reaction) > allowance)):
        raise ComparisonError('stored UFL global reaction exceeds the unchanged frozen allowance')
    for name in ('inlet', 'outlet'):
        flux = audit['ufl_boundary_fluxes'][name]
        if not math.isfinite(flux) or abs(flux-accuracy['analytic_checks'][f'{name}_flux']) > accuracy['analytic_checks']['flux_arithmetic_allowance']:
            raise ComparisonError('stored UFL boundary flux exceeds the unchanged input-only allowance')
    return record, reference, coordinates, primary, associated, accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, default=Path(__file__).with_name('results')/'correctness.json')
    args = parser.parse_args()
    print(json.dumps(recheck(args.record)[-1], indent=2, allow_nan=False))
