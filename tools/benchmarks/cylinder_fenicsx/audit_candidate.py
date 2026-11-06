"""Independent FEniCSx assembly-only reconstruction of retained Eqiora cold fields."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dolfinx import fem
import numpy as np
from accuracy import check_trace_and_flux
from field_artifact import read_candidate
from budget import compare
from common_basis import canonical_fields
from fenicsx_adapter import prepare, boundary_flux_forms
from reference_reader import load_reference
from reference_system import build_system, observations
from shared_measurement import ComparisonError, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-manifest', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--candidate-field', type=Path, required=True)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--candidate-worker-receipt', type=Path, required=True)
    parser.add_argument('--candidate-worker-receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ComparisonError('assembly-only audit output must be fresh')
    if digest(args.candidate_field) != args.candidate_sha256:
        raise ComparisonError('candidate cold field hash mismatch')
    if digest(args.candidate_worker_receipt) != args.candidate_worker_receipt_sha256:
        raise ComparisonError('candidate worker/source/artifact receipt hash mismatch')
    receipt = json.loads(args.candidate_worker_receipt.read_text())
    if receipt['terminal_exit_code'] != 0:
        raise ComparisonError('candidate worker did not pass; no completed-comparison audit claim')
    reference = load_reference(args.reference_manifest, args.reference_sha256)
    coordinates, primary, associated = read_candidate(args.candidate_field, args.candidate_sha256)
    cold_accuracy = reference.candidate(coordinates, primary, associated)
    fields = {'velocity_vertex': primary, 'velocity_cell_bubble': associated['velocity_cell_bubble']['values'],
              'pressure_vertex': associated['pressure_vertex']['values']}
    prepared = prepare(reference.data)
    directory = args.output.parent/(args.output.stem+'-jit')
    if directory.exists():
        raise ComparisonError('assembly-only audit requires a fresh JIT directory')
    jit = {'cache_dir': str(directory), 'cffi_extra_compile_args': ['-O3']}
    system = build_system(prepared, reference.data, rule='duffy-3', jit_options=jit)
    canonical = canonical_fields(fields)
    source = system.transformation.createVecRight(); source.array[:] = canonical
    nodal = system.transformation.createVecLeft(); system.transformation.mult(source, nodal)
    function = fem.Function(prepared.mixed_space); function.x.array[:] = nodal.array
    function.x.scatter_forward()
    row = observations(system, function, fields, canonical, jit_options=jit)
    row['scaled_residual_inf'] = cold_accuracy['scaled_residual_inf']
    imported_comparison = compare(reference.data, row, reference.reference, reference.manifest)
    allowance = reference.reference['global_reaction_allowance']
    if np.any(np.abs(row['global_essential_reaction_on_domain']) > allowance):
        raise ComparisonError('independent reconstructed global essential-union reaction exceeds frozen allowance')
    analytic = check_trace_and_flux(reference.data, fields,
                                    reference.manifest['residual_control']['absolute_scaled_residual_ceiling'])
    fluxes = {name: float(fem.assemble_scalar(fem.form(expression, jit_options=jit)))
              for name, expression in boundary_flux_forms(prepared, function.sub(0)).items()}
    for name in fluxes:
        if abs(fluxes[name]-analytic[f'{name}_flux']) > analytic['flux_arithmetic_allowance']:
            raise ComparisonError(f'independent UFL {name} flux differs from retained cold trace')
    report = {'state': 'assembly-only independent reconstruction passed', 'pde_solves': 0,
              'scope': 'FEniCSx reconstruction from retained Eqiora cold fields; not native Eqiora union-force export',
              'candidate_worker_actual_harness_revision': receipt['actual_harness_revision'],
              'candidate_worker_artifact': receipt['artifact'],
              'candidate_worker_receipt_sha256': args.candidate_worker_receipt_sha256,
              'candidate_field_sha256': args.candidate_sha256,
              'frozen_reference_manifest_sha256': reference.manifest_sha256,
              'cold_candidate_accuracy': cold_accuracy, 'imported_field_force_agreement': imported_comparison,
              'reconstructed_force_on_domain': row['force_on_domain'],
              'reconstructed_force_on_selection': row['force_on_selection'],
              'reconstructed_global_essential_reaction_on_domain': row['global_essential_reaction_on_domain'],
              'frozen_global_reaction_allowance': allowance,
              'matrix_virtual_work_on_domain_diagnostic': row['matrix_virtual_work_on_domain'],
              'ufl_boundary_fluxes': fluxes, 'analytic_checks': analytic,
              'audit_source_sha256': digest(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for item in (source, nodal, system.matrix, system.rhs, system.transformation, system.raw_nodal_matrix): item.destroy()
    print('ASSEMBLY_ONLY_AUDIT_PASS: imported cold fields, UFL force/flux/global union, zero PDE solves')


if __name__ == '__main__':
    try:
        main()
    except ComparisonError as error:
        print(json.dumps({'error': str(error), 'assembly_only_audit_passed': False}))
        raise SystemExit(2)
