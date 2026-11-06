"""Fresh ordinary FEniCSx MINI execution against the designated-A frozen policy."""
import time
IMPORT_START = time.perf_counter_ns()
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dolfinx import fem
from dolfinx.fem import petsc
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from accuracy import MU, U
from common_basis import canonical_fields, layout
from external_residual import residual_inf
from fenicsx_adapter import prepare as prepare_mesh, export_fields, observation_linear_forms
from mesh_levels import mesh_level
from reference_reader import associated, load_reference
from reference_system import prepare_forms, assemble_system, factor_solve, recover_mixed, destroy_system
from shared_measurement import ComparisonError, clock, environment, finish, memory
from worker import arguments
IMPORT_SECONDS = (time.perf_counter_ns()-IMPORT_START)/1e9
PROBLEM_ROOT = Path(__file__).resolve().parent


def prepare(args):
    reference = load_reference(args.reference_manifest, args.reference_sha256)
    if args.case_label is not None and (reference.data.get('mesh_level') != args.case_label
            or reference.data.get('mesh_controls') != mesh_level(args.case_label).controls()):
        raise ComparisonError('worker case label/controls differ from the independently frozen mesh')
    env = environment('fenicsx', IMPORT_SECONDS, protocol_root=PROBLEM_ROOT)
    env.update(petsc_version=PETSc.Sys.getVersionInfo(), mpi_library=MPI.Get_library_version(),
               actual_factorization='mumps', frozen_reference_manifest_sha256=reference.manifest_sha256,
               reference_harness_revision=reference.launch['harness_revision'])
    baseline = memory()
    prepared, mesh_time = clock(lambda: prepare_mesh(reference.data))
    jit_directory = args.output.parent/'fresh-jit-cache'
    if jit_directory.exists(): raise ComparisonError('ordinary FEniCSx worker requires a fresh JIT directory')
    jit_directory.mkdir(parents=True)
    jit = {'cache_dir': str(jit_directory), 'cffi_extra_compile_args': ['-O3']}

    def compile_inputs():
        forms = prepare_forms(prepared, reference.data, rule='duffy-3', jit_options=jit)
        observers = {name: fem.form(expression, jit_options=jit)
                     for name, expression in observation_linear_forms(prepared).items()}
        return forms, observers
    (forms, observers), preparation = clock(compile_inputs)
    common = layout(reference.data)
    report = {'dimension': 2, 'case_label': args.case_label, 'problem': 'finite-channel steady Stokes cylinder',
              'mesh': {'vertices': len(prepared.coordinates), 'cells': len(prepared.cells),
                       'digest': reference.data['mesh_digest'], 'geometry_digest': reference.data['geometry_digest'],
                       'topology': reference.data['cells'], 'boundary_edges': reference.data['boundaries']},
              'environment': env, 'memory_after_import': baseline, 'memory_after_preparation': memory(),
              'mesh_measurement': mesh_time, 'preparation_measurement': preparation,
              'dofs': {'total': common['size'], 'free': len(common['free']), 'stored_algebraic': len(common['free'])},
              'primary_field': {'name': 'velocity', 'association': 'vertex', 'unit': 'm/s'},
              'expected_execution_calls': {'cold_candidate': 1, 'separate_profile': 1,
                                            'warmups': 2 if args.mode == 'benchmark' else 0,
                                            'timed_samples': 7 if args.mode == 'benchmark' else 0},
              'retained_inputs': ['mesh', 'space', 'compiled UFL forms', 'immutable basis transformation',
                                  'common layout', 'immutable P1 observation lifts'],
              'ordinary_scope': 'fresh matrix/congruence/reduction/lifting, fresh MUMPS factor/solve and true residual, fresh mixed-field recovery and system cleanup; observers/field inspection untimed'}

    def execute(*, diagnostic=False):
        if diagnostic:
            system, assembly_time = clock(lambda: assemble_system(forms))
            (canonical, summary), solve_time = clock(lambda: factor_solve(system, 'mumps'))
            function, recovery_time = clock(lambda: recover_mixed(system, canonical))
            _, cleanup_time = clock(lambda: destroy_system(system))
        else:
            system = assemble_system(forms)
            canonical, summary = factor_solve(system, 'mumps')
            function = recover_mixed(system, canonical)
            destroy_system(system)
            assembly_time = solve_time = recovery_time = cleanup_time = None
        if (not np.isfinite(summary['own_scaled_free_residual_inf'])
                or summary['own_scaled_free_residual_inf'] > reference.manifest['residual_control']['absolute_scaled_residual_ceiling']):
            raise ComparisonError('fresh MUMPS solve failed the unchanged scaled residual ceiling')
        return function, summary, {'assembly': assembly_time, 'linear_solve': solve_time,
                                    'field_recovery': recovery_time, 'system_cleanup': cleanup_time}

    def extract(result):
        function, summary, _ = result
        fields = export_fields(prepared, function)
        summary = dict(summary, external_scaled_residual_inf=residual_inf(reference.operator, canonical_fields(fields)))
        if summary['external_scaled_residual_inf'] > reference.manifest['residual_control']['absolute_scaled_residual_ceiling']:
            raise ComparisonError('fresh ordinary field failed the frozen external-operator residual ceiling')
        observed = {}
        for name, form in observers.items():
            vector = petsc.assemble_vector(form)
            vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            observed[name] = float(np.dot(vector.array, function.x.array)); vector.destroy()
        reaction = np.array([observed['force_x'], observed['force_y']])
        return prepared.coordinates, fields['velocity_vertex'], summary, associated(
            fields, reaction, -reaction, observed['inlet_flux'], observed['outlet_flux'])

    def diagnostic():
        result = execute(diagnostic=True)
        extract(result)
        phases = result[2]
        if args.mode == 'correctness-only':
            return {'required_phases_available': True, 'phase_inventory': list(phases)}
        return {'required_phases_available': True, **phases,
                'assembly_scope': 'fresh raw stiffness, basis congruence, free matrix/RHS allocation and symmetric lifting',
                'linear_solve_scope': 'new KSP/LU/MUMPS setup, factorization, zero-start solve, physical/scaled true residual, summary and solver-vector cleanup',
                'field_recovery_scope': 'new mixed Function, hierarchical-to-nodal transformation and scatter',
                'system_cleanup_scope': 'fresh raw/reduced matrices and RHS destruction; immutable basis retained',
                'observation_scope': 'fresh force/flux UFL functional-vector assembly/dots and complete field exports outside ordinary clocks'}
    return reference, report, execute, extract, diagnostic


def main():
    args = arguments('fenicsx')
    if args.output.exists(): raise ComparisonError('worker output must be fresh')
    reference, report, execute, extract, diagnostic = prepare(args)
    finish(args, 'fenicsx', report, execute, extract, diagnostic, accuracy=reference.candidate)


if __name__ == '__main__':
    try:
        main()
    except ComparisonError as error:
        print(json.dumps({'error': str(error), 'complete_comparison': False}))
        raise SystemExit(2)
