"""One ordinary installed Eqiora worker against the frozen external cylinder reference."""
import time
IMPORT_START = time.perf_counter_ns()
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eqiora
import numpy as np
from common_basis import layout
from public_mesh import author_geometry, compile_model, export_fields, export_mesh
from reference_reader import associated, load_reference
from shared_measurement import ROOT, ComparisonError, clock, digest, environment, finish, memory
IMPORT_SECONDS = (time.perf_counter_ns()-IMPORT_START)/1e9
PROBLEM_ROOT = Path(__file__).resolve().parent


def prepare(args):
    if Path(eqiora.__file__).resolve().is_relative_to(ROOT):
        raise ComparisonError('requires the installed optimized wheel, not repository Python source')
    if not args.wheel.is_file() or not re.fullmatch('[0-9a-f]{40}', args.revision):
        raise ComparisonError('requires wheel artifact and actual full producer source ID')
    reference = load_reference(args.reference_manifest, args.reference_sha256)
    env = environment('eqiora', IMPORT_SECONDS, protocol_root=PROBLEM_ROOT)
    env.update(wheel_sha256=digest(args.wheel), actual_producer_source_declared=args.revision,
               rust_build_profile_declared=args.build_profile, module_path=eqiora.__file__,
               frozen_reference_manifest_sha256=reference.manifest_sha256,
               reference_harness_revision=reference.launch['harness_revision'])
    baseline = memory()

    def reopen():
        geometry = author_geometry(eqiora)
        mesh = eqiora.meshing.Mesh.from_bytes((reference.directory/reference.data['mesh_artifact']).read_bytes())
        exported = export_mesh(geometry, mesh)
        for key in ('mesh_digest', 'geometry_digest', 'coordinates', 'cells', 'boundaries'):
            if exported[key] != reference.data[key]:
                raise ComparisonError(f'reopened public mesh differs from frozen {key}')
        return geometry, mesh
    (geometry, mesh), mesh_time = clock(reopen)

    def make_plan():
        model = compile_model(eqiora, geometry)
        if model.digest != reference.data['model_digest'] or model.package_compilation_digest != reference.data['package_compilation_digest']:
            raise ComparisonError('ordinary packaged model compilation differs from the reference mesh preparation')
        plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.MiniP1(), scaling=None,
                              solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.SparseLu,
            preconditioner=eqiora.solve.Preconditioner.Identity, reduction=eqiora.solve.Reduction.Fast,
            provider=eqiora.solve.SolverProvider.faer(), relative_tolerance=1e-6,
            absolute_tolerance=1e-13, maximum_iterations=10000))
        if plan.execution.workers != 1 or plan.execution.placement != 'host-serial':
            raise ComparisonError('candidate must use the ordinary one-worker host-serial placement')
        return model, plan
    (model, plan), preparation = clock(make_plan)
    common = layout(reference.data)
    report = {'dimension': 2, 'problem': 'finite-channel steady Stokes cylinder',
              'mesh': {'vertices': mesh.vertex_count, 'cells': mesh.cell_count, 'digest': mesh.digest,
                       'geometry_digest': geometry.digest, 'topology': reference.data['cells'],
                       'boundary_edges': reference.data['boundaries']},
              'environment': env, 'memory_after_import': baseline, 'memory_after_preparation': memory(),
              'mesh_measurement': mesh_time, 'preparation_measurement': preparation,
              'model_digest': model.digest, 'package_compilation_digest': model.package_compilation_digest,
              'plan_sha256': hashlib.sha256(plan.to_bytes()).hexdigest(),
              'dofs': {'total': common['size'], 'free': len(common['free']), 'stored_algebraic': len(common['free'])},
              'primary_field': {'name': 'velocity', 'association': 'vertex', 'unit': 'm/s'},
              'associated_units': {'velocity_cell_bubble': 'm/s', 'pressure_vertex': 'Pa',
                                   'cylinder_force_on_domain': 'N/m', 'cylinder_force_on_selection': 'N/m',
                                   'inlet_flux': 'm^2/s', 'outlet_flux': 'm^2/s'},
              'expected_execution_calls': {'cold_candidate': 1, 'separate_profile': 1, 'warmups': 0, 'timed_samples': 0},
              'global_union_check': 'separate independent FEniCSx reconstruction from retained cold fields; no native Eqiora union-export claim'}

    def summary(result):
        native = result.solve
        row = {key: getattr(native, key) for key in (
            'algorithm', 'completed_iterations', 'initial_residual_norm', 'reported_residual_norm',
            'true_residual_norm', 'residual_target', 'backend', 'adapter', 'verification_adapter',
            'reduction', 'relative_tolerance', 'absolute_tolerance', 'maximum_iterations')}
        row.update(preconditioner=native.preconditioner.kind, convergence_reason=str(native.reason))
        if (row['algorithm'] != 'sparse-lu' or row['preconditioner'] != 'identity' or row['reduction'] != 'fast'
                or (row['relative_tolerance'], row['absolute_tolerance'], row['maximum_iterations']) != (1e-6, 1e-13, 10000)):
            raise ComparisonError('native candidate solve policy differs from the shipped example controls')
        if not np.isfinite(row['true_residual_norm']) or row['true_residual_norm'] > row['residual_target']:
            raise ComparisonError('native candidate failed its own accepted residual contract')
        return row

    def extract(result):
        # Standard product artifacts retain the cold result even if a public
        # observer unexpectedly fails before shared candidate serialization.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent/'candidate-plan.bin').write_bytes(plan.to_bytes())
        (args.output.parent/'candidate-result.bin').write_bytes(result.to_bytes())
        fields = export_fields(plan, result)
        force = result.boundary_force(geometry.selection('cylinder'))
        inlet = result.boundary_flux(geometry.selection('inlet'))
        outlet = result.boundary_flux(geometry.selection('outlet'))
        if (tuple(force.dimension) != (1, 0, -2, 0, 0, 0, 0) or force.frame != 'spatial-cartesian'
                or tuple(inlet.dimension) != (0, 2, -1, 0, 0, 0, 0)
                or tuple(outlet.dimension) != (0, 2, -1, 0, 0, 0, 0)):
            raise ComparisonError('public cylinder force/flux units or frame differ from the declared intrinsic-2D problem')
        row = summary(result)
        row['public_force_identity'] = {'digest': force.digest, 'source_digest': force.source_digest,
                                        'mesh_digest': force.mesh_digest, 'geometry_digest': force.geometry_digest}
        return (np.asarray(mesh.coordinates), fields['velocity_vertex'], row,
                associated(fields, force.on_domain, force.on_selection, inlet.value, outlet.value))

    def diagnostic():
        result = eqiora.run(plan, profile=True)
        native = summary(result)
        inventory = [{'path': list(phase.path), 'fields': dict(phase.fields), 'calls': phase.calls}
                     for phase in result.profile.phases]
        return {'required_phases_available': all(any(p['path'][-1] == name for p in inventory)
                                                for name in ('assembly', 'linear_solve')),
                'phase_inventory': inventory, 'solve': native,
                'scope': 'one separate native profile call; cold fields remain the only compared candidate'}
    return reference, report, lambda: eqiora.run(plan), extract, diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-manifest', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('correctness-only',), required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--build-profile', choices=('release',), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ComparisonError('candidate output must be fresh; no overwrite or best-of-run selection')
    reference, report, execute, extract, diagnostic = prepare(args)
    finish(args, 'eqiora', report, execute, extract, diagnostic, accuracy=reference.candidate)


if __name__ == '__main__':
    try:
        main()
    except (ComparisonError, eqiora.ValidationError) as error:
        print(json.dumps({'error': str(error), 'complete_comparison': False}))
        raise SystemExit(2)
