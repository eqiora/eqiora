"""Ordinary installed Eqiora vector elasticity adapter; shared runner owns timing."""
import time
IMPORT_START = time.perf_counter_ns()
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eqiora
import numpy as np
from shared_measurement import clock, digest, environment, finish, memory
from accuracy import ComparisonError, check_accuracy, check_equilibrium
from worker import arguments
IMPORT_SECONDS = (time.perf_counter_ns()-IMPORT_START)/1e9
PROTOCOL_ROOT = Path(__file__).resolve().parent
ROOT = PROTOCOL_ROOT.parents[2]


def prepare(args):
    if not args.wheel or not args.revision or args.build_profile != 'release':
        raise ComparisonError('requires optimized installed wheel, actual producer revision and release profile')
    if Path(eqiora.__file__).resolve().is_relative_to(ROOT):
        raise ComparisonError('use installed wheel, not repository Python source')
    env = environment('eqiora', IMPORT_SECONDS, protocol_root=PROTOCOL_ROOT)
    env.update(wheel_sha256=digest(args.wheel), source_revision_declared=args.revision,
               rust_build_profile_declared=args.build_profile, module_path=eqiora.__file__,
               blas=np.show_config(mode='dicts'))
    baseline = memory()
    names = ('x_lower', 'x_upper', 'y_lower', 'y_upper')

    def make_mesh():
        graph = eqiora.geometry.GeometryGraph()
        rectangle = graph.rectangle(x_bounds=(0., 1.), y_bounds=(0., 1.))
        geometry = graph.build(rectangle, named_topology={'body': rectangle.region,
            **dict(zip(names, rectangle.boundaries, strict=True))})
        mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
            geometry, eqiora.meshing.CartesianMesher(cells=(args.cells, args.cells))))
        return geometry, mesh
    (geometry, mesh), mesh_time = clock(make_mesh)
    source = PROTOCOL_ROOT/'manufactured.eqi'

    def make_plan(text):
        model = eqiora.compile(source=text, geometry=geometry, entry='ManufacturedElasticity2d', bindings={
            'body': geometry.selection('body'),
            **{name: (geometry.selection(name), geometry.selection('body')) for name in names}})
        plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.ConjugateGradient,
            preconditioner=eqiora.solve.Preconditioner.Identity, reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(), relative_tolerance=1e-12,
            absolute_tolerance=1e-14, maximum_iterations=10000))
        if plan.execution.workers != 1 or plan.execution.placement != 'host-serial':
            raise ComparisonError('requires ordinary one-worker host-serial policy')
        return model, plan

    (model, plan), preparation = clock(lambda: make_plan(source.read_text()))
    field = plan.capability.fields[0]
    env['resolved_execution'] = {name: getattr(plan.execution, name) for name in
                                ('scalar_type', 'provider', 'provider_version', 'placement', 'workers')}
    env['linear_provider'] = {'id': plan.solve.provider.id,
                              'implementation_version': plan.solve.provider.implementation_version}

    def summary(result):
        evidence = eqiora.solid.linear_elasticity_evidence(result)
        native = evidence.solve
        row = {name: getattr(native, name) for name in (
            'algorithm', 'preconditioner', 'completed_iterations', 'initial_residual_norm',
            'reported_residual_norm', 'true_residual_norm', 'residual_target',
            'backend', 'adapter', 'verification_adapter', 'reduction',
            'relative_tolerance', 'absolute_tolerance', 'maximum_iterations')}
        row['convergence_reason'] = str(native.reason)
        if (row['algorithm'] != 'cg' or row['preconditioner'] != 'identity'
                or (row['relative_tolerance'], row['absolute_tolerance'], row['maximum_iterations']) != (1e-12, 1e-14, 10000)):
            raise ComparisonError('Eqiora effective solve policy differs from protocol')
        target = max(1e-14, 1e-12*row['initial_residual_norm'])
        if row['residual_target'] != target or not np.isfinite(row['true_residual_norm']) or row['true_residual_norm'] > target:
            raise ComparisonError('Eqiora true residual fails independent target')
        return row

    def extract(result):
        return np.asarray(plan.mesh.coordinates), result.output(field).values('vertex').numpy().reshape(-1, 2), summary(result)

    def diagnostic():
        result = eqiora.run(plan, profile=True)
        summary(result)
        # Separate equilibrium solve follows cold execution, preserving its HWM scope.
        _, probe_plan = make_plan((PROTOCOL_ROOT/'equilibrium.eqi').read_text())
        probe_result = eqiora.run(probe_plan)
        probe = eqiora.solid.linear_elasticity_evidence(probe_result)
        check_equilibrium(probe.constrained_reaction, probe.integrated_body_force)
        report['equilibrium_probe'] = {'scope': 'separate untimed affine-potential solve after cold execution',
                                      'reaction': list(probe.constrained_reaction), 'body_force': list(probe.integrated_body_force),
                                      'solve': summary(probe_result), 'model_sha256': digest(PROTOCOL_ROOT/'equilibrium.eqi')}
        phases = [{'path': list(p.path), 'fields': dict(p.fields), 'calls': p.calls,
                   'inclusive_seconds': p.inclusive_seconds, 'self_seconds': p.self_seconds} for p in result.profile.phases]
        available = all(any(p['path'][-1] == name for p in phases) for name in ('assembly', 'linear_solve'))
        if args.mode == 'correctness-only':
            return {'required_phases_available': available,
                    'phase_inventory': [{k: p[k] for k in ('path', 'fields', 'calls')} for p in phases]}
        return {'required_phases_available': available, 'phases': phases,
                'source': 'separate profile=True diagnostic execution; no inferred ordinary timing partitions'}

    report = {'environment': env, 'memory_after_import': baseline, 'memory_after_preparation': memory(),
              'mesh_measurement': mesh_time, 'preparation_measurement': preparation,
              'model_sha256': digest(source), 'plan_sha256': hashlib.sha256(plan.to_bytes()).hexdigest(),
              'package_compilation_digest': model.package_compilation_digest,
              'dofs': {'total': 2*(args.cells+1)**2, 'free': 2*(args.cells-1)**2,
                       'stored_algebraic': 2*(args.cells-1)**2}}
    return report, lambda: eqiora.run(plan), extract, diagnostic


def main(args):
    report, execute, extract, diagnostic = prepare(args)
    finish(args, 'eqiora', report, execute, extract, diagnostic, accuracy=check_accuracy)


if __name__ == '__main__':
    main(arguments())
