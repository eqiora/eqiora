"""External-only A/B/C reference batch; launch only after protocol authorisation."""
import time
IMPORT_START = time.perf_counter_ns()
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import basix
from petsc4py import PETSc
import numpy as np
from accuracy import MU, U, H, physical_norms, discrete_inlet_flux, boundary_flux, check_trace_and_flux
from budget import check_falsifiers, freeze_budget, residual_ceiling
from fenicsx_adapter import prepare, export_fields
from external_residual import residual_inf, validate_operator
from common_basis import canonical_fields
from reference_system import (build_system, observations, quadrature,
                              scaled_operator, factor_solve, recover_mixed)
from shared_measurement import ComparisonError, digest, environment
from topology import cylinder_geometry_report
IMPORT_SECONDS = (time.perf_counter_ns()-IMPORT_START)/1e9
PROBLEM_ROOT = Path(__file__).resolve().parent


def serializable_observation(row):
    return {**row, 'fields': {key: np.asarray(value).tolist() for key, value in row['fields'].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ComparisonError('reference batch output must be new; prior attempts are retained')
    if any(key in os.environ for key in ('PETSC_OPTIONS', 'PETSC_OPTIONS_YAML', 'SLEPC_OPTIONS')):
        raise ComparisonError('external solver option injection is outside the frozen protocol')
    if not all(PETSc.Sys.hasExternalPackage(name) for name in ('mumps', 'superlu')):
        raise ComparisonError('two declared distinct factorization implementations are required')
    if not args.mesh.is_file():
        raise ComparisonError('public mesh export file is missing')
    data = json.loads(args.mesh.read_text())
    geometry_audit = cylinder_geometry_report(data)
    env = environment('fenicsx', IMPORT_SECONDS, protocol_root=PROBLEM_ROOT)
    env.update(petsc_version=PETSc.Sys.getVersionInfo(), numpy_version=np.__version__, basix_version=basix.__version__,
               installed_petsc_options=PETSc.Options().getAll())
    env['factorization_packages'] = {}
    for package in (Path(sys.prefix)/'conda-meta').glob('*.json'):
        metadata = json.loads(package.read_text())
        if any(name in metadata.get('name', '') for name in ('petsc', 'mumps', 'superlu')):
            env['factorization_packages'][metadata['name']] = {key: metadata.get(key) for key in ('version', 'build', 'sha256', 'channel')}
    prepared = prepare(data)
    args.output.mkdir(parents=True)
    shutil.copyfile(args.mesh, args.output/'public-mesh.json')
    mesh_artifact = args.mesh.parent/data['mesh_artifact']
    if digest(mesh_artifact) != data['mesh_artifact_sha256']:
        raise ComparisonError('public mesh artifact does not match its exported digest')
    shutil.copyfile(mesh_artifact, args.output/data['mesh_artifact'])
    policy = [('A', 'mumps', 'duffy-3'), ('B', 'superlu', 'duffy-3'), ('C', 'mumps', 'basix-degree-6')]
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROBLEM_ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROBLEM_ROOT, text=True).strip()
    if dirty:
        raise ComparisonError('freeze a clean committed harness before reference calibration')
    launch = {'harness_revision': source, 'harness_clean': True, 'mesh_file_sha256': digest(args.mesh),
              'environment': env, 'policy': policy, 'geometry_audit': geometry_audit,
              'no_eqiora_output_read': True, 'candidate_started': False,
              'quadrature': {name: {key: value.tolist() if hasattr(value, 'tolist') else value
                                   for key, value in quadrature(name).items()}
                             for name in ('duffy-3', 'basix-degree-6')}}
    (args.output/'reference-launch.json').write_text(json.dumps(launch, indent=2, allow_nan=False)+'\n')
    references, reference_system = {}, None
    for label, implementation, rule in policy:
        directory = args.output/label; directory.mkdir()
        jit = {'cache_dir': str(directory/'jit'), 'cffi_extra_compile_args': ['-O3']}
        system = build_system(prepared, data, rule=rule, jit_options=jit)
        canonical, summary = factor_solve(system, implementation)
        function = recover_mixed(system, canonical)
        fields = export_fields(prepared, function)
        # Residual acceptance uses the exported common hierarchy, including any
        # representation rounding, and the same designated-A operator for all runs.
        canonical = canonical_fields(fields)
        if reference_system is None:
            reference_system = system
            reference_matrix, reference_rhs = scaled_operator(system)
            csr = reference_matrix.getValuesCSR()
            max_row_nonzeros = max(np.count_nonzero(csr[2][csr[0][i]:csr[0][i+1]])
                                   for i in range(len(csr[0])-1))
            control = residual_ceiling(max_row_nonzeros, reference_matrix.norm(PETSc.NormType.NORM_INFINITY),
                                       float(np.max(np.abs(canonical[system.free]/system.scales))),
                                       reference_rhs.norm(PETSc.NormType.NORM_INFINITY))
            viewer = PETSc.Viewer().createBinary(str(args.output/'reference-scaled-matrix.petsc'), 'w', comm=prepared.domain.comm)
            reference_matrix.view(viewer); viewer.destroy()
            viewer = PETSc.Viewer().createBinary(str(args.output/'reference-scaled-rhs.petsc'), 'w', comm=prepared.domain.comm)
            reference_rhs.view(viewer); viewer.destroy()
            operator = {'indptr': csr[0], 'indices': csr[1], 'values': csr[2],
                        'rhs': reference_rhs.array.copy(), 'free': system.free, 'scales': system.scales}
            validate_operator(operator, canonical_size=len(canonical), expected_free=system.free, expected_scales=system.scales)
            np.savez(args.output/'reference-system.npz', **operator)
        if not np.array_equal(system.free, reference_system.free) or not np.array_equal(system.scales, reference_system.scales):
            raise ComparisonError('external batch did not preserve the common free basis/scaling')
        observation = observations(system, function, fields, canonical, jit_options=jit)
        observation['solve'] = summary
        observation['scaled_residual_inf'] = residual_inf(operator, canonical)
        observation['analytic_checks'] = check_trace_and_flux(
            data, fields, control['absolute_scaled_residual_ceiling'])
        global_allowance = len(prepared.coordinates)*MU*U*(control['absolute_scaled_residual_ceiling']+256*np.finfo(float).eps)
        observation['global_reaction_allowance'] = global_allowance
        if np.any(np.abs(observation['global_essential_reaction_on_domain']) > global_allowance):
            raise ComparisonError('external union-of-essential-vertices reaction balance failed')
        observation['physical_rms'] = physical_norms(data, fields)
        observation['inlet_flux'] = boundary_flux(data, fields, 'inlet')
        observation['outlet_flux'] = boundary_flux(data, fields, 'outlet')
        observation['analytic_discrete_inlet_flux'] = discrete_inlet_flux(data)
        references[label] = observation
        (directory/'observation.json').write_text(json.dumps(serializable_observation(observation), indent=2, allow_nan=False)+'\n')
    frozen = freeze_budget(data, references, control)
    frozen['falsifiers_rejected'] = check_falsifiers(data, references['A'], frozen)
    frozen.update(state='frozen-before-candidate', launch_sha256=digest(args.output/'reference-launch.json'),
                  mesh_sha256=digest(args.output/'public-mesh.json'),
                  mesh_identity=data['mesh_digest'], geometry_identity=data['geometry_digest'],
                  mesh_artifact_sha256=data['mesh_artifact_sha256'],
                  common_basis='original vertex velocity components, original cell hierarchical bubble components, original pressure vertices',
                  essential_boundary_names=['inlet', 'walls', 'cylinder'], pressure_constraint='none',
                  variable_scales={'velocity_and_bubbles': U, 'pressure': MU*U/H},
                  residual_norm='infinity, same exported dimensionless designated-A CSR operator for references and candidate',
                  designated_reference_file='A/observation.json',
                  reference_file_sha256={label: digest(args.output/label/'observation.json') for label in references},
                  common_operator_files={path.name: digest(path) for path in args.output.iterdir()
                                         if path.suffix in ('.petsc', '.npz')},
                  no_eqiora_output_read=True, candidate_started=False)
    # A completed immutable manifest is written only after every gate and mutant.
    (args.output/'reference-manifest.json').write_text(json.dumps(frozen, indent=2, allow_nan=False)+'\n')
    print('REFERENCE_BATCH_PASS: external-only empirical budget frozen; no Eqiora candidate executed')


if __name__ == '__main__':
    try:
        main()
    except ComparisonError as error:
        print(json.dumps({'error': str(error), 'reference_frozen': False}))
        raise SystemExit(2)
