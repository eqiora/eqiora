"""Read the frozen external reference before a candidate; never recalibrate it."""
from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
import numpy as np
from accuracy import U, P, MU, H, FORCE_SCALE, check_trace_and_flux, physical_norms
from budget import compare, check_force_pair
from common_basis import canonical_fields, layout
from external_residual import load_operator, residual_inf
from shared_measurement import ComparisonError, digest
from topology import cylinder_geometry_report

ASSOCIATIONS = {'velocity_cell_bubble': 'cell-bubble', 'pressure_vertex': 'vertex',
                'cylinder_force_on_domain': 'boundary:cylinder',
                'cylinder_force_on_selection': 'boundary:cylinder',
                'inlet_flux': 'boundary:inlet', 'outlet_flux': 'boundary:outlet'}


@dataclass
class Reference:
    directory: Path
    manifest: dict
    manifest_sha256: str
    data: dict
    reference: dict
    operator: dict
    launch: dict

    def candidate(self, coordinates, primary, associated):
        if not np.array_equal(np.asarray(coordinates), np.asarray(self.data['coordinates'])):
            raise ComparisonError('candidate coordinates/order differ from the frozen public mesh')
        if set(associated) != set(ASSOCIATIONS):
            raise ComparisonError('candidate must retain every declared MINI/force/flux association')
        for name, association in ASSOCIATIONS.items():
            if associated[name]['association'] != association:
                raise ComparisonError(f'wrong candidate association for {name}')
        fields = {'velocity_vertex': np.asarray(primary),
                  'velocity_cell_bubble': np.asarray(associated['velocity_cell_bubble']['values']),
                  'pressure_vertex': np.asarray(associated['pressure_vertex']['values'])}
        physical_norms(self.data, fields)  # Finite and complete inventory gate.
        row = {'fields': fields,
               'force_on_domain': np.asarray(associated['cylinder_force_on_domain']['values']),
               'force_on_selection': np.asarray(associated['cylinder_force_on_selection']['values']),
               'scaled_residual_inf': residual_inf(self.operator, canonical_fields(fields))}
        check_force_pair(row)
        checks = check_trace_and_flux(self.data, fields,
                                      self.manifest['residual_control']['absolute_scaled_residual_ceiling'])
        # Public observable fluxes must agree with integration of the exported
        # physical trace, using the already frozen input-only arithmetic margin.
        for name in ('inlet', 'outlet'):
            actual = np.asarray(associated[f'{name}_flux']['values'])
            if actual.shape != (1,) or not np.isfinite(actual).all() or abs(actual[0]-checks[f'{name}_flux']) > checks['flux_arithmetic_allowance']:
                raise ComparisonError(f'public {name} flux differs from exported MINI trace')
        differences = compare(self.data, row, self.reference, self.manifest)
        return {'frozen_reference_manifest_sha256': self.manifest_sha256,
                'agreement': differences, 'scaled_residual_inf': row['scaled_residual_inf'],
                'scaled_residual_ceiling': self.manifest['residual_control']['absolute_scaled_residual_ceiling'],
                'analytic_checks': checks, 'physical_rms': physical_norms(self.data, fields),
                'force_on_domain': row['force_on_domain'].tolist(),
                'force_on_selection': row['force_on_selection'].tolist(),
                'scope': 'shared affine-mesh empirical agreement; no continuum-drag or timing claim'}


def _verified(directory, name, expected):
    path = directory/name
    if not path.is_file() or digest(path) != expected:
        raise ComparisonError(f'frozen reference artifact digest mismatch: {name}')
    return path


def load_reference(path, expected_sha256):
    path = Path(path)
    if not path.is_file() or digest(path) != expected_sha256:
        raise ComparisonError('frozen reference manifest digest mismatch')
    manifest = json.loads(path.read_text()); directory = path.parent
    if (manifest.get('state') != 'frozen-before-candidate' or manifest.get('designated_reference') != 'A'
            or manifest.get('reference_order') != ['A', 'B', 'C']
            or manifest.get('no_eqiora_output_read') is not True or manifest.get('candidate_started') is not False
            or manifest.get('common_basis') != 'original vertex velocity components, original cell hierarchical bubble components, original pressure vertices'
            or manifest.get('pressure_constraint') != 'none'
            or manifest.get('essential_boundary_names') != ['inlet', 'walls', 'cylinder']
            or manifest.get('variable_scales') != {'velocity_and_bubbles': U, 'pressure': P}
            or manifest.get('residual_norm') != 'infinity, same exported dimensionless designated-A CSR operator for references and candidate'):
        raise ComparisonError('reference policy/basis/scaling differs from the reviewed protocol')
    constants = {'spread_multiplier': 10, 'broad_cap_fraction': 1e-4, 'mutant_separation': 20,
                 'residual_gamma_multiplier': 100}
    if manifest.get('constants') != constants:
        raise ComparisonError('reference empirical constants differ from the frozen protocol')
    for name, expected in manifest['common_operator_files'].items(): _verified(directory, name, expected)
    if 'reference-system.npz' not in manifest['common_operator_files']:
        raise ComparisonError('reference is missing its portable designated-A operator')
    for label, expected in manifest['reference_file_sha256'].items(): _verified(directory, f'{label}/observation.json', expected)
    if set(manifest['reference_file_sha256']) != {'A', 'B', 'C'}:
        raise ComparisonError('reference file inventory differs from the A/B/C batch')
    data = json.loads(_verified(directory, 'public-mesh.json', manifest['mesh_sha256']).read_text())
    launch = json.loads(_verified(directory, 'reference-launch.json', manifest['launch_sha256']).read_text())
    if (launch.get('policy') != [['A', 'mumps', 'duffy-3'], ['B', 'superlu', 'duffy-3'], ['C', 'mumps', 'basix-degree-6']]
            or launch.get('harness_clean') is not True or launch.get('no_eqiora_output_read') is not True
            or launch.get('candidate_started') is not False
            or not re.fullmatch('[0-9a-f]{40}', launch.get('harness_revision', ''))
            or launch.get('mesh_file_sha256') != manifest['mesh_sha256']):
        raise ComparisonError('reference launch differs from the frozen external-only A/B/C policy')
    if (data['mesh_digest'] != manifest['mesh_identity'] or data['geometry_digest'] != manifest['geometry_identity']
            or data['mesh_artifact_sha256'] != manifest['mesh_artifact_sha256']):
        raise ComparisonError('reference geometry/mesh identities differ')
    _verified(directory, data['mesh_artifact'], data['mesh_artifact_sha256'])
    cylinder_geometry_report(data)
    common = layout(data)
    operator = load_operator(directory/'reference-system.npz', canonical_size=common['size'],
                             expected_free=common['free'], expected_scales=common['scales'])
    if manifest['designated_reference_file'] != 'A/observation.json':
        raise ComparisonError('designated reference must be the predeclared A file')
    reference = json.loads((directory/manifest['designated_reference_file']).read_text())
    physical_norms(data, reference['fields']); check_force_pair(reference)
    control = manifest['residual_control']; eps = np.finfo(float).eps
    if (not all(math.isfinite(control[k]) for k in ('epsilon', 'm', 'gamma', 'scale_D', 'absolute_scaled_residual_ceiling'))
            or not isinstance(control['m'], int) or control['m'] < 2 or control['epsilon'] != eps or control['m']*eps >= 1 or control['scale_D'] <= 0
            or control['gamma'] != control['m']*eps/(1-control['m']*eps)
            or control['absolute_scaled_residual_ceiling'] != 100*control['gamma']*control['scale_D']):
        raise ComparisonError('reference residual ceiling is inconsistent with its frozen formula')
    scales = dict(velocity=U, pressure=P, force_x=FORCE_SCALE, force_y=FORCE_SCALE)
    if set(manifest['metrics']) != set(scales):
        raise ComparisonError('reference must freeze all four physical metrics')
    for name, metric in manifest['metrics'].items():
        values = [metric[k] for k in ('input_scale', 'reference_amplitude', 'scale', 'external_spread',
                                      'tolerance', 'broad_cap', 'mutant_distance', 'separation_cap')]
        if (not all(math.isfinite(v) and v >= 0 for v in values) or metric['input_scale'] != scales[name]
                or metric['scale'] != max(metric['input_scale'], metric['reference_amplitude'])
                or metric['tolerance'] != max(math.sqrt(eps)*metric['scale'], 10*metric['external_spread'])
                or metric['broad_cap'] != 1e-4*metric['scale'] or metric['separation_cap'] != metric['mutant_distance']/20
                or metric['tolerance'] > min(metric['broad_cap'], metric['separation_cap'])):
            raise ComparisonError(f'frozen {name} budget does not satisfy its declared caps/formula')
    return Reference(directory, manifest, expected_sha256, data, reference, operator, launch)


def associated(fields, force_on_domain, force_on_selection, inlet_flux, outlet_flux):
    values = {'velocity_cell_bubble': fields['velocity_cell_bubble'], 'pressure_vertex': fields['pressure_vertex'],
              'cylinder_force_on_domain': force_on_domain, 'cylinder_force_on_selection': force_on_selection,
              'inlet_flux': [inlet_flux], 'outlet_flux': [outlet_flux]}
    return {name: {'association': ASSOCIATIONS[name], 'values': np.asarray(value)} for name, value in values.items()}
