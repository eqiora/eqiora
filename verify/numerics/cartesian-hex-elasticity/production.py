"""Ordinary public Hex producer; independent comparisons live in reference/."""
from hashlib import sha256 as _sha256
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIDES = tuple(f'{axis}_{side}' for axis in 'xyz' for side in ('lower', 'upper'))
DIMENSION_M = (0, 1, 0, 0, 0, 0, 0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return _sha256(data).hexdigest()


def write(directory, name, value):
    data = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()
    with (directory/name).open('xb') as stream:
        stream.write(data)
    return sha256(data)


def binary64(value):
    value = float(value)
    require(math.isfinite(value), 'nonfinite public value')
    require(value != 0 or math.copysign(1., value) > 0, 'negative zero public value')
    return value.hex()


def geometry_mesh(eqiora):
    graph = eqiora.geometry.GeometryGraph()
    box = graph.cuboid(x_bounds=(0., 2.), y_bounds=(0., 3.), z_bounds=(0., 4.))
    geometry = graph.build(box, named_topology={'body': box.region, **dict(zip(SIDES, box.boundaries, strict=True))})
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(geometry, eqiora.meshing.CartesianMesher(cells=(2, 2, 2))))
    return geometry, mesh


def prepare(eqiora, directory, *, source=None):
    geometry, mesh = geometry_mesh(eqiora)
    source = (HERE/'model.eqi').read_bytes() if source is None else source
    write(directory, 'model.eqi', source)
    write(directory, 'mesh.json', mesh.to_bytes())
    model = eqiora.compile(source=source.decode(), filename='cartesian-hex-elasticity.eqi', entry='CartesianHexElasticity', geometry=geometry,
        bindings={'body': geometry.selection('body'), **{side: (geometry.selection(side), geometry.selection('body')) for side in SIDES}, 'mu': 3., 'lambda': 2.})
    write(directory, 'model.json', model.to_bytes())
    field = model.field('definition.displacement')
    observables = tuple(model.observable('definition.'+name) for name in ('energy', 'left_x', 'left_y', 'left_z'))
    plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(), quadrature=eqiora.fem.GaussLegendre(2), solve=eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.ConjugateGradient, preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Reproducible, provider=eqiora.solve.SolverProvider.reference(),
        relative_tolerance=1e-14, absolute_tolerance=1e-14, maximum_iterations=10000))
    write(directory, 'plan.json', plan.to_bytes())
    authenticate_plan(plan, model, field)
    replay = eqiora.Plan.from_bytes(plan.to_bytes())
    require(replay.identity == plan.identity and replay.to_bytes() == plan.to_bytes(), 'Plan canonical replay changed')
    # Reopened Models use exact IDs rather than original caller support aliases.
    reopened = eqiora.Model.from_bytes(model.to_bytes())
    replay_field = reopened.field(field.id)
    replay_observables = tuple(reopened.observable(item.id) for item in observables)
    require(replay_field == field and replay_observables == observables, 'Model replay lost exact selected IDs')
    require(reopened.to_bytes() == model.to_bytes() and reopened.digest == model.digest, 'Model canonical replay changed')
    authenticate_plan(replay, reopened, replay_field, authored_source_identity=model.authored_formulations[0].source_identity)
    return model, replay, replay_field, replay_observables


def authenticate_plan(plan, model, field, *, authored_source_identity=None):
    wire = json.loads(plan.to_bytes())
    require(wire.get('assembly_quadrature') == {'kind': 'gauss-legendre', 'points_per_axis': 2}, 'effective assembly quadrature differs')
    require(wire.get('spatial') == {'kind': 'uniform', 'policy': 'q1'}, 'effective discrete space differs')
    require(plan.model_digest == model.digest and plan.mesh_digest == plan.mesh.digest, 'Plan lineage differs')
    require(plan.fields == (field,), 'Plan field inventory differs')
    import eqiora
    require(plan.formulation.requested == eqiora.FormulationSelectionMode.Authored, 'authored weak form was dropped')
    if authored_source_identity is None:
        require(len(model.authored_formulations) == 1, 'fresh Model lost authored weak form inventory')
        authored_source_identity = model.authored_formulations[0].source_identity
    require(plan.formulation.requested_source_identity == authored_source_identity, 'selected authored weak form identity differs')
    require(plan.execution.workers == 1 and plan.execution.placement == 'host-serial', 'unexpected execution placement')
    mesh = plan.mesh
    require(mesh.dimension == 3, 'Mesh dimension differs')
    require(json.loads(mesh.to_bytes())['resources']['mesh']['kind'] == 'structured-cartesian', 'retained Mesh family differs')
    coordinates = tuple(tuple(float(v) for v in p) for p in mesh.coordinates)
    expected = set(itertools.product((0., 1., 2.), (0., 1.5, 3.), (0., 2., 4.)))
    require(len(coordinates) == 27 and set(coordinates) == expected, 'physical vertex bijection differs')
    cells = tuple(tuple(int(v) for v in cell) for cell in mesh.cells)
    require(len(cells) == 8, 'retained Hex count differs')
    anchors = set()
    for cell in cells:
        require(len(cell) == 8 and len(set(cell)) == 8, 'retained cell is not an eight-node Hex')
        require(all(0 <= v < 27 for v in cell), 'cell vertex index out of bounds')
        points = [coordinates[v] for v in cell]
        lower = tuple(min(p[a] for p in points) for a in range(3))
        require(lower not in anchors, 'duplicate Hex')
        anchors.add(lower)
        # Authenticate local x+2y+4z corner convention using physical coordinates,
        # independently of the product vertex inventory's z-fast storage order.
        for corner, point in enumerate(points):
            require(point == tuple(lower[a]+(1., 1.5, 2.)[a]*((corner >> a) & 1) for a in range(3)), 'wrong local corner/physical-axis map')
    require(anchors == set(itertools.product((0., 1.), (0., 1.5), (0., 2.))), 'Hex anchors differ')
    entities = tuple(plan.field_coefficient_entities(field))
    require(len(entities) == 27 and all(e[0] == 0 for e in entities), 'Q1 coefficient entities differ')
    return coordinates, entities


def extract(plan, model, result, field, observables):
    coordinates, entities = authenticate_plan(plan, model, field)
    wire = json.loads(result.to_bytes())
    content = wire['content']
    require(content['plan_identity'] == plan.identity and result.model_digest == model.digest, 'Result lineage differs')
    require(type(wire['identity']) is str and wire['identity'], 'canonical Result identity absent')
    output = result.output(field)
    require(output.field == field and output.mesh.digest == plan.mesh.digest, 'public output lineage differs')
    require(output.value_type.shape == [3] and output.value_type.array_rank == 0 and output.value_type.frame == 'spatial_cartesian', 'displacement component frame/shape differs')
    require(tuple(output.coefficient_dimension) == DIMENSION_M, 'displacement coefficient dimension differs')
    require(tuple(output.associations) == ('vertex',) and output.coefficient_count('vertex') == 27, 'complete vertex block differs')
    values = output.values('vertex').numpy(copy=False).reshape(27, 3)
    records = []
    for entity, components in zip(entities, values, strict=True):
        vertices = tuple(plan.mesh.entity_vertices(entity))
        require(len(vertices) == 1, 'vertex coefficient has invalid closure')
        point = coordinates[vertices[0]]
        for axis, value in zip('xyz', components, strict=True):
            records.append({'coordinate_m': [binary64(v) for v in point], 'component': axis, 'value_m': binary64(value)})
    observations = tuple(result.observe(item, quadrature_points=2) for item in observables)
    for item, observation, dimension in zip(observables, observations, ((1, 2, -2, 0, 0, 0, 0),)*1+((1, 1, -2, 0, 0, 0, 0),)*3, strict=True):
        require(tuple(observation.value_type.dimension.exponents) == dimension, 'public integral physical dimension differs')
        require(observation.observable_id == item.id and observation.result_identity == wire['identity'], 'public observation lineage differs')
        quadrature = ('GaussLegendre', 3 if item == observables[0] else 2, 2)
        require(observation.quadratures and set(observation.quadratures.values()) == {quadrature}, 'effective observation quadrature differs')
    return {'identity': {'dimension': 3, 'family': 'Hex', 'order': 'Q1', 'geometry': 'affine-cartesian', 'cells': [2, 2, 2], 'quadrature': [2, 2, 2], 'units': ['m', 'J', 'N']},
            'lineage': {'mesh': plan.mesh.digest, 'model': model.digest, 'result': wire['identity']}, 'coefficients': records,
            'energy_J': binary64(observations[0].value), 'left_traction_N': [binary64(v.value) for v in observations[1:]]}
