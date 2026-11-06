"""Prepare and export the shipped cylinder example through public Eqiora APIs."""
from importlib.resources import files
import numpy as np
from mesh_levels import mesh_level

BOUNDARIES = ('inlet', 'outlet', 'walls', 'cylinder')


def author_geometry(eqiora):
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0., 2.2), y_bounds=(0., .41))
    circle = graph.circle(center=(.2, .2), radius=.05)
    fluid = graph.subtract(rectangle, circle)
    geometry = graph.build(fluid, named_topology={
        'fluid': fluid.region, 'inlet': rectangle.boundaries[0],
        'outlet': rectangle.boundaries[1], 'walls': rectangle.boundaries[2:4],
        'cylinder': circle.boundaries[0]})
    return geometry


def geometry_and_mesh(eqiora, *, level='standard'):
    geometry = author_geometry(eqiora)
    controls = mesh_level(level).controls()
    mesh = eqiora.meshing.generate(geometry, eqiora.meshing.GmshMesher(**controls))
    return geometry, mesh


def export_mesh(geometry, mesh):
    return {'coordinates': np.asarray(mesh.coordinates).tolist(),
            'cells': np.asarray(mesh.cells).tolist(),
            'boundaries': {name: [list(mesh.entity_vertices(entity)) for entity in
                                 mesh.selection_entities(geometry.selection(name))]
                           for name in BOUNDARIES},
            'mesh_digest': mesh.digest, 'geometry_digest': geometry.digest,
            'production_lineage_digest': mesh.production_lineage_digest}


def compile_model(eqiora, geometry):
    return eqiora.compile(path=files(eqiora).joinpath('examples', 'steady-flow-past-cylinder.eqi'),
                          geometry=geometry, entry='SteadyFlowPastCylinder', bindings={
        'fluid': geometry.selection('fluid'),
        **{name: geometry.selection(name) for name in BOUNDARIES},
        'dynamic_viscosity': .001, 'inlet_speed': .3, 'channel_height': .41})


def export_fields(plan, result):
    """Return public MINI hierarchical coefficients in original mesh order."""
    velocity = result.output(plan.capability.velocity)
    pressure = result.output(plan.capability.pressure)
    return {'velocity_vertex': velocity.values('vertex').numpy().reshape(-1, 2),
            'velocity_cell_bubble': velocity.values('cell-bubble').numpy().reshape(-1, 2),
            'pressure_vertex': pressure.values('vertex').numpy()}
