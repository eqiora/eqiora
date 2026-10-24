from importlib.resources import files
import eqiora

def prepare_plans():
    geometry_graph = eqiora.geometry.GeometryGraph()
    rectangle = geometry_graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
    circle = geometry_graph.circle(center=(0.2, 0.2), radius=0.05)
    fluid = geometry_graph.subtract(rectangle, circle)
    geometry = geometry_graph.build(fluid, named_topology={'fluid': fluid.region, 'inlet': rectangle.boundaries[0], 'outlet': rectangle.boundaries[1], 'walls': rectangle.boundaries[2:4], 'cylinder': circle.boundaries[0]})
    mesh_request = eqiora.meshing.GmshMesher(maximum_boundary_error=0.0001, maximum_target_size=0.025, minimum_mean_ratio=1e-05, maximum_boundary_facets=50)
    mesh_plan = eqiora.meshing.resolve(geometry, mesh_request)
    mesh = eqiora.meshing.generate(mesh_plan)
    source_root = files(eqiora).joinpath('examples')
    parameters = {'dynamic_viscosity': 0.001, 'zero_pressure': 0.0, 'inlet_speed': 0.3, 'channel_height': geometry.bounds[1][1] - geometry.bounds[1][0]}
    steady_model = eqiora.compile(path=source_root.joinpath('steady-flow-past-cylinder.eqi'), geometry=geometry, parameters=parameters)
    linear = eqiora.solve.Linear(relative_tolerance=1e-06, absolute_tolerance=1e-09, maximum_iterations=20000)
    steady_plan = eqiora.resolve(steady_model, mesh=mesh, spatial=eqiora.fem.MiniP1(), solve=linear, scaling=None)
    model = eqiora.compile(path=source_root.joinpath('transient-flow-past-cylinder.eqi'), geometry=geometry, parameters={'density': 1.0, **parameters})
    plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.MiniP1(), temporal=eqiora.time.BackwardEuler(0.01), solve=eqiora.solve.Newton(linear=linear), scaling=eqiora.fluid.IncompressibleScaling(length_m=0.41, velocity_m_per_s=0.3, pressure_pa=0.09))
    return (steady_plan, plan, mesh, geometry)
