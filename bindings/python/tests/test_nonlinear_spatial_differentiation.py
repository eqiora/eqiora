"""Nonlinear material sensitivities use ordinary Plan/State and collection owners."""
import eqiora as q
import numpy as np
import pytest


def newton_policy():
    linear = q.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-14, maximum_iterations=100,
        algorithm=q.solve.LinearSolver.SparseLu, preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer())
    return q.solve.Newton(
        linear=linear, relative_tolerance=0., absolute_tolerance=1e-11,
        maximum_iterations=16, maximum_line_search_steps=16)


def material_plan(property_law):
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 1.))
    geometry = graph.build(interval, named_topology={
        "body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1],
    })
    body = geometry.selection("body")
    supports = """support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)"""
    equations = """variable u:1 on body in smooth;
        relation balance on body {-div((1+p*u)*grad(u))=0;}
        relation lower on left {trace(u)=1;}
        relation upper on right {trace(u)=r;}"""
    if property_law:
        source = f"""
        property contract Law(input value:1):1 {{derivatives first_partials;}}
        property release Release:Law {{
            analytic {{value=1+value;source_unit:1=1;}}
            validity value>=-0.5 and value<=4;outside reject;branch single;
            citation org.example.manufactured;license spdx.CC0_1_0;
        }}
        component Equation({supports},parameter p:1,parameter r:1,property response:Law) {{
            {equations.replace('(1+p*u)', 'response(value=p*u)')}
        }}
        public model Main({supports}) {{parameter p:1=1;parameter r:1=2;
            instance physics:Equation(body=body,left=left,right=right,p=p,r=r,response=Release);
        }}"""
    else:
        source = f"public model Main({supports}) {{parameter p:1=1;parameter r:1=2;{equations}}}"
    model = q.compile(source=source, entry="Main", geometry=geometry, bindings={
        "body": body, "left": (geometry.selection("left"), body),
        "right": (geometry.selection("right"), body),
    })
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(2,))))
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), solve=newton_policy())
    plan = q.Plan.from_bytes(plan.to_bytes())
    field = plan.fields[0]
    seed = q.State.initial(plan, fields=(q.InitialField(field, vertex_values=np.array([1., 1.5, 2.])),))
    return model, plan, q.State.from_bytes(plan, seed.to_bytes())


def exact(p, r):
    # q(u)=u+p*u²/2 is affine in x and each P1 cell flux integrates exactly.
    # At the midpoint R_u=4(1+p*u), R_p=2u²-1-r² and R_r=-2(1+p*r).
    mean_q = (1+r)/2 + p*(1+r*r)/4
    u = mean_q if p == 0 else (np.sqrt(1+2*p*mean_q)-1)/p
    jacobian = 4*(1+p*u)
    return u, -(2*u*u-1-r*r)/jacobian, 2*(1+p*r)/jacobian


@pytest.mark.parametrize("property_law", [False, True])
def test_nonlinear_spatial_products_and_recorded_collection(property_law):
    model, plan, seed = material_plan(property_law)
    before = seed.to_bytes()
    inputs = tuple(plan.model.parameter(model.parameter(name).id) for name in ("p", "r"))
    program = q.diff.compile(plan, inputs=inputs, outputs=(plan.fields[0],), state=seed)
    points = [(0., 2.), (1., 2.), (1., 3.)]
    for p, r in points:
        u, dp, dr = exact(p, r)
        point = program.evaluate((p, r))
        np.testing.assert_allclose(point.primal().output[0], [1., u, r], rtol=0, atol=1e-10)
        np.testing.assert_allclose(point.accepted_unknowns.numpy(), [u], rtol=0, atol=1e-10)
        np.testing.assert_allclose(point.jvp((1., 0.)).tangent[0], [0., dp, 0.], rtol=0, atol=1e-10)
        np.testing.assert_allclose(point.jvp((0., 1.)).tangent[0], [0., dr, 1.], rtol=0, atol=1e-10)
        reverse = point.vjp((np.array([2., 3., 5.]),))
        np.testing.assert_allclose(reverse.input_cotangent, [3*dp, 3*dr+5], rtol=0, atol=1e-10)
        assert reverse.evidence.initial_state_identity == seed.digest
        assert (point.primal().evidence.primal_solve.completed_iterations == 0) == (p == 0)
    if property_law:
        with pytest.raises(q.EqioraError):
            program.evaluate((-1., 2.))
        # A rejected operating point cannot alter the retained seed or default point.
        np.testing.assert_allclose(program.primal().output[0], [1., exact(1., 2.)[0], 2.], rtol=0, atol=1e-10)
    collection = program.map(tuple(np.asarray(values) for values in zip(*points)), point_shape=(3,)).execute_recorded()
    assert collection.complete
    encoded = collection.to_bytes()
    del collection, program, model
    reopened = q.EvaluationCollection.from_bytes(encoded)
    assert reopened.to_bytes() == encoded
    for member, (p, r) in zip(reopened[0], points, strict=True):
        np.testing.assert_allclose(member.result.output[0], [1., exact(p, r)[0], r], rtol=0, atol=1e-10)
        assert member.result.evidence.initial_state_identity == seed.digest
    assert seed.to_bytes() == before
    with pytest.raises(q.EqioraError):
        q.diff.compile(plan, inputs=inputs, outputs=(plan.fields[0],))


def tensor_material_plan(property_law):
    graph = q.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0., 1.), y_bounds=(0., 1.))
    sides = ("left", "right", "bottom", "top")
    geometry = graph.build(rectangle, named_topology={
        "body": rectangle.region, **dict(zip(sides, rectangle.boundaries, strict=True)),
    })
    body = geometry.selection("body")
    supports = "support body:volume(ambient_dimension=2)," + ",".join(
        f"support {side}:boundary(parent=body)" for side in sides)
    law = "(1+T)*delta(0,1)+p"
    declarations = f"operator k(input T:1,input p:1):tensor<1,2,2>={law};"
    property_argument = ""
    binding = ""
    if property_law:
        declarations = f"""
        property contract Conductivity(input T:1,input p:1):tensor<1,2,2> {{
            derivatives first_partials;
        }}
        property release Measured:Conductivity {{
            analytic {{value={law};source_unit:1=1;}}
            validity T>=0 and T<=20 and p>=0 and p<=3;outside reject;branch single;
            citation org.example.manufactured;license spdx.CC0_1_0;
        }}"""
        property_argument = ",property k:Conductivity"
        binding = ",k=Measured"
    exact = "p*(1+coordinate(0)/1[m]+2*coordinate(1)/1[m])"
    flux = "contract(k(T=u,p=p),grad(u),axes=((1,0),))"
    source = f"""{declarations}
    component Equation({supports},parameter p:1{property_argument}) {{
        variable u:1 on body in smooth;
        relation balance on body {{-div({flux})=-5*p*p/1[m^2];}}
        relation lower_x on left {{trace(u)={exact};}}
        relation lower_y on bottom {{trace(u)={exact};}}
        relation upper_x on right {{normal({flux})=((1+{exact})*p+3*p*p)/1[m];}}
        relation upper_y on top {{normal({flux})=(2*(1+{exact})*p+3*p*p)/1[m];}}
    }}
    public model Main({supports}) {{parameter p:1=1;
        instance physics:Equation(body=body,left=left,right=right,bottom=bottom,top=top,p=p{binding});
    }}"""
    model = q.compile(source=source, entry="Main", geometry=geometry, bindings={
        "body": body, **{side: (geometry.selection(side), body) for side in sides},
    })
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(2, 2))))
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), solve=newton_policy())
    plan = q.Plan.from_bytes(plan.to_bytes())
    points = np.stack(np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3), indexing="ij"), axis=-1).reshape(-1, 2)
    profile = 1 + points @ np.array([1., 2.])
    seed = q.State.initial(plan, fields=(q.InitialField(
        plan.fields[0], vertex_values=profile-.25*points[:, 0]*points[:, 1]),))
    return model, plan, q.State.from_bytes(plan, seed.to_bytes()), profile


@pytest.mark.parametrize("property_law", [False, True])
def test_tensor_material_solution_and_total_parameter_products(property_law):
    model, plan, seed, profile = tensor_material_plan(property_law)
    before = seed.to_bytes()
    program = q.diff.compile(plan, inputs=(plan.model.parameter(model.parameter("p").id),),
                             outputs=(plan.fields[0],), state=seed)
    # K=(1+u)I+pJ, where J has every entry equal to one. For u=p(1+x+2y),
    # grad(u)=p(1,2), div(K grad(u))=5p² and J grad(u)=(3p,3p).
    # The nonzero off-diagonal flux and both natural traces matter. Q1 represents
    # this affine solution exactly; du/dp=1+x+2y includes boundary and state terms.
    weights = np.arange(1, 10)/16
    for p in [.5, 1., 2.]:
        point = program.evaluate((p,))
        np.testing.assert_allclose(point.primal().output[0], p*profile, rtol=0, atol=1e-10)
        np.testing.assert_allclose(point.jvp((1.,)).tangent[0], profile, rtol=0, atol=1e-10)
        np.testing.assert_allclose(point.vjp((weights,)).input_cotangent,
                                   [np.dot(profile, weights)], rtol=0, atol=1e-10)
    if property_law:
        with pytest.raises(q.EqioraError):
            program.evaluate((4.,))
        np.testing.assert_allclose(program.primal().output[0], profile, rtol=0, atol=1e-10)
    assert seed.to_bytes() == before
