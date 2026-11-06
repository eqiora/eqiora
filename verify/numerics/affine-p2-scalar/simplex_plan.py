"""Prepare and authenticate an ordinary public P2 Plan, without calling run."""
from pathlib import Path
from source_model import source
from simplex_production import inputs, authenticate_plan, require, sha256
from quadratic_output import write_new

HERE=Path(__file__).resolve().parent


def prepare(eqiora,dimension,n,profile,directory):
    authored=source(dimension,profile)
    msh=inputs.msh_bytes(dimension,n)
    source_sha=write_new(directory/'model.eqi',authored.encode())
    msh_sha=write_new(directory/'mesh.msh',msh)
    geometry=inputs.geometry(eqiora,dimension)
    mesh=eqiora.meshing.Mesh.import_msh(geometry,msh,coordinate_unit='m')
    mesh_sha=write_new(directory/'mesh.json',mesh.to_bytes())
    names=[f'q{i}_{side}' for i in range(dimension) for side in ('lower','upper')]
    model=eqiora.compile(source=authored,filename=f'affine-p2-{profile}.eqi',entry='AffineP2',geometry=geometry,
        bindings={'body':geometry.selection('body'),**{name:(geometry.selection(name),geometry.selection('body')) for name in names}})
    model_sha=write_new(directory/'model.json',model.to_bytes())
    field=model.field('definition.u')
    points=3 if profile=='patch' or dimension==2 else 5
    plan=eqiora.resolve(model,mesh=mesh,spatial=eqiora.fem.P2(),quadrature=eqiora.fem.SimplexDuffyGaussLegendre(dimension,points),
        solve=eqiora.solve.Linear(algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
            preconditioner=eqiora.solve.Preconditioner.Identity,reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),relative_tolerance=1e-14,absolute_tolerance=1e-14,maximum_iterations=100000))
    require(plan.fields==(field,), 'source-selected field differs from actual Plan inventory')
    require(plan.mesh_digest==mesh.digest,'Plan did not retain original caller Mesh')
    require(plan.execution.workers==1 and plan.execution.placement=='host-serial','unexpected actual execution policy')
    raw=plan.to_bytes();plan_sha=write_new(directory/'plan.json',raw)
    reopened=eqiora.Plan.from_bytes(raw)
    require(reopened.to_bytes()==raw and reopened.identity==plan.identity,'Plan replay changed canonical settings')
    require(reopened.fields==(field,), 'Plan replay changed actual field identity')
    _,actual_mesh,_,expected,_,_,_=authenticate_plan(reopened,dimension,n,profile)
    entities=tuple(reopened.field_coefficient_entities(field))
    require(entities==tuple(e for _,selected in expected for e in selected),'actual P2 field entity inventory differs')
    require(actual_mesh.digest==mesh.digest,'reopened actual Mesh identity differs')
    receipt={'schema':1,'status':'preflight-passed-no-run','dimension':dimension,'N':n,'profile':profile,
        'source_sha256':source_sha,'msh_input_sha256':msh_sha,'mesh_sha256':mesh_sha,'model_sha256':model_sha,'plan_sha256':plan_sha,
        'geometry_digest':geometry.digest,'model_digest':model.digest,'mesh_digest':mesh.digest,'plan_identity':plan.identity,
        'coefficient_entities':[list(e) for e in entities],'effective_duffy_points_per_axis':points,
        'actual_plan_workers':plan.execution.workers,'actual_plan_placement':plan.execution.placement,
        'source_closure_sha256':{str(p.relative_to(HERE.parent)):sha256(p.read_bytes()) for p in [HERE/'source_model.py',HERE/'fixtures.py',HERE/'simplex_plan.py',HERE/'simplex_production.py',HERE.parent/'quadratic_output.py',HERE/'reference'/'exact.py']},
        'reference_manifest_sha256':sha256((HERE/'reference'/'data'/'manifest.json').read_bytes()),
        'ordinary_run_attempted':0,'ordinary_run_completed':0,'plan_reopens':1}
    write_new(directory/'preflight.json',receipt)
    return reopened,field,receipt
