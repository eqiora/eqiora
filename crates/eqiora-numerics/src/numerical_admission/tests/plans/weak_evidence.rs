use super::*;
use eqiora_compiler::AuthoredFormulationProjection;
use num_complex::Complex64 as C;

mod actions;
mod coordinate_replay;
mod profiles;

const SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../verify/numerics/complex-weak-forms/models/affine.eqi"
));

fn geometry(reflected: bool) -> CanonicalGeometryV1 {
    let graph = GeometryGraph::new();
    let interval = graph.interval([0., 6.]).unwrap();
    graph
        .build(
            &interval,
            &BTreeMap::from([
                ("body".into(), vec![interval.region().into()]),
                (
                    "left".into(),
                    vec![interval.boundaries()[usize::from(reflected)].into()],
                ),
                (
                    "right".into(),
                    vec![interval.boundaries()[usize::from(!reflected)].into()],
                ),
            ]),
        )
        .unwrap()
}

fn compile(
    source: &str,
    geometry: &CanonicalGeometryV1,
) -> Result<(KernelProgram, AuthoredFormulationProjection), Diagnostic> {
    let bindings = ["body", "left", "right"].map(|name| {
        (
            name,
            StaticBindingValue::GeometrySupport {
                geometry,
                selection: geometry.entity_set(name).unwrap(),
                parent: (name != "body").then(|| geometry.entity_set("body").unwrap()),
            },
        )
    });
    let compiled = CompiledModel::compile_selected("affine-wave.eqi", source, "Wave", &bindings)
        .map_err(|errors| errors.into_iter().next().unwrap())?;
    let projection = compiled
        .authored_formulations()
        .next()
        .unwrap()
        .projection()
        .clone();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[geometry]).unwrap();
    Ok((program, projection))
}

fn resolve(source: &str) -> Result<ResolvedCommonPlan, Diagnostic> {
    resolve_on(
        source,
        geometry(false),
        LinearSolver::BiConjugateGradientStabilized,
    )
}

fn resolve_on(
    source: &str,
    geometry: CanonicalGeometryV1,
    algorithm: LinearSolver,
) -> Result<ResolvedCommonPlan, Diagnostic> {
    let (program, projection) = compile(source, &geometry)?;
    let model = ModelEnvelope::from_program(&program).unwrap();
    ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[2]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            algorithm,
            1e-12,
            1e-14,
            NonZeroUsize::new(64).unwrap(),
        )),
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        Some(&projection),
    )
}

#[test]
fn affine_complex_weak_form_has_independent_volume_and_boundary_solution() {
    let (program, projection) = compile(SOURCE, &geometry(false)).unwrap();
    actions::check(&program, &projection);
    profiles::check();
    let plan = replay_plan(resolve(SOURCE).unwrap(), &REFERENCE_LINEAR_SOLVER);
    let result = plan
        .as_scalar()
        .unwrap()
        .run_result(&REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let (_, values, shape) = result.field_block(0, 0).unwrap();
    assert_eq!(shape, &[3]);
    // u=1+3i+(2-i)x is exactly Q1. Its second derivative vanishes,
    // q*u=-2+4i+(3+i)x, and the outward right flux is a*(2-i)=18+6i.
    for (actual, expected) in
        values
            .as_chunks::<2>()
            .0
            .iter()
            .zip([C::new(1., 3.), C::new(7., 0.), C::new(13., -3.)])
    {
        assert!((C::new(actual[0], actual[1]) - expected).norm() < 1e-10);
    }
    let bytes = result.to_bytes().unwrap();
    assert_eq!(
        crate::CommonResult::from_bytes(&bytes, &plan)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    // A has diagonal (6+6i,3+3i), so it is not Hermitian. A valid
    // sesquilinear form must not grant the Hermitian-positive CG profile.
    let error = resolve_on(SOURCE, geometry(false), LinearSolver::ConjugateGradient).unwrap_err();
    for required in [
        "solver backend does not support the exact",
        "ConjugateGradient",
        "General",
    ] {
        assert!(error.message().contains(required), "{error:?}");
    }
    let real = SOURCE
        .replace("complex<m^2>", "m^2")
        .replace("complex<1/m>", "1/m")
        .replace("complex<m>", "m")
        .replace("complex<1>", "1")
        .replace("math.complex(6[m^2],6[m^2])", "6[m^2]")
        .replace("math.complex(1,1)", "1")
        .replace("math.complex(-2,4)", "1")
        .replace("math.complex(3[1/m],1[1/m])", "2[1/m]")
        .replace("math.complex(18[m],6[m])", "12[m]")
        .replace("math.complex(1,3)", "1");
    let real = resolve(&real)
        .unwrap()
        .as_scalar()
        .unwrap()
        .run_result(&REFERENCE_LINEAR_SOLVER)
        .unwrap();
    // The real specialization is u=1+2x, f=1+2x and right flux 12.
    for (actual, expected) in real.field_block(0, 0).unwrap().1.iter().zip([1., 7., 13.]) {
        assert!((actual - expected).abs() < 1e-10);
    }
    let reflected = SOURCE
        .replace("math.complex(-2,4)", "math.complex(16,10)")
        .replace(
            "math.complex(3[1/m],1[1/m])",
            "math.complex(-3[1/m],-1[1/m])",
        );
    let reflected = resolve_on(
        &reflected,
        geometry(true),
        LinearSolver::BiConjugateGradientStabilized,
    )
    .unwrap()
    .as_scalar()
    .unwrap()
    .run_result(&REFERENCE_LINEAR_SOLVER)
    .unwrap();
    // Reflect x -> 6-x. The prescribed end moves to x=6; the same
    // outward flux now acts on x=0 with normal -1 and derivative -2+i.
    for (actual, expected) in reflected
        .field_block(0, 0)
        .unwrap()
        .1
        .as_chunks::<2>()
        .0
        .iter()
        .zip([C::new(13., -3.), C::new(7., 0.), C::new(1., 3.)])
    {
        assert!((C::new(actual[0], actual[1]) - expected).norm() < 1e-10);
    }
    for (from, to) in [
        ("inner(grad(eta),a*grad(u))", "inner(a*grad(u),grad(eta))"),
        ("inner(eta,q*u)", "inner(eta,math.conj(q)*u)"),
        ("inner(trace(eta),g)", "inner(trace(eta),-g)"),
        (
            "inner(eta,f+s*coordinate(0))",
            "inner(eta,math.conj(f+s*coordinate(0)))",
        ),
    ] {
        assert!(resolve(&SOURCE.replace(from, to)).is_err(), "accepted {to}");
    }
}
