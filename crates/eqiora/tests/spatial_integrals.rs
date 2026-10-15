//! Independent affine volume and parent-outward boundary integrals on a 3-4-5 triangle.
use eqiora::api::ModelDocument;
use eqiora::artifact::{GmshMeshPolicyV1, ModelEnvelope};
use eqiora::compiler::StaticBindingValue;
use eqiora::geometry::{CanonicalGeometryV1, NamedEntitySet, PlanarFace, PlanarRegion};
use eqiora::solver::{
    LinearSolver, REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER, ReductionPolicy, SolverPlan,
};
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonLinearRequest, CommonResult, CommonSolvePolicy,
    CommonSpatialPolicy, ResolvedCommonPlan,
};
use std::{collections::HashMap, num::NonZeroUsize};

const SOURCE: &str = r#"public component Heat(
    support body:volume(ambient_dimension=2),
    support base:boundary(parent=body),
    support diagonal:boundary(parent=body),
    support left:boundary(parent=body)
) {
    variable temperature:K on body in smooth;
    relation balance on body {-div(grad(temperature))=0[K/m^2];}
    relation fixed_base on base {trace(temperature)=300[K]+2[K/m]*coordinate(0)-3[K/m]*coordinate(1);}
    relation fixed_diagonal on diagonal {trace(temperature)=300[K]+2[K/m]*coordinate(0)-3[K/m]*coordinate(1);}
    relation fixed_left on left {trace(temperature)=300[K]+2[K/m]*coordinate(0)-3[K/m]*coordinate(1);}
    observable base_flux:W/m=integral(normal(-2[W/(m*K)]*grad(temperature)),measure(base));
    observable diagonal_flux:W/m=integral(normal(-2[W/(m*K)]*grad(temperature)),measure(diagonal));
    observable left_flux:W/m=integral(normal(-2[W/(m*K)]*grad(temperature)),measure(left));
    observable net_flux:W/m=base_flux+diagonal_flux+left_flux;
    observable diagonal_temperature:K*m=integral(trace(temperature),measure(diagonal));
    observable volume_temperature:K*m^2=integral(temperature,measure(body));
}"#;

fn geometry() -> CanonicalGeometryV1 {
    CanonicalGeometryV1::from_region(
        &PlanarRegion::new(
            vec![[0., 0.], [4., 0.], [0., 3.]],
            vec![PlanarFace::new(vec![0, 1, 2], vec![])],
            vec![
                NamedEntitySet::new("body", 2, vec![0]),
                NamedEntitySet::new("base", 1, vec![0]),
                NamedEntitySet::new("diagonal", 1, vec![1]),
                NamedEntitySet::new("left", 1, vec![2]),
            ],
            1e-12,
        )
        .unwrap(),
    )
    .unwrap()
}

fn document(geometry: &CanonicalGeometryV1, source: &str) -> ModelDocument {
    let body = geometry.entity_set("body").unwrap();
    let bindings = ["body", "base", "diagonal", "left"].map(|name| {
        (
            name,
            StaticBindingValue::GeometrySupport {
                geometry,
                selection: geometry.entity_set(name).unwrap(),
                parent: (name != "body").then_some(body),
            },
        )
    });
    ModelDocument::compile_selected("integrals.eqi", source, "Heat", &bindings).unwrap()
}

fn resolve(
    model: &ModelEnvelope,
    geometry: &CanonicalGeometryV1,
    permuted: bool,
    center: [f64; 2],
) -> ResolvedCommonPlan {
    let mut msh = format!(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n0 0 0\n4 0 0\n0 3 0\n{} {} 0\n$EndNodes\n$Elements\n1 3 1 3\n2 1 2 3\n",
        center[0], center[1]
    );
    for (index, mut cell) in [[1, 2, 4], [2, 3, 4], [3, 1, 4]].into_iter().enumerate() {
        if permuted {
            cell.rotate_left(1);
        }
        msh += &format!("{} {} {} {}\n", index + 1, cell[0], cell[1], cell[2]);
    }
    msh += "$EndElements\n";
    let resources = AuthenticatedCommonMesh::gmsh_4152(
        geometry.clone(),
        GmshMeshPolicyV1::explicit(1e-12, 0.01, 8, 5.).unwrap(),
        msh.into_bytes(),
    )
    .unwrap();
    ResolvedCommonPlan::resolve(
        model,
        resources,
        CommonSpatialPolicy::P1,
        CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(
                SolverPlan::new(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-11,
                    1e-13,
                    NonZeroUsize::new(1000).unwrap(),
                )
                .unwrap()
                .with_reduction(ReductionPolicy::Reproducible),
                REFERENCE_SOLVER_PROVIDER,
            )
            .unwrap(),
        ),
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap()
}

#[test]
fn affine_triangle_integrals_retain_measure_orientation_and_accepted_occurrence() {
    let geometry = geometry();
    let document = document(&geometry, SOURCE);
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let domain = |name: &str| document.aliases()[name].downcast().unwrap();
    let observable = |name: &str| {
        document.aliases()[&format!("definition.{name}")]
            .downcast()
            .unwrap()
    };
    let line = eqiora::meshing::simplex_duffy_gauss_legendre(1, 2).unwrap();
    let area = eqiora::meshing::simplex_duffy_gauss_legendre(2, 2).unwrap();
    for permuted in [false, true] {
        let plan = resolve(&model, &geometry, permuted, [4. / 3., 1.]);
        let result = plan
            .as_spatial()
            .unwrap()
            .run_result(&REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let accepted = result.to_bytes().unwrap();
        let replay = CommonResult::from_bytes(&accepted, &plan).unwrap();
        for result in [&result, &replay] {
            // T=300+2*x-3*y is harmonic and exactly affine. The physical heat
            // flux is -2*grad(T)=(-4,6) W/m². Edge lengths are 4,5,3 metres;
            // their outward normals are (0,-1),(3/5,4/5),(-1,0).
            // T averages 299.5 K on the diagonal; area=6, vertex mean=899/3 K.
            for (name, support, expected, rule) in [
                ("base_flux", "base", -24., &line),
                ("diagonal_flux", "diagonal", 12., &line),
                ("left_flux", "left", 12., &line),
                ("diagonal_temperature", "diagonal", 1497.5, &line),
                ("volume_temperature", "body", 1798., &area),
            ] {
                let rules = HashMap::from([(domain(support), rule.clone())]);
                assert_eq!(
                    result.observation_quadratures(observable(name), 2).unwrap(),
                    rules
                );
                let sample = result.observe(&model, observable(name), &rules).unwrap();
                let value = sample.value().real_scalar_value().unwrap();
                assert!((value.value() - expected).abs() < 1e-9, "{name}: {value:?}");
                let exponents = match name {
                    "diagonal_temperature" => [0, 1, 0, 0, 1, 0, 0],
                    "volume_temperature" => [0, 2, 0, 0, 1, 0, 0],
                    _ => [1, 1, -3, 0, 0, 0, 0],
                };
                assert_eq!(
                    value.dim(),
                    eqiora::DimExponents::from_integers(exponents).unwrap()
                );
                assert_eq!(sample.result_identity(), result.identity());
                assert_eq!(sample.quadratures(), &rules);
            }
            let rules = ["base", "diagonal", "left"]
                .map(|name| (domain(name), line.clone()))
                .into();
            let net = result
                .observe(&model, observable("net_flux"), &rules)
                .unwrap();
            assert!(net.value().real_scalar_value().unwrap().value().abs() < 1e-9);
            let wrong = HashMap::from([(
                domain("diagonal"),
                eqiora::meshing::QuadratureRule::gauss_legendre(2).unwrap(),
            )]);
            assert!(
                result
                    .observe(&model, observable("diagonal_flux"), &wrong)
                    .unwrap_err()
                    .message()
                    .contains("reference cell")
            );
            assert_eq!(result.to_bytes().unwrap(), accepted);
        }
        let foreign = resolve(&model, &geometry, permuted, [1., 1.]);
        assert!(CommonResult::from_bytes(&accepted, &foreign).is_err());
    }
}
