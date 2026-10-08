use std::collections::BTreeSet;

use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_schema::kernel::{DomainKind, KernelNode};
use eqiora_sem::KernelProgram;

use super::elasticity::derive_cartesian_q1_elasticity_form_2d;
use super::scalar::derive_candidate;
use super::vocabulary::{
    BoundaryTreatment, FormulationKind, FormulationRule, PrimalGalerkinCorrespondence,
};

const POISSON: &str =
    include_str!("../../../../crates/eqiora-api/packages/org.example.poisson/src/main.eqi");
const ELASTICITY: &str =
    include_str!("../../../../verify/solid/isotropic-elasticity-2d/models/linear-load.eqi");

#[test]
fn poisson_and_elasticity_share_one_typed_primal_galerkin_vocabulary() {
    let poisson_program = compile_program("poisson.eqi", POISSON);
    let poisson_domain = poisson_program
        .nodes()
        .find_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::CartesianBox { .. }) =>
            {
                Some(domain.id().erase())
            }
            _ => None,
        })
        .expect("Poisson has one Cartesian volume");
    let poisson = derive_candidate(&poisson_program, poisson_domain)
        .expect("Poisson derivation is valid")
        .expect("Poisson selects the shared formulation");

    let elasticity_program = compile_program("elasticity.eqi", ELASTICITY);
    let elasticity = derive_cartesian_q1_elasticity_form_2d(&elasticity_program)
        .expect("elasticity derivation is valid");

    assert_shared_primal_galerkin(poisson.correspondence());
    assert_shared_primal_galerkin(elasticity.correspondence());
    assert_ne!(
        poisson.correspondence().law,
        elasticity.correspondence().law
    );
}

fn assert_shared_primal_galerkin(correspondence: &PrimalGalerkinCorrespondence) {
    assert_eq!(
        correspondence.formulation.kind,
        FormulationKind::PrimalGalerkin
    );
    assert_eq!(
        correspondence.formulation.boundary_treatment,
        BoundaryTreatment::CompleteEssential
    );
    assert_eq!(correspondence.formulation.trial, correspondence.law.unknown);
    assert_eq!(correspondence.formulation.test, correspondence.law.unknown);
    assert_eq!(
        correspondence.formulation.rules,
        [
            FormulationRule::TestPairing,
            FormulationRule::DivergenceByParts,
            FormulationRule::ZeroTestTraceDischarge,
            FormulationRule::SourcePairing,
        ]
    );
    assert_eq!(correspondence.law.relations.len(), 5);
    assert_eq!(
        correspondence
            .law
            .relations
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len(),
        correspondence.law.relations.len()
    );
}

fn compile_program(name: &str, source: &str) -> KernelProgram {
    let mut compiled = compile(name, source).expect("source compiles");
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("transaction commits");
    KernelProgram::from_snapshot(&store.snapshot(), model).expect("kernel program projects")
}

#[test]
fn boundary_flux_compares_exact_parent_coordinate_restrictions() {
    let coefficient = "(1 + wave_number * coordinate(0))";
    let boundary = format!("normal({coefficient} * grad(potential)) = wave_number;");
    let source = POISSON
        .replace(
            "-div(grad(potential))",
            &format!("-div({coefficient} * grad(potential))"),
        )
        .replace(
            "relation y_upper_value on y_upper { trace(potential) = 0; }",
            &format!("relation y_upper_value on y_upper {{ {boundary} }}"),
        );
    for (source, admitted) in [
        (source.clone(), true),
        (
            source.replace(
                &boundary,
                &boundary.replace("coordinate(0)", "coordinate(1)"),
            ),
            false,
        ),
    ] {
        let program = compile_program("coordinate-flux.eqi", &source);
        let domain = program
            .nodes()
            .find_map(|node| match node {
                KernelNode::Domain(domain)
                    if matches!(domain.kind(), DomainKind::CartesianBox { .. }) =>
                {
                    Some(domain.id().erase())
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(
            derive_candidate(&program, domain).unwrap().is_some(),
            admitted
        );
    }
}

#[test]
fn decoded_complex_dependence_resolves_the_live_model() {
    use eqiora_compiler::{AuthoredFormulationProjection, CompiledModel, StaticBindingValue};
    use eqiora_geometry::GeometryGraph;
    let graph = GeometryGraph::new();
    let interval = graph.interval([0.0, 1.0]).unwrap();
    let geometry = graph
        .build(
            &interval,
            &std::collections::BTreeMap::from([
                ("body".to_owned(), vec![interval.region().into()]),
                ("left".to_owned(), vec![interval.boundaries()[0].into()]),
                ("right".to_owned(), vec![interval.boundaries()[1].into()]),
            ]),
        )
        .unwrap();
    let compiled = CompiledModel::compile_selected(
        "complex-form.eqi",
        r#"
public component ComplexForm(support body:volume(ambient_dimension=1)) {
    parameter q:complex<1/m^2> = math.complex(3,-1);
    variable u:complex<1> on body;
    relation wave on body { -div(grad(u))+q*u=0; }
    form weak for wave {
        test eta:1 for u;
        integrate(body,inner(grad(eta),grad(u))+inner(eta,q*u))=0;
    }
}
"#,
        "ComplexForm",
        &[(
            "body",
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("body").unwrap(),
                parent: None,
            },
        )],
    )
    .unwrap_or_else(|errors| panic!("{errors:?}"));
    let projection = compiled
        .authored_formulations()
        .next()
        .unwrap()
        .projection()
        .clone();
    let decoded = AuthoredFormulationProjection::decode(projection.canonical_bytes()).unwrap();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry]).unwrap();
    super::check_authored_dependence(&decoded, &program).unwrap();
    // Equal-looking wire leaves are not authority for a different live Model.
    let foreign = compile_program(
        "foreign.eqi",
        "model Foreign(){parameter q:complex<1/m^2> = math.complex(3,-1); variable u:complex<1/m^2>; relation law{u=q;}}",
    );
    let error = super::check_authored_dependence(&decoded, &foreign).unwrap_err();
    assert!(error.message().contains("not a live Field or Parameter"));
}
