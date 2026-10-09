use std::collections::BTreeSet;

use eqiora::api::ModelDocument;
use eqiora::entity::kinds;
use eqiora::graph::EdgeKind;
use eqiora::kernel::{
    ActivationKind, ConnectionSemantics, ExprDag, ExprId, ExprNode, KernelNode, SymbolRef,
};
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageDependencyV1, PackageManifestV1, PackageReleaseV1, PackageSourcesV1,
    PackagedModelDocument, QualifiedName, ResolutionRecordV1, SourceFileV1,
    prepare_package_release_v1,
};
use eqiora::{DimExponents, DynQuantity, Entity};

#[path = "support/embedded_package.rs"]
mod embedded_package;

const ROOT_SOURCE: &str =
    include_str!("../../../verify/solid/packaged-elastic-boundary-2d/models/coupled.eqi");
const ROOT_SOURCE_PERMUTED: &str =
    include_str!("../../../verify/solid/packaged-elastic-boundary-2d/models/coupled-permuted.eqi");
const PACKAGE_NAME: &str = "Eqiora.Solid.LinearElasticity";
const ROOT_NAME: &str = "org.eqiora.verify.packaged_elastic_boundary_2d";
const ROOT_VERSION: &str = "0.1.0";
const SIDES: [(&str, &str); 4] = [
    ("axis=0,side=lower", "x_lower"),
    ("axis=0,side=upper", "x_upper"),
    ("axis=1,side=lower", "y_lower"),
    ("axis=1,side=upper", "y_upper"),
];

fn mechanics_package() -> PackageReleaseV1 {
    prepare_package_release_v1(
        embedded_package::public_sources("Eqiora.Mechanics.Interfaces"),
        &[],
    )
    .expect("prepare current mechanics package")
}

fn package_release() -> PackageReleaseV1 {
    prepare_package_release_v1(
        embedded_package::public_sources(PACKAGE_NAME),
        &[mechanics_package()],
    )
    .expect("prepare current elasticity package")
}

fn root_sources(dependency: &PackageReleaseV1, alias: &str, source: &str) -> PackageSourcesV1 {
    let model_path = NormalizedRelativePath::parse("src/main.eqi").expect("root model path");
    let requirement = PackageDependencyV1::new(
        dependency
            .package_identity()
            .expect("elasticity package identity"),
    );
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse(ROOT_NAME).expect("root package name"),
        ExactVersion::parse(ROOT_VERSION).expect("root package version"),
        vec![requirement],
        vec![BundleEntryV1::new(
            model_path.clone(),
            BundleRoleV1::ModelSource,
        )],
    )
    .expect("root package manifest");
    let source = format!(
        "import {PACKAGE_NAME}.linear_elasticity as {alias};\n{}",
        source.replace("solid.", &format!("{alias}."))
    );
    if alias != "solid" {
        assert!(!source.contains("solid."));
    }
    PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            model_path,
            BundleRoleV1::ModelSource,
            source.into_bytes(),
        )],
    )
    .expect("closed root author sources")
}

fn root_release(dependency: &PackageReleaseV1, alias: &str, source: &str) -> PackageReleaseV1 {
    prepare_package_release_v1(
        root_sources(dependency, alias, source),
        &[dependency.clone(), mechanics_package()],
    )
    .expect("prepare exact elasticity-boundary root")
}

fn compile_locked(dependency: &PackageReleaseV1, root: &PackageReleaseV1) -> PackagedModelDocument {
    let resolution =
        ResolutionRecordV1::from_exact_releases(root, &[dependency.clone(), mechanics_package()])
            .expect("exact package resolution");
    let mut store = InMemoryPackageStore::default();
    store.insert(dependency).expect("insert dependency release");
    store
        .insert(&mechanics_package())
        .expect("insert mechanics release");
    store.insert(root).expect("insert root release");
    PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("compile exact packaged elasticity boundary")
}

fn typed<I: Entity>(model: &ModelDocument, name: &str) -> eqiora::Id<I> {
    model.aliases()[name]
        .downcast()
        .unwrap_or_else(|| panic!("`{name}` has the wrong entity kind"))
}

fn expression_node(dag: &ExprDag, id: ExprId) -> &ExprNode {
    dag.node(id).expect("validated expression ID")
}

fn subtraction(dag: &ExprDag, id: ExprId) -> (ExprId, ExprId) {
    match expression_node(dag, id) {
        ExprNode::Sub(left, right) => (*left, *right),
        node => panic!("expected subtraction, found {node:?}"),
    }
}

fn addition(dag: &ExprDag, id: ExprId) -> (ExprId, ExprId) {
    match expression_node(dag, id) {
        ExprNode::Add(left, right) => (*left, *right),
        node => panic!("expected addition, found {node:?}"),
    }
}

fn multiplication(dag: &ExprDag, id: ExprId) -> (ExprId, ExprId) {
    match expression_node(dag, id) {
        ExprNode::Mul(left, right) => (*left, *right),
        node => panic!("expected multiplication, found {node:?}"),
    }
}

fn unary_operand(
    dag: &ExprDag,
    id: ExprId,
    select: impl FnOnce(&ExprNode) -> Option<ExprId>,
    name: &str,
) -> ExprId {
    select(expression_node(dag, id))
        .unwrap_or_else(|| panic!("expected {name}, found {:?}", expression_node(dag, id)))
}

fn assert_symbol(dag: &ExprDag, id: ExprId, expected: SymbolRef) {
    assert_eq!(
        expression_node(dag, id),
        &ExprNode::Symbol(expected),
        "the exact semantic symbol must occupy this expression position"
    );
}

fn assert_isotropic_boundary_relation(
    residuals: &ExprDag,
    displacement: eqiora::Id<kinds::Field>,
    mu: eqiora::Id<kinds::Parameter>,
    lambda: eqiora::Id<kinds::Parameter>,
    port: eqiora::Id<kinds::Port>,
) {
    let [trace_root, traction_root] = residuals.roots() else {
        panic!("one boundary interface must have exactly two residual roots");
    };

    let (displacement_trace, port_trace) = subtraction(residuals, *trace_root);
    let traced_displacement = unary_operand(
        residuals,
        displacement_trace,
        |node| match node {
            ExprNode::Trace { value, .. } => Some(*value),
            _ => None,
        },
        "displacement trace",
    );
    assert_symbol(
        residuals,
        traced_displacement,
        SymbolRef::Field(displacement),
    );
    assert_symbol(residuals, port_trace, SymbolRef::PortTrace(port));

    let (outward_traction, port_flux) = subtraction(residuals, *traction_root);
    let stress = unary_operand(
        residuals,
        outward_traction,
        |node| match node {
            ExprNode::NormalComponent { value, .. } => Some(*value),
            _ => None,
        },
        "parent-outward normal component",
    );
    assert_symbol(residuals, port_flux, SymbolRef::PortFlux(port));

    let (shear_stress, volumetric_stress) = addition(residuals, stress);
    let (twice_mu, symmetric_gradient) = multiplication(residuals, shear_stress);
    let (two, mu_symbol) = multiplication(residuals, twice_mu);
    assert_eq!(
        expression_node(residuals, two),
        &ExprNode::Constant(
            DynQuantity::new(2.0, DimExponents::DIMENSIONLESS)
                .try_into()
                .unwrap()
        )
    );
    assert_symbol(residuals, mu_symbol, SymbolRef::Parameter(mu));
    let gradient = unary_operand(
        residuals,
        symmetric_gradient,
        |node| match node {
            ExprNode::SymmetricPart(value) => Some(*value),
            _ => None,
        },
        "symmetric part",
    );
    let gradient_field = unary_operand(
        residuals,
        gradient,
        |node| match node {
            ExprNode::Gradient(value) => Some(*value),
            _ => None,
        },
        "displacement gradient",
    );
    assert_symbol(residuals, gradient_field, SymbolRef::Field(displacement));

    let (lambda_symbol, lifted_divergence) = multiplication(residuals, volumetric_stress);
    assert_symbol(residuals, lambda_symbol, SymbolRef::Parameter(lambda));
    let divergence = unary_operand(
        residuals,
        lifted_divergence,
        |node| match node {
            ExprNode::IsotropicLift(value) => Some(*value),
            _ => None,
        },
        "isotropic lift",
    );
    let divergence_field = unary_operand(
        residuals,
        divergence,
        |node| match node {
            ExprNode::Divergence(value) => Some(*value),
            _ => None,
        },
        "displacement divergence",
    );
    assert_symbol(residuals, divergence_field, SymbolRef::Field(displacement));
}

#[test]
fn exact_package_elaborates_isotropic_boundary_meaning_without_a_realization() {
    let dependency = package_release();
    assert_ne!(ROOT_SOURCE, ROOT_SOURCE_PERMUTED);
    let canonical_root = root_release(&dependency, "solid", ROOT_SOURCE);
    let reordered_root = root_release(&dependency, "mechanics", ROOT_SOURCE_PERMUTED);
    assert_eq!(
        canonical_root
            .package_identity()
            .expect("canonical root identity"),
        reordered_root
            .package_identity()
            .expect("equivalent root identity"),
        "import alias spelling and exterior order cannot enter root meaning"
    );

    let canonical = compile_locked(&dependency, &canonical_root);
    let reordered = compile_locked(&dependency, &reordered_root);
    assert_eq!(
        canonical.model().canonical_json().expect("canonical Model"),
        reordered
            .model()
            .canonical_json()
            .expect("equivalent Model"),
        "import alias spelling and exterior order cannot enter Model bytes"
    );

    let model = canonical.model();
    let displacement = typed::<kinds::Field>(model, "displacement");
    let mu = typed::<kinds::Parameter>(model, "solid_boundary.mu");
    let lambda = typed::<kinds::Parameter>(model, "solid_boundary.lambda");
    let family_ports = SIDES
        .iter()
        .map(|(side, _)| typed::<kinds::Port>(model, &format!("solid_boundary.mechanical[{side}]")))
        .collect::<Vec<_>>();
    let family_relations = SIDES
        .iter()
        .map(|(side, _)| {
            typed::<kinds::Relation>(model, &format!("solid_boundary.boundary_interface[{side}]"))
        })
        .collect::<Vec<_>>();
    assert_eq!(family_ports.len(), 4);
    assert_eq!(family_relations.len(), 4);
    assert_eq!(
        family_ports
            .iter()
            .map(|port| port.erase())
            .collect::<BTreeSet<_>>()
            .len(),
        4
    );
    assert_eq!(
        family_relations
            .iter()
            .map(|relation| relation.erase())
            .collect::<BTreeSet<_>>()
            .len(),
        4
    );
    assert_eq!(
        model
            .aliases()
            .iter()
            .filter(|(name, id)| {
                name.starts_with("solid_boundary.mechanical[")
                    && id.downcast::<kinds::Port>().is_some()
            })
            .count(),
        4,
        "the complete 2D exterior generates exactly four Ports"
    );
    assert_eq!(
        model
            .aliases()
            .iter()
            .filter(|(name, id)| {
                name.starts_with("solid_boundary.boundary_interface[")
                    && id.downcast::<kinds::Relation>().is_some()
            })
            .count(),
        4,
        "the complete 2D exterior generates exactly four Relations"
    );

    let activations = family_relations
        .iter()
        .map(|relation| {
            let matches = model
                .program()
                .edges()
                .iter()
                .filter(|edge| edge.kind() == EdgeKind::Activates && edge.to() == relation.erase())
                .collect::<Vec<_>>();
            assert_eq!(matches.len(), 1, "each family Relation has one Activation");
            assert!(
                matches[0].from().downcast::<kinds::Activation>().is_some(),
                "Activates edge starts at an Activation"
            );
            matches[0].from()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(activations.len(), 4, "family Activations remain distinct");
    for activation in activations {
        let KernelNode::Activation(definition) = model
            .program()
            .node(activation)
            .expect("generated family Activation")
        else {
            panic!("Activates edge selects an Activation node");
        };
        assert!(matches!(definition.kind(), ActivationKind::Continuous));
    }

    let connections = family_ports
        .iter()
        .map(|port| {
            let matches = model
                .program()
                .edges()
                .iter()
                .filter(|edge| edge.kind() == EdgeKind::Connects && edge.to() == port.erase())
                .collect::<Vec<_>>();
            assert_eq!(matches.len(), 1, "each family Port enters one maximal set");
            assert!(
                matches[0].from().downcast::<kinds::Connection>().is_some(),
                "Connects edge starts at a Connection"
            );
            matches[0].from()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        connections.len(),
        4,
        "one connection set closes each exact side"
    );
    for connection in connections {
        let KernelNode::Connection(definition) = model
            .program()
            .node(connection)
            .expect("exact boundary Connection")
        else {
            panic!("Connects edge selects a Connection node");
        };
        assert_eq!(definition.semantics(), ConnectionSemantics::Conserving);
        assert_eq!(
            model
                .program()
                .edges()
                .iter()
                .filter(|edge| { edge.kind() == EdgeKind::Connects && edge.from() == connection })
                .count(),
            2,
            "each exact interface joins one package Port to one terminal Port"
        );
    }

    for (side, terminal) in SIDES {
        let family_port_name = format!("solid_boundary.mechanical[{side}]");
        let family_port = typed::<kinds::Port>(model, &family_port_name);
        let terminal_port = typed::<kinds::Port>(model, &format!("{terminal}_terminal.mechanical"));
        let (KernelNode::Port(family_definition), KernelNode::Port(terminal_definition)) = (
            model
                .program()
                .node(family_port.erase())
                .expect("generated family Port"),
            model
                .program()
                .node(terminal_port.erase())
                .expect("terminal Port"),
        ) else {
            panic!("exact aliases select ordinary Port nodes");
        };
        assert_eq!(
            family_definition.payload(),
            terminal_definition.payload(),
            "the nominal Connector and exact Boundary are shared"
        );

        let relation_name = format!("solid_boundary.boundary_interface[{side}]");
        let relation = typed::<kinds::Relation>(model, &relation_name);
        let KernelNode::Relation(definition) = model
            .program()
            .node(relation.erase())
            .expect("generated boundary Relation")
        else {
            panic!("exact alias selects an ordinary Relation node");
        };
        let residuals = &model
            .program()
            .numerical_residuals(definition.id().erase())
            .unwrap();
        assert_isotropic_boundary_relation(residuals, displacement, mu, lambda, family_port);
    }
}
