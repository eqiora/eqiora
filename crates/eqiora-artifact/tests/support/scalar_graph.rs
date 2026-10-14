//! Forward projection for the single-Field Poisson artifact fixtures.
use eqiora_artifact::ModelEnvelope;
use eqiora_core::entity::kinds;
use eqiora_graph::EdgeKind;
use eqiora_realization::{PortableRealizationGraph, ResolvedRealization};
use eqiora_schema::kernel::KernelNode;
use eqiora_solver::LinearOperatorProperties;

pub fn graph(model: &ModelEnvelope, resolved: &ResolvedRealization) -> PortableRealizationGraph {
    let program = model.to_program().unwrap();
    let fields = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Field(field) => Some(field.id()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        fields.len(),
        1,
        "this fixture has exactly one physical Field"
    );
    let field = fields[0];
    let supports = program
        .edges()
        .iter()
        .filter(|edge| edge.from() == field.erase() && edge.kind() == EdgeKind::DefinedOn)
        .filter_map(|edge| edge.to().downcast::<kinds::Domain>())
        .collect::<Vec<_>>();
    assert_eq!(supports.len(), 1);
    resolved
        .portable_graph(
            supports[0],
            field,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap()
}
