use super::*;

pub(super) fn model_path(model: OntologyId<Model>) -> GraphPath {
    GraphPath::new(["ontology-view", "eqiora.model/v1", &model.to_string()])
}

pub(super) fn kernel_path(id: RawId) -> GraphPath {
    GraphPath::new(["semantic", &format!("{:?}", id.kind()), &id.to_string()])
}

pub(super) fn expression_path(owner: RawId, expression_id: u32) -> GraphPath {
    GraphPath::new([
        "semantic".to_owned(),
        format!("{:?}", owner.kind()),
        owner.to_string(),
        "expression".to_owned(),
        expression_id.to_string(),
    ])
}

pub(super) fn kernel_error(id: RawId, message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_KERNEL_DEFINITION, message).with_graph_path(kernel_path(id))
}

pub(super) fn clock_error(id: RawId, message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_CLOCK, message).with_graph_path(kernel_path(id))
}

pub(super) fn relation_dimension_error(id: RawId, message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_RELATION_DIMENSION, message).with_graph_path(kernel_path(id))
}
