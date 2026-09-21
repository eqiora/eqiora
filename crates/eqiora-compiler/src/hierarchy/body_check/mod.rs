//! Occurrence-free type checking for reusable package definitions.
//!
//! Definitions are checked without inventing Model occurrences, graph IDs,
//! transactions, parameter values, instance paths, or provenance. Expression
//! rules come from the same identity-parametric kernel contract used by the
//! semantic oracle.

mod property;
use eqiora_core::Diagnostic;
use std::collections::{BTreeMap, BTreeSet};

use eqiora_lang::{Expr, ExprKind, NamePath, TextRange};

use crate::connection_sets::ConnectionFragment;
use crate::connection_sets::ConnectionSetLimits;

use super::field_slots::FieldInterface;
use super::parameters::SymbolicParameterMap;
use super::preflight::{ComponentDefinition, DefinitionKey, Elaborator, ModelDefinition};
use super::supports::SupportInterface;

mod completion;
pub(crate) use completion::CompletionIndex;
mod component;
mod expression;
mod indexed;
pub(super) use expression::DependencyActivation;
mod model;
mod scope;
pub(super) use scope::{field_expression_type, resolve_value_shape};

pub(super) fn validate_component_body(
    elaborator: &Elaborator<'_>,
    definition: &ComponentDefinition<'_>,
    parameters: &SymbolicParameterMap,
    supports: &SupportInterface,
    fields: &FieldInterface,
) -> Result<DefinitionBodyProof, Vec<Diagnostic>> {
    component::validate(elaborator, definition, parameters, supports, fields)
}

pub(super) fn validate_model_body(
    elaborator: &Elaborator<'_>,
    definition: &ModelDefinition<'_>,
    compile_time_values: &SymbolicParameterMap,
) -> Result<DefinitionBodyProof, Vec<Diagnostic>> {
    model::validate(elaborator, definition, compile_time_values)
}

/// A physical endpoint selected and type-checked in one definition body.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum ResolvedPhysicalEndpoint {
    Local(String),
    Child { instance: String, port: String },
}

impl ResolvedPhysicalEndpoint {
    pub(super) fn from_path(path: &NamePath) -> Option<Self> {
        let segments = path.segments().collect::<Vec<_>>();
        match segments.as_slice() {
            [name] => Some(Self::Local((*name).to_owned())),
            [instance, port] => Some(Self::Child {
                instance: (*instance).to_owned(),
                port: (*port).to_owned(),
            }),
            _ => None,
        }
    }

    pub(super) fn from_key(key: &[String]) -> Option<Self> {
        match key {
            [port] => Some(Self::Local(port.clone())),
            [instance, port] => Some(Self::Child {
                instance: instance.clone(),
                port: port.clone(),
            }),
            [instance, ordinal, port] if ordinal.parse::<u32>().is_ok() => Some(Self::Child {
                instance: format!("{instance}[{ordinal}]"),
                port: port.clone(),
            }),
            _ => None,
        }
    }

    pub(super) fn from_expression(expression: &Expr) -> Option<Self> {
        match expression.kind() {
            ExprKind::Name(name) => Some(Self::Local(name.clone())),
            ExprKind::Path(path) => Self::from_path(path),
            _ => None,
        }
    }

    pub(super) fn display(&self) -> String {
        match self {
            Self::Local(name) => name.clone(),
            Self::Child { instance, port } => format!("{instance}.{port}"),
        }
    }
}

pub(super) type PhysicalEndpointSelections = BTreeSet<ResolvedPhysicalEndpoint>;
pub(super) type PhysicalConnectionFragment = ConnectionFragment<ResolvedPhysicalEndpoint>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LocalPhysicalPortProof {
    pub(super) public: bool,
    pub(super) range: TextRange,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ChildInstanceProof {
    pub(super) definition: DefinitionKey,
    pub(super) range: TextRange,
}

/// Successful typed-body proof consumed by later physical closure.
#[derive(Clone, Debug)]
pub(super) struct DefinitionBodyProof {
    pub(super) file: String,
    pub(super) range: TextRange,
    pub(super) connection_limits: ConnectionSetLimits,
    pub(super) local_physical_ports: BTreeMap<String, LocalPhysicalPortProof>,
    pub(super) children: BTreeMap<String, ChildInstanceProof>,
    pub(super) relation_endpoints: Vec<PhysicalEndpointSelections>,
    pub(super) physical_connection_fragments: Vec<PhysicalConnectionFragment>,
    /// Occurrence-independent endpoints selected by Connections whose exact
    /// equivalence class depends on a statically elaborated boundary member.
    ///
    /// The definition pass can discharge these endpoints' one-membership
    /// obligation, but only occurrence expansion may normalize the complete
    /// connection set after exact Boundary identities are known.
    pub(super) deferred_connection_memberships: PhysicalEndpointSelections,
}

impl DefinitionBodyProof {
    pub(super) fn new(
        file: &str,
        range: TextRange,
        connection_limits: ConnectionSetLimits,
    ) -> Self {
        Self {
            file: file.to_owned(),
            range,
            connection_limits,
            local_physical_ports: BTreeMap::new(),
            children: BTreeMap::new(),
            relation_endpoints: Vec::new(),
            physical_connection_fragments: Vec::new(),
            deferred_connection_memberships: BTreeSet::new(),
        }
    }
}

#[derive(Default)]
pub(super) struct DefinitionBodyProofs {
    pub(super) components: BTreeMap<DefinitionKey, Vec<DefinitionBodyProof>>,
    pub(super) models: BTreeMap<DefinitionKey, DefinitionBodyProof>,
}

fn validate_clock(
    file: &str,
    _range: TextRange,
    period: &eqiora_lang::Expr,
    phase: &eqiora_lang::Expr,
) -> Result<(), Diagnostic> {
    crate::units::lower_clock(file, period, phase).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_identity::LocalSourceIdentity;

    fn validate_model(source: &str, model: &str) -> Result<(), Vec<Diagnostic>> {
        let document = eqiora_lang::parse("models.eqi", source)
            .into_document()
            .expect("test source parses");
        let identity = LocalSourceIdentity::from_document(&document).expect("source identity");
        let elaborator = Elaborator::new(
            "models.eqi",
            source.len(),
            &document,
            identity,
            super::super::HierarchyLimits::default(),
        )?;
        let definition = elaborator
            .models()
            .find_map(|(_, definition)| {
                (definition.declaration.name() == model).then(|| definition.clone())
            })
            .expect("selected Model exists");
        validate_model_body(&elaborator, &definition, &SymbolicParameterMap::new()).map(|_| ())
    }

    #[test]
    fn valid_spatial_contract_is_accepted() {
        let source = r#"
model Poisson() {
  domain body = box(0, 1, 0, 1);
  domain wall = boundary(body, axis = 0, side = lower);
  variable u: 1 on body; initial { u = 0; }
  relation balance on body { -div(grad(u)) = 0; }
  relation boundary on wall { trace(u) = 0; }
}
"#;
        validate_model(source, "Poisson").expect("valid spatial model");
    }

    #[test]
    fn spatial_operator_falsifiers_fail_without_occurrences() {
        let cases = [
            (
                "grad parameter",
                "model M() { domain d = box(0,1); parameter p: 1 = 1; relation r on d { grad(p) = 0; } }",
                "gradient operand has no spatial Domain support",
            ),
            (
                "div scalar",
                "model M() { domain d = box(0,1); variable u: 1 on d; initial { u = 0; } relation r on d { div(u) = 0; } }",
                "divergence requires a spatial tensor operand",
            ),
            (
                "symmetric vector",
                "model M() { domain d = box(0,1,0,1); variable u: vector<1, 2> on d; relation r on d { symmetric_part(u) = 0; } }",
                "symmetric_part requires an exact [d,d] spatial Cartesian tensor",
            ),
            (
                "symmetric nonsquare",
                "model M() { domain d = box(0,1,0,1); variable a: array<array<1, 3>, 2> on d; relation r on d { symmetric_part(a) = 0; } }",
                "symmetric_part requires an exact [d,d] spatial Cartesian tensor",
            ),
            (
                "lift vector",
                "model M() { domain d = box(0,1,0,1); variable u: vector<1, 2> on d; relation r on d { isotropic_lift(u) = 0; } }",
                "isotropic_lift requires an invariant scalar",
            ),
            (
                "lift parameter",
                "model M() { domain d = box(0,1,0,1); parameter p: 1 = 1; relation r on d { isotropic_lift(p) = 0; } }",
                "isotropic_lift requires a Cartesian volume operand",
            ),
            (
                "coordinate without domain",
                "model M() { relation r { coordinate(0) = 0; } }",
                "coordinate operator requires a Cartesian Relation scope",
            ),
            (
                "coordinate outside domain",
                "model M() { domain d = box(0,1); relation r on d { coordinate(1) = 0; } }",
                "coordinate axis 1 is outside Domain dimension 1",
            ),
        ];
        for (name, source, message) in cases {
            let diagnostics = validate_model(source, "M").expect_err(name);
            assert!(
                diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.message().contains(message)),
                "{name}: {diagnostics:?}"
            );
        }
    }

    #[test]
    fn support_shape_and_boundary_falsifiers_fail_closed() {
        let cases = [
            (
                "mixed domains",
                "model M() { domain a = box(0,1); domain b = box(0,1); variable x: 1 on a; initial { x = 0; } variable y: 1 on b; initial { y = 0; } relation r on a { x + y = 0; } }",
                "incompatible supports",
            ),
            (
                "relation support",
                "model M() { domain d = box(0,1); relation r on d { 1 = 0; } }",
                "residual support",
            ),
            (
                "trace on volume",
                "model M() { domain d = box(0,1); variable u: 1 on d; initial { u = 0; } relation r on d { trace(u) = 0; } }",
                "boundary Domain",
            ),
            (
                "normal scalar",
                "model M() { domain d = box(0,1); domain w = boundary(d, axis = 0, side = lower); variable u: 1 on d; initial { u = 0; } relation r on w { normal(u) = 0; } }",
                "normal component requires a spatial tensor",
            ),
            (
                "nested boundary",
                "model M() { domain d = box(0,1); domain w = boundary(d, axis = 0, side = lower); domain nested = boundary(w, axis = 0, side = lower); relation r { 0 = 0; } }",
                "Cartesian boundary parent must be a Cartesian box Domain",
            ),
        ];
        for (name, source, message) in cases {
            let diagnostics = validate_model(source, "M").expect_err(name);
            assert!(
                diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.message().contains(message)),
                "{name}: {diagnostics:?}"
            );
            assert!(
                diagnostics
                    .iter()
                    .all(|diagnostic| diagnostic.graph_path().is_none())
            );
        }
    }

    #[test]
    fn scalar_connection_contract_falsifiers_fail_without_occurrences() {
        let cases = [
            (
                "signal dimensions",
                "model M() { port out: signal output m; port sink: signal input s; connect out -> sink; }",
                "dimension-matched inputs",
            ),
            (
                "signal source",
                "model M() { port out: signal output 1; port sink: signal input 1; connect sink -> out; }",
                "source before `->`",
            ),
            (
                "conserving families",
                "model M() { domain d = scalar_physical(across potential: 1, through flow: 1); port causal: signal input 1; port physical: d; connect causal, physical; }",
                "cannot mix",
            ),
        ];
        for (name, source, message) in cases {
            let diagnostics = validate_model(source, "M").expect_err(name);
            assert!(
                diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.message().contains(message)),
                "{name}: {diagnostics:?}"
            );
            assert!(
                diagnostics
                    .iter()
                    .all(|diagnostic| diagnostic.source_span().is_some())
            );
        }
    }

    #[test]
    fn root_exact_boundary_selection_is_deferred_to_occurrence_expansion() {
        let source = r#"
public connector BoundaryScalar {
  trace value: 1;
  flux flux: 1;
  shape [];
  frame invariant;
  pairing euclidean_boundary_duality;
  orientation parent_outward;
}
component BoundaryLaw(support body: volume(ambient_dimension = 1), support exterior: complete_exterior(parent = body), port boundary[side in exterior]: BoundaryScalar over side) {


  
}
component BoundaryTerminal(support body: volume(ambient_dimension = 1), support face: boundary(parent = body), port boundary: BoundaryScalar over face) {


  
}
model M() {
  domain body = box(0, 1);
  domain lower = boundary(body, axis = 0, side = lower);
  domain upper = boundary(body, axis = 0, side = upper);
  instance law: BoundaryLaw(
    body = body,
    exterior = boundaries(lower, upper)
  );
  instance environment: BoundaryTerminal(
    body = body,
    face = lower
  );
  connect law.boundary[side = lower], environment.boundary;
}
"#;
        validate_model(source, "M")
            .expect("root exact selector belongs to occurrence expansion, not family scope");
    }
}
