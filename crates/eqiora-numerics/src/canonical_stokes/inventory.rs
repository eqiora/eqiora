//! Shared graph inventories; constitutive and family admission remain at the callers.
use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::{Diagnostic, DimExponents, RawId, ValueFrame, ValueShape};
use eqiora_graph::EdgeKind;
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, KernelNode, SymbolRef};
use eqiora_sem::KernelProgram;

use super::expression::is_divergence_of_field;
use super::support::{
    continuum_representation, has_edge, lowering_error, model_lowering_error,
    require_continuous_relation, typed_relation, unique_root,
};

pub(super) const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
pub(super) const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
pub(super) const VELOCITY_POTENTIAL: DimExponents =
    DimExponents::from_integers([0, 2, -1, 0, 0, 0, 0]).expect("bounded dimension");

pub(super) struct FlowFields {
    pub(super) count: usize,
    pub(super) velocities: Vec<RawId>,
    pub(super) pressures: Vec<RawId>,
    pub(super) coefficients: BTreeSet<RawId>,
    representations: BTreeSet<RawId>,
}

impl FlowFields {
    pub(super) fn collect(
        program: &KernelProgram,
        domain: RawId,
        dimension: u32,
        coefficient_dimensions: &[DimExponents],
    ) -> Self {
        let vector = ValueShape::new([dimension]).expect("admitted flow vector shape");
        let mut inventory = Self {
            count: 0,
            velocities: Vec::new(),
            pressures: Vec::new(),
            coefficients: BTreeSet::new(),
            representations: BTreeSet::new(),
        };
        for node in program.nodes() {
            let KernelNode::Field(field) = node else {
                continue;
            };
            let id = field.id().erase();
            if !has_edge(program, id, domain, EdgeKind::DefinedOn) {
                continue;
            }
            let Some(representation) = continuum_representation(program, id) else {
                continue;
            };
            inventory.count += 1;
            inventory.representations.insert(representation);
            if field.shape() == &vector
                && field.frame() == ValueFrame::SpatialCartesian
                && field.dimension() == VELOCITY
            {
                inventory.velocities.push(id);
            }
            if field.shape().is_scalar() && field.frame() == ValueFrame::Invariant {
                if field.dimension() == PRESSURE {
                    inventory.pressures.push(id);
                }
                if coefficient_dimensions.contains(&field.dimension()) {
                    inventory.coefficients.insert(id);
                }
            }
        }
        inventory
    }

    /// Call only after the family has accounted for every classified Field.
    pub(super) fn representation(&self, domain: RawId, message: &str) -> Result<RawId, Diagnostic> {
        if self.representations.len() != 1 {
            return Err(lowering_error(domain, message));
        }
        Ok(*self
            .representations
            .first()
            .expect("nonempty representation inventory"))
    }
}

pub(super) fn typed_volume_relations(
    program: &KernelProgram,
    relations: &[RawId],
) -> Result<BTreeMap<RawId, TypedResidual<RawId>>, Diagnostic> {
    for relation in relations {
        require_continuous_relation(program, *relation)?;
    }
    relations
        .iter()
        .map(|relation| Ok((*relation, typed_relation(program, *relation)?)))
        .collect()
}

pub(super) fn incompressibility_relations(
    relations: &[RawId],
    typed: &BTreeMap<RawId, TypedResidual<RawId>>,
    velocity: RawId,
) -> Vec<RawId> {
    relations
        .iter()
        .copied()
        .filter(|relation| {
            typed
                .get(relation)
                .and_then(|value| {
                    unique_root(value.expression(), *relation)
                        .ok()
                        .map(|root| (value.expression(), root))
                })
                .is_some_and(|(expression, root)| {
                    is_divergence_of_field(expression, root, velocity)
                })
        })
        .collect()
}

pub(super) type VolumeAssignment<M> = (RawId, Option<(RawId, RawId, ExprId)>, (RawId, M));

/// Shared role search; each momentum recognizer proves its own equations.
pub(super) fn volume_assignments<M: Copy>(
    relations: &[RawId],
    typed: &BTreeMap<RawId, TypedResidual<RawId>>,
    scalars: &[RawId],
    incompressibility: RawId,
    definition: impl Fn(&ExprDag, ExprId, RawId, RawId) -> Result<Option<ExprId>, Diagnostic>,
    momentum: impl Fn(
        &TypedResidual<RawId>,
        ExprId,
        RawId,
        Option<RawId>,
        RawId,
    ) -> Result<Option<M>, Diagnostic>,
) -> Result<Vec<VolumeAssignment<M>>, Diagnostic> {
    let mut candidates = Vec::new();
    for &pressure in scalars {
        for force in scalars
            .iter()
            .copied()
            .filter(|field| *field != pressure)
            .map(Some)
            .chain([None])
        {
            let definitions = if let Some(field) = force {
                let mut definitions = Vec::new();
                for &relation in relations {
                    let expression = typed[&relation].expression();
                    let root = unique_root(expression, relation)?;
                    if let Some(node) = definition(expression, root, field, relation)? {
                        definitions.push(Some((field, relation, node)));
                    }
                }
                definitions
            } else {
                vec![None]
            };
            if definitions.len() != 1 {
                continue;
            }
            let mut momenta = Vec::new();
            for &relation in relations {
                if relation == incompressibility
                    || definitions[0].is_some_and(|(_, definition, _)| relation == definition)
                {
                    continue;
                }
                let residual = &typed[&relation];
                let root = unique_root(residual.expression(), relation)?;
                if let Some(parts) = momentum(residual, root, pressure, force, relation)? {
                    momenta.push((relation, parts));
                }
            }
            if momenta.len() == 1 {
                candidates.push((pressure, definitions[0], momenta[0]));
            }
        }
    }
    Ok(candidates)
}

pub(super) struct FlowClosure<'a> {
    pub(super) domains: &'a BTreeSet<RawId>,
    pub(super) fields: &'a BTreeSet<RawId>,
    pub(super) relations: &'a BTreeSet<RawId>,
    pub(super) representation: RawId,
    pub(super) ports: &'a BTreeSet<RawId>,
    pub(super) connections: &'a BTreeSet<RawId>,
    pub(super) additional_parameters: &'a BTreeSet<RawId>,
}

impl FlowClosure<'_> {
    pub(super) fn validate(
        &self,
        program: &KernelProgram,
        context: &str,
    ) -> Result<(), Diagnostic> {
        let activations = program
            .edges()
            .iter()
            .filter(|edge| {
                edge.kind() == EdgeKind::Activates && self.relations.contains(&edge.to())
            })
            .map(|edge| edge.from())
            .collect::<BTreeSet<_>>();
        let mut parameters = parameters_referenced_by(program, self.relations);
        parameters.extend(self.additional_parameters);
        for node in program.nodes() {
            let admitted = match node {
                KernelNode::Domain(value) => self.domains.contains(&value.id().erase()),
                KernelNode::Representation(value) => value.id().erase() == self.representation,
                KernelNode::Field(value) => self.fields.contains(&value.id().erase()),
                KernelNode::Parameter(value) => parameters.contains(&value.id().erase()),
                KernelNode::Relation(value) => self.relations.contains(&value.id().erase()),
                KernelNode::Activation(value) => activations.contains(&value.id().erase()),
                KernelNode::Port(value) => self.ports.contains(&value.id().erase()),
                KernelNode::Connection(value) => self.connections.contains(&value.id().erase()),
                _ => false,
            };
            if !admitted {
                return Err(model_lowering_error(
                    program,
                    format!(
                        "{context} would ignore unexpected {:?} node {}",
                        node.kind(),
                        node.id()
                    ),
                ));
            }
        }
        Ok(())
    }
}

fn parameters_referenced_by(
    program: &KernelProgram,
    relations: &BTreeSet<RawId>,
) -> BTreeSet<RawId> {
    relations
        .iter()
        .copied()
        .flat_map(|relation| match program.node(relation) {
            Some(KernelNode::Relation(definition)) => definition.expression().nodes().iter(),
            _ => unreachable!("admitted Relations were already inspected"),
        })
        .filter_map(|node| match node {
            ExprNode::Symbol(SymbolRef::Parameter(parameter)) => Some(parameter.erase()),
            _ => None,
        })
        .collect()
}
