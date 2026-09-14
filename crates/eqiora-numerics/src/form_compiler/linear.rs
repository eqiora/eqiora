//! Checked linear scalar equations with one shared Q1 element evaluator.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::{Diagnostic, DynQuantity, RawId, ScalarDomain, ValueFrame, ValueType};
use eqiora_schema::kernel::{DomainKind, ExprNode, KernelNode, RelationMeaning, SymbolRef};
use eqiora_sem::KernelProgram;

use super::equation_roles::{EquationRoles, Role};
use super::region::{BoundRegionForm, CompiledRegionForm, ScalarRow};
use super::scalar::{continuous_activations, require_closed_dag, typed_relation};

mod binding;
mod boundary;
pub(super) mod data;
mod lowering;
mod temporal;

use data::{Context, Data};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompiledLinearBlockForm {
    domain: RawId,
    dimension: usize,
    fields: Vec<(RawId, ValueType)>,
    relations: Vec<RawId>,
    residual_types: Vec<ValueType>,
    dependencies: BTreeMap<RawId, BTreeSet<RawId>>,
    boundary_laws: BTreeMap<RawId, BTreeMap<RawId, super::region::RegionBoundaryLaw>>,
    volume: CompiledRegionForm,
    step: Option<DynQuantity>,
    initial: BTreeMap<RawId, Data>,
    storage: BTreeMap<RawId, Data>,
}

impl CompiledLinearBlockForm {
    pub(crate) fn derive(
        program: &KernelProgram,
        domain: RawId,
        dimension: usize,
        interface_boundaries: &BTreeSet<RawId>,
    ) -> Result<Self, Diagnostic> {
        let Some(KernelNode::Domain(definition)) = program.node(domain) else {
            return Err(invalid("linear block support is not a Domain"));
        };
        if !(1..=3).contains(&dimension)
            || !matches!(
                definition.kind(),
                DomainKind::CartesianBox { .. } | DomainKind::GeometryRegion { .. }
            )
            || matches!(definition.kind(), DomainKind::CartesianBox { .. })
                && program
                    .resolved_cartesian_bounds(definition.id())
                    .map_err(|_| invalid("unresolved Cartesian bounds"))?
                    .len()
                    != dimension
        {
            return Err(invalid(
                "linear block requires a dimension-matched 1D–3D Cartesian Domain",
            ));
        }
        let roles = EquationRoles::derive(program, [domain])?;
        for relation in roles.relations.keys() {
            let typed = typed_relation(program, *relation)?;
            match program.node(*relation) {
                Some(KernelNode::Relation(definition))
                    if matches!(definition.meaning(), RelationMeaning::Conservation(_)) =>
                {
                    let RelationMeaning::Conservation(law) = definition.meaning() else {
                        unreachable!()
                    };
                    temporal::require_closed_law(typed.expression(), *law)?;
                }
                _ => require_closed_dag(typed.expression(), *relation)?,
            }
            for node_type in typed.node_types() {
                if let Some(support) = &node_type.support
                    && (*support.domain() != domain || support.dimensions() != dimension)
                {
                    return Err(invalid(
                        "linear equation support or coordinate dimension differs from its Domain",
                    ));
                }
            }
        }
        for (_, value_type) in roles.fields.values() {
            require_scalar(value_type)?;
        }
        let mut residuals = BTreeMap::new();
        for (relation, role) in &roles.relations {
            match role.kind {
                Role::Residual { tested } => {
                    residuals.insert(tested, *relation);
                }
                Role::Coefficient { .. } => {}
                Role::Kinematic { .. } => {
                    return Err(invalid(
                        "stationary linear block cannot eliminate dynamic state",
                    ));
                }
            }
        }
        if residuals.is_empty() {
            return Err(invalid("linear block requires at least one unknown"));
        }
        let fields = residuals
            .keys()
            .map(|field| (*field, roles.fields[field].1.clone()))
            .collect::<Vec<_>>();
        let coefficients = coefficients(program, dimension, &roles)?;
        let mut rows = Vec::new();
        let mut storage = BTreeMap::new();
        let mut residual_types = Vec::new();
        for (field, relation) in &residuals {
            let typed = typed_relation(program, *relation)?;
            let root = typed.expression().roots()[0];
            let value_type = typed
                .node_type(root)
                .expect("typed root")
                .value_type
                .clone();
            require_scalar(&value_type)?;
            let context = Context {
                program,
                dag: typed.expression(),
                owner: *relation,
                dimension,
                coefficients: &coefficients,
            };
            let mut row = match program.node(*relation) {
                Some(KernelNode::Relation(definition)) => match definition.meaning() {
                    RelationMeaning::Conservation(terms) => context.conservation(*terms)?,
                    _ => context.terms(root, 0)?,
                },
                _ => context.terms(root, 0)?,
            };
            // Conservation fixes the physical outward flux orientation. Equations
            // may reverse their entire row, but a Law must retain its sign.
            let physical_balance = matches!(program.node(*relation),
                Some(KernelNode::Relation(definition))
                    if matches!(definition.meaning(), RelationMeaning::Conservation(_)));
            if !physical_balance && context.diffusion_orientation(root, 0)? == Some(1) {
                row = row.scale(Data::constant(dimension, -1.0))?;
            }
            if row.diffusion.len() != 1
                || !row.diffusion.contains_key(field)
                || row
                    .reaction
                    .keys()
                    .any(|trial| !residuals.contains_key(trial))
            {
                return Err(invalid(
                    "linear row requires its unique principal diffusion and exact unknown trial Fields",
                ));
            }
            if !row.storage.is_empty() {
                if row.storage.len() != 1
                    || !row.storage.contains_key(field)
                    || residuals.len() != 1
                {
                    return Err(invalid(
                        "first scalar Backward Euler requires storage of its single exact Field",
                    ));
                }
                storage.insert(*field, row.storage[field].clone());
            }
            rows.push(row);
            residual_types.push(value_type);
        }
        let volume_rows = residuals
            .iter()
            .zip(rows)
            .zip(&residual_types)
            .map(|(((field, relation), mut row), residual_type)| ScalarRow {
                relation: *relation,
                field: *field,
                residual_type: residual_type.clone(),
                diffusion: row
                    .diffusion
                    .remove(field)
                    .expect("admitted principal diffusion"),
                reaction: row.reaction,
                storage: row.storage,
                forcing: row.constant.multiply(Data::constant(dimension, -1.0)),
            })
            .collect();
        let initial = temporal::initial_values(
            program,
            domain,
            dimension,
            &storage,
            &coefficients,
            !storage.is_empty(),
        )?;
        let volume = CompiledRegionForm::scalar(domain, dimension, roles.clone(), volume_rows)?;
        let boundary = boundary::derive(
            program,
            domain,
            dimension,
            &fields,
            &volume,
            interface_boundaries,
        )?;
        let all_relations = roles
            .relations
            .keys()
            .copied()
            .chain(boundary.dependencies.keys().copied())
            .collect();
        continuous_activations(program, &all_relations)?;
        let dependencies = roles
            .relations
            .iter()
            .map(|(id, role)| (*id, role.dependencies.clone()))
            .chain(boundary.dependencies)
            .collect();
        Ok(Self {
            domain,
            dimension,
            fields,
            relations: residuals.values().copied().collect(),
            residual_types,
            dependencies,
            boundary_laws: boundary.fields,
            volume,
            step: None,
            initial,
            storage,
        })
    }

    pub(crate) const fn domain(&self) -> RawId {
        self.domain
    }
    pub(crate) const fn dimension(&self) -> usize {
        self.dimension
    }
    /// Stable exact Field identity order; local basis DOFs are contiguous per Field.
    pub(crate) fn fields(&self) -> &[(RawId, ValueType)] {
        &self.fields
    }
    pub(crate) fn boundary_laws(
        &self,
    ) -> &BTreeMap<RawId, BTreeMap<RawId, super::region::RegionBoundaryLaw>> {
        &self.boundary_laws
    }

    pub(crate) fn volume(&self) -> Result<BoundRegionForm, Diagnostic> {
        let time = self.step.map(|step| super::region::RegionTimeBinding {
            step,
            states: Vec::new(),
        });
        self.volume.bind_scalar(time.as_ref())
    }
}

pub(super) fn coefficients(
    program: &KernelProgram,
    dimension: usize,
    roles: &EquationRoles,
) -> Result<BTreeMap<RawId, Data>, Diagnostic> {
    let mut known = BTreeMap::new();
    let mut pending = roles
        .relations
        .iter()
        .filter_map(|(relation, role)| match role.kind {
            Role::Coefficient { field } => Some((*relation, field)),
            _ => None,
        })
        .collect::<Vec<_>>();
    while !pending.is_empty() {
        let count = pending.len();
        let mut next = Vec::new();
        for (relation, field) in pending {
            let typed = typed_relation(program, relation)?;
            let dag = typed.expression();
            let mut root = dag.roots()[0];
            while let Some(ExprNode::Neg(value)) = dag.node(root) {
                root = *value;
            }
            let Some(ExprNode::Sub(a, b)) = dag.node(root) else {
                return Err(invalid("coefficient definition lost solved form"));
            };
            let target = |node| matches!(dag.node(node),Some(ExprNode::Symbol(SymbolRef::Field(id))) if id.erase() == field);
            let rhs = if target(*a) {
                *b
            } else if target(*b) {
                *a
            } else {
                return Err(invalid("coefficient definition lost its Field"));
            };
            let context = Context {
                program,
                dag,
                owner: relation,
                dimension,
                coefficients: &known,
            };
            match context.data(rhs, 0) {
                Ok(value) => {
                    known.insert(field, value);
                }
                Err(_) => next.push((relation, field)),
            }
        }
        if next.len() == count {
            return Err(invalid(
                "unsupported, cyclic or unresolved scalar coefficient definitions",
            ));
        }
        pending = next;
    }
    Ok(known)
}

fn require_scalar(value_type: &ValueType) -> Result<(), Diagnostic> {
    if value_type.scalar_domain() != ScalarDomain::Real
        || !value_type.shape().is_scalar()
        || value_type.frame() != ValueFrame::Invariant
        || value_type.array_rank() != 0
    {
        return Err(invalid(
            "linear block execution requires invariant real scalar values",
        ));
    }
    Ok(())
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(
        eqiora_core::diagnostic::codes::INVALID_DISCRETIZATION,
        message,
    )
}

#[cfg(test)]
mod tests;
