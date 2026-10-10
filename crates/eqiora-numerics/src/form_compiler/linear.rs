//! Checked linear equations with shared region assembly and exact boundary coverage.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::{Diagnostic, DynQuantity, RawId, ScalarDomain, ValueFrame, ValueType};
use eqiora_schema::kernel::{DomainKind, ExprNode, KernelNode, RelationMeaning, SymbolRef};
use eqiora_sem::KernelProgram;

use super::equation_roles::{EquationRoles, Role};
use super::region::{BoundRegionForm, CompiledRegionForm, PrescribedDatum, ScalarRow};
use super::scalar::{continuous_activations, require_closed_dag, typed_relation};

mod binding;
mod boundary;
pub(crate) use boundary::InterfaceBoundary;
pub(super) mod data;
mod lowering;
pub(crate) mod motion;
pub(crate) mod observation;
mod support;
mod temporal;
pub(super) use temporal::require_closed_law;

use crate::spatial_expression::Coefficient;
use data::{Context, Data};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompiledLinearBlockForm<S: Coefficient> {
    domain: RawId,
    dimension: usize,
    fields: Vec<(RawId, ValueType)>,
    relations: Vec<RawId>,
    residual_types: Vec<ValueType>,
    dependencies: BTreeMap<RawId, BTreeSet<RawId>>,
    boundary_laws: BTreeMap<RawId, BTreeMap<RawId, super::region::RegionBoundaryLaw<S>>>,
    volume: CompiledRegionForm<S>,
    step: Option<DynQuantity>,
    initial: BTreeMap<RawId, PrescribedDatum<S>>,
    storage: BTreeMap<RawId, Data<S>>,
    motion: Option<motion::StorageMotion>,
    kinematics: Vec<(eqiora_realization::BackwardEulerStatePair, ValueType)>,
}

impl<S: Coefficient> CompiledLinearBlockForm<S> {
    pub(crate) fn derive(
        program: &KernelProgram,
        domain: RawId,
        dimension: usize,
        interface_boundaries: &BTreeSet<InterfaceBoundary>,
    ) -> Result<Self, Diagnostic> {
        Self::derive_at_time(program, domain, dimension, interface_boundaries, None)
    }

    /// Bind prescribed coefficient and boundary data at one explicit physical Time.
    /// Geometry and consecutive storage history remain the caller's responsibility.
    pub(crate) fn derive_at_time(
        program: &KernelProgram,
        domain: RawId,
        dimension: usize,
        interface_boundaries: &BTreeSet<InterfaceBoundary>,
        time_s: Option<f64>,
    ) -> Result<Self, Diagnostic> {
        Self::derive_temporal(
            program,
            domain,
            dimension,
            interface_boundaries,
            time_s,
            None,
        )
    }

    pub(crate) fn derive_over_step(
        program: &KernelProgram,
        domain: RawId,
        previous_time_s: f64,
        current_time_s: f64,
        step_s: f64,
    ) -> Result<Self, Diagnostic> {
        Self::derive_temporal(
            program,
            domain,
            2,
            &BTreeSet::new(),
            Some(current_time_s),
            Some((previous_time_s, step_s)),
        )
    }

    fn derive_temporal(
        program: &KernelProgram,
        domain: RawId,
        dimension: usize,
        interface_boundaries: &BTreeSet<InterfaceBoundary>,
        time_s: Option<f64>,
        previous: Option<(f64, f64)>,
    ) -> Result<Self, Diagnostic> {
        if time_s.is_some_and(|time| !time.is_finite()) {
            return Err(invalid(
                "linear coefficient evaluation requires finite physical Time",
            ));
        }
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
        let motion = motion::StorageMotion::select(program, domain)?;
        let chart = motion
            .as_ref()
            .map(|motion| motion.bind(program, time_s.unwrap_or(0.0)))
            .transpose()?;
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
            support::require_equation_support(&typed, domain, dimension)?;
        }
        let kinematics = roles
            .relations
            .iter()
            .filter_map(|(relation, role)| match role.kind {
                Role::Kinematic { state, rate } => Some(
                    eqiora_realization::BackwardEulerStatePair::new(
                        relation.downcast().expect("Relation"),
                        state.downcast().expect("Field"),
                        rate.downcast().expect("Field"),
                    )
                    .map(|pair| (pair, roles.fields[&state].1.clone())),
                ),
                _ => None,
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let scalar_profile = kinematics.is_empty()
            && roles
                .fields
                .values()
                .all(|(_, value_type)| value_type.shape().is_scalar());
        if scalar_profile {
            for (_, value_type) in roles.fields.values() {
                require_scalar::<S>(value_type)?;
            }
        }
        let mut residuals = BTreeMap::new();
        for (relation, role) in &roles.relations {
            match role.kind {
                Role::Residual { tested } => {
                    residuals.insert(tested, *relation);
                }
                Role::Coefficient { .. } => {}
                Role::Kinematic { .. } => {}
            }
        }
        if residuals.is_empty() {
            return Err(invalid("linear block requires at least one unknown"));
        }
        let fields = residuals
            .keys()
            .map(|field| (*field, roles.fields[field].1.clone()))
            .collect::<Vec<_>>();
        let coefficients = coefficients_at_time(program, dimension, &roles, time_s)?;
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
            residual_types.push(value_type.clone());
            if !scalar_profile {
                continue;
            }
            require_scalar::<S>(&value_type)?;
            let context = Context {
                time_s,
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
                row = row.scale(Data::constant(dimension, <S as From<f64>>::from(-1.0)))?;
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
            if row.transport.keys().any(|(trial, _)| *trial != *field)
                || (!row.transport.is_empty() && !interface_boundaries.is_empty())
            {
                return Err(invalid(
                    "scalar transport requires its exact local trial without interface quotients",
                ));
            }
            if let (Some(motion), Some(chart)) = (&motion, &chart) {
                motion.bind_transport(&context, *field, &mut row, chart, previous)?;
                row = row.on_uniform_chart(chart)?;
            }
            if !row.storage.is_empty() {
                if row.storage.len() != 1 || !row.storage.contains_key(field) {
                    return Err(invalid(
                        "scalar Backward Euler requires each row to store its exact Field",
                    ));
                }
                storage.insert(*field, row.storage[field].clone());
            }
            rows.push(row);
        }
        let (volume, initial) = if scalar_profile {
            let volume_rows = residuals
                .iter()
                .zip(rows)
                .zip(&residual_types)
                .map(
                    |(((field, relation), mut row), residual_type)| ScalarRow::<S> {
                        relation: *relation,
                        field: *field,
                        residual_type: residual_type.clone(),
                        diffusion: row
                            .diffusion
                            .remove(field)
                            .expect("admitted principal diffusion"),
                        reaction: row.reaction,
                        storage: row.storage,
                        transport: row.transport,
                        forcing: row
                            .constant
                            .multiply(Data::constant(dimension, <S as From<f64>>::from(-1.0))),
                    },
                )
                .collect();
            let initial_coefficients =
                coefficients_at_time(program, dimension, &roles, time_s.map(|_| 0.0))?;
            let initial = temporal::initial_values(
                program,
                domain,
                dimension,
                &storage,
                &initial_coefficients,
                &fields.iter().map(|(field, _)| *field).collect(),
                time_s.map(|_| 0.0),
            )?;
            let volume =
                CompiledRegionForm::<S>::scalar(domain, dimension, roles.clone(), volume_rows)?;
            (volume, initial)
        } else {
            let volume = CompiledRegionForm::<S>::derive(program, domain, dimension)?;
            if motion.is_some() {
                return Err(invalid(
                    "moving vector storage requires explicit geometric history",
                ));
            }
            storage = volume.linear_storage()?;
            let initial_coefficients =
                coefficients_at_time(program, dimension, &roles, time_s.map(|_| 0.0))?;
            let initial = temporal::initial_values(
                program,
                domain,
                dimension,
                &storage,
                &initial_coefficients,
                &fields
                    .iter()
                    .map(|(field, _)| *field)
                    .chain(kinematics.iter().map(|(pair, _)| pair.state().erase()))
                    .collect(),
                time_s.map(|_| 0.0),
            )?;
            (volume, initial)
        };
        if motion.is_some() && storage.len() != fields.len() {
            return Err(invalid(
                "moving algebraic constraints require explicit geometric State acceptance",
            ));
        }
        let mut boundary = boundary::derive(
            program,
            domain,
            dimension,
            &fields,
            &volume,
            interface_boundaries,
            time_s.or_else(|| (!kinematics.is_empty()).then_some(0.0)),
        )?;
        if let Some(chart) = &chart {
            for law in boundary
                .fields
                .values_mut()
                .flat_map(|laws| laws.values_mut())
            {
                law.on_uniform_chart(chart)?;
            }
        }
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
            motion,
            kinematics,
        })
    }

    pub(crate) fn algebraic_fields(&self) -> impl Iterator<Item = RawId> + '_ {
        self.fields
            .iter()
            .map(|(field, _)| *field)
            .filter(|field| !self.storage.contains_key(field))
    }

    pub(crate) fn kinematics(&self) -> &[(eqiora_realization::BackwardEulerStatePair, ValueType)] {
        &self.kinematics
    }

    pub(crate) fn represented_fields(&self) -> Vec<(RawId, ValueType)> {
        let mut fields = self.fields.clone();
        fields.extend(
            self.kinematics
                .iter()
                .map(|(pair, ty)| (pair.state().erase(), ty.clone())),
        );
        fields.sort_by_key(|(field, _)| *field);
        fields
    }

    pub(crate) fn motion(&self) -> Option<&motion::StorageMotion> {
        self.motion.as_ref()
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
    ) -> &BTreeMap<RawId, BTreeMap<RawId, super::region::RegionBoundaryLaw<S>>> {
        &self.boundary_laws
    }

    /// Cartesian Q1 binding retains exact Field shapes and coherent-SI normalization.
    /// Interface closure uses the same exact constitutive witness as exterior laws.
    pub(crate) fn require_interface_flux(
        &self,
        program: &KernelProgram,
        support: RawId,
        relation: RawId,
        field: RawId,
        normal: eqiora_schema::kernel::ExprId,
    ) -> Result<(), Diagnostic> {
        self.volume
            .require_boundary_flux(program, support, relation, field, normal, false)
    }

    pub(crate) fn volume(&self) -> Result<BoundRegionForm<S>, Diagnostic> {
        self.bind_space(
            eqiora_meshing::ReferenceCell::hypercube(self.dimension)?,
            eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
        )
    }
}

pub(super) fn coefficients<S: crate::spatial_expression::Coefficient>(
    program: &KernelProgram,
    dimension: usize,
    roles: &EquationRoles,
) -> Result<BTreeMap<RawId, Data<S>>, Diagnostic> {
    coefficients_at_time(program, dimension, roles, None)
}

pub(super) fn coefficients_at_time<S: crate::spatial_expression::Coefficient>(
    program: &KernelProgram,
    dimension: usize,
    roles: &EquationRoles,
    time_s: Option<f64>,
) -> Result<BTreeMap<RawId, Data<S>>, Diagnostic> {
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
            let target = |node| matches!(dag.node(node),Some(ExprNode::Symbol(SymbolRef::Field(id))) if id.erase() == field);
            if target(root) {
                known.insert(
                    field,
                    Data::constant(dimension, <S as From<f64>>::from(0.0)),
                );
                continue;
            }
            let Some(ExprNode::Sub(a, b)) = dag.node(root) else {
                return Err(invalid("coefficient definition lost solved form"));
            };
            let rhs = if target(*a) {
                *b
            } else if target(*b) {
                *a
            } else {
                return Err(invalid("coefficient definition lost its Field"));
            };
            let context = Context {
                time_s,
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

fn require_scalar<S: Coefficient>(value_type: &ValueType) -> Result<(), Diagnostic> {
    if !matches!(
        value_type.scalar_domain(),
        ScalarDomain::Real | ScalarDomain::Complex
    ) || (value_type.scalar_domain() == ScalarDomain::Complex
        && S::DOMAIN != ScalarDomain::Complex)
        || !value_type.shape().is_scalar()
        || value_type.frame() != ValueFrame::Invariant
        || value_type.array_rank() != 0
    {
        return Err(invalid(
            "linear block execution requires invariant scalar values matching its coefficient representation",
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
