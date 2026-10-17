//! Checked spatial equations with shared region assembly and exact boundary coverage.

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
mod error_estimate;
mod lowering;
pub(crate) mod motion;
pub(crate) mod observation;
mod support;
mod temporal;
pub(super) use temporal::require_closed_law;

use crate::spatial_expression::Coefficient;
use data::{Components, Context, Data};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompiledSpatialForm<S: Coefficient> {
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

impl<S: Coefficient> CompiledSpatialForm<S> {
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
                "spatial coefficient evaluation requires finite physical Time",
            ));
        }
        let Some(KernelNode::Domain(definition)) = program.node(domain) else {
            return Err(invalid("spatial form support is not a Domain"));
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
                "spatial form requires a dimension-matched 1D–3D Cartesian Domain",
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
            return Err(invalid("spatial form requires at least one unknown"));
        }
        let fields = residuals
            .keys()
            .map(|field| (*field, roles.fields[field].1.clone()))
            .collect::<Vec<_>>();
        let scalar_profile = kinematics.is_empty()
            && fields
                .iter()
                .all(|(_, value_type)| value_type.shape().is_scalar());
        if scalar_profile {
            for (_, value_type) in &fields {
                require_scalar::<S>(value_type)?;
            }
        }
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
                    "spatial row requires its unique principal diffusion and exact unknown trial Fields",
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
            storage = volume.stored_fields()?;
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
        if volume.is_nonlinear() && S::DOMAIN != eqiora_core::ScalarDomain::Real {
            return Err(invalid("nonlinear spatial execution requires real Fields"));
        }
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
    pub(crate) fn formulation_rules(&self) -> BTreeSet<&'static str> {
        let mut rules = self.volume.formulation_rules();
        if self.boundary_laws.values().any(|laws| !laws.is_empty()) {
            rules.insert(super::vocabulary::MixedFormulationRule::ExplicitBoundaryLaw.id());
        }
        rules
    }

    pub(crate) fn constraint_pairs(&self) -> impl Iterator<Item = (RawId, RawId)> + '_ {
        self.volume.constraint_pairs()
    }

    /// The paired SI volume form is symmetric; strong traces remove matching
    /// rows/columns and prescribed natural fluxes contribute only to the load.
    /// Backward-Euler state elimination is included; moving geometry is not.
    pub(crate) fn is_nonlinear(&self) -> bool {
        self.volume.is_nonlinear()
    }

    pub(crate) fn has_symmetric_operator(&self) -> bool {
        use crate::canonical_boundary::PhysicalBoundaryQuantity;
        self.motion.is_none()
            && !self.is_nonlinear()
            && self.volume.volume_test_orientations().is_some_and(|signs| {
                self.constraint_pairs().next().is_some() || signs.values().all(|sign| *sign == 1.0)
            })
            && self
                .boundary_laws
                .values()
                .flat_map(|laws| laws.values())
                .all(|law| match law.quantity {
                    PhysicalBoundaryQuantity::Trace => law.trace_field == Some(law.tested),
                    PhysicalBoundaryQuantity::Flux => law.trace_field.is_none(),
                })
    }

    pub(crate) fn interface_test_orientation(
        &self,
        field: RawId,
        scaling: &eqiora_realization::SystemScaling,
    ) -> Option<f64> {
        match scaling {
            eqiora_realization::SystemScaling::Dimensional => self
                .constraint_pairs()
                .next()
                .and_then(|_| self.volume.volume_test_orientations())
                .map_or(Some(1.0), |signs| signs.get(&field).copied()),
            eqiora_realization::SystemScaling::SymmetricCongruence(_) => {
                self.volume.volume_test_orientations()?.get(&field).copied()
            }
        }
    }

    pub(crate) fn has_zero_multiplier_block(&self) -> bool {
        self.volume.has_zero_multiplier_block()
    }

    pub(crate) fn requires_boundary(&self, field: RawId) -> bool {
        self.volume.requires_boundary(field)
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

    /// Use the already compiled physical row without rebinding it as a nodal form.
    pub(crate) fn conservative_row(
        &self,
    ) -> Result<super::region::ConservativeRow<'_, S>, Diagnostic> {
        if !self.storage.is_empty() || self.motion.is_some() || !self.kinematics.is_empty() {
            return Err(invalid(
                "steady conservative assembly cannot discard storage or motion",
            ));
        }
        let row = self.volume.conservative_row()?;
        if self.fields.len() != 1 || self.fields[0].0 != row.field() {
            return Err(invalid(
                "scalar conservative row differs from its exact Field inventory",
            ));
        }
        Ok(row)
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
) -> Result<BTreeMap<RawId, Components<S>>, Diagnostic> {
    coefficients_at_time(program, dimension, roles, None)
}

pub(super) fn coefficients_at_time<S: crate::spatial_expression::Coefficient>(
    program: &KernelProgram,
    dimension: usize,
    roles: &EquationRoles,
    time_s: Option<f64>,
) -> Result<BTreeMap<RawId, Components<S>>, Diagnostic> {
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
            let rhs = match dag.node(root) {
                _ if target(root) => None,
                Some(ExprNode::Sub(a, b)) if target(*a) => Some(*b),
                Some(ExprNode::Sub(a, b)) if target(*b) => Some(*a),
                _ => return Err(invalid("coefficient definition lost solved form")),
            };
            let context = Context {
                time_s,
                program,
                dag,
                owner: relation,
                dimension,
                coefficients: &known,
            };
            match Components::prescribed(&context, &roles.fields[&field].1, rhs) {
                Ok(value) => {
                    known.insert(field, value);
                }
                Err(_) => next.push((relation, field)),
            }
        }
        if next.len() == count {
            return Err(invalid(
                "unsupported, cyclic or unresolved coefficient definitions",
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
            "spatial form execution requires invariant scalar values matching its coefficient representation",
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
