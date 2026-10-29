//! Semantic agreement before execution, not matrix authentication or a persisted artifact.
use super::*;
use sha2::{Digest, Sha256};

impl SolverSelection {
    /// Fingerprint the exact planning assumptions and selected implementation.
    /// This does not authenticate matrix coefficients, RHS values, or deployment.
    ///
    /// # Errors
    /// Rejects a count that cannot be represented by the fixed-width encoding.
    pub fn planning_agreement_fingerprint(&self) -> Result<[u8; 32], Diagnostic> {
        let mut out = Agreement(Sha256::new());
        out.string("eqiora.solver-planning.collective-agreement")?;
        out.string(POLICY_ID)?;
        out.tag(domain(self.scalar_domain));
        out.tag(storage(self.scalar_type));
        out.tag(match self.objective {
            SolverPlanningObjective::Robust => 0,
            SolverPlanningObjective::Fast => 1,
            SolverPlanningObjective::LowMemory => 2,
        });
        out.string(self.candidate_id)?;
        out.tag(algorithm(self.plan.algorithm()));
        out.tag(preconditioner(self.plan.preconditioner()));
        out.tag(reduction(self.plan.reduction()));
        out.0
            .update(self.plan.relative_tolerance().to_bits().to_be_bytes());
        out.0
            .update(self.plan.absolute_tolerance().to_bits().to_be_bytes());
        out.count(self.plan.maximum_iterations().get())?;
        let profile = &self.profile;
        out.tag(properties(profile.facts.properties));
        out.tag(match profile.facts.orientation {
            LinearOperatorOrientation::Normal => 0,
            LinearOperatorOrientation::Transposed => 1,
            LinearOperatorOrientation::ConjugateTransposed => 2,
        });
        out.tag(u8::from(profile.facts.canonical_csr));
        out.option(profile.facts.complete_diagonal, |out, value| {
            out.tag(u8::from(value));
            Ok(())
        })?;
        out.option(profile.required_reduction, |out, value| {
            out.tag(reduction(value));
            Ok(())
        })?;
        out.option(profile.structure.as_ref(), |out, structure| {
            out.count(structure.blocks().count())?;
            for block in structure.blocks() {
                let field = match block {
                    crate::AlgebraicBlock::Field(field) => {
                        out.tag(0);
                        field
                    }
                    crate::AlgebraicBlock::ConstraintMultiplier { field } => {
                        out.tag(1);
                        field
                    }
                };
                out.0.update(field.ulid().to_bytes());
            }
            out.count(structure.constraints().len())?;
            for constraint in structure.constraints() {
                match constraint {
                    crate::AlgebraicConstraint::ZeroIntegral { field } => {
                        out.tag(0);
                        out.0.update(field.ulid().to_bytes());
                    }
                }
            }
            Ok(())
        })?;
        out.option(profile.size, |out, size| {
            out.count(size.rows())?;
            out.count(size.stored_slots())?;
            out.tag(u8::from(size.is_exact()));
            Ok(())
        })?;
        out.option(profile.cost_request, |out, cost| {
            out.count(cost.solves().get())?;
            out.count(cost.structures().get())?;
            out.count(cost.numeric_preparations().get())?;
            out.optional_count(cost.identity_iterations().map(NonZeroUsize::get))?;
            out.optional_count(cost.jacobi_iterations().map(NonZeroUsize::get))?;
            out.optional_count(cost.numeric_payload_limit().map(NonZeroUsize::get))?;
            out.tag(u8::from(cost.declared_horizon()));
            Ok(())
        })?;
        out.option(profile.run_forecast, |out, run| {
            out.option(run.common_work, |out, value| {
                out.0.update(value.to_be_bytes());
                Ok(())
            })?;
            out.optional_count(run.retained_numeric_bytes)?;
            out.option(run.prepared_provider, Agreement::solver_provider)
        })?;
        out.solver_provider(self.provider)?;
        out.provider(
            profile.execution_provider.id().as_str(),
            profile.execution_provider.implementation_version(),
            profile.execution_provider.libraries(),
        )?;
        out.string(profile.execution.adapter().as_str())?;
        match profile.execution.topology() {
            crate::ExecutionTopology::Host { workers } => {
                out.tag(0);
                out.count(workers.get())?;
            }
            crate::ExecutionTopology::Distributed {
                ranks,
                workers_per_partition,
            } => {
                out.tag(1);
                out.count(ranks.get())?;
                out.count(workers_per_partition.get())?;
            }
            crate::ExecutionTopology::Cuda { device } => {
                out.tag(2);
                out.0.update(device.to_be_bytes());
            }
        }
        out.count(self.capabilities.combinations().len())?;
        for capability in self.capabilities.combinations() {
            let SolverCapability {
                scalar_domain,
                algorithm: solver,
                operator_properties,
                preconditioner: policy,
                reduction: order,
                scalar_type,
            } = capability;
            out.tag(domain(*scalar_domain));
            out.tag(storage(*scalar_type));
            out.tag(algorithm(*solver));
            out.tag(properties(*operator_properties));
            out.tag(preconditioner(*policy));
            out.tag(reduction(*order));
        }
        Ok(out.0.finalize().into())
    }
}

struct Agreement(Sha256);
impl Agreement {
    fn tag(&mut self, value: u8) {
        self.0.update([value]);
    }
    fn count(&mut self, value: usize) -> Result<(), Diagnostic> {
        self.0.update(
            u64::try_from(value)
                .map_err(|_| invalid_catalog("agreement.count-overflow"))?
                .to_be_bytes(),
        );
        Ok(())
    }
    fn string(&mut self, value: &str) -> Result<(), Diagnostic> {
        self.count(value.len())?;
        self.0.update(value.as_bytes());
        Ok(())
    }
    fn option<T>(
        &mut self,
        value: Option<T>,
        encode: impl FnOnce(&mut Self, T) -> Result<(), Diagnostic>,
    ) -> Result<(), Diagnostic> {
        match value {
            Some(value) => {
                self.tag(1);
                encode(self, value)
            }
            None => {
                self.tag(0);
                Ok(())
            }
        }
    }
    fn optional_count(&mut self, value: Option<usize>) -> Result<(), Diagnostic> {
        self.option(value, Self::count)
    }
    fn solver_provider(&mut self, provider: SolverProvider) -> Result<(), Diagnostic> {
        self.provider(
            provider.id().as_str(),
            provider.implementation_version(),
            provider.libraries(),
        )
    }
    fn provider(
        &mut self,
        id: &str,
        version: &str,
        libraries: &[ProviderLibrary],
    ) -> Result<(), Diagnostic> {
        self.string(id)?;
        self.string(version)?;
        self.count(libraries.len())?;
        for library in libraries {
            self.string(library.name())?;
            self.string(library.version())?;
        }
        Ok(())
    }
}

fn domain(value: ScalarDomain) -> u8 {
    match value {
        ScalarDomain::Boolean => 0,
        ScalarDomain::Enum => 1,
        ScalarDomain::Integer => 2,
        ScalarDomain::Real => 3,
        ScalarDomain::Complex => 4,
    }
}
fn storage(value: ScalarType) -> u8 {
    match value {
        ScalarType::F32 => 0,
        ScalarType::F64 => 1,
    }
}
fn algorithm(value: LinearSolver) -> u8 {
    match value {
        LinearSolver::ConjugateGradient => 0,
        LinearSolver::MinimumResidual => 1,
        LinearSolver::BiConjugateGradientStabilized => 2,
        LinearSolver::SparseLu => 3,
    }
}
fn preconditioner(value: PreconditionerPolicy) -> u8 {
    match value {
        PreconditionerPolicy::Identity => 0,
        PreconditionerPolicy::Jacobi => 1,
    }
}
fn reduction(value: ReductionPolicy) -> u8 {
    match value {
        ReductionPolicy::Reproducible => 0,
        ReductionPolicy::Fast => 1,
    }
}
fn properties(value: LinearOperatorProperties) -> u8 {
    match value {
        LinearOperatorProperties::General => 0,
        LinearOperatorProperties::Symmetric => 1,
        LinearOperatorProperties::ComplexSymmetric => 2,
        LinearOperatorProperties::Hermitian => 3,
        LinearOperatorProperties::HermitianPositiveDefinite => 4,
        LinearOperatorProperties::SymmetricPositiveDefinite => 5,
        LinearOperatorProperties::SymmetricIndefinite => 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selected(profile: SolverPlanningProfile) -> SolverSelection {
        SolverSelection::resolve_host::<f64>(
            profile,
            SolverPlanningObjective::Robust,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
            &crate::REFERENCE_LINEAR_SOLVER,
            None,
        )
        .unwrap()
    }
    fn fingerprint(selection: &SolverSelection) -> [u8; 32] {
        selection.planning_agreement_fingerprint().unwrap()
    }
    fn assert_changed(original: &SolverSelection, changed: &SolverSelection) {
        assert_eq!(original.candidate_id, changed.candidate_id);
        assert_eq!(original.plan.algorithm(), changed.plan.algorithm());
        assert_eq!(
            original.plan.preconditioner(),
            changed.plan.preconditioner()
        );
        assert_eq!(original.plan.reduction(), changed.plan.reduction());
        assert_ne!(fingerprint(original), fingerprint(changed));
    }

    #[test]
    pub(super) fn same_tuple_controls_objective_and_profile_facts_must_agree() {
        let original = selected(SolverPlanningProfile::general_canonical_csr());
        let changes: &[fn(&mut SolverSelection)] = &[
            |s| s.objective = SolverPlanningObjective::Fast,
            |s| s.scalar_domain = ScalarDomain::Complex,
            |s| s.scalar_type = ScalarType::F32,
            |s| {
                s.plan = catalog_plan(
                    s.plan.algorithm(),
                    s.plan.preconditioner(),
                    s.plan.reduction(),
                    2e-12,
                    1e-14,
                    s.plan.maximum_iterations(),
                )
                .unwrap()
            },
            |s| {
                s.plan = catalog_plan(
                    s.plan.algorithm(),
                    s.plan.preconditioner(),
                    s.plan.reduction(),
                    1e-12,
                    2e-14,
                    s.plan.maximum_iterations(),
                )
                .unwrap()
            },
            |s| {
                s.plan = catalog_plan(
                    s.plan.algorithm(),
                    s.plan.preconditioner(),
                    s.plan.reduction(),
                    1e-12,
                    1e-14,
                    NonZeroUsize::new(101).unwrap(),
                )
                .unwrap()
            },
            |s| s.profile.facts.properties = LinearOperatorProperties::Symmetric,
            |s| s.profile.facts.orientation = LinearOperatorOrientation::Transposed,
            |s| s.profile.facts.canonical_csr = false,
            |s| s.profile.facts.complete_diagonal = Some(false),
            |s| s.profile.facts.complete_diagonal = None,
            |s| s.profile.required_reduction = Some(ReductionPolicy::Reproducible),
            |s| s.profile.size = Some(SolverPlanningSize::exact(2, 4)),
            |s| s.profile.cost_request = Some(SolverCostRequest::default()),
            |s| s.profile = s.profile.clone().with_run_forecast(None, None, None),
        ];
        for change in changes {
            let mut changed = original.clone();
            change(&mut changed);
            assert_changed(&original, &changed);
        }
        assert_eq!(fingerprint(&original), fingerprint(&original.clone()));
        let mut audit_only = original.clone();
        audit_only.reasons.clear();
        audit_only.costs.clear();
        assert_eq!(fingerprint(&original), fingerprint(&audit_only));
    }

    #[test]
    pub(super) fn size_certainty_each_cost_forecast_and_provider_snapshot_must_agree() {
        let original = selected(
            SolverPlanningProfile::general_canonical_csr()
                .with_size(SolverPlanningSize::exact(2, 4))
                .with_cost_request(SolverCostRequest::default())
                .with_run_forecast(None, None, None),
        );
        let nz = |n| NonZeroUsize::new(n).unwrap();
        let mut changed = original.clone();
        changed.profile.size = SolverPlanningSize::dense_bound(2);
        assert_changed(&original, &changed);
        for size in [
            SolverPlanningSize::exact(3, 4),
            SolverPlanningSize::exact(2, 5),
        ] {
            let mut changed = original.clone();
            changed.profile.size = Some(size);
            assert_changed(&original, &changed);
        }
        for cost in [
            SolverCostRequest::new(nz(1), nz(1), nz(1)).unwrap(),
            SolverCostRequest::new(nz(2), nz(1), nz(1)).unwrap(),
            SolverCostRequest::default().with_iteration_forecasts(Some(nz(2)), None),
            SolverCostRequest::default().with_iteration_forecasts(None, Some(nz(2))),
            SolverCostRequest::default().with_numeric_payload_limit(nz(1000)),
        ] {
            let mut changed = original.clone();
            changed.profile.cost_request = Some(cost);
            assert_changed(&original, &changed);
        }
        let base_cost = SolverCostRequest::new(nz(3), nz(1), nz(2)).unwrap();
        let mut horizon = original.clone();
        horizon.profile.cost_request = Some(base_cost);
        for cost in [
            SolverCostRequest::new(nz(3), nz(2), nz(2)).unwrap(),
            SolverCostRequest::new(nz(3), nz(1), nz(3)).unwrap(),
        ] {
            let mut changed = horizon.clone();
            changed.profile.cost_request = Some(cost);
            assert_changed(&horizon, &changed);
        }
        for (work, retained, prepared) in [
            (Some(1), None, None),
            (None, Some(1), None),
            (None, None, Some(REFERENCE_PROVIDER)),
        ] {
            let mut changed = original.clone();
            changed.profile = changed.profile.with_run_forecast(work, retained, prepared);
            assert_changed(&original, &changed);
        }
        let mut changed = original.clone();
        changed.capabilities = crate::SolverCapabilities::exact([SolverCapability {
            scalar_domain: ScalarDomain::Real,
            scalar_type: ScalarType::F64,
            algorithm: original.plan.algorithm(),
            operator_properties: LinearOperatorProperties::General,
            preconditioner: original.plan.preconditioner(),
            reduction: original.plan.reduction(),
        }])
        .unwrap();
        assert_changed(&original, &changed);
        changed = original.clone();
        changed.provider = SolverProvider::new(original.provider.id(), "different-release", &[]);
        assert_changed(&original, &changed);
        changed = original.clone();
        changed.profile.execution_provider =
            ExecutionProvider::new(original.execution_provider().id(), "different-release", &[]);
        assert_changed(&original, &changed);
        changed = original.clone();
        changed.profile.execution =
            crate::ExecutionReport::host(original.execution_provider().id(), nz(2));
        assert_changed(&original, &changed);
    }

    #[test]
    pub(super) fn exact_float_bits_topology_and_provider_libraries_are_retained() {
        let original = selected(SolverPlanningProfile::general_canonical_csr());
        let mut zero = original.clone();
        zero.plan = catalog_plan(
            original.plan.algorithm(),
            original.plan.preconditioner(),
            original.plan.reduction(),
            1e-12,
            0.0,
            original.plan.maximum_iterations(),
        )
        .unwrap();
        let mut negative_zero = zero.clone();
        negative_zero.plan = catalog_plan(
            original.plan.algorithm(),
            original.plan.preconditioner(),
            original.plan.reduction(),
            1e-12,
            -0.0,
            original.plan.maximum_iterations(),
        )
        .unwrap();
        assert_eq!(zero.plan, negative_zero.plan);
        assert_changed(&zero, &negative_zero);
        let id = original.execution_provider().id();
        for report in [
            crate::ExecutionReport::distributed(id, NonZeroUsize::MIN),
            crate::ExecutionReport::cuda(id, 0),
            crate::ExecutionReport::cuda(id, 1),
        ] {
            let mut changed = original.clone();
            changed.profile.execution = report;
            assert_changed(&original, &changed);
        }
        let mut cuda0 = original.clone();
        cuda0.profile.execution = crate::ExecutionReport::cuda(id, 0);
        let mut cuda1 = cuda0.clone();
        cuda1.profile.execution = crate::ExecutionReport::cuda(id, 1);
        assert_changed(&cuda0, &cuda1);
        const LIBRARIES: &[ProviderLibrary] = &[ProviderLibrary::new("runtime", "1")];
        let mut changed = original.clone();
        changed.provider = SolverProvider::new(
            original.provider.id(),
            original.provider.implementation_version(),
            LIBRARIES,
        );
        assert_changed(&original, &changed);
        changed = original.clone();
        changed.profile.execution_provider = ExecutionProvider::new(
            id,
            original.execution_provider().implementation_version(),
            LIBRARIES,
        );
        assert_changed(&original, &changed);
    }

    #[test]
    pub(super) fn normalized_field_order_agrees_but_exact_fields_and_gauges_do_not() {
        use eqiora_core::Id;
        let a = Id::new();
        let b = Id::new();
        let foreign = Id::new();
        let structure =
            |fields, constraints| crate::AlgebraicStructure::new(fields, constraints).unwrap();
        let original = selected(SolverPlanningProfile::general_canonical_csr());
        let mut base = original.clone();
        base.profile.structure = Some(structure([a, b], vec![]));
        let mut reordered = base.clone();
        reordered.profile.structure = Some(structure([b, a], vec![]));
        assert_eq!(fingerprint(&base), fingerprint(&reordered));
        for changed_structure in [
            structure([a, foreign], vec![]),
            structure(
                [a, b],
                vec![crate::AlgebraicConstraint::ZeroIntegral { field: a }],
            ),
            structure(
                [a, b],
                vec![crate::AlgebraicConstraint::ZeroIntegral { field: b }],
            ),
        ] {
            let mut changed = base.clone();
            changed.profile.structure = Some(changed_structure);
            assert_changed(&base, &changed);
        }
        assert_changed(&original, &base);
    }
}

#[cfg(test)]
pub(crate) fn registered_planning_agreement_falsifiers() {
    tests::same_tuple_controls_objective_and_profile_facts_must_agree();
    tests::size_certainty_each_cost_forecast_and_provider_snapshot_must_agree();
    tests::exact_float_bits_topology_and_provider_libraries_are_retained();
    tests::normalized_field_order_agrees_but_exact_fields_and_gauges_do_not();
}
