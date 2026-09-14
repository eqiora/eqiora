use eqiora_core::{Diagnostic, ScalarType};
use eqiora_solver::{
    CanonicalCsrAgreementFingerprintV1, CanonicalCsrSystemView, DiagonalAvailability,
    LinearOperatorProperties, LinearSolver, PreconditionerPolicy, SolveReport, SolverPlan,
};

use crate::agreement::{
    DistributedAdmissionFingerprintV1, DistributedLayoutAgreementIdentityV1,
    PartitionAgreementIdentityV1, distributed_admission_fingerprint,
    distributed_layout_agreement_identity, partition_agreement_identity,
};
use crate::allocation::{realization_copy, realization_vector};
use crate::csr::{DistributedCsr, OwnedLinearSystemShard};
use crate::error::{invalid_realization, solve_failed};
use crate::partition::{Partition, PartitionId};

/// Complete algebraic source lowered into one validated distributed layout.
///
/// The complete matrix is represented by its Eqiora-owned CSR agreement
/// identity while [`DistributedCsr`] owns the derived row shards. The finite
/// complete RHS and its rank-local projections are retained here. Mesh,
/// assembly, transport, reconstruction, and solver policy remain outside this
/// contract.
#[derive(Debug, PartialEq)]
pub struct DistributedLinearSystem {
    operator: DistributedCsr,
    complete_right_hand_side: Vec<f64>,
    local_right_hand_sides: Vec<Vec<f64>>,
    properties: LinearOperatorProperties,
    system_identity: CanonicalCsrAgreementFingerprintV1,
    partition_identity: PartitionAgreementIdentityV1,
    layout_identity: DistributedLayoutAgreementIdentityV1,
}

impl DistributedLinearSystem {
    /// Derive distributed shards, layouts, halo, RHS projections, and all L2
    /// identities from one canonical complete view and one owner map.
    ///
    /// # Errors
    /// Returns `EQ0807` when dimensions or scalar types disagree, or when the
    /// derived distributed CSR violates its closed contract.
    pub fn from_complete(
        complete: &CanonicalCsrSystemView,
        partition: Partition,
    ) -> Result<Self, Diagnostic> {
        if partition.space().scalar_type() != ScalarType::F64
            || partition.space().dimension().get() != complete.rows()
            || complete.rows() != complete.columns()
        {
            return Err(invalid_realization(format!(
                "complete {}x{} f64 CSR and partition {:?} dimension {} do not agree",
                complete.rows(),
                complete.columns(),
                partition.space().scalar_type(),
                partition.space().dimension()
            )));
        }
        let operator = DistributedCsr::from_global_csr(
            partition,
            complete.row_offsets(),
            complete.column_indices(),
            complete.values(),
        )?;
        let complete_right_hand_side = realization_copy(
            complete.right_hand_side(),
            "complete distributed right-hand side",
        )?;
        let mut local_right_hand_sides =
            realization_vector(operator.layouts().len(), "rank-local RHS table")?;
        for layout in operator.layouts() {
            let mut local = realization_vector(layout.owned().len(), "one rank-local RHS")?;
            for &global in layout.owned() {
                local.push(complete_right_hand_side[global]);
            }
            local_right_hand_sides.push(local);
        }
        let system_identity = complete.agreement_fingerprint();
        let partition_identity = partition_agreement_identity(operator.partition())?;
        let layout_identity =
            distributed_layout_agreement_identity(system_identity, partition_identity, &operator)?;
        Ok(Self {
            operator,
            complete_right_hand_side,
            local_right_hand_sides,
            properties: complete.properties(),
            system_identity,
            partition_identity,
            layout_identity,
        })
    }

    /// Promote already assembled owned rows into the unique distributed
    /// system for their accepted owner map.
    ///
    /// Every shard is compared bit-for-bit with the complete verifier before
    /// it can enter the operator. The supplied partition must own exactly the
    /// rows carried by the shards; there is no balancing or secondary
    /// partition choice in this constructor.
    ///
    /// # Errors
    /// Returns `EQ0807` for a missing, duplicate, misowned, reordered, or
    /// numerically different shard, or for disagreement with the complete
    /// canonical system.
    pub fn from_owned_shards(
        complete: &CanonicalCsrSystemView,
        partition: Partition,
        shards: Vec<OwnedLinearSystemShard>,
    ) -> Result<Self, Diagnostic> {
        let (operator, local_right_hand_sides) =
            DistributedCsr::from_owned_shards(complete, partition, shards)?.into_parts();
        let complete_right_hand_side = realization_copy(
            complete.right_hand_side(),
            "complete distributed right-hand side",
        )?;
        let system_identity = complete.agreement_fingerprint();
        let partition_identity = partition_agreement_identity(operator.partition())?;
        let layout_identity =
            distributed_layout_agreement_identity(system_identity, partition_identity, &operator)?;
        Ok(Self {
            operator,
            complete_right_hand_side,
            local_right_hand_sides,
            properties: complete.properties(),
            system_identity,
            partition_identity,
            layout_identity,
        })
    }

    /// Derived distributed CSR, including partition, shards, and halo plan.
    #[must_use]
    pub const fn operator(&self) -> &DistributedCsr {
        &self.operator
    }

    /// Unique owner map.
    #[must_use]
    pub const fn partition(&self) -> &Partition {
        self.operator.partition()
    }

    /// Complete finite RHS in global-index order.
    #[must_use]
    pub fn complete_right_hand_side(&self) -> &[f64] {
        &self.complete_right_hand_side
    }

    /// Asserted mathematical properties inherited from the complete view.
    #[must_use]
    pub const fn properties(&self) -> LinearOperatorProperties {
        self.properties
    }

    /// Exact complete-system algebraic agreement identity.
    #[must_use]
    pub const fn system_identity(&self) -> CanonicalCsrAgreementFingerprintV1 {
        self.system_identity
    }

    /// Exact owner-map agreement identity.
    #[must_use]
    pub const fn partition_identity(&self) -> PartitionAgreementIdentityV1 {
        self.partition_identity
    }

    /// Exact derived layout/halo agreement identity.
    #[must_use]
    pub const fn layout_identity(&self) -> DistributedLayoutAgreementIdentityV1 {
        self.layout_identity
    }

    /// Confirm that a supplied complete view is the exact captured algebraic
    /// source, including CSR, RHS, and property assertion.
    #[must_use]
    pub fn matches_complete(&self, complete: &CanonicalCsrSystemView) -> bool {
        self.system_identity == complete.agreement_fingerprint()
    }

    /// Borrow one rank-local problem in that layout's explicit owned order.
    ///
    /// # Errors
    /// Returns `EQ0802` for an unknown partition or an internal projection
    /// contradiction.
    pub fn local_problem(
        &self,
        partition: PartitionId,
    ) -> Result<DistributedLinearProblem<'_>, Diagnostic> {
        let right_hand_side = self
            .local_right_hand_sides
            .get(partition.index())
            .ok_or_else(|| solve_failed("distributed system names an unknown partition"))?;
        DistributedLinearProblem::new(&self.operator, partition, right_hand_side, self.properties)
    }

    /// Validate the distributed numerical policy and derive the exact
    /// fixed-size collective admission fingerprint.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the algorithm accepts the asserted operator,
    /// the distributed contract implements the requested preconditioner, and
    /// an available positive Jacobi diagonal exists when requested.
    pub fn admission_fingerprint(
        &self,
        plan: SolverPlan,
    ) -> Result<DistributedAdmissionFingerprintV1, Diagnostic> {
        self.validate_plan(plan)?;
        distributed_admission_fingerprint(
            self.system_identity,
            self.partition_identity,
            self.layout_identity,
            plan,
        )
    }

    fn validate_plan(&self, plan: SolverPlan) -> Result<(), Diagnostic> {
        if !plan.algorithm().accepts(self.properties) {
            return Err(invalid_realization(
                "distributed solver algorithm does not accept the asserted operator properties",
            ));
        }
        match plan.algorithm() {
            LinearSolver::ConjugateGradient => {}
            LinearSolver::MinimumResidual
                if plan.preconditioner() == PreconditionerPolicy::Identity => {}
            LinearSolver::MinimumResidual => {
                return Err(invalid_realization(
                    "distributed MINRES currently admits identity preconditioning only",
                ));
            }
            LinearSolver::BiConjugateGradientStabilized => {
                return Err(invalid_realization(
                    "distributed BiCGSTAB is not implemented",
                ));
            }
            LinearSolver::SparseLu => {
                return Err(invalid_realization(
                    "distributed sparse LU is not implemented",
                ));
            }
        }
        if plan.preconditioner() == PreconditionerPolicy::Jacobi {
            for partition in 0..self.operator.partition().count().get() {
                let shard = self
                    .operator
                    .shard(PartitionId::new(partition))
                    .ok_or_else(|| invalid_realization("validated partition has no CSR shard"))?;
                let mut diagonal =
                    realization_vector(shard.layout().owned().len(), "Jacobi admission buffer")?;
                diagonal.resize(shard.layout().owned().len(), 0.0);
                if shard.diagonal(&mut diagonal)? != DiagonalAvailability::Available
                    || diagonal
                        .iter()
                        .any(|value| *value <= 0.0 || !value.is_finite())
                {
                    return Err(invalid_realization(format!(
                        "partition {partition} lacks a finite positive Jacobi diagonal"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// One rank-local view of a validated distributed linear problem.
#[derive(Debug)]
pub struct DistributedLinearProblem<'a> {
    operator: &'a DistributedCsr,
    partition: PartitionId,
    right_hand_side: &'a [f64],
    initial_guess: Option<&'a [f64]>,
    properties: LinearOperatorProperties,
}

impl<'a> DistributedLinearProblem<'a> {
    /// Bind local vectors to one immutable distributed operator artifact.
    ///
    /// # Errors
    /// Returns `EQ0802` for an unknown partition, local shape mismatch, or
    /// non-finite right-hand-side value.
    pub fn new(
        operator: &'a DistributedCsr,
        partition: PartitionId,
        right_hand_side: &'a [f64],
        properties: LinearOperatorProperties,
    ) -> Result<Self, Diagnostic> {
        let layout = operator
            .layouts()
            .get(partition.index())
            .ok_or_else(|| solve_failed("distributed problem names an unknown partition"))?;
        if right_hand_side.len() != layout.owned().len()
            || right_hand_side.iter().any(|value| !value.is_finite())
        {
            return Err(solve_failed(format!(
                "partition {} right-hand side must contain {} finite owned values",
                partition.index(),
                layout.owned().len()
            )));
        }
        Ok(Self {
            operator,
            partition,
            right_hand_side,
            initial_guess: None,
            properties,
        })
    }

    /// Attach a rank-local initial guess in owned-index order.
    ///
    /// # Errors
    /// Returns `EQ0802` for a local shape mismatch or non-finite value.
    pub fn with_initial_guess(mut self, initial_guess: &'a [f64]) -> Result<Self, Diagnostic> {
        if initial_guess.len() != self.right_hand_side.len()
            || initial_guess.iter().any(|value| !value.is_finite())
        {
            return Err(solve_failed(
                "distributed initial guess must match the local right-hand side and be finite",
            ));
        }
        self.initial_guess = Some(initial_guess);
        Ok(self)
    }

    /// Immutable distributed operator artifact.
    #[must_use]
    pub const fn operator(&self) -> &'a DistributedCsr {
        self.operator
    }

    /// Rank-local partition identity.
    #[must_use]
    pub const fn partition(&self) -> PartitionId {
        self.partition
    }

    /// Rank-local right-hand side in owned-index order.
    #[must_use]
    pub const fn right_hand_side(&self) -> &'a [f64] {
        self.right_hand_side
    }

    /// Explicit rank-local initial guess, or `None` for zero.
    #[must_use]
    pub const fn initial_guess(&self) -> Option<&'a [f64]> {
        self.initial_guess
    }

    /// Asserted global operator properties.
    #[must_use]
    pub const fn properties(&self) -> LinearOperatorProperties {
        self.properties
    }
}

/// Rank-local values paired with globally accepted solve evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalLinearSolution {
    partition: PartitionId,
    values: Vec<f64>,
    report: SolveReport,
}

impl LocalLinearSolution {
    /// Pair finite owned values with an accepted global solve report.
    ///
    /// # Errors
    /// Returns `EQ0802` when values contradict the problem's local layout.
    pub fn new(
        problem: &DistributedLinearProblem<'_>,
        values: Vec<f64>,
        report: SolveReport,
    ) -> Result<Self, Diagnostic> {
        if values.len() != problem.right_hand_side.len()
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(solve_failed(
                "distributed solution must contain one finite value per owned row",
            ));
        }
        Ok(Self {
            partition: problem.partition,
            values,
            report,
        })
    }

    /// Rank-local partition identity.
    #[must_use]
    pub const fn partition(&self) -> PartitionId {
        self.partition
    }

    /// Accepted values in owned-index order.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Global convergence and execution evidence.
    #[must_use]
    pub const fn report(&self) -> &SolveReport {
        &self.report
    }
}
