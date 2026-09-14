use eqiora_core::{Diagnostic, ScalarType};
use eqiora_solver::{
    CanonicalCsrAgreementFingerprintV1, LinearSolver, PreconditionerPolicy, ReductionPolicy,
    SolverPlan,
};
use sha2::{Digest, Sha256};

use crate::csr::DistributedCsr;
use crate::error::invalid_realization;
use crate::partition::Partition;

const PARTITION_AGREEMENT_DOMAIN_V1: &[u8] = b"eqiora.partition-agreement/v1\0";
const DISTRIBUTED_LAYOUT_AGREEMENT_DOMAIN_V1: &[u8] = b"eqiora.distributed-layout-agreement/v1\0";
const DISTRIBUTED_ADMISSION_DOMAIN_V1: &[u8] = b"eqiora.distributed-admission/v1\0";

/// Fixed-size L2 identity for one complete unique-owner partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PartitionAgreementIdentityV1([u8; 32]);

impl PartitionAgreementIdentityV1 {
    /// Raw SHA-256 bytes for fixed-size collective comparison.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Fixed-size L2 identity for derived local layouts and halo exchanges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DistributedLayoutAgreementIdentityV1([u8; 32]);

impl DistributedLayoutAgreementIdentityV1 {
    /// Raw SHA-256 bytes for fixed-size collective comparison.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Fixed-size L2 identity for collective system/partition/plan admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DistributedAdmissionFingerprintV1([u8; 32]);

impl DistributedAdmissionFingerprintV1 {
    /// Raw SHA-256 bytes for fixed-size collective comparison.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

pub(crate) fn partition_agreement_identity(
    partition: &Partition,
) -> Result<PartitionAgreementIdentityV1, Diagnostic> {
    let mut hash = Sha256::new();
    hash.update(PARTITION_AGREEMENT_DOMAIN_V1);
    hash.update([scalar_tag(partition.space().scalar_type())]);
    update_count(
        &mut hash,
        partition.space().dimension().get(),
        "global dimension",
    )?;
    update_count(&mut hash, partition.count().get(), "partition count")?;
    update_count(&mut hash, partition.owners().len(), "owner count")?;
    for owner in partition.owners() {
        update_count(&mut hash, owner.index(), "owner partition")?;
    }
    Ok(PartitionAgreementIdentityV1(hash.finalize().into()))
}

pub(crate) fn distributed_layout_agreement_identity(
    system: CanonicalCsrAgreementFingerprintV1,
    partition: PartitionAgreementIdentityV1,
    operator: &DistributedCsr,
) -> Result<DistributedLayoutAgreementIdentityV1, Diagnostic> {
    let mut hash = Sha256::new();
    hash.update(DISTRIBUTED_LAYOUT_AGREEMENT_DOMAIN_V1);
    hash.update(system.as_bytes());
    hash.update(partition.as_bytes());
    update_count(&mut hash, operator.layouts().len(), "local-layout count")?;
    for layout in operator.layouts() {
        update_count(&mut hash, layout.partition().index(), "layout partition")?;
        update_indices(&mut hash, layout.owned(), "owned index")?;
        update_indices(&mut hash, layout.ghosts(), "ghost index")?;
    }
    update_count(
        &mut hash,
        operator.halo().exchanges().len(),
        "halo-exchange count",
    )?;
    for exchange in operator.halo().exchanges() {
        update_count(&mut hash, exchange.owner().index(), "halo owner")?;
        update_count(&mut hash, exchange.receiver().index(), "halo receiver")?;
        update_indices(&mut hash, exchange.indices(), "halo index")?;
    }
    Ok(DistributedLayoutAgreementIdentityV1(hash.finalize().into()))
}

pub(crate) fn distributed_admission_fingerprint(
    system: CanonicalCsrAgreementFingerprintV1,
    partition: PartitionAgreementIdentityV1,
    layout: DistributedLayoutAgreementIdentityV1,
    plan: SolverPlan,
) -> Result<DistributedAdmissionFingerprintV1, Diagnostic> {
    let mut hash = Sha256::new();
    hash.update(DISTRIBUTED_ADMISSION_DOMAIN_V1);
    hash.update(system.as_bytes());
    hash.update(partition.as_bytes());
    hash.update(layout.as_bytes());
    hash.update([linear_solver_tag(plan.algorithm())]);
    hash.update([match plan.preconditioner() {
        PreconditionerPolicy::Identity => 0,
        PreconditionerPolicy::Jacobi => 1,
    }]);
    hash.update([match plan.reduction() {
        ReductionPolicy::Reproducible => 0,
        ReductionPolicy::Fast => 1,
    }]);
    hash.update(plan.relative_tolerance().to_bits().to_be_bytes());
    hash.update(plan.absolute_tolerance().to_bits().to_be_bytes());
    update_count(
        &mut hash,
        plan.maximum_iterations().get(),
        "maximum iteration count",
    )?;
    Ok(DistributedAdmissionFingerprintV1(hash.finalize().into()))
}

const fn linear_solver_tag(algorithm: LinearSolver) -> u8 {
    match algorithm {
        LinearSolver::ConjugateGradient => 0,
        LinearSolver::BiConjugateGradientStabilized => 1,
        LinearSolver::MinimumResidual => 2,
        LinearSolver::SparseLu => 3,
    }
}

fn scalar_tag(scalar_type: ScalarType) -> u8 {
    match scalar_type {
        ScalarType::F32 => 0,
        ScalarType::F64 => 1,
    }
}

fn update_indices(
    hash: &mut Sha256,
    indices: &[usize],
    name: &'static str,
) -> Result<(), Diagnostic> {
    update_count(hash, indices.len(), name)?;
    for &index in indices {
        update_count(hash, index, name)?;
    }
    Ok(())
}

fn update_count(hash: &mut Sha256, value: usize, name: &'static str) -> Result<(), Diagnostic> {
    let value = u64::try_from(value)
        .map_err(|_| invalid_realization(format!("distributed {name} exceeds portable u64")))?;
    hash.update(value.to_be_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distributed_solver_tags_are_additive_and_frozen() {
        assert_eq!(linear_solver_tag(LinearSolver::ConjugateGradient), 0);
        assert_eq!(
            linear_solver_tag(LinearSolver::BiConjugateGradientStabilized),
            1
        );
        assert_eq!(linear_solver_tag(LinearSolver::MinimumResidual), 2);
        assert_eq!(linear_solver_tag(LinearSolver::SparseLu), 3);
    }
}
