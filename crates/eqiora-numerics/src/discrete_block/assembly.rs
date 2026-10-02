use super::*;

impl DiscreteBlockSystem {
    pub(crate) fn checked_backend<'a>(
        &'a self,
        inner: &'a dyn AssemblyBackend,
    ) -> super::CheckedBlockAssemblyBackend<'a> {
        super::CheckedBlockAssemblyBackend {
            system: self,
            inner,
            validated_materializations: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub(crate) fn bind_materialization(
        &self,
        system: &CanonicalCsrSystemView,
        report: &AssemblyReport,
    ) -> Result<BlockMaterialization, Diagnostic> {
        self.validate_report(report)?;
        if system.properties() != self.required_properties {
            return Err(invalid(
                "captured CSR properties differ from the block-system requirement",
            ));
        }
        Ok(BlockMaterialization {
            block_identity: self.identity,
            csr_fingerprint: system.agreement_fingerprint(),
            rows: system.rows(),
            packet_count: self.packet_count,
        })
    }
}

/// One exact association between semantic block structure and canonical CSR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockMaterialization {
    block_identity: BlockSystemIdentity,
    csr_fingerprint: CanonicalCsrAgreementFingerprintV1,
    rows: usize,
    packet_count: usize,
}

impl BlockMaterialization {
    pub(crate) fn validate(self, system: &CanonicalCsrSystemView) -> Result<(), Diagnostic> {
        if self.rows != system.rows()
            || self.csr_fingerprint != system.agreement_fingerprint()
            || self.packet_count == 0
            || self.block_identity.0 == [0; 32]
        {
            return Err(invalid(
                "the captured CSR no longer agrees with its exact block materialization",
            ));
        }
        Ok(())
    }
}

/// Adapter proving assembly passed through the exact block packet/target shape.
pub(crate) struct CheckedBlockAssemblyBackend<'a> {
    system: &'a DiscreteBlockSystem,
    inner: &'a dyn AssemblyBackend,
    validated_materializations: std::sync::atomic::AtomicUsize,
}

impl CheckedBlockAssemblyBackend<'_> {
    pub(crate) fn validated_materialization_count(&self) -> usize {
        self.validated_materializations
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl fmt::Debug for CheckedBlockAssemblyBackend<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CheckedBlockAssemblyBackend")
            .field("block_identity", &self.system.identity)
            .field("inner", &self.inner)
            .finish()
    }
}

impl AssemblyBackend for CheckedBlockAssemblyBackend<'_> {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork,
    ) -> Result<AssemblyResult, Diagnostic> {
        if plan.target_count() != self.system.target_count
            || work.packet_count() != self.system.packet_count
        {
            return Err(invalid(
                "assembly plan/work shape differs from the exact discrete block system",
            ));
        }
        let checked_work = CheckedBlockAssemblyWork {
            system: self.system,
            inner: work,
        };
        let result = self.inner.assemble(plan, &checked_work)?;
        self.system.validate_report(result.report())?;
        let primary = result
            .systems()
            .get(self.system.primary_target)
            .ok_or_else(|| {
                invalid("assembly result omits the block system's primary materialization target")
            })?;
        let canonical = CanonicalCsrSystemView::new(primary, self.system.required_properties)?;
        let materialization = self
            .system
            .bind_materialization(&canonical, result.report())?;
        materialization.validate(&canonical)?;
        #[allow(deprecated, reason = "try_update requires Rust 1.95; MSRV is 1.89")]
        self.validated_materializations
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |count| count.checked_add(1),
            )
            .map_err(|_| invalid("validated block-materialization count overflows usize"))?;
        Ok(result)
    }
}

struct CheckedBlockAssemblyWork<'a> {
    system: &'a DiscreteBlockSystem,
    inner: &'a dyn AssemblyWork,
}

impl fmt::Debug for CheckedBlockAssemblyWork<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CheckedBlockAssemblyWork")
            .field("block_identity", &self.system.identity)
            .field("inner", &self.inner)
            .finish()
    }
}

impl AssemblyWork for CheckedBlockAssemblyWork<'_> {
    fn packet_set_identity(&self) -> AssemblyPacketSetIdentityV1 {
        self.inner.packet_set_identity()
    }

    fn packet_count(&self) -> usize {
        self.inner.packet_count()
    }

    fn evaluate(&self, packet_index: usize) -> Result<AssemblyPacket, Diagnostic> {
        let packet = self.inner.evaluate(packet_index)?;
        self.system.validate_packet(packet_index, &packet)?;
        Ok(packet)
    }
}
