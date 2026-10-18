//! Real scalar storage of retained complex-valued spatial equations.
use super::*;
use crate::finalized_spatial::FinalizedLinearCore;
use crate::region_assembly::mapping::RegionRealCoordinates;
use eqiora_solver::CanonicalCsrSystemView;
use num_complex::Complex64 as C;

pub(in crate::numerical_admission) enum StaticSpatialSystem<P: Coefficient> {
    Real(StaticRegionSystem<f64, P>),
    Complex(StaticRegionSystem<C, P>),
}

impl<P: Coefficient> StaticSpatialSystem<P> {
    pub(in crate::numerical_admission) fn mapping(&self) -> &RegionDofMap<P> {
        match self {
            Self::Real(system) => &system.mapping,
            Self::Complex(system) => &system.mapping,
        }
    }
}

/// Assembly precedes solver selection; mathematical Field identity stays in the map.
pub(in crate::numerical_admission) struct RealStaticAssembly {
    mapping: RegionDofMap<C>,
    canonical: CanonicalCsrSystemView<f64>,
    assembly: StaticAssembly<f64, C>,
    report: eqiora_assembly::AssemblyReport,
}

struct ComplexStaticMatrices {
    reduced: CanonicalCsrSystemView<C>,
    full: Option<eqiora_assembly::LinearSystem<C>>,
}

impl ExecutableSpatialEquations<C> {
    pub(in crate::numerical_admission) fn assemble_real_static(
        &self,
        resources: &NativeMeshResources,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
    ) -> Result<RealStaticAssembly, Diagnostic> {
        if selected
            .iter()
            .flat_map(|domain| domain.field_spaces())
            .any(|field| field.space() == Space::cell_constant())
        {
            let mapping = self.static_tpfa_mapping(resources, selected, scaling)?;
            let properties = self.tpfa_operator_properties(None)?;
            let (prepared, result) = self
                .with_tpfa_work::<crate::form_compiler::region::coordinates::Real, _>(
                    resources,
                    &mapping,
                    properties,
                    None,
                    |plan, work| {
                        eqiora_assembly::AssemblyBackend::assemble(
                            &REFERENCE_ASSEMBLY_BACKEND,
                            plan,
                            work,
                        )
                    },
                )?;
            let assembly = prepared.finish(result);
            return Ok(RealStaticAssembly {
                mapping,
                canonical: assembly.canonical_system()?.as_ref().clone(),
                report: assembly.assembly_report().clone(),
                assembly: StaticAssembly::FiniteVolume,
            });
        }
        // A real-linear map makes no complex symmetry or definiteness promise.
        let properties = LinearOperatorProperties::General;
        let (mapping, input) = self.static_assembly(resources, selected, scaling, properties)?;
        let coordinates = RegionRealCoordinates::new(&mapping)?;
        let (reduced, full, report) = match resources {
            NativeMeshResources::Cartesian { mesh, .. } => {
                coordinates.assemble_static(mesh.mesh(), &input)?
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => {
                coordinates.assemble_static(mesh.mesh(), &input)?
            }
            _ => return Err(invalid("static system requires its admitted Mesh")),
        };
        let system =
            eqiora_solver::CanonicalCsrSystemView::new(&reduced, reduced.rhs(), properties)?;
        Ok(RealStaticAssembly {
            mapping,
            canonical: system,
            assembly: StaticAssembly::FiniteElement(Box::new((full, input))),
            report,
        })
    }

    pub(in crate::numerical_admission) fn finalize_complex_static(
        &self,
        workers: NonZeroUsize,
        solver: eqiora_solver::SolverPlan,
        resources: &NativeMeshResources,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
    ) -> Result<StaticSpatialSystem<C>, Diagnostic> {
        self.assemble_real_static(resources, selected, scaling)?
            .finalize(workers, solver)
    }
}

impl RealStaticAssembly {
    fn complex_systems(&self) -> Result<Option<ComplexStaticMatrices>, Diagnostic> {
        // Adjacent pairing is justified by physical types, never just an even size.
        if self.mapping.keys().any(|key| {
            self.mapping
                .field_layout(key.field)
                .expect("mapped Field")
                .1
                .value_type
                .scalar_domain()
                != eqiora_core::ScalarDomain::Complex
        }) {
            return Ok(None);
        }
        let Some(reduced) = self.canonical.complex_coordinate_system()? else {
            return Ok(None);
        };
        let StaticAssembly::FiniteElement(retained) = &self.assembly else {
            return Ok(Some(ComplexStaticMatrices {
                reduced,
                full: None,
            }));
        };
        let full = CanonicalCsrSystemView::new(
            &retained.0,
            retained.0.rhs(),
            LinearOperatorProperties::General,
        )?;
        let Some(full) = full.complex_coordinate_system()? else {
            return Ok(None);
        };
        let matrix = eqiora_assembly::CsrMatrix::from_sorted_csr(
            full.rows(),
            full.rows(),
            full.row_offsets().to_vec(),
            full.column_indices().to_vec(),
            full.values().to_vec(),
        )?;
        Ok(Some(ComplexStaticMatrices {
            reduced,
            full: Some(eqiora_assembly::LinearSystem::new(
                matrix,
                full.right_hand_side().to_vec(),
            )?),
        }))
    }

    pub(in crate::numerical_admission) fn scalar_domain(
        &self,
    ) -> Result<eqiora_core::ScalarDomain, Diagnostic> {
        Ok(if self.complex_systems()?.is_some() {
            eqiora_core::ScalarDomain::Complex
        } else {
            eqiora_core::ScalarDomain::Real
        })
    }

    pub(in crate::numerical_admission) fn finalize_real(
        self,
        workers: NonZeroUsize,
        solver: SolverPlan,
    ) -> StaticRegionSystem<f64, C> {
        StaticRegionSystem::new(
            self.mapping,
            finalized(self.canonical, workers, solver),
            self.report,
            self.assembly,
        )
    }

    fn finalize(
        self,
        workers: NonZeroUsize,
        solver: SolverPlan,
    ) -> Result<StaticSpatialSystem<C>, Diagnostic> {
        Ok(match self.complex_systems()? {
            Some(ComplexStaticMatrices { reduced, full }) => {
                StaticSpatialSystem::Complex(StaticRegionSystem::new(
                    self.mapping,
                    finalized(reduced, workers, solver),
                    self.report,
                    match self.assembly {
                        StaticAssembly::FiniteElement(retained) => StaticAssembly::FiniteElement(
                            Box::new((full.expect("projected full system"), retained.1)),
                        ),
                        StaticAssembly::FiniteVolume => StaticAssembly::FiniteVolume,
                    },
                ))
            }
            None => StaticSpatialSystem::Real(self.finalize_real(workers, solver)),
        })
    }
}

fn finalized<S: Coefficient + Send + Sync>(
    canonical: CanonicalCsrSystemView<S>,
    workers: NonZeroUsize,
    solver: SolverPlan,
) -> FinalizedLinearCore<S> {
    FinalizedLinearCore::new(
        solver,
        VectorLayoutKind::Replicated,
        Target::HostCpu { threads: workers },
        std::sync::Arc::new(canonical),
    )
}
