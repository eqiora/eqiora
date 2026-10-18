//! Real scalar storage of retained complex-valued spatial equations.
use super::*;
use crate::finalized_spatial::FinalizedLinearCore;
use crate::region_assembly::mapping::RegionRealCoordinates;
use crate::region_assembly::mapping::RegionSolveInput;
use eqiora_solver::CanonicalCsrSystemView;
use num_complex::Complex64 as C;

pub(in crate::numerical_admission) enum ComplexSpatialSystem {
    Real(StaticRegionSystem<f64, C>),
    Complex(StaticRegionSystem<C>),
}

/// Assembly precedes solver selection; mathematical Field identity stays in the map.
pub(in crate::numerical_admission) struct RealStaticAssembly {
    mapping: RegionDofMap<C>,
    canonical: CanonicalCsrSystemView<f64>,
    full: eqiora_assembly::LinearSystem<f64>,
    input: RegionSolveInput<C>,
    report: eqiora_assembly::AssemblyReport,
}

struct ComplexStaticMatrices {
    reduced: CanonicalCsrSystemView<C>,
    full: eqiora_assembly::LinearSystem<C>,
}

impl ExecutableSpatialEquations<C> {
    pub(in crate::numerical_admission) fn assemble_real_static(
        &self,
        resources: &NativeMeshResources,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
    ) -> Result<RealStaticAssembly, Diagnostic> {
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
            full,
            input,
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
        properties: LinearOperatorProperties,
    ) -> Result<ComplexSpatialSystem, Diagnostic> {
        if selected
            .iter()
            .flat_map(|domain| domain.field_spaces())
            .any(|field| field.space() == Space::cell_constant())
        {
            return self
                .finalize_static(workers, solver, resources, selected, scaling, properties)
                .map(ComplexSpatialSystem::Complex);
        }
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
        let full = CanonicalCsrSystemView::new(
            &self.full,
            self.full.rhs(),
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
            full: eqiora_assembly::LinearSystem::new(matrix, full.right_hand_side().to_vec())?,
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
            StaticAssembly::FiniteElement(Box::new((self.full, self.input))),
        )
    }

    fn finalize(
        self,
        workers: NonZeroUsize,
        solver: SolverPlan,
    ) -> Result<ComplexSpatialSystem, Diagnostic> {
        Ok(match self.complex_systems()? {
            Some(ComplexStaticMatrices { reduced, full }) => {
                ComplexSpatialSystem::Complex(StaticRegionSystem::new(
                    self.mapping,
                    finalized(reduced, workers, solver),
                    self.report,
                    StaticAssembly::FiniteElement(Box::new((full, self.input))),
                ))
            }
            None => ComplexSpatialSystem::Real(self.finalize_real(workers, solver)),
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
