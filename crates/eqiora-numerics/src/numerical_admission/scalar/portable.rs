//! One exact numerical graph derived from retained linear admission resources.
use super::*;

pub(super) fn resolve_common_linear_portable<S: crate::spatial_expression::Coefficient>(
    admission: &NativeNumericalAdmission,
    lowered: &ExecutableLinearEquations<S>,
) -> Result<PortableRealizationGraph, Diagnostic> {
    let mesh = match admission.resources() {
        NativeMeshResources::Cartesian { mesh, .. } => {
            let artifact = mesh.artifact_reference()?;
            let cells = admission.resources().cartesian_cells()?;
            let nonzero = |count| NonZeroUsize::new(count).expect("validated Cartesian cells");
            match cells.as_slice() {
                [x] => MeshPolicy::SuppliedCartesian1d {
                    artifact,
                    cells: [nonzero(*x)],
                },
                [x, y] => MeshPolicy::SuppliedCartesian {
                    artifact,
                    cells: [nonzero(*x), nonzero(*y)],
                },
                [x, y, z] => MeshPolicy::SuppliedCartesian3d {
                    artifact,
                    cells: [nonzero(*x), nonzero(*y), nonzero(*z)],
                },
                _ => return Err(invalid("linear Plan requires one to three Cartesian axes")),
            }
        }
        NativeMeshResources::GmshSimplicial { mesh, .. } => MeshPolicy::ImportedSimplicial {
            artifact: mesh.artifact_reference()?,
        },
        _ => {
            return Err(invalid(
                "linear Plan has no admitted finite-dimensional Mesh",
            ));
        }
    };
    let (method, quadrature) = match admission.spatial {
        NativeSpatialPolicy::LinearFiniteElement(space)
            if space == Space::continuous_lagrange(std::num::NonZeroU16::MIN) =>
        {
            (
                DiscretizationMethod::ContinuousGalerkin,
                if matches!(
                    admission.resources(),
                    NativeMeshResources::GmshSimplicial { .. }
                ) {
                    QuadraturePolicy::SimplexDuffyGaussLegendre {
                        spatial_dimension: NonZeroUsize::new(2).unwrap(),
                        points_per_axis: NonZeroUsize::new(3).unwrap(),
                    }
                } else {
                    QuadraturePolicy::GaussLegendre {
                        points_per_axis: NonZeroUsize::new(2).expect("two is non-zero"),
                    }
                },
            )
        }
        NativeSpatialPolicy::LinearFiniteElement(space)
            if matches!(
                space.family(),
                SpaceFamily::TetrahedralEdge | SpaceFamily::TetrahedralFace
            ) =>
        {
            (
                DiscretizationMethod::ContinuousGalerkin,
                QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension: NonZeroUsize::new(3).unwrap(),
                    points_per_axis: NonZeroUsize::new(3).unwrap(),
                },
            )
        }
        NativeSpatialPolicy::ScalarTpfa(_) => (
            DiscretizationMethod::CellCenteredFiniteVolume,
            QuadraturePolicy::CellCentroid,
        ),
        NativeSpatialPolicy::LinearFiniteElement(_)
        | NativeSpatialPolicy::CoordinateCellConstant
        | NativeSpatialPolicy::StokesMiniP1(_)
        | NativeSpatialPolicy::TransientMiniP1(_)
        | NativeSpatialPolicy::TransientCellCentered(_) => {
            return Err(invalid(
                "common scalar portable graph received a non-scalar spatial policy",
            ));
        }
    };
    let discretization = Discretization::new(method, mesh, quadrature);
    validate_resources(admission.spatial, admission.resources())?;
    let solver = admission.linear.solver;
    admission.linear.capabilities.require_problem(
        solver,
        lowered
            .fields()
            .first()
            .ok_or_else(|| invalid("scalar Plan has no unknown Fields"))?
            .1
            .scalar_domain(),
        ScalarType::F64,
        admission.operator_properties,
    )?;
    let kinematic_step = if lowered
        .regions
        .iter()
        .any(|region| !region.form.kinematics().is_empty())
    {
        let step = admission
            .temporal
            .ok_or_else(|| invalid("kinematic graph requires its explicit temporal Plan"))?
            .step();
        let mut states = Vec::new();
        for region in &lowered.regions {
            let selected = admission
                .discretizations
                .iter()
                .find(|binding| binding.domain() == region.domain_id())
                .ok_or_else(|| invalid("Region has no selected Field spaces"))?;
            let reference = if matches!(
                admission.resources(),
                NativeMeshResources::GmshSimplicial { .. }
            ) {
                eqiora_meshing::ReferenceCell::simplex(region.form.dimension())?
            } else {
                eqiora_meshing::ReferenceCell::hypercube(region.form.dimension())?
            };
            let form = region
                .form
                .bind_backward_euler(step)?
                .bind_field_spaces(reference, selected.field_spaces())?;
            states.extend(
                form.time_binding()
                    .expect("bound temporal form")
                    .states
                    .iter()
                    .copied(),
            );
        }
        Some(eqiora_realization::BackwardEulerStep::new(step, states)?)
    } else {
        None
    };
    PortableRealizationGraph::linear_regions(
        RealizationLineage::explicit(
            admission.program().model(),
            SemanticRevision::new(admission.program().revision().0),
            RealizationRevision::new(COMMON_SCALAR_REALIZATION_REVISION),
        ),
        admission.discretizations.clone(),
        lowered.quotients()?,
        kinematic_step.as_ref(),
        discretization,
        admission.operator_properties,
        ScalarType::F64,
        VectorLayoutKind::Replicated,
        solver,
        Target::HostCpu {
            threads: admission.linear.workers,
        },
        ExecutionSchedule::Offline,
    )
}
