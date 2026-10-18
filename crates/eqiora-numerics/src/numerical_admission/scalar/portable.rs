//! One exact numerical graph derived from retained spatial admission resources.
use super::*;

pub(super) fn resolve_common_spatial_portable<S: crate::spatial_expression::Coefficient>(
    admission: &NativeNumericalAdmission,
    lowered: &ExecutableSpatialEquations<S>,
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
                _ => return Err(invalid("spatial Plan requires one to three Cartesian axes")),
            }
        }
        NativeMeshResources::GmshSimplicial { mesh, .. }
        | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => {
            MeshPolicy::ImportedSimplicial {
                artifact: mesh.artifact_reference()?,
            }
        }
        _ => {
            return Err(invalid(
                "spatial Plan has no admitted finite-dimensional Mesh",
            ));
        }
    };
    let (method, quadrature) = match &admission.spatial {
        NativeSpatialPolicy::FiniteElement(_) => {
            let dimension = admission.resources().geometry()?.ambient_dimension();
            let simplex = matches!(
                admission.resources(),
                NativeMeshResources::GmshSimplicial { .. }
                    | NativeMeshResources::AffineTriangleSimplicial { .. }
            );
            let reference = if simplex {
                eqiora_meshing::ReferenceCell::simplex(dimension)?
            } else {
                eqiora_meshing::ReferenceCell::hypercube(dimension)?
            };
            let mut equations = lowered.clone();
            if let Some(temporal) = admission.temporal {
                for region in &mut equations.regions {
                    region.form = region.form.bind_backward_euler(temporal.step())?;
                }
            }
            let forms = equations.bind_spaces(
                reference,
                &admission.discretizations,
                &admission.system_scaling,
            )?;
            let points_per_axis = NonZeroUsize::new(regions::quadrature_points(
                reference,
                &admission.discretizations,
                &forms,
            )?)
            .expect("positive Gauss point count");
            let quadrature = if simplex {
                QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension: NonZeroUsize::new(dimension).unwrap(),
                    points_per_axis,
                }
            } else {
                QuadraturePolicy::GaussLegendre { points_per_axis }
            };
            (DiscretizationMethod::ContinuousGalerkin, quadrature)
        }
        NativeSpatialPolicy::ScalarTpfa(_) => (
            DiscretizationMethod::CellCenteredFiniteVolume,
            QuadraturePolicy::CellCentroid,
        ),
        _ => {
            return Err(invalid(
                "common spatial portable graph received a foreign spatial policy",
            ));
        }
    };
    let discretization = Discretization::new(method, mesh, quadrature);
    validate_resources(&admission.spatial, admission.resources())?;
    let solver = admission.linear.solver;
    admission.linear.capabilities.require_problem(
        solver,
        admission.linear.scalar_domain,
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
                    | NativeMeshResources::AffineTriangleSimplicial { .. }
            ) {
                eqiora_meshing::ReferenceCell::simplex(region.form.dimension())?
            } else {
                eqiora_meshing::ReferenceCell::hypercube(region.form.dimension())?
            };
            let form = region.form.bind_backward_euler(step)?.bind_field_spaces(
                reference,
                selected.field_spaces(),
                &admission.system_scaling,
            )?;
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
    PortableRealizationGraph::spatial_regions(
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
        admission.nonlinear,
        Target::HostCpu {
            threads: admission.linear.workers,
        },
        ExecutionSchedule::Offline,
        admission.system_scaling.clone(),
    )
}
