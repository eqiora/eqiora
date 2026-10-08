//! Coordinate cell balances share the ordinary scalar Plan, assembly, and solver owners.
use super::super::*;
use super::*;
use eqiora_assembly::DofId;
use eqiora_assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPlan, AssemblyTarget,
    IndexedAssemblyWork, LocalContribution, LocalUnknown, TargetAssemblyMap,
};
use eqiora_realization::{DomainFieldDiscretization, FieldSpaceBinding};

impl CommonScalarPlan {
    pub(in crate::numerical_admission) fn from_coordinate_admission(
        model: &ModelEnvelope,
        admission: NativeNumericalAdmission,
    ) -> Result<Self, Diagnostic> {
        let NativeMeshResources::Coordinates(grid) = admission.resources() else {
            return Err(invalid(
                "coordinate Plan requires an authenticated factor grid",
            ));
        };
        let RecognizedNativeModel::Coordinates(projection) = admission.recognized_model() else {
            return Err(invalid("coordinate Plan lost its cell-integrated equality"));
        };
        let cells = (0..grid.source.factors.len())
            .map(|axis| {
                grid.mesh
                    .mesh()
                    .axis_cell_count(axis)
                    .expect("authenticated grid axis")
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let fields = projection.fields().into_boxed_slice();
        let portable = portable(&admission, &cells)?;
        let realization_digest = hex_bytes(&portable.digest()?);
        let (digests, bytes) = static_plan_identity_lineage(&admission, &realization_digest)?;
        let reference = model.artifact_reference()?;
        let lineage = CommonSpatialPlanLineage::new(
            domain_separated_identity(b"eqiora.coordinate-cell-plan/v1\0", &bytes),
            reference.model().ulid().to_string(),
            reference.semantic_revision().get(),
            digests,
            realization_digest,
        );
        Ok(Self {
            admission,
            portable,
            formulation: None,
            authored_formulation: None,
            lineage,
            fields,
            cells,
        })
    }
}

pub(in crate::numerical_admission) fn portable(
    admission: &NativeNumericalAdmission,
    cells: &[usize],
) -> Result<PortableRealizationGraph, Diagnostic> {
    let NativeMeshResources::Coordinates(grid) = admission.resources() else {
        return Err(invalid(
            "coordinate portable realization requires its exact grid",
        ));
    };
    let RecognizedNativeModel::Coordinates(projection) = admission.recognized_model() else {
        return Err(invalid(
            "coordinate portable realization requires its exact Field",
        ));
    };
    if admission.spatial != NativeSpatialPolicy::CoordinateCellConstant
        || admission.temporal.is_some()
        || admission.nonlinear.is_some()
    {
        return Err(invalid(
            "coordinate cell projection requires a static linear cell-constant policy",
        ));
    }
    let artifact = grid.mesh.artifact_reference()?;
    let count = |value| NonZeroUsize::new(value).expect("authenticated nonzero cell count");
    let mesh = match cells {
        [x] => MeshPolicy::SuppliedCartesian1d {
            artifact,
            cells: [count(*x)],
        },
        [x, y] => MeshPolicy::SuppliedCartesian {
            artifact,
            cells: [count(*x), count(*y)],
        },
        [x, y, z] => MeshPolicy::SuppliedCartesian3d {
            artifact,
            cells: [count(*x), count(*y), count(*z)],
        },
        _ => {
            return Err(invalid(
                "coordinate cell projection requires one to three axes",
            ));
        }
    };
    PortableRealizationGraph::linear_regions(
        RealizationLineage::explicit(
            admission.program().model(),
            SemanticRevision::new(admission.program().revision().0),
            RealizationRevision::new(COMMON_SCALAR_REALIZATION_REVISION),
        ),
        [DomainFieldDiscretization::new(
            parse_domain(&grid.source.domain)?,
            projection
                .fields()
                .into_iter()
                .map(|(id, _)| FieldSpaceBinding::new(id, Space::cell_constant())),
            [],
        )?],
        [],
        Discretization::new(
            DiscretizationMethod::CellCenteredFiniteVolume,
            mesh,
            QuadraturePolicy::GaussLegendre {
                points_per_axis: count(2),
            },
        ),
        LinearOperatorProperties::SymmetricPositiveDefinite,
        ScalarType::F64,
        VectorLayoutKind::Replicated,
        admission.linear.solver,
        Target::HostCpu {
            threads: admission.linear.workers,
        },
        ExecutionSchedule::Offline,
    )
}

pub(in crate::numerical_admission) fn execute(
    admission: &NativeNumericalAdmission,
    projection: &CellEquations,
    backend: &dyn LinearSolverBackend,
) -> Result<CommonScalarRunOutput<f64>, Diagnostic> {
    let NativeMeshResources::Coordinates(grid) = admission.resources() else {
        return Err(invalid("coordinate cell execution requires its exact grid"));
    };
    let CellEquations::Projection(projection) = projection else {
        let CellEquations::Diffusion(diffusion) = projection else {
            unreachable!()
        };
        return super::diffusion_plan::execute(admission, grid, diffusion, backend);
    };
    let structure = eqiora_solver::AlgebraicStructure::new([projection.field], [])?;
    let backend = admission
        .linear
        .checked_backend(backend, Some(&structure))?;
    let averages = projection.cell_values(grid)?;
    // Divide each cell-integrated equality by its positive measure. This diagonal
    // normalization preserves the equation and never interprets velocity as a length.
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(averages.len())?])?;
    let target = plan.target_id(0).expect("one cell-balance target");
    let work = IndexedAssemblyWork::new(averages.len(), |cell: usize| {
        let dof = DofId::new(cell);
        AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![1.0], vec![averages[cell]])?,
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(vec![Some(dof)], vec![LocalUnknown::Free(dof)])?,
            )],
        )
    });
    let (systems, assembly_report) = REFERENCE_ASSEMBLY_BACKEND
        .assemble(&plan, &work)?
        .into_parts();
    let canonical = Arc::new(eqiora_solver::CanonicalCsrSystemView::new(
        &systems[0],
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )?);
    let core = crate::finalized_spatial::FinalizedLinearCore::new(
        admission.linear.solver,
        VectorLayoutKind::Replicated,
        Target::HostCpu {
            threads: admission.linear.workers,
        },
        canonical,
    );
    let solution = LinearSolveRequest::new(&backend, admission.linear.solver)
        .solve(&core.linear_problem()?)?;
    core.validate_solution(&solution)?;
    let (values, solve_report) = solution.into_parts();
    Ok(CommonScalarRunOutput {
        fields: vec![(projection.field, projection.value_type.clone(), values)],
        nullspace: None,
        solve_report,
        assembly_report,
    })
}
