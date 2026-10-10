//! Conservative radial cell balances; the zero-area center carries the declared zero flux.
use super::super::*;
use super::{CoordinateGrid, diffusion::Diffusion};
use eqiora_assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPlan, AssemblyTarget, DofId,
    IndexedAssemblyWork, LocalContribution, LocalUnknown, TargetAssemblyMap,
};

pub(super) fn execute(
    admission: &NativeNumericalAdmission,
    grid: &CoordinateGrid,
    equation: &Diffusion,
    backend: &dyn LinearSolverBackend,
) -> Result<CommonLinearRunOutput<f64>, Diagnostic> {
    let axis = grid.mesh.mesh().axis_coordinates(0).expect("radial axis");
    let count = axis.len() - 1;
    if count > 1_048_576 {
        return Err(invalid("radial conservation exceeds 1048576 cells"));
    }
    let centers = axis
        .windows(2)
        .map(|pair| pair[0] + (pair[1] - pair[0]) * 0.5)
        .collect::<Vec<_>>();
    let structure = eqiora_solver::AlgebraicStructure::new([equation.concentration], [])?;
    let backend = admission
        .linear
        .checked_backend(backend, Some(&structure))?;
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(count)?])?;
    let target = plan.target_id(0).expect("one radial concentration target");
    let work = IndexedAssemblyWork::new(2 * count, |index: usize| {
        let (dofs, matrix, rhs) = if index < count {
            let lower = axis[index];
            let upper = axis[index + 1];
            // The common 4*pi factor cancels from both sides of the conservation law.
            // Keep the radial r^2 measure, including the first cell at the center.
            let volume = (upper - lower) * (upper * upper + upper * lower + lower * lower) / 3.0;
            if !volume.is_finite() || volume <= 0.0 {
                return Err(invalid(
                    "radial assembly requires a positive finite radial cell measure",
                ));
            }
            (vec![index], vec![0.0], vec![equation.production * volume])
        } else if index < 2 * count - 1 {
            let left = index - count;
            let right = left + 1;
            let face = axis[right];
            let conductance = equation.diffusivity * face * face / (centers[right] - centers[left]);
            (
                vec![left, right],
                vec![conductance, -conductance, -conductance, conductance],
                vec![0.0, 0.0],
            )
        } else {
            let radius = axis[count];
            let conductance =
                equation.diffusivity * radius * radius / (radius - centers[count - 1]);
            (
                vec![count - 1],
                vec![conductance],
                vec![conductance * equation.surface],
            )
        };
        if matrix.iter().chain(&rhs).any(|value| !value.is_finite()) {
            return Err(invalid("radial cell assembly coefficients must be finite"));
        }
        let size = dofs.len();
        let unknowns = dofs
            .iter()
            .map(|index| LocalUnknown::Free(DofId::new(*index)))
            .collect();
        let rows = dofs
            .into_iter()
            .map(|index| Some(DofId::new(index)))
            .collect();
        AssemblyPacket::new(
            LocalContribution::new(size, size, matrix, rhs)?,
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(rows, unknowns)?,
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
    let mut faces = Vec::with_capacity(count + 1);
    faces.push(0.0); // The retained center condition, not an extrapolated cell value.
    for face in 1..count {
        faces.push(
            -equation.diffusivity * (values[face] - values[face - 1])
                / (centers[face] - centers[face - 1]),
        );
    }
    faces.push(
        -equation.diffusivity * (equation.surface - values[count - 1])
            / (axis[count] - centers[count - 1]),
    );
    let flux = faces
        .windows(2)
        .map(|pair| pair[0] * 0.5 + pair[1] * 0.5)
        .collect::<Vec<_>>();
    if flux.iter().any(|value| !value.is_finite()) {
        return Err(invalid("radial flux reconstruction must be finite"));
    }
    Ok(CommonLinearRunOutput {
        reactions: None,
        fields: vec![
            (
                equation.concentration,
                equation.concentration_type.clone(),
                values,
                eqiora_realization::Space::cell_constant(),
            ),
            (
                equation.flux,
                equation.flux_type.clone(),
                flux,
                eqiora_realization::Space::cell_constant(),
            ),
        ],
        nullspace: None,
        solve_report,
        assembly_report,
    })
}
