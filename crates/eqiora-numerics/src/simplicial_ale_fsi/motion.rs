//! Fixed-topology P1 harmonic mesh motion on affine simplices.
//!
//! The action is sealed against one immutable reference mesh and its exact
//! fluid/solid partition. Solid displacement is the only driver: its trace on
//! the conforming interface supplies Dirichlet data, the fluid exterior is
//! fixed, and the remaining fluid vertices are the componentwise harmonic
//! extension. The precomputed influence matrix is linear, so primal and JVP
//! evaluation necessarily use the same action.

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, Id, ScalarType, entity::kinds};
use eqiora_meshing::P1HarmonicCoordinateRelation;
use eqiora_meshing::{SimplicialMesh, VertexId};
use eqiora_realization::P1HarmonicMeshMotionPolicy;
use eqiora_solver::{
    DiagonalAvailability, LinearOperator, LinearOperatorProperties, LinearProblem,
    LinearSolveRequest, LinearSolver, SolveReport,
};
use std::collections::{BTreeMap, BTreeSet};

use crate::simplicial_fsi::FixedReferenceFsiPartition;

const RESIDUAL_ULPS: f64 = 16_384.0;
const MAX_DENSE_MOTION_COEFFICIENTS: usize = 8_000_000;

/// A sealed linear map from absolute solid displacement to ALE mesh motion.
///
/// This is deliberately a bounded CPU-reference realization. It owns an exact
/// clone of the admitted reference mesh and partition so the influence action
/// cannot silently be replayed against different topology, coordinates, or
/// material membership. No current coordinates or independently supplied mesh
/// velocity are part of the contract.
#[derive(Debug, Clone, PartialEq)]
pub struct P1HarmonicMeshMotionAction<const D: usize> {
    partition: FixedReferenceFsiPartition<D>,
    policy: P1HarmonicMeshMotionPolicy,
    relation: P1HarmonicCoordinateRelation<D>,
    influence: Vec<f64>,
    influence_solve_reports: Vec<SolveReport>,
}

impl<const D: usize> P1HarmonicMeshMotionAction<D> {
    /// Seal the unique P1 harmonic extension on one reference partition.
    ///
    /// Solid vertices are driven by their absolute displacement. Fluid-only
    /// vertices on the physical exterior are fixed to exact zero. A vertex
    /// that is both exterior and on the material interface is coherently owned
    /// by the solid/interface driver, rather than receiving two constraints.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the resolved backend implements the exact
    /// conjugate-gradient/SPD/f64 plan, `EQ0803` unless the mesh and its exact
    /// partition/Dirichlet closure define the harmonic coordinate relation,
    /// `EQ0801` when the bounded first-slice action cannot be materialized, or
    /// `EQ0802` when the common solver cannot accept an influence column.
    pub fn new(
        mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        policy: P1HarmonicMeshMotionPolicy,
        solver: LinearSolveRequest<'_>,
    ) -> Result<Self, Diagnostic> {
        if solver.plan() != policy.solver()
            || partition.domains().collect::<Vec<_>>().len() != 2
            || partition
                .domains()
                .any(|domain| domain != policy.fluid_domain() && domain != policy.solid_domain())
            || partition.quotients().any(|quotient| {
                quotient.source()
                    != eqiora_realization::ConformingTraceSource::ConservingConnection(
                        policy.interface(),
                    )
            })
            || partition.quotients().next().is_none()
        {
            return Err(invalid(
                "one harmonic motion policy requires its complete exact Domain/Connection inventory",
            ));
        }
        if solver.plan().algorithm() != LinearSolver::ConjugateGradient {
            return Err(invalid_realization(
                "P1 harmonic ALE mesh motion requires the resolved conjugate-gradient policy",
            ));
        }
        solver.backend().capabilities().require_problem(
            solver.plan(),
            eqiora_core::ScalarDomain::Real,
            ScalarType::F64,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )?;
        let replayed = FixedReferenceFsiPartition::<D>::new(
            mesh,
            partition.domains().map(|domain| {
                (
                    domain,
                    partition
                        .domain_cells(domain)
                        .expect("exact Domain")
                        .to_vec(),
                )
            }),
            &partition.quotients().collect::<Vec<_>>(),
        )?;
        if &replayed != partition {
            return Err(invalid(
                "P1 harmonic ALE partition cache differs from exact reference-mesh replay",
            ));
        }

        let facets = partition
            .traces()
            .iter()
            .flat_map(|trace| {
                trace
                    .facets
                    .iter()
                    .map(|facet| eqiora_meshing::FacetId::new(facet.facet.index()))
            })
            .collect::<BTreeSet<_>>();
        let relation = P1HarmonicCoordinateRelation::<D>::new(
            mesh,
            partition
                .domain_cells(policy.fluid_domain())
                .expect("exact motion Domain")
                .to_vec(),
            partition
                .domain_cells(policy.solid_domain())
                .expect("exact driver Domain")
                .to_vec(),
            facets.into_iter().collect(),
        )?;
        if relation.fluid_interior_vertices().is_empty() {
            return Err(invalid(
                "P1 harmonic ALE first slice requires at least one genuinely solved fluid-interior vertex",
            ));
        }
        let interior_count = relation.fluid_interior_vertices().len();
        let driver_count = relation.driver_vertices().len();
        let square = interior_count
            .checked_mul(interior_count)
            .ok_or_else(|| invalid("P1 harmonic ALE dense reference width overflows usize"))?;
        let coupling = interior_count
            .checked_mul(driver_count)
            .ok_or_else(|| invalid("P1 harmonic ALE influence width overflows usize"))?;
        let peak_coefficients = square
            .checked_mul(2)
            .and_then(|value| {
                coupling
                    .checked_mul(2)
                    .and_then(|coupling| value.checked_add(coupling))
            })
            .ok_or_else(|| invalid("P1 harmonic ALE dense reference storage overflows usize"))?;
        if peak_coefficients > MAX_DENSE_MOTION_COEFFICIENTS {
            return Err(invalid(format!(
                "P1 harmonic ALE bounded dense reference action requires at most {MAX_DENSE_MOTION_COEFFICIENTS} peak coefficients, got {peak_coefficients}",
            )));
        }
        let operator = DenseSpdOperator::new(relation.interior_stiffness(), interior_count)?;
        let mut influence = vec![0.0; interior_count * driver_count];
        let mut influence_solve_reports = Vec::with_capacity(driver_count);
        for driver in 0..driver_count {
            let rhs = (0..interior_count)
                .map(|row| -relation.driver_stiffness()[row * driver_count + driver])
                .collect::<Vec<_>>();
            let problem = LinearProblem::new(
                &operator,
                &rhs,
                LinearOperatorProperties::SymmetricPositiveDefinite,
            )?;
            let (column, report) = solver.solve(&problem)?.into_parts();
            for (row, value) in column.into_iter().enumerate() {
                influence[row * driver_count + driver] = value;
            }
            influence_solve_reports.push(report);
        }
        validate_influence_residual(
            relation.interior_stiffness(),
            relation.driver_stiffness(),
            &influence,
            interior_count,
            driver_count,
            &influence_solve_reports,
        )?;

        Ok(Self {
            partition: partition.clone(),
            policy,
            relation,
            influence,
            influence_solve_reports,
        })
    }

    /// Immutable reference mesh against which this action was sealed.
    #[must_use]
    pub const fn reference_mesh(&self) -> &SimplicialMesh {
        self.relation.reference_mesh()
    }

    /// Exact conforming material partition against which this action was sealed.
    #[must_use]
    pub const fn partition(&self) -> &FixedReferenceFsiPartition<D> {
        &self.partition
    }

    /// Interface vertices whose trace drives the fluid mesh, in canonical order.
    #[must_use]
    pub fn driver_vertices(&self) -> &[VertexId] {
        self.relation.driver_vertices()
    }

    /// Fluid-only physical-exterior vertices fixed to exact zero.
    #[must_use]
    pub fn fixed_exterior_vertices(&self) -> &[VertexId] {
        self.relation.fixed_exterior_vertices()
    }

    /// Genuine fluid-interior vertices solved by harmonic extension.
    #[must_use]
    pub fn fluid_interior_vertices(&self) -> &[VertexId] {
        self.relation.fluid_interior_vertices()
    }

    /// Accepted common-solver evidence for each interface influence column.
    ///
    /// Reports follow [`Self::driver_vertices`] order. Consequently the
    /// backend, plan, execution, and independent residual verification that
    /// produced the sealed map remain auditable without retaining a backend
    /// reference inside the action.
    #[must_use]
    pub fn influence_solve_reports(&self) -> &[SolveReport] {
        &self.influence_solve_reports
    }

    /// Fail closed if a caller attempts to reuse the action with another root.
    ///
    /// # Errors
    /// Returns `EQ0801` unless mesh coordinates, topology, quality policy, and
    /// the replayed exact partition all equal the sealed reference.
    pub fn validate_reference(
        &self,
        mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
    ) -> Result<(), Diagnostic> {
        let replayed = FixedReferenceFsiPartition::<D>::new(
            mesh,
            partition.domains().map(|domain| {
                (
                    domain,
                    partition
                        .domain_cells(domain)
                        .expect("exact Domain")
                        .to_vec(),
                )
            }),
            &partition.quotients().collect::<Vec<_>>(),
        )?;
        if mesh != self.relation.reference_mesh()
            || partition != &self.partition
            || replayed != *partition
        {
            return Err(invalid(
                "P1 harmonic ALE motion action cannot be replayed against a different reference root",
            ));
        }
        Ok(())
    }

    /// Apply the sealed map to one absolute solid displacement field.
    ///
    /// The input uses reference-mesh vertex order and must be exact zero outside
    /// the solid closure. The returned field covers every mesh vertex: solid
    /// values are copied exactly, fluid exterior values remain exact zero, and
    /// fluid-interior values satisfy the admitted P1 Laplace equations.
    ///
    /// # Errors
    /// Returns `EQ0801` for an incompatible field shape, or `EQ0803` for
    /// non-finite data, non-zero data outside the solid closure, overflow, or
    /// failure of the harmonic residual certificate.
    pub fn apply(
        &self,
        field: Id<kinds::Field>,
        values: &BTreeMap<VertexId, [f64; D]>,
    ) -> Result<Vec<[f64; D]>, Diagnostic> {
        self.apply_linear(field, values)
    }

    /// Exact JVP of the same sealed action on the complete exact driver inventory.
    pub fn apply_jvp(
        &self,
        field: Id<kinds::Field>,
        values: &BTreeMap<VertexId, [f64; D]>,
    ) -> Result<Vec<[f64; D]>, Diagnostic> {
        self.apply_linear(field, values)
    }

    /// Exact admitted motion policy; no geometry position identifies its driver.
    pub const fn policy(&self) -> P1HarmonicMeshMotionPolicy {
        self.policy
    }

    fn apply_linear(
        &self,
        field: Id<kinds::Field>,
        values: &BTreeMap<VertexId, [f64; D]>,
    ) -> Result<Vec<[f64; D]>, Diagnostic> {
        if field != self.policy.solid_displacement()
            || !values
                .keys()
                .copied()
                .eq(self.relation.solid_vertices().iter().copied())
            || values.values().flatten().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "harmonic driver differs from exact Field/vertex inventory or is nonfinite",
            ));
        }
        let mut solid_input = vec![[0.0; D]; self.relation.reference_mesh().vertices().len()];
        for (&vertex, &value) in values {
            solid_input[vertex.index()] = value;
        }
        let mut displacement = vec![[0.0; D]; solid_input.len()];
        for vertex in self.relation.solid_vertices() {
            displacement[vertex.index()] = solid_input[vertex.index()];
        }
        let driver_count = self.relation.driver_vertices().len();
        for (row, vertex) in self.relation.fluid_interior_vertices().iter().enumerate() {
            for component in 0..D {
                displacement[vertex.index()][component] = self
                    .relation
                    .driver_vertices()
                    .iter()
                    .enumerate()
                    .map(|(driver, source)| {
                        self.influence[row * driver_count + driver]
                            * solid_input[source.index()][component]
                    })
                    .sum();
            }
        }
        let current_coordinates = self
            .relation
            .reference_mesh()
            .vertices()
            .iter()
            .zip(&displacement)
            .map(|(reference, displacement)| {
                reference
                    .iter()
                    .zip(displacement)
                    .map(|(reference, displacement)| reference + displacement)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let residual_targets = self
            .influence_solve_reports
            .iter()
            .map(SolveReport::residual_target)
            .collect::<Vec<_>>();
        self.relation.validate_current_coordinates(
            &solid_input,
            &current_coordinates,
            &residual_targets,
        )?;
        Ok(displacement)
    }
}

fn validate_influence_residual(
    interior: &[f64],
    driver: &[f64],
    influence: &[f64],
    interior_count: usize,
    driver_count: usize,
    reports: &[SolveReport],
) -> Result<(), Diagnostic> {
    if reports.len() != driver_count {
        return Err(invalid(
            "P1 harmonic ALE influence evidence does not cover every driver column",
        ));
    }
    for column in 0..driver_count {
        let mut residual_norm = 0.0_f64;
        let mut rounding_norm = 0.0_f64;
        for row in 0..interior_count {
            let mut residual = driver[row * driver_count + column];
            let mut scale = residual.abs();
            for inner in 0..interior_count {
                let term = interior[row * interior_count + inner]
                    * influence[inner * driver_count + column];
                residual += term;
                scale += term.abs();
            }
            residual_norm = residual_norm.hypot(residual);
            rounding_norm =
                rounding_norm.hypot(RESIDUAL_ULPS * f64::EPSILON * scale.max(f64::MIN_POSITIVE));
        }
        let tolerance = reports[column].residual_target() + rounding_norm;
        if !residual_norm.is_finite() || !tolerance.is_finite() || residual_norm > tolerance {
            return Err(invalid(format!(
                "P1 harmonic ALE influence column {column} fails independent Laplace reapplication: {residual_norm:e} > {tolerance:e}",
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct DenseSpdOperator<'a> {
    matrix: &'a [f64],
    size: usize,
}

impl<'a> DenseSpdOperator<'a> {
    fn new(matrix: &'a [f64], size: usize) -> Result<Self, Diagnostic> {
        let square = size
            .checked_mul(size)
            .ok_or_else(|| invalid("P1 harmonic ALE dense operator width overflows usize"))?;
        if size == 0 || matrix.len() != square || matrix.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "P1 harmonic ALE Dirichlet Laplacian has an invalid dense reference layout",
            ));
        }
        for row in 0..size {
            if matrix[row * size + row] <= 0.0 {
                return Err(invalid(
                    "P1 harmonic ALE Dirichlet Laplacian has a non-positive diagonal",
                ));
            }
            for column in 0..row {
                if matrix[row * size + column] != matrix[column * size + row] {
                    return Err(invalid(
                        "P1 harmonic ALE Dirichlet Laplacian is not exactly symmetric",
                    ));
                }
            }
        }
        Ok(Self { matrix, size })
    }
}

impl LinearOperator for DenseSpdOperator<'_> {
    type Scalar = f64;

    fn rows(&self) -> usize {
        self.size
    }

    fn columns(&self) -> usize {
        self.size
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        if input.len() != self.size
            || output.len() != self.size
            || input.iter().any(|value| !value.is_finite())
        {
            return Err(solve_failed(
                "P1 harmonic ALE dense operator action has an invalid finite shape",
            ));
        }
        for (row, result) in output.iter_mut().enumerate() {
            *result = self.matrix[row * self.size..(row + 1) * self.size]
                .iter()
                .zip(input)
                .map(|(coefficient, value)| coefficient * value)
                .sum();
            if !result.is_finite() {
                return Err(solve_failed(
                    "P1 harmonic ALE dense operator action produced a non-finite value",
                ));
            }
        }
        Ok(())
    }

    fn diagonal(&self, output: &mut [f64]) -> Result<DiagonalAvailability, Diagnostic> {
        if output.len() != self.size {
            return Err(solve_failed(
                "P1 harmonic ALE dense operator diagonal has an invalid shape",
            ));
        }
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.matrix[row * self.size + row];
        }
        Ok(DiagonalAvailability::Available)
    }
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}

fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[cfg(test)]
mod tests;
