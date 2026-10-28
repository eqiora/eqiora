//! Fixed-topology P1 harmonic mesh motion on affine simplices.
//!
//! The action is sealed against one immutable reference mesh and its exact
//! Domain partition. Exact displacement Fields drive selected Domains; their
//! conforming traces supply Dirichlet data for the harmonic Domains. Remaining
//! harmonic exterior vertices are fixed, and free vertices use componentwise
//! harmonic extension. The precomputed influence matrix is linear, so primal and JVP
//! evaluation necessarily use the same action.

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, Id, ScalarType, entity::kinds};
use eqiora_meshing::P1HarmonicCoordinateRelation;
use eqiora_meshing::{SimplicialMesh, VertexId};
use eqiora_realization::P1HarmonicMeshMotionPolicy;
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperatorProperties, LinearSolveRequest, SolveReport,
};
use std::collections::{BTreeMap, BTreeSet};

use crate::simplicial_fsi::FixedReferenceFsiPartition;

const RESIDUAL_ULPS: f64 = 16_384.0;
const MAX_DENSE_MOTION_COEFFICIENTS: usize = 8_000_000;

/// A sealed linear map from exact displacement Fields to ALE mesh motion.
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
    /// Driven vertices use their Domain's absolute displacement Field; shared
    /// driven vertices require equal values. Harmonic-only physical exterior
    /// vertices are fixed to zero. An exterior/interface vertex is driven by
    /// its displacement Field, giving it one coherent coordinate constraint.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the resolved backend implements the exact
    /// requested SPD/f64 policy tuple, `EQ0803` unless the mesh and its exact
    /// partition/Dirichlet closure define the harmonic coordinate relation,
    /// `EQ0801` when the bounded first-slice action cannot be materialized, or
    /// `EQ0802` when the common solver cannot accept an influence column.
    pub fn new(
        mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        policy: P1HarmonicMeshMotionPolicy,
        solver: LinearSolveRequest<'_>,
    ) -> Result<Self, Diagnostic> {
        let harmonic = policy
            .harmonic_domains()
            .iter()
            .map(|domain| domain.erase())
            .collect::<BTreeSet<_>>();
        let domains = harmonic
            .iter()
            .copied()
            .chain(policy.drivers().iter().map(|(domain, _)| domain.erase()))
            .collect::<BTreeSet<_>>();
        let interfaces = policy
            .interfaces()
            .iter()
            .copied()
            .map(eqiora_realization::ConformingTraceSource::ConservingConnection)
            .collect::<std::collections::HashSet<_>>();
        let available = partition
            .quotients()
            .filter(|quotient| {
                let [left, right] = quotient.endpoints();
                harmonic.contains(&left.domain().erase())
                    != harmonic.contains(&right.domain().erase())
            })
            .map(|quotient| quotient.source())
            .collect::<std::collections::HashSet<_>>();
        if solver.plan() != policy.solver()
            || domains != partition.domains().map(|domain| domain.erase()).collect()
            || interfaces != available
        {
            return Err(invalid(
                "harmonic motion requires its complete exact Domain and interface inventory",
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
            .filter(|trace| interfaces.contains(&trace.quotient.source()))
            .flat_map(|trace| {
                trace
                    .facets
                    .iter()
                    .map(|facet| eqiora_meshing::FacetId::new(facet.facet.index()))
            })
            .collect::<BTreeSet<_>>();
        let cells = |domains: Vec<Id<kinds::Domain>>| {
            let mut cells = domains
                .into_iter()
                .flat_map(|domain| {
                    partition
                        .domain_cells(domain)
                        .expect("validated Domain")
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            cells.sort_by_key(|cell| cell.index());
            cells
        };
        let relation = P1HarmonicCoordinateRelation::<D>::new(
            mesh,
            cells(policy.harmonic_domains().to_vec()),
            cells(policy.drivers().iter().map(|(domain, _)| *domain).collect()),
            facets.into_iter().collect(),
        )?;
        if relation.fluid_interior_vertices().is_empty() {
            return Err(invalid(
                "P1 harmonic ALE first slice requires at least one solved harmonic-interior vertex",
            ));
        }
        let interior_count = relation.fluid_interior_vertices().len();
        let driver_count = relation.driver_vertices().len();
        require_motion_coefficient_budget(interior_count, driver_count)?;
        let first_rhs = (0..interior_count)
            .map(|row| -relation.driver_stiffness()[row * driver_count])
            .collect::<Vec<_>>();
        let operator =
            capture_interior_stiffness(relation.interior_stiffness(), interior_count, &first_rhs)?;
        let mut influence = vec![0.0; interior_count * driver_count];
        let mut influence_solve_reports = Vec::with_capacity(driver_count);
        for driver in 0..driver_count {
            let rhs = (0..interior_count)
                .map(|row| -relation.driver_stiffness()[row * driver_count + driver])
                .collect::<Vec<_>>();
            let problem = operator.linear_problem_with_right_hand_side(&rhs)?;
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

    /// Interface vertices whose trace drives harmonic extension, in canonical order.
    #[must_use]
    pub fn driver_vertices(&self) -> &[VertexId] {
        self.relation.driver_vertices()
    }

    /// Harmonic-only physical-exterior vertices fixed to exact zero.
    #[must_use]
    pub fn fixed_exterior_vertices(&self) -> &[VertexId] {
        self.relation.fixed_exterior_vertices()
    }

    /// Free harmonic vertices solved by harmonic extension.
    #[must_use]
    pub fn harmonic_interior_vertices(&self) -> &[VertexId] {
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

    /// Apply the sealed map to the complete exact displacement Field inventory.
    /// Shared driven vertices must agree; harmonic exterior values remain zero.
    ///
    /// # Errors
    /// Rejects foreign/missing Fields or vertices, inconsistent shared values,
    /// nonfinite data, overflow, or a failed harmonic residual certificate.
    pub fn apply<'a>(
        &self,
        values: impl IntoIterator<Item = (Id<kinds::Field>, &'a BTreeMap<VertexId, [f64; D]>)>,
    ) -> Result<Vec<[f64; D]>, Diagnostic> {
        self.apply_linear(values)
    }

    /// Exact JVP of the same sealed action on the complete exact driver inventory.
    pub fn apply_jvp<'a>(
        &self,
        values: impl IntoIterator<Item = (Id<kinds::Field>, &'a BTreeMap<VertexId, [f64; D]>)>,
    ) -> Result<Vec<[f64; D]>, Diagnostic> {
        self.apply_linear(values)
    }

    /// Exact numerical motion policy.
    pub const fn policy(&self) -> &P1HarmonicMeshMotionPolicy {
        &self.policy
    }

    fn apply_linear<'a>(
        &self,
        values: impl IntoIterator<Item = (Id<kinds::Field>, &'a BTreeMap<VertexId, [f64; D]>)>,
    ) -> Result<Vec<[f64; D]>, Diagnostic> {
        let mut fields = BTreeMap::new();
        for (field, values) in values {
            if fields.insert(field.erase(), values).is_some() {
                return Err(invalid("harmonic motion repeats an exact driver Field"));
            }
        }
        if fields.keys().copied().collect::<BTreeSet<_>>()
            != self
                .policy
                .drivers()
                .iter()
                .map(|(_, field)| field.erase())
                .collect()
        {
            return Err(invalid(
                "harmonic motion requires every exact driver Field once",
            ));
        }
        let mut assigned = vec![None; self.relation.reference_mesh().vertices().len()];
        for &(domain, field) in self.policy.drivers() {
            let field_values = fields[&field.erase()];
            if !field_values.keys().copied().eq(self
                .partition
                .domain_vertices(domain)
                .expect("validated driven Domain")
                .iter()
                .copied())
                || field_values
                    .values()
                    .flatten()
                    .any(|value| !value.is_finite())
            {
                return Err(invalid(
                    "harmonic driver differs from exact Domain vertices or is nonfinite",
                ));
            }
            for (&vertex, &value) in field_values {
                match assigned[vertex.index()] {
                    Some(old) if old != value => {
                        return Err(invalid(
                            "harmonic driver Fields disagree at a shared driven vertex",
                        ));
                    }
                    Some(_) => {}
                    None => assigned[vertex.index()] = Some(value),
                }
            }
        }
        let driver_input = assigned
            .into_iter()
            .map(|value| value.unwrap_or([0.; D]))
            .collect::<Vec<_>>();
        let mut displacement = vec![[0.0; D]; driver_input.len()];
        for vertex in self.relation.solid_vertices() {
            displacement[vertex.index()] = driver_input[vertex.index()];
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
                            * driver_input[source.index()][component]
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
            &driver_input,
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

/// Bound matrix coefficient storage before CSR capture or influence allocation.
/// This is not a bound on indices, RHS/Krylov vectors, or total allocator bytes.
fn require_motion_coefficient_budget(
    interior_count: usize,
    driver_count: usize,
) -> Result<(), Diagnostic> {
    let square = interior_count
        .checked_mul(interior_count)
        .ok_or_else(|| invalid("P1 harmonic ALE dense reference width overflows usize"))?;
    let coupling = interior_count
        .checked_mul(driver_count)
        .ok_or_else(|| invalid("P1 harmonic ALE influence width overflows usize"))?;
    // The dense relation, finalized CSR, and canonical coefficient copy coexist
    // during capture (at most 3N²); validation has no coefficient workspace.
    // The relation's driver matrix and influence map need at most 2NM more.
    let coefficients = square
        .checked_mul(3)
        .and_then(|value| {
            coupling
                .checked_mul(2)
                .and_then(|coupling| value.checked_add(coupling))
        })
        .ok_or_else(|| invalid("P1 harmonic ALE dense reference storage overflows usize"))?;
    if coefficients > MAX_DENSE_MOTION_COEFFICIENTS {
        return Err(invalid(format!(
            "P1 harmonic ALE bounded reference action requires at most {MAX_DENSE_MOTION_COEFFICIENTS} matrix coefficients, got {coefficients}",
        )));
    }
    Ok(())
}

/// Capture the already assembled Laplacian once for every influence RHS.
/// All providers execute the same canonical coefficient action.
fn capture_interior_stiffness(
    matrix: &[f64],
    size: usize,
    right_hand_side: &[f64],
) -> Result<CanonicalCsrSystemView, Diagnostic> {
    validate_interior_stiffness(matrix, size)?;
    let mut offsets = Vec::with_capacity(size + 1);
    let nonzeros = matrix.iter().filter(|&&value| value != 0.).count();
    let mut columns = Vec::with_capacity(nonzeros);
    let mut values = Vec::with_capacity(nonzeros);
    offsets.push(0);
    for row in matrix.chunks_exact(size) {
        for (column, &value) in row.iter().enumerate() {
            if value != 0. {
                columns.push(column);
                values.push(value);
            }
        }
        offsets.push(values.len());
    }
    let storage =
        eqiora_assembly::CsrMatrix::from_sorted_csr(size, size, offsets, columns, values)?;
    CanonicalCsrSystemView::new(
        &storage,
        right_hand_side,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
}

fn validate_interior_stiffness(matrix: &[f64], size: usize) -> Result<(), Diagnostic> {
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
    Ok(())
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}

#[cfg(test)]
mod tests;
