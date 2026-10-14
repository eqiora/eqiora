//! Fixed-topology affine-simplex geometry states and consecutive actions.
//!
//! The bounded contract admits intrinsic dimensions two and three. A state owns only
//! coordinates and recomputed quality evidence; connectivity always comes
//! from one separately supplied immutable [`SimplicialMesh`].  A consecutive
//! action derives velocity and every endpoint metric quantity from two
//! accepted states, so callers cannot pair coordinates with an independently
//! authored mesh velocity or geometric correction.

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::{
    AffineGeometryLinearization, AffineGeometryMap, GeometryMap, MeshEntity, MeshGeometry,
    MeshQualityReport, MeshTopology, SimplicialMesh,
};

/// Absolute coordinates over one immutable dimension-typed simplex topology.
///
/// This value contains no connectivity.  Construction and every later replay
/// rebuild the geometry with the exact ordered cells and quality policy of the
/// supplied reference mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedTopologyGeometryState<const D: usize> {
    coordinates: Vec<Vec<f64>>,
    quality_report: MeshQualityReport,
}

impl<const D: usize> FixedTopologyGeometryState<D> {
    /// Validate absolute coordinates against one immutable reference mesh.
    ///
    /// # Errors
    /// Returns `EQ0803` unless the reference is intrinsic `D` and the supplied
    /// coordinates rebuild its exact ordered topology with positive
    /// orientation and the reference quality gate.
    pub fn new(reference: &SimplicialMesh, coordinates: Vec<Vec<f64>>) -> Result<Self, Diagnostic> {
        let mesh = rebuild::<D>(reference, &coordinates)?;
        Ok(Self {
            coordinates,
            quality_report: mesh.quality_report(),
        })
    }

    /// The reference mesh's own accepted coordinate state.
    ///
    /// # Errors
    /// Returns `EQ0803` unless `reference` is intrinsic `D`.
    pub fn reference(reference: &SimplicialMesh) -> Result<Self, Diagnostic> {
        Self::new(reference, reference.vertices().to_vec())
    }

    /// Absolute coordinates in immutable reference-vertex order.
    #[must_use]
    pub fn coordinates(&self) -> &[Vec<f64>] {
        &self.coordinates
    }

    /// Quality evidence recomputed using the reference mesh's policy.
    #[must_use]
    pub const fn quality_report(&self) -> MeshQualityReport {
        self.quality_report
    }

    /// Rebuild the accepted current mesh from immutable reference topology.
    ///
    /// This is a validation operation, not a new mesh identity: no
    /// connectivity is accepted from this state.
    ///
    /// # Errors
    /// Preserves the constructor's dimension, orientation, and quality checks.
    pub fn reconstruct_mesh(
        &self,
        reference: &SimplicialMesh,
    ) -> Result<SimplicialMesh, Diagnostic> {
        let mesh = rebuild::<D>(reference, &self.coordinates)?;
        if mesh.quality_report() != self.quality_report {
            return Err(invalid(
                "fixed-topology geometry-state quality evidence differs on replay",
            ));
        }
        Ok(mesh)
    }
}

/// One affine cell's derived endpoint geometry and mesh-motion action.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedTopologyCellGeometryAction<const D: usize> {
    cell: MeshEntity,
    previous_map: AffineGeometryMap,
    current_map: AffineGeometryMap,
    velocity_origin: [f64; D],
    reference_velocity_gradient: Vec<f64>,
    current_velocity_gradient: Vec<f64>,
    current_velocity_divergence: f64,
    endpoint_metric_rate: f64,
    metric_identity_defect: f64,
    minimum_path_signed_measure_scale: f64,
}

impl<const D: usize> FixedTopologyCellGeometryAction<D> {
    /// Exact immutable-topology cell owning this action.
    #[must_use]
    pub const fn cell(&self) -> MeshEntity {
        self.cell
    }

    /// Previous accepted affine cell map.
    #[must_use]
    pub const fn previous_map(&self) -> &AffineGeometryMap {
        &self.previous_map
    }

    /// Current accepted affine cell map used by endpoint weak forms.
    #[must_use]
    pub const fn current_map(&self) -> &AffineGeometryMap {
        &self.current_map
    }

    /// Mesh velocity at a point of the canonical reference simplex.
    ///
    /// # Errors
    /// Returns `EQ0803` unless `reference` is a finite point in that simplex.
    pub fn mesh_velocity(&self, reference: &[f64]) -> Result<[f64; D], Diagnostic> {
        if !self.current_map.reference_cell().contains(reference) {
            return Err(invalid(
                "fixed-topology mesh velocity requires a point in the reference simplex",
            ));
        }
        let mut velocity = self.velocity_origin;
        for (row, velocity) in velocity.iter_mut().enumerate() {
            for (column, coordinate) in reference.iter().enumerate() {
                *velocity += self.reference_velocity_gradient[row * D + column] * coordinate;
            }
        }
        if velocity.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "fixed-topology mesh-velocity evaluation is non-finite",
            ));
        }
        Ok(velocity)
    }

    /// Row-major `grad_X(w)` on the immutable reference cell.
    #[must_use]
    pub fn reference_velocity_gradient(&self) -> &[f64] {
        &self.reference_velocity_gradient
    }

    /// Row-major `grad_x(w) = grad_X(w) F_1^-1` on the current cell.
    #[must_use]
    pub fn current_velocity_gradient(&self) -> &[f64] {
        &self.current_velocity_gradient
    }

    /// Current spatial mesh-velocity divergence.
    #[must_use]
    pub const fn current_velocity_divergence(&self) -> f64 {
        self.current_velocity_divergence
    }

    /// Derived coefficient `0.5 * div_x(w)` paired with ALE skew convection.
    #[must_use]
    pub const fn skew_gcl_correction(&self) -> f64 {
        0.5 * self.current_velocity_divergence
    }

    /// Endpoint derivative of the signed affine metric `J` along this action.
    #[must_use]
    pub const fn endpoint_metric_rate(&self) -> f64 {
        self.endpoint_metric_rate
    }

    /// Independently evaluated `dJ/dt - J div_x(w)` defect.
    #[must_use]
    pub const fn metric_identity_defect(&self) -> f64 {
        self.metric_identity_defect
    }

    /// Minimum signed Jacobian over the complete linear path.
    #[must_use]
    pub const fn minimum_path_signed_measure_scale(&self) -> f64 {
        self.minimum_path_signed_measure_scale
    }
}

/// Sealed consecutive action over one immutable dimension-typed simplex topology.
///
/// Vertex velocities and all cell-local metric data are derived from the two
/// states and the positive duration.  There is intentionally no constructor
/// parameter for connectivity, velocity, velocity divergence, or a GCL term.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedTopologyGeometryAction<const D: usize> {
    previous: FixedTopologyGeometryState<D>,
    current: FixedTopologyGeometryState<D>,
    current_mesh: SimplicialMesh,
    time_step: f64,
    vertex_velocities: Vec<Vec<f64>>,
    cells: Vec<FixedTopologyCellGeometryAction<D>>,
    minimum_path_signed_measure_scale: f64,
}

impl<const D: usize> FixedTopologyGeometryAction<D> {
    /// Derive one endpoint action from consecutive accepted coordinate states.
    ///
    /// The linear path is checked analytically. Cell determinants are
    /// quadratic in 2D and cubic in 3D, so every interior stationary point is
    /// included rather than trusting the two accepted endpoints.
    ///
    /// # Errors
    /// Returns `EQ0803` for an invalid duration, stale state, failed endpoint
    /// metric identity, or a degenerate/inverting geometry path.
    pub fn new(
        reference: &SimplicialMesh,
        previous: &FixedTopologyGeometryState<D>,
        current: &FixedTopologyGeometryState<D>,
        time_step: f64,
    ) -> Result<Self, Diagnostic> {
        if !time_step.is_finite() || time_step <= 0.0 {
            return Err(invalid(
                "fixed-topology geometry action requires a finite positive time step",
            ));
        }
        let previous_mesh = previous.reconstruct_mesh(reference)?;
        let current_mesh = current.reconstruct_mesh(reference)?;
        let vertex_velocities = previous
            .coordinates()
            .iter()
            .zip(current.coordinates())
            .map(|(previous, current)| {
                previous
                    .iter()
                    .zip(current)
                    .map(|(previous, current)| (current - previous) / time_step)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if vertex_velocities
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "fixed-topology geometry action produced a non-finite mesh velocity",
            ));
        }

        let cell_count = reference
            .entity_count(D)
            .expect("accepted simplex mesh owns cells");
        let mut cells = Vec::new();
        cells
            .try_reserve_exact(cell_count)
            .map_err(|_| invalid("fixed-topology cell-action allocation failed"))?;
        let mut minimum_path_signed_measure_scale = f64::INFINITY;
        for cell_index in 0..cell_count {
            let cell = MeshEntity::new(D, cell_index);
            let previous_map = previous_mesh
                .geometry_map(cell)
                .ok_or_else(|| invalid("previous fixed-topology cell map is unavailable"))?;
            let current_map = current_mesh
                .geometry_map(cell)
                .ok_or_else(|| invalid("current fixed-topology cell map is unavailable"))?;
            let linearized = current_mesh.linearized_geometry_map(cell, &vertex_velocities)?;
            let cell_action = cell_action::<D>(cell, previous_map, current_map, linearized)?;
            minimum_path_signed_measure_scale = minimum_path_signed_measure_scale
                .min(cell_action.minimum_path_signed_measure_scale());
            cells.push(cell_action);
        }
        Ok(Self {
            previous: previous.clone(),
            current: current.clone(),
            current_mesh,
            time_step,
            vertex_velocities,
            cells,
            minimum_path_signed_measure_scale,
        })
    }

    /// Previous accepted absolute-coordinate state.
    #[must_use]
    pub const fn previous(&self) -> &FixedTopologyGeometryState<D> {
        &self.previous
    }

    /// Current accepted absolute-coordinate state.
    #[must_use]
    pub const fn current(&self) -> &FixedTopologyGeometryState<D> {
        &self.current
    }

    /// Current mesh reconstructed solely from immutable reference topology.
    #[must_use]
    pub const fn current_mesh(&self) -> &SimplicialMesh {
        &self.current_mesh
    }

    /// Positive interval between the consecutive states.
    #[must_use]
    pub const fn time_step(&self) -> f64 {
        self.time_step
    }

    /// Derived vertex velocities in immutable reference order.
    #[must_use]
    pub fn vertex_velocities(&self) -> &[Vec<f64>] {
        &self.vertex_velocities
    }

    /// Derived action for one valid cell index.
    #[must_use]
    pub fn cell(&self, cell: usize) -> Option<&FixedTopologyCellGeometryAction<D>> {
        self.cells.get(cell)
    }

    /// All derived cell actions in immutable reference-cell order.
    #[must_use]
    pub fn cells(&self) -> &[FixedTopologyCellGeometryAction<D>] {
        &self.cells
    }

    /// Minimum signed Jacobian over every cell and the entire linear path.
    #[must_use]
    pub const fn minimum_path_signed_measure_scale(&self) -> f64 {
        self.minimum_path_signed_measure_scale
    }
}

fn rebuild<const D: usize>(
    reference: &SimplicialMesh,
    coordinates: &[Vec<f64>],
) -> Result<SimplicialMesh, Diagnostic> {
    require_supported_dimension::<D>()?;
    if reference.topological_dimension() != D {
        return Err(invalid(format!(
            "fixed-topology geometry state requires an intrinsic {D}D simplex reference",
        )));
    }
    if coordinates.len() != reference.vertices().len()
        || coordinates.iter().any(|coordinate| {
            coordinate.len() != D || coordinate.iter().any(|value| !value.is_finite())
        })
    {
        return Err(invalid(format!(
            "fixed-topology geometry state requires {} finite {D}D coordinates",
            reference.vertices().len(),
        )));
    }
    SimplicialMesh::new(
        D,
        coordinates.to_vec(),
        reference.cells().to_vec(),
        reference.quality_gate(),
    )
}

fn cell_action<const D: usize>(
    cell: MeshEntity,
    previous_map: AffineGeometryMap,
    current_map: AffineGeometryMap,
    linearized: AffineGeometryLinearization,
) -> Result<FixedTopologyCellGeometryAction<D>, Diagnostic> {
    let velocity_origin: [f64; D] = linearized
        .origin_tangent()
        .try_into()
        .map_err(|_| invalid("fixed-topology velocity origin has the wrong dimension"))?;
    let reference_velocity_gradient = linearized.jacobian_tangent().to_vec();
    if reference_velocity_gradient.len() != D * D {
        return Err(invalid(
            "fixed-topology velocity gradient has the wrong dimension",
        ));
    }
    let inverse = current_map.inverse_jacobian()?;
    let current_velocity_gradient = (0..D * D)
        .map(|entry| {
            let row = entry / D;
            let column = entry % D;
            (0..D)
                .map(|axis| {
                    reference_velocity_gradient[row * D + axis] * inverse[axis * D + column]
                })
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let current_velocity_divergence = (0..D)
        .map(|axis| current_velocity_gradient[axis * D + axis])
        .sum::<f64>();
    let endpoint_metric_rate = linearized.measure_scale_tangent();
    let metric_identity_defect =
        endpoint_metric_rate - current_map.measure_scale() * current_velocity_divergence;
    let metric_scale = endpoint_metric_rate
        .abs()
        .max((current_map.measure_scale() * current_velocity_divergence).abs())
        .max(current_map.measure_scale())
        .max(f64::MIN_POSITIVE);
    let metric_tolerance = 4_096.0 * f64::EPSILON * metric_scale;
    if !current_velocity_divergence.is_finite()
        || current_velocity_gradient
            .iter()
            .any(|value| !value.is_finite())
        || !endpoint_metric_rate.is_finite()
        || !metric_identity_defect.is_finite()
        || metric_identity_defect.abs() > metric_tolerance
    {
        return Err(invalid(
            "fixed-topology endpoint affine metric identity is not satisfied",
        ));
    }
    let minimum_path_signed_measure_scale =
        minimum_path_determinant::<D>(previous_map.jacobian(), current_map.jacobian())?;
    Ok(FixedTopologyCellGeometryAction {
        cell,
        previous_map,
        current_map,
        velocity_origin,
        reference_velocity_gradient,
        current_velocity_gradient,
        current_velocity_divergence,
        endpoint_metric_rate,
        metric_identity_defect,
        minimum_path_signed_measure_scale,
    })
}

fn minimum_path_determinant<const D: usize>(
    previous: &[f64],
    current: &[f64],
) -> Result<f64, Diagnostic> {
    require_supported_dimension::<D>()?;
    if previous.len() != D * D || current.len() != D * D {
        return Err(invalid(
            "fixed-topology affine Jacobian has the wrong path dimension",
        ));
    }
    let delta = previous
        .iter()
        .zip(current)
        .map(|(previous, current)| current - previous)
        .collect::<Vec<_>>();
    let coefficients = if D == 2 {
        let constant = determinant_2d(previous);
        let linear = delta[0] * previous[3] + previous[0] * delta[3]
            - delta[1] * previous[2]
            - previous[1] * delta[2];
        [constant, linear, determinant_2d(&delta), 0.0]
    } else {
        determinant_path_coefficients_3d(previous, &delta)
    };
    let endpoint = evaluate_cubic(&coefficients, 1.0);
    let mut minimum = coefficients[0].min(endpoint);
    for stationary in stationary_points(&coefficients) {
        if stationary > 0.0 && stationary < 1.0 {
            minimum = minimum.min(evaluate_cubic(&coefficients, stationary));
        }
    }
    let polynomial_scale = coefficients
        .iter()
        .map(|coefficient| coefficient.abs())
        .sum::<f64>()
        .max(f64::MIN_POSITIVE);
    let tolerance = 512.0 * f64::EPSILON * polynomial_scale;
    if !minimum.is_finite() || minimum <= tolerance {
        return Err(invalid(
            "fixed-topology linear geometry path is degenerate or changes orientation",
        ));
    }
    Ok(minimum)
}

fn determinant_path_coefficients_3d(previous: &[f64], delta: &[f64]) -> [f64; 4] {
    let column =
        |matrix: &[f64], index: usize| [matrix[index], matrix[3 + index], matrix[6 + index]];
    let a = [
        column(previous, 0),
        column(previous, 1),
        column(previous, 2),
    ];
    let b = [column(delta, 0), column(delta, 1), column(delta, 2)];
    [
        determinant_columns_3d(a[0], a[1], a[2]),
        determinant_columns_3d(b[0], a[1], a[2])
            + determinant_columns_3d(a[0], b[1], a[2])
            + determinant_columns_3d(a[0], a[1], b[2]),
        determinant_columns_3d(b[0], b[1], a[2])
            + determinant_columns_3d(b[0], a[1], b[2])
            + determinant_columns_3d(a[0], b[1], b[2]),
        determinant_columns_3d(b[0], b[1], b[2]),
    ]
}

fn stationary_points(coefficients: &[f64; 4]) -> Vec<f64> {
    let a = 3.0 * coefficients[3];
    let b = 2.0 * coefficients[2];
    let c = coefficients[1];
    if a == 0.0 {
        return if b == 0.0 { Vec::new() } else { vec![-c / b] };
    }
    let discriminant = b.mul_add(b, -4.0 * a * c);
    let discriminant_scale = b.abs().mul_add(b.abs(), (4.0 * a * c).abs());
    let tolerance = 64.0 * f64::EPSILON * discriminant_scale.max(f64::MIN_POSITIVE);
    if discriminant < -tolerance {
        return Vec::new();
    }
    let root = discriminant.max(0.0).sqrt();
    let q = -0.5 * (b + root.copysign(b));
    if q == 0.0 {
        vec![-b / (2.0 * a)]
    } else {
        vec![q / a, c / q]
    }
}

fn evaluate_cubic(coefficients: &[f64; 4], value: f64) -> f64 {
    coefficients[0]
        + value * (coefficients[1] + value * (coefficients[2] + value * coefficients[3]))
}

fn determinant_2d(matrix: &[f64]) -> f64 {
    matrix[0] * matrix[3] - matrix[1] * matrix[2]
}

fn determinant_columns_3d(first: [f64; 3], second: [f64; 3], third: [f64; 3]) -> f64 {
    first[0] * (second[1] * third[2] - second[2] * third[1])
        - second[0] * (first[1] * third[2] - first[2] * third[1])
        + third[0] * (first[1] * second[2] - first[2] * second[1])
}

fn require_supported_dimension<const D: usize>() -> Result<(), Diagnostic> {
    if matches!(D, 2 | 3) {
        Ok(())
    } else {
        Err(invalid(
            "fixed-topology geometry actions admit dimensions two and three",
        ))
    }
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_MESH, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshQualityGate;

    fn gate() -> MeshQualityGate {
        MeshQualityGate::new(0.05).unwrap()
    }

    fn reference_triangle() -> SimplicialMesh {
        SimplicialMesh::new(
            2,
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![0, 1, 2]],
            gate(),
        )
        .unwrap()
    }

    fn reference_tetrahedron() -> SimplicialMesh {
        SimplicialMesh::new(
            3,
            vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            vec![vec![0, 1, 2, 3]],
            MeshQualityGate::new(0.01).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn derives_one_coherent_endpoint_motion_and_metric_identity() {
        let reference = reference_triangle();
        let previous = FixedTopologyGeometryState::<2>::reference(&reference).unwrap();
        let current = FixedTopologyGeometryState::<2>::new(
            &reference,
            vec![vec![0.1, -0.1], vec![1.3, 0.0], vec![0.2, 1.1]],
        )
        .unwrap();
        let action =
            FixedTopologyGeometryAction::<2>::new(&reference, &previous, &current, 0.5).unwrap();
        let cell = action.cell(0).unwrap();

        assert_eq!(action.vertex_velocities()[0], [0.2, -0.2]);
        assert_eq!(cell.mesh_velocity(&[0.0, 0.0]).unwrap(), [0.2, -0.2]);
        let endpoint_velocity = cell.mesh_velocity(&[1.0, 0.0]).unwrap();
        assert!((endpoint_velocity[0] - 0.6).abs() < 1.0e-15);
        assert_eq!(endpoint_velocity[1], 0.0);
        assert!(cell.current_velocity_divergence().abs() > 1.0e-12);
        assert!(cell.metric_identity_defect().abs() < 1.0e-14);
        assert!(cell.endpoint_metric_rate().is_finite());
        assert!(cell.minimum_path_signed_measure_scale() > 0.0);

        // These are the two geometry densities paired by the differential
        // ALE free-stream proof.  The correction is derived from this action,
        // not accepted as an independent compensating value.
        let constant_state_mesh_skew = -0.5 * cell.current_velocity_divergence();
        assert_eq!(constant_state_mesh_skew + cell.skew_gcl_correction(), 0.0);
    }

    #[test]
    fn zero_motion_is_exact_and_reconstructs_reference_topology() {
        let reference = reference_triangle();
        let state = FixedTopologyGeometryState::<2>::reference(&reference).unwrap();
        let action =
            FixedTopologyGeometryAction::<2>::new(&reference, &state, &state, 0.25).unwrap();
        assert_eq!(action.current_mesh(), &reference);
        assert!(
            action
                .vertex_velocities()
                .iter()
                .flatten()
                .all(|value| *value == 0.0)
        );
        let cell = action.cell(0).unwrap();
        assert_eq!(cell.reference_velocity_gradient(), &[0.0; 4]);
        assert_eq!(cell.current_velocity_gradient(), &[0.0; 4]);
        assert_eq!(cell.current_velocity_divergence(), 0.0);
        assert_eq!(cell.skew_gcl_correction(), 0.0);
        assert_eq!(cell.endpoint_metric_rate(), 0.0);
        assert_eq!(cell.metric_identity_defect(), 0.0);
    }

    #[test]
    fn rejects_path_collapse_even_when_both_endpoints_are_positive() {
        let reference = reference_triangle();
        let previous = FixedTopologyGeometryState::<2>::reference(&reference).unwrap();
        let rotated = FixedTopologyGeometryState::<2>::new(
            &reference,
            vec![vec![0.0, 0.0], vec![-1.0, 0.0], vec![0.0, -1.0]],
        )
        .unwrap();
        assert!(previous.quality_report().minimum_signed_measure_scale() > 0.0);
        assert!(rotated.quality_report().minimum_signed_measure_scale() > 0.0);
        let error = FixedTopologyGeometryAction::<2>::new(&reference, &previous, &rotated, 1.0)
            .unwrap_err();
        assert!(error.message().contains("path is degenerate"));
    }

    #[test]
    fn tetrahedral_action_derives_metric_identity_and_complete_velocity() {
        let reference = reference_tetrahedron();
        let previous = FixedTopologyGeometryState::<3>::reference(&reference).unwrap();
        let current = FixedTopologyGeometryState::<3>::new(
            &reference,
            vec![
                vec![0.1, -0.1, 0.2],
                vec![1.3, -0.1, 0.2],
                vec![0.1, 1.0, 0.2],
                vec![0.1, -0.1, 1.1],
            ],
        )
        .unwrap();
        let action =
            FixedTopologyGeometryAction::<3>::new(&reference, &previous, &current, 0.5).unwrap();
        let cell = action.cell(0).unwrap();

        assert_eq!(action.vertex_velocities()[0], [0.2, -0.2, 0.4]);
        assert_eq!(cell.mesh_velocity(&[0.0; 3]).unwrap(), [0.2, -0.2, 0.4]);
        assert_eq!(cell.reference_velocity_gradient().len(), 9);
        assert_eq!(cell.current_velocity_gradient().len(), 9);
        assert!(cell.current_velocity_divergence().abs() > 1.0e-12);
        assert!(cell.metric_identity_defect().abs() < 1.0e-13);
        assert!(cell.minimum_path_signed_measure_scale() > 0.0);
    }

    #[test]
    fn tetrahedral_path_rejects_an_interior_inversion_with_positive_endpoints() {
        let reference = reference_tetrahedron();
        let previous = FixedTopologyGeometryState::<3>::new(
            &reference,
            vec![
                vec![0.0, 0.0, 0.0],
                vec![-0.2, 0.0, 0.0],
                vec![0.0, -0.7, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        )
        .unwrap();
        let current = FixedTopologyGeometryState::<3>::new(
            &reference,
            vec![
                vec![0.0, 0.0, 0.0],
                vec![0.8, 0.0, 0.0],
                vec![0.0, 0.3, 0.0],
                vec![0.0, 0.0, 2.0],
            ],
        )
        .unwrap();
        assert!(previous.quality_report().minimum_signed_measure_scale() > 0.0);
        assert!(current.quality_report().minimum_signed_measure_scale() > 0.0);

        let error = FixedTopologyGeometryAction::<3>::new(&reference, &previous, &current, 1.0)
            .unwrap_err();
        assert!(error.message().contains("path is degenerate"));
    }

    #[test]
    fn rejects_invalid_states_and_durations_without_accepting_connectivity_or_velocity() {
        let reference = reference_triangle();
        let wrong_count =
            FixedTopologyGeometryState::<2>::new(&reference, vec![vec![0.0, 0.0], vec![1.0, 0.0]])
                .unwrap_err();
        assert!(wrong_count.message().contains("3 finite 2D coordinates"));

        let inverted = FixedTopologyGeometryState::<2>::new(
            &reference,
            vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]],
        )
        .unwrap_err();
        assert_eq!(inverted.code(), codes::INVALID_MESH);

        let low_quality_reference = SimplicialMesh::new(
            2,
            reference.vertices().to_vec(),
            reference.cells().to_vec(),
            MeshQualityGate::new(0.5).unwrap(),
        )
        .unwrap();
        let low_quality = FixedTopologyGeometryState::<2>::new(
            &low_quality_reference,
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0e-4, 1.0e-3]],
        )
        .unwrap_err();
        assert!(low_quality.message().contains("mean-ratio quality"));

        let state = FixedTopologyGeometryState::<2>::reference(&reference).unwrap();
        let duration =
            FixedTopologyGeometryAction::<2>::new(&reference, &state, &state, 0.0).unwrap_err();
        assert!(duration.message().contains("positive time step"));
    }
}
