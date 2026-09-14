//! State, material, scale, load, and boundary contracts.

use eqiora_meshing::CellId;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

use eqiora_core::Diagnostic;
use eqiora_meshing::{
    AffineGeometryMap, GeometryMap, MeshEntity, MeshTopology, QuadratureRule, SimplicialMesh,
    VertexId,
};

use super::partition::FixedReferenceFsiPartition;
use super::{invalid, required_quadrature_exactness};
use crate::linear_elasticity::IsotropicElasticityMaterial;

/// Positive material data for the bounded linear FSI realization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedReferenceFsiMaterial<const D: usize> {
    fluid_density: f64,
    fluid_dynamic_viscosity: f64,
    solid_density: f64,
    solid: IsotropicElasticityMaterial<D>,
}

impl<const D: usize> FixedReferenceFsiMaterial<D> {
    /// Construct stable Newtonian-fluid and isotropic-linear-solid data.
    ///
    /// # Errors
    /// Rejects non-finite/non-positive densities, viscosity, or shear modulus,
    /// and rejects a Lamé pair unless `lambda + 2 mu / D` is positive.
    pub fn new(
        fluid_density: f64,
        fluid_dynamic_viscosity: f64,
        solid_density: f64,
        solid_shear_modulus: f64,
        solid_first_lame_parameter: f64,
    ) -> Result<Self, Diagnostic> {
        require_supported_dimension::<D>()?;
        let solid =
            IsotropicElasticityMaterial::<D>::new(solid_shear_modulus, solid_first_lame_parameter);
        let Some(solid) = solid else {
            return Err(invalid(
                "fixed-reference FSI material data must be finite and coercive in its admitted dimension",
            ));
        };
        Self::from_admitted_solid(fluid_density, fluid_dynamic_viscosity, solid_density, solid)
    }

    pub(crate) fn from_admitted_solid(
        fluid_density: f64,
        fluid_dynamic_viscosity: f64,
        solid_density: f64,
        solid: IsotropicElasticityMaterial<D>,
    ) -> Result<Self, Diagnostic> {
        if !fluid_density.is_finite()
            || fluid_density <= 0.0
            || !fluid_dynamic_viscosity.is_finite()
            || fluid_dynamic_viscosity <= 0.0
            || !solid_density.is_finite()
            || solid_density <= 0.0
        {
            return Err(invalid(
                "fixed-reference FSI material data must be finite and coercive in its admitted dimension",
            ));
        }
        Ok(Self {
            fluid_density,
            fluid_dynamic_viscosity,
            solid_density,
            solid,
        })
    }

    /// Fluid mass density.
    #[must_use]
    pub const fn fluid_density(self) -> f64 {
        self.fluid_density
    }

    /// Fluid dynamic viscosity.
    #[must_use]
    pub const fn fluid_dynamic_viscosity(self) -> f64 {
        self.fluid_dynamic_viscosity
    }

    /// Solid mass density.
    #[must_use]
    pub const fn solid_density(self) -> f64 {
        self.solid_density
    }

    /// Solid shear modulus `mu`.
    #[must_use]
    pub const fn solid_shear_modulus(self) -> f64 {
        self.solid.shear_modulus()
    }

    /// Solid first Lamé parameter `lambda`.
    #[must_use]
    pub const fn solid_first_lame_parameter(self) -> f64 {
        self.solid.first_lame_parameter()
    }

    pub(crate) const fn solid_material(self) -> IsotropicElasticityMaterial<D> {
        self.solid
    }
}

/// Characteristic profile defining the dimensionless monolithic algebra.
///
/// For velocity scale `U`, pressure scale `P`, and length scale `L`, physical
/// unknowns satisfy `x = D x_hat` and the captured system is the direct
/// congruence `A_hat = D^T A D / Theta`, `b_hat = D^T b / Theta`, with
/// `Theta = P U L^(D - 1)` in ambient dimension `D`. The solver therefore
/// never receives a dimensionally mixed saddle matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedReferenceFsiScale<const D: usize> {
    length: f64,
    velocity: f64,
    pressure: f64,
}

impl<const D: usize> FixedReferenceFsiScale<D> {
    /// Construct finite positive characteristic length, velocity, and pressure.
    ///
    /// # Errors
    /// Returns `EQ0801` when any scale is non-finite or non-positive.
    pub fn new(length: f64, velocity: f64, pressure: f64) -> Result<Self, Diagnostic> {
        require_supported_dimension::<D>()?;
        if [length, velocity, pressure]
            .into_iter()
            .any(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(invalid(
                "fixed-reference FSI scales must be finite and positive",
            ));
        }
        let value = Self {
            length,
            velocity,
            pressure,
        };
        if [value.action(), value.energy(), value.power()]
            .into_iter()
            .any(|scale| !scale.is_finite() || scale <= 0.0)
        {
            return Err(invalid(
                "fixed-reference FSI derived dimensional scales must remain finite and positive",
            ));
        }
        Ok(value)
    }

    /// Characteristic length.
    #[must_use]
    pub const fn length(self) -> f64 {
        self.length
    }

    /// Characteristic velocity.
    #[must_use]
    pub const fn velocity(self) -> f64 {
        self.velocity
    }

    /// Characteristic pressure.
    #[must_use]
    pub const fn pressure(self) -> f64 {
        self.pressure
    }

    pub(crate) fn action(self) -> f64 {
        self.pressure * self.length.powi((D - 1) as i32)
    }

    pub(crate) fn energy(self) -> f64 {
        self.pressure * self.length.powi(D as i32)
    }

    pub(crate) fn power(self) -> f64 {
        self.pressure * self.velocity * self.length.powi((D - 1) as i32)
    }
}

/// Complete time/material/scale selection for one backward-Euler step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedReferenceFsiStepConfig<const D: usize> {
    time_step: f64,
    material: FixedReferenceFsiMaterial<D>,
    scale: FixedReferenceFsiScale<D>,
    load: FixedReferenceFsiLoad,
}

impl<const D: usize> FixedReferenceFsiStepConfig<D> {
    /// Bind one positive time step to admitted material and scale contracts.
    ///
    /// # Errors
    /// Returns `EQ0801` when `time_step` is non-finite or non-positive.
    pub fn new(
        time_step: f64,
        material: FixedReferenceFsiMaterial<D>,
        scale: FixedReferenceFsiScale<D>,
        load: FixedReferenceFsiLoad,
    ) -> Result<Self, Diagnostic> {
        if !time_step.is_finite() || time_step <= 0.0 {
            return Err(invalid(
                "fixed-reference FSI time step must be finite and positive",
            ));
        }
        Ok(Self {
            time_step,
            material,
            scale,
            load,
        })
    }

    /// Backward-Euler step width.
    #[must_use]
    pub const fn time_step(self) -> f64 {
        self.time_step
    }

    /// Material selection.
    #[must_use]
    pub const fn material(self) -> FixedReferenceFsiMaterial<D> {
        self.material
    }

    /// Acceptance scales.
    #[must_use]
    pub const fn scale(self) -> FixedReferenceFsiScale<D> {
        self.scale
    }

    /// Explicit v1 load policy.
    #[must_use]
    pub const fn load(self) -> FixedReferenceFsiLoad {
        self.load
    }
}

/// Bounded load vocabulary for the first CPU reference realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FixedReferenceFsiLoad {
    /// No fluid body force, solid body force, or prescribed traction.
    #[default]
    Zero,
}

/// Private complete exterior-facet role stored by an admitted prepared step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreparedFsiExteriorFacetDisposition {
    EssentialVelocity,
    NaturalOutflow,
}

#[derive(Debug, Clone, PartialEq)]
struct PreparedFsiBoundaryData<const D: usize> {
    previous_endpoint_words: [u64; 4],
    previous_time_bits: u64,
    current_endpoint_words: [u64; 4],
    current_time_bits: u64,
    previous_physical: Vec<[Option<f64>; D]>,
    current_physical: Vec<[Option<f64>; D]>,
    previous_quotient: Vec<[Option<f64>; D]>,
    current_quotient: Vec<[Option<f64>; D]>,
    exterior_facets: Vec<(MeshEntity, PreparedFsiExteriorFacetDisposition)>,
    canonical_velocity_scale: bool,
}

// Construction in the prepared-boundary owner rejects every non-finite word,
// so derived `PartialEq` is reflexive for every admitted value.
impl<const D: usize> Eq for PreparedFsiBoundaryData<D> {}

/// Homogeneous essential velocity closure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedReferenceFsiBoundary<const D: usize> {
    fixed_zero_velocity_vertices: Vec<VertexId>,
    prepared_velocity: Option<Arc<PreparedFsiBoundaryData<D>>>,
}

impl<const D: usize> FixedReferenceFsiBoundary<D> {
    /// Construct the complete homogeneous exterior-velocity closure.
    ///
    /// The interface remains live.  An interface endpoint that also belongs
    /// to the exterior is naturally constrained once by this inventory.
    ///
    /// # Errors
    /// Returns `EQ0801` if the mesh does not have the admitted dimension.
    pub fn homogeneous_exterior(mesh: &SimplicialMesh) -> Result<Self, Diagnostic> {
        require_mesh_dimension::<D>(mesh)?;
        let fixed_zero_velocity_vertices = (0..mesh.vertices().len())
            .filter(|&vertex| {
                mesh.is_boundary_entity(MeshEntity::new(0, vertex))
                    .expect("accepted vertex owns boundary classification")
            })
            .map(VertexId::new)
            .collect();
        Ok(Self {
            fixed_zero_velocity_vertices,
            prepared_velocity: None,
        })
    }

    /// Vertices carrying an exact zero velocity trace.
    #[must_use]
    pub fn fixed_zero_velocity_vertices(&self) -> &[VertexId] {
        &self.fixed_zero_velocity_vertices
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_prepared_velocity(
        previous_endpoint_words: [u64; 4],
        previous_time_bits: u64,
        current_endpoint_words: [u64; 4],
        current_time_bits: u64,
        previous_physical: Vec<[Option<f64>; D]>,
        current_physical: Vec<[Option<f64>; D]>,
        previous_quotient: Vec<[Option<f64>; D]>,
        current_quotient: Vec<[Option<f64>; D]>,
        exterior_facets: Vec<(MeshEntity, bool)>,
        canonical_velocity_scale: bool,
    ) -> Self {
        let prepared = Arc::new(PreparedFsiBoundaryData {
            previous_endpoint_words,
            previous_time_bits,
            current_endpoint_words,
            current_time_bits,
            previous_physical,
            current_physical,
            previous_quotient,
            current_quotient,
            exterior_facets: exterior_facets
                .into_iter()
                .map(|(facet, essential)| {
                    (
                        facet,
                        if essential {
                            PreparedFsiExteriorFacetDisposition::EssentialVelocity
                        } else {
                            PreparedFsiExteriorFacetDisposition::NaturalOutflow
                        },
                    )
                })
                .collect(),
            canonical_velocity_scale,
        });
        let fixed_zero_velocity_vertices = prepared
            .current_quotient
            .iter()
            .enumerate()
            .filter_map(|(vertex, components)| {
                components
                    .iter()
                    .any(Option::is_some)
                    .then_some(VertexId::new(vertex))
            })
            .collect();
        Self {
            fixed_zero_velocity_vertices,
            prepared_velocity: Some(prepared),
        }
    }

    pub(crate) fn prepared_previous_endpoint(&self) -> Option<([u64; 4], u64)> {
        self.prepared_velocity.as_ref().map(|prepared| {
            (
                prepared.previous_endpoint_words,
                prepared.previous_time_bits,
            )
        })
    }

    pub(crate) fn prepared_current_endpoint(&self) -> Option<([u64; 4], u64)> {
        self.prepared_velocity
            .as_ref()
            .map(|prepared| (prepared.current_endpoint_words, prepared.current_time_bits))
    }

    pub(crate) fn prepared_previous_physical(&self) -> Option<&[[Option<f64>; D]]> {
        self.prepared_velocity
            .as_deref()
            .map(|prepared| prepared.previous_physical.as_slice())
    }

    pub(crate) fn prepared_current_physical(&self) -> Option<&[[Option<f64>; D]]> {
        self.prepared_velocity
            .as_deref()
            .map(|prepared| prepared.current_physical.as_slice())
    }

    pub(crate) fn prepared_previous_quotient(&self) -> Option<&[[Option<f64>; D]]> {
        self.prepared_velocity
            .as_deref()
            .map(|prepared| prepared.previous_quotient.as_slice())
    }

    pub(crate) fn prepared_current_quotient(&self) -> Option<&[[Option<f64>; D]]> {
        self.prepared_velocity
            .as_deref()
            .map(|prepared| prepared.current_quotient.as_slice())
    }

    pub(crate) fn prepared_uses_canonical_velocity_scale(&self) -> bool {
        self.prepared_velocity
            .as_ref()
            .is_some_and(|prepared| prepared.canonical_velocity_scale)
    }

    pub(crate) fn prepared_exterior_facets(&self) -> Option<Vec<(MeshEntity, bool)>> {
        self.prepared_velocity.as_deref().map(|prepared| {
            prepared
                .exterior_facets
                .iter()
                .map(|(facet, disposition)| {
                    (
                        *facet,
                        *disposition == PreparedFsiExteriorFacetDisposition::EssentialVelocity,
                    )
                })
                .collect()
        })
    }
}

/// Complete previous-step state for the fixed-reference spaces.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedReferenceFsiState<const D: usize> {
    vertex_velocity: Vec<[f64; D]>,
    fluid_cell_bubble_velocity: BTreeMap<CellId, [f64; D]>,
    solid_displacement: Vec<[f64; D]>,
}

impl<const D: usize> FixedReferenceFsiState<D> {
    /// Admit one finite state in the exact partition layout.
    ///
    /// Solid displacement is represented in mesh-vertex order to make the
    /// shared trace explicit; entries outside the solid closure must be exact
    /// zero.  Fluid bubble entries are keyed by the exact owned `CellId` inventory.
    ///
    /// # Errors
    /// Returns `EQ0801` for an incompatible shape, non-finite coefficient, or
    /// non-zero displacement outside the solid closure.
    pub fn new(
        mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        vertex_velocity: Vec<[f64; D]>,
        fluid_cell_bubble_velocity: BTreeMap<CellId, [f64; D]>,
        solid_displacement: Vec<[f64; D]>,
    ) -> Result<Self, Diagnostic> {
        require_mesh_dimension::<D>(mesh)?;
        let vertex_count = mesh.vertices().len();
        if vertex_velocity.len() != vertex_count
            || solid_displacement.len() != vertex_count
            || !fluid_cell_bubble_velocity
                .keys()
                .copied()
                .eq(partition.fluid_cells().iter().copied())
            || vertex_velocity
                .iter()
                .chain(fluid_cell_bubble_velocity.values())
                .chain(&solid_displacement)
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "fixed-reference FSI state must be finite and match its exact partition layout",
            ));
        }
        let solid = partition
            .solid_vertices()
            .iter()
            .map(|vertex| vertex.index())
            .collect::<BTreeSet<_>>();
        if solid_displacement
            .iter()
            .enumerate()
            .any(|(vertex, value)| !solid.contains(&vertex) && *value != [0.0; D])
        {
            return Err(invalid(
                "fixed-reference FSI displacement must be exact zero outside the solid closure",
            ));
        }
        Ok(Self {
            vertex_velocity,
            fluid_cell_bubble_velocity,
            solid_displacement,
        })
    }

    /// Previous shared mesh-vertex velocity coefficients.
    #[must_use]
    pub fn vertex_velocity(&self) -> &[[f64; D]] {
        &self.vertex_velocity
    }

    /// Previous fluid MINI bubble coefficients keyed by exact fluid `CellId`.
    #[must_use]
    pub fn fluid_cell_bubble_velocity(&self) -> &BTreeMap<CellId, [f64; D]> {
        &self.fluid_cell_bubble_velocity
    }

    /// Previous solid P1 displacement in mesh-vertex order.
    #[must_use]
    pub fn solid_displacement(&self) -> &[[f64; D]] {
        &self.solid_displacement
    }
}

pub(crate) fn validate_problem<const D: usize>(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &FixedReferenceFsiBoundary<D>,
    previous: &FixedReferenceFsiState<D>,
    config: FixedReferenceFsiStepConfig<D>,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    validate_problem_common(mesh, partition, previous, config, quadrature)?;
    if boundary.prepared_velocity.is_some() {
        return Ok(());
    }
    let fixed = boundary
        .fixed_zero_velocity_vertices
        .iter()
        .map(|vertex| vertex.index())
        .collect::<BTreeSet<_>>();
    if fixed.len() != boundary.fixed_zero_velocity_vertices.len()
        || fixed.iter().any(|vertex| *vertex >= mesh.vertices().len())
    {
        return Err(invalid(
            "fixed-reference FSI boundary inventory must contain unique valid vertices",
        ));
    }
    for vertex in 0..mesh.vertices().len() {
        if mesh
            .is_boundary_entity(MeshEntity::new(0, vertex))
            .expect("accepted vertex owns boundary classification")
            && !fixed.contains(&vertex)
        {
            return Err(invalid(
                "fixed-reference FSI v1 requires homogeneous velocity on the complete exterior",
            ));
        }
    }
    if fixed.iter().any(|&vertex| {
        previous.vertex_velocity[vertex]
            .iter()
            .any(|value| value.to_bits() != 0.0_f64.to_bits())
    }) {
        return Err(invalid(
            "fixed-reference FSI previous state violates the homogeneous velocity closure",
        ));
    }
    Ok(())
}

fn validate_problem_common<const D: usize>(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    previous: &FixedReferenceFsiState<D>,
    config: FixedReferenceFsiStepConfig<D>,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    require_mesh_dimension::<D>(mesh)?;
    let replayed = FixedReferenceFsiPartition::<D>::new(
        mesh,
        partition.fluid_cells().to_vec(),
        partition.solid_cells().to_vec(),
        partition.interface_facets().to_vec(),
    )?;
    if &replayed != partition {
        return Err(invalid(
            "fixed-reference FSI partition cache differs from exact mesh replay",
        ));
    }
    if config.load != FixedReferenceFsiLoad::Zero {
        return Err(invalid(
            "fixed-reference FSI v1 admits only the explicit zero-load policy",
        ));
    }
    let required_exactness = required_quadrature_exactness::<D>();
    if quadrature.reference_cell() != eqiora_meshing::ReferenceCell::simplex(D)?
        || quadrature.polynomial_exactness().unwrap_or(0) < required_exactness
    {
        return Err(invalid(format!(
            "fixed-reference FSI requires matching simplex quadrature exact through degree {required_exactness}",
        )));
    }
    FixedReferenceFsiState::<D>::new(
        mesh,
        partition,
        previous.vertex_velocity.clone(),
        previous.fluid_cell_bubble_velocity.clone(),
        previous.solid_displacement.clone(),
    )?;
    Ok(())
}

pub(super) fn require_mesh_dimension<const D: usize>(
    mesh: &SimplicialMesh,
) -> Result<(), Diagnostic> {
    require_supported_dimension::<D>()?;
    if mesh.topological_dimension() != D
        || mesh
            .vertices()
            .iter()
            .any(|coordinates| coordinates.len() != D)
    {
        return Err(invalid(format!(
            "fixed-reference FSI requires one intrinsic {D}D affine-simplex mesh",
        )));
    }
    Ok(())
}

pub(super) fn require_local_geometry_dimension<const D: usize>(
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    require_supported_dimension::<D>()?;
    let required_exactness = required_quadrature_exactness::<D>();
    if geometry.reference_cell() != quadrature.reference_cell()
        || geometry.reference_cell().dimension() != D
        || geometry.physical_dimension() != D
        || quadrature.polynomial_exactness().unwrap_or(0) < required_exactness
    {
        return Err(invalid(format!(
            "fixed-reference FSI cell requires a matching affine simplex and degree-{required_exactness} quadrature",
        )));
    }
    Ok(())
}

fn require_supported_dimension<const D: usize>() -> Result<(), Diagnostic> {
    if matches!(D, 2 | 3) {
        Ok(())
    } else {
        Err(invalid(
            "fixed-reference FSI reference contracts admit dimensions two and three",
        ))
    }
}
