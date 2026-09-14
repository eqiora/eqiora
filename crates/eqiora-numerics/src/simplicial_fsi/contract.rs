//! State, material, scale, load, and boundary contracts.

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

/// Exact Domain/Field-owned coefficients for transient kinetic, viscous and elastic terms.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedReferenceFsiMaterial<const D: usize> {
    pub(super) densities: BTreeMap<eqiora_core::RawId, (eqiora_core::RawId, f64)>,
    pub(super) viscosities: BTreeMap<eqiora_core::RawId, (eqiora_core::RawId, f64)>,
    pub(super) elasticities:
        BTreeMap<eqiora_core::RawId, (eqiora_core::RawId, IsotropicElasticityMaterial<D>)>,
}

impl<const D: usize> FixedReferenceFsiMaterial<D> {
    /// Bind coherent-SI density/viscosity and admitted elastic witnesses to exact Fields.
    /// Densities own velocity/rate Fields; elastic witnesses own displacement states.
    /// The executing Plan authenticates the complete role and Domain inventory.
    /// # Errors
    /// Rejects repeated Fields, nonpositive or nonfinite coefficients and incompatible ownership.
    pub(crate) fn new(
        densities: impl IntoIterator<
            Item = (
                eqiora_core::Id<eqiora_core::entity::kinds::Domain>,
                eqiora_core::Id<eqiora_core::entity::kinds::Field>,
                f64,
            ),
        >,
        viscosities: impl IntoIterator<
            Item = (
                eqiora_core::Id<eqiora_core::entity::kinds::Domain>,
                eqiora_core::Id<eqiora_core::entity::kinds::Field>,
                f64,
            ),
        >,
        elasticities: impl IntoIterator<
            Item = (
                eqiora_core::Id<eqiora_core::entity::kinds::Domain>,
                eqiora_core::Id<eqiora_core::entity::kinds::Field>,
                IsotropicElasticityMaterial<D>,
            ),
        >,
    ) -> Result<Self, Diagnostic> {
        require_supported_dimension::<D>()?;
        let collect = |values: Vec<_>| -> Result<BTreeMap<_, _>, Diagnostic> {
            let mut result = BTreeMap::new();
            for (domain, field, value) in values {
                if !f64::is_finite(value)
                    || value <= 0.0
                    || result.insert(field, (domain, value)).is_some()
                {
                    return Err(invalid(
                        "transient coefficient is nonpositive, nonfinite or repeats an exact Field",
                    ));
                }
            }
            Ok(result)
        };
        let densities = collect(
            densities
                .into_iter()
                .map(|(domain, field, value)| (domain.erase(), field.erase(), value))
                .collect(),
        )?;
        let viscosities = collect(
            viscosities
                .into_iter()
                .map(|(domain, field, value)| (domain.erase(), field.erase(), value))
                .collect(),
        )?;
        if densities.is_empty()
            || viscosities.iter().any(|(field, (domain, _))| {
                densities
                    .get(field)
                    .is_none_or(|(owner, _)| owner != domain)
            })
        {
            return Err(invalid(
                "viscous term has no matching exact kinetic Domain/Field owner",
            ));
        }
        let mut elastic = BTreeMap::new();
        for (domain, field, material) in elasticities {
            if densities.contains_key(&field.erase())
                || elastic
                    .insert(field.erase(), (domain.erase(), material))
                    .is_some()
            {
                return Err(invalid(
                    "elastic state coefficient repeats an exact Field role",
                ));
            }
        }
        Ok(Self {
            densities,
            viscosities,
            elasticities: elastic,
        })
    }
    /// Density of one exact velocity/rate Field, if admitted.
    pub fn density(
        &self,
        field: eqiora_core::Id<eqiora_core::entity::kinds::Field>,
    ) -> Option<f64> {
        self.densities.get(&field.erase()).map(|(_, value)| *value)
    }
    /// Viscosity of one exact velocity Field, if admitted.
    pub fn viscosity(
        &self,
        field: eqiora_core::Id<eqiora_core::entity::kinds::Field>,
    ) -> Option<f64> {
        self.viscosities
            .get(&field.erase())
            .map(|(_, value)| *value)
    }
    /// Existing admitted elasticity witness of one exact displacement state.
    pub(crate) fn elasticity(
        &self,
        field: eqiora_core::Id<eqiora_core::entity::kinds::Field>,
    ) -> Option<IsotropicElasticityMaterial<D>> {
        self.elasticities
            .get(&field.erase())
            .map(|(_, material)| *material)
    }

    pub(crate) fn kinetic_fields(
        &self,
    ) -> impl Iterator<Item = eqiora_core::Id<eqiora_core::entity::kinds::Field>> + '_ {
        self.densities
            .keys()
            .map(|field| field.downcast().expect("typed material Field"))
    }

    #[cfg(test)]
    pub(crate) fn density_entries(
        &self,
    ) -> impl Iterator<Item = (eqiora_core::RawId, eqiora_core::RawId, f64)> + '_ {
        self.densities
            .iter()
            .map(|(field, (domain, value))| (*domain, *field, *value))
    }

    #[cfg(test)]
    pub(crate) fn viscosity_entries(
        &self,
    ) -> impl Iterator<Item = (eqiora_core::RawId, eqiora_core::RawId, f64)> + '_ {
        self.viscosities
            .iter()
            .map(|(field, (domain, value))| (*domain, *field, *value))
    }

    #[cfg(test)]
    pub(crate) fn elasticity_entries(
        &self,
    ) -> impl Iterator<
        Item = (
            eqiora_core::RawId,
            eqiora_core::RawId,
            IsotropicElasticityMaterial<D>,
        ),
    > + '_ {
        self.elasticities
            .iter()
            .map(|(field, (domain, value))| (*domain, *field, *value))
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
#[derive(Debug, Clone, PartialEq)]
pub struct FixedReferenceFsiStepConfig<const D: usize> {
    time_step: f64,
    material: Arc<FixedReferenceFsiMaterial<D>>,
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
            material: Arc::new(material),
            scale,
            load,
        })
    }

    /// Backward-Euler step width.
    #[must_use]
    pub const fn time_step(&self) -> f64 {
        self.time_step
    }

    /// Material selection.
    #[must_use]
    pub fn material(&self) -> &FixedReferenceFsiMaterial<D> {
        &self.material
    }

    /// Acceptance scales.
    #[must_use]
    pub const fn scale(&self) -> FixedReferenceFsiScale<D> {
        self.scale
    }

    /// Explicit v1 load policy.
    #[must_use]
    pub const fn load(&self) -> FixedReferenceFsiLoad {
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

pub use super::state::FixedReferenceFsiState;

pub(crate) fn validate_problem<const D: usize>(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &FixedReferenceFsiBoundary<D>,
    previous: &FixedReferenceFsiState<D>,
    config: &FixedReferenceFsiStepConfig<D>,
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
    for &field in config.material().densities.keys() {
        let values = previous
            .fields
            .get(&field)
            .ok_or_else(|| invalid("history omits exact velocity Field"))?;
        if values.coefficients.iter().any(|(key, value)| {
            key.entity.dimension() == 0
                && fixed.contains(&key.entity.index())
                && value.to_bits() != 0.0_f64.to_bits()
        }) {
            return Err(invalid(
                "previous exact velocity Field violates homogeneous exterior closure",
            ));
        }
    }
    Ok(())
}

fn validate_problem_common<const D: usize>(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    previous: &FixedReferenceFsiState<D>,
    config: &FixedReferenceFsiStepConfig<D>,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    require_mesh_dimension::<D>(mesh)?;
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
    if previous.fields.iter().any(|(&id, field)| {
        partition
            .domain_cells(field.domain.downcast().expect("Domain"))
            .is_none()
            || field.coefficients.iter().any(|(key, value)| {
                key.field != id
                    || !value.is_finite()
                    || mesh
                        .entity_count(key.entity.dimension())
                        .is_none_or(|count| key.entity.index() >= count)
            })
    }) {
        return Err(invalid(
            "physical State has stale Domain/entity ownership or nonfinite values",
        ));
    }
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
