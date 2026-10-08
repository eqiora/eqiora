//! Method-neutral linear-continuum elasticity meaning shared by realizations.

use eqiora_core::{Diagnostic, RawId};

use crate::canonical_boundary::{BoundaryRelationBinding, CartesianBoundaryInventory};
use crate::linear_elasticity::IsotropicElasticityMaterial;
use crate::spatial_expression::ScalarSpatialExpression;

/// Constitutive reduction carried by an admitted isotropic continuum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IsotropicElasticityReduction {
    /// Intrinsic two-dimensional elasticity with a two-dimensional stress law.
    IntrinsicTwoDimensional,
    /// Full three-dimensional elasticity.
    FullThreeDimensional,
}

/// Exact integration-measure identity of an admitted isotropic continuum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElasticityIntegrationMeasure {
    /// Area integration producing resultants per unit out-of-plane thickness.
    PerUnitOutOfPlaneThickness,
    /// Three-dimensional volume integration.
    Volume,
}

/// One method-neutral isotropic small-strain continuum in `D` dimensions.
///
/// This descriptor owns the exact displacement/stress relation, Lamé
/// lineage, conservative load, reduction and integration-measure identities,
/// and complete boundary meaning. Element maps, spaces, quadrature, time
/// integration, algebra, solvers, and execution targets remain outside it.
#[derive(Debug, Clone, PartialEq)]
pub struct IsotropicElasticityContinuum<const D: usize> {
    domain: RawId,
    displacement: RawId,
    load_potential: RawId,
    load_definition_relation: RawId,
    equilibrium_relation: RawId,
    bounds: [[f64; 2]; D],
    reduction: IsotropicElasticityReduction,
    integration_measure: ElasticityIntegrationMeasure,
    material: IsotropicElasticityMaterial<D>,
    shear_modulus: ScalarSpatialExpression<f64>,
    first_lame_parameter: ScalarSpatialExpression<f64>,
    load_potential_expression: ScalarSpatialExpression<f64>,
    boundary_inventory: CartesianBoundaryInventory<D>,
    boundary_relations: Vec<BoundaryRelationBinding>,
    tractions: std::collections::BTreeMap<RawId, [f64; D]>,
}

impl<const D: usize> IsotropicElasticityContinuum<D> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        domain: RawId,
        displacement: RawId,
        load_potential: RawId,
        load_definition_relation: RawId,
        equilibrium_relation: RawId,
        bounds: [[f64; 2]; D],
        material: IsotropicElasticityMaterial<D>,
        shear_modulus: ScalarSpatialExpression<f64>,
        first_lame_parameter: ScalarSpatialExpression<f64>,
        load_potential_expression: ScalarSpatialExpression<f64>,
        boundary_inventory: CartesianBoundaryInventory<D>,
        boundary_relations: Vec<BoundaryRelationBinding>,
        tractions: std::collections::BTreeMap<RawId, [f64; D]>,
    ) -> Option<Self> {
        let (reduction, integration_measure) = match D {
            2 => (
                IsotropicElasticityReduction::IntrinsicTwoDimensional,
                ElasticityIntegrationMeasure::PerUnitOutOfPlaneThickness,
            ),
            3 => (
                IsotropicElasticityReduction::FullThreeDimensional,
                ElasticityIntegrationMeasure::Volume,
            ),
            _ => return None,
        };
        Some(Self {
            domain,
            displacement,
            load_potential,
            load_definition_relation,
            equilibrium_relation,
            bounds,
            reduction,
            integration_measure,
            material,
            shear_modulus,
            first_lame_parameter,
            load_potential_expression,
            boundary_inventory,
            boundary_relations,
            tractions,
        })
    }

    pub(crate) fn tractions(&self) -> &std::collections::BTreeMap<RawId, [f64; D]> {
        &self.tractions
    }

    /// Canonical volume Domain.
    #[must_use]
    pub const fn domain(&self) -> RawId {
        self.domain
    }

    /// Canonical spatial-vector displacement Field.
    #[must_use]
    pub const fn displacement(&self) -> RawId {
        self.displacement
    }

    /// Canonical scalar conservative-load potential Field.
    #[must_use]
    pub const fn load_potential(&self) -> RawId {
        self.load_potential
    }

    /// Exact Relation defining the conservative-load potential.
    #[must_use]
    pub const fn load_definition_relation(&self) -> RawId {
        self.load_definition_relation
    }

    /// Exact static balance or dynamic momentum Relation owning the stress law.
    #[must_use]
    pub const fn equilibrium_relation(&self) -> RawId {
        self.equilibrium_relation
    }

    /// Physical Cartesian bounds in coherent SI coordinates.
    #[must_use]
    pub const fn bounds(&self) -> &[[f64; 2]; D] {
        &self.bounds
    }

    /// Explicit constitutive reduction identity.
    #[must_use]
    pub const fn reduction(&self) -> IsotropicElasticityReduction {
        self.reduction
    }

    /// Explicit integration-measure identity.
    #[must_use]
    pub const fn integration_measure(&self) -> ElasticityIntegrationMeasure {
        self.integration_measure
    }

    /// Shear modulus `mu` in coherent SI units.
    #[must_use]
    pub fn shear_modulus(&self) -> f64 {
        self.material.shear_modulus()
    }

    /// First Lamé parameter `lambda` in coherent SI units.
    #[must_use]
    pub fn first_lame_parameter(&self) -> f64 {
        self.material.first_lame_parameter()
    }

    pub(crate) const fn material(&self) -> IsotropicElasticityMaterial<D> {
        self.material
    }

    /// Constant expression retaining the exact `mu` Parameter lineage.
    #[must_use]
    pub const fn shear_modulus_expression(&self) -> &ScalarSpatialExpression<f64> {
        &self.shear_modulus
    }

    /// Constant expression retaining the exact `lambda` Parameter lineage.
    #[must_use]
    pub const fn first_lame_parameter_expression(&self) -> &ScalarSpatialExpression<f64> {
        &self.first_lame_parameter
    }

    /// Immutable scalar tape defining the conservative-load potential.
    #[must_use]
    pub const fn load_potential_expression(&self) -> &ScalarSpatialExpression<f64> {
        &self.load_potential_expression
    }

    /// Complete package-neutral meaning of every Cartesian side.
    #[must_use]
    pub const fn boundary_inventory(&self) -> &CartesianBoundaryInventory<D> {
        &self.boundary_inventory
    }

    /// Canonically ordered exact Relations admitted by boundary normalization.
    pub(crate) fn boundary_relations(&self) -> &[BoundaryRelationBinding] {
        &self.boundary_relations
    }

    /// Evaluate the conservative body-force gradient from the canonical tape.
    ///
    /// # Errors
    /// Preserves the tape's exact shape and finite-evaluation diagnostics.
    pub fn conservative_body_force(&self, coordinates: &[f64]) -> Result<[f64; D], Diagnostic> {
        self.load_potential_expression
            .evaluate_gradient(coordinates)
    }
}
