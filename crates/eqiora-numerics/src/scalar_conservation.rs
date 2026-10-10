//! Private, method-neutral recognition of bounded scalar-conservation meaning.
//!
//! This projection deliberately has no physics-family tag. A mathematically
//! identical scalar balance may describe temperature, concentration, axial
//! displacement, or another scalar quantity. Numerical admission and user-
//! visible capability names remain outside this module.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::ValueFrame;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id, OntologyId, RawId};
use eqiora_graph::EdgeKind;
use eqiora_schema::Model;
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{
    ActivationKind, BoundarySide, CartesianBoundaryEmbedding, ConnectionSemantics, ExprDag, ExprId,
    ExprNode, KernelNode, SymbolRef,
};
use eqiora_sem::KernelProgram;

use crate::additive_residual::{AdditiveResidualView, SignedOpaqueLeaf};
use crate::canonical::{
    boundary_parent, continuum_fields_on, lower_flux_coefficient, lowering_error,
    model_lowering_error, relation_expression, relations_on, unique_root,
    validate_positive_affine_coefficient,
};
use crate::spatial_expression::{self, ScalarSpatialExpression};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScalarTermLineage {
    relation: RawId,
    expression: ExprId,
}

impl ScalarTermLineage {
    pub(crate) const fn relation(&self) -> RawId {
        self.relation
    }
    pub(crate) const fn expression(&self) -> ExprId {
        self.expression
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScalarExteriorLineage {
    relation: RawId,
    operator_expression: ExprId,
    datum_expression: Option<ExprId>,
    datum_negative: bool,
    robin_coefficient_expression: Option<ExprId>,
    robin_trace_expression: Option<ExprId>,
}

impl ScalarExteriorLineage {
    pub(crate) const fn relation(&self) -> RawId {
        self.relation
    }
    pub(crate) const fn operator_expression(&self) -> ExprId {
        self.operator_expression
    }
    pub(crate) const fn datum_expression(&self) -> Option<ExprId> {
        self.datum_expression
    }
    /// Sign of the retained datum in the admitted outward conormal value.
    pub(crate) const fn datum_negative(&self) -> bool {
        self.datum_negative
    }
    pub(crate) const fn robin_coefficient_expression(&self) -> Option<ExprId> {
        self.robin_coefficient_expression
    }
    pub(crate) const fn robin_trace_expression(&self) -> Option<ExprId> {
        self.robin_trace_expression
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScalarStorageMeaning {
    coefficient: ScalarSpatialExpression<f64>,
    lineage: ScalarTermLineage,
}

impl ScalarStorageMeaning {
    pub(crate) const fn coefficient(&self) -> &ScalarSpatialExpression<f64> {
        &self.coefficient
    }
    pub(crate) const fn lineage(&self) -> ScalarTermLineage {
        self.lineage
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScalarFluxMeaning {
    coefficient: ScalarSpatialExpression<f64>,
    lineage: ScalarTermLineage,
}

impl ScalarFluxMeaning {
    pub(crate) const fn coefficient(&self) -> &ScalarSpatialExpression<f64> {
        &self.coefficient
    }
    pub(crate) const fn lineage(&self) -> ScalarTermLineage {
        self.lineage
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VolumetricSourceMeaning {
    expression: ScalarSpatialExpression<f64>,
    dimension: DimExponents,
    integrated_dimension: DimExponents,
    lineage: ScalarTermLineage,
}

impl VolumetricSourceMeaning {
    pub(crate) const fn expression(&self) -> &ScalarSpatialExpression<f64> {
        &self.expression
    }
    pub(crate) const fn dimension(&self) -> DimExponents {
        self.dimension
    }
    pub(crate) const fn integrated_dimension(&self) -> DimExponents {
        self.integrated_dimension
    }
    pub(crate) const fn lineage(&self) -> ScalarTermLineage {
        self.lineage
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ScalarExteriorLaw {
    PrescribedTrace {
        value: ScalarSpatialExpression<f64>,
        lineage: ScalarExteriorLineage,
    },
    PrescribedOutwardFlux {
        value: ScalarSpatialExpression<f64>,
        lineage: ScalarExteriorLineage,
    },
    Robin {
        trace_coefficient: ScalarSpatialExpression<f64>,
        value: ScalarSpatialExpression<f64>,
        lineage: ScalarExteriorLineage,
    },
    ZeroOutwardFlux {
        lineage: ScalarExteriorLineage,
    },
}

impl ScalarExteriorLaw {
    pub(crate) const fn lineage(&self) -> ScalarExteriorLineage {
        match self {
            Self::PrescribedTrace { lineage, .. }
            | Self::PrescribedOutwardFlux { lineage, .. }
            | Self::Robin { lineage, .. }
            | Self::ZeroOutwardFlux { lineage } => *lineage,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScalarExteriorBoundary {
    boundary: RawId,
    parent: RawId,
    axis: usize,
    side: BoundarySide,
    law: ScalarExteriorLaw,
}

impl ScalarExteriorBoundary {
    pub(crate) const fn boundary(&self) -> RawId {
        self.boundary
    }
    pub(crate) const fn parent(&self) -> RawId {
        self.parent
    }
    pub(crate) const fn axis(&self) -> usize {
        self.axis
    }
    pub(crate) const fn side(&self) -> BoundarySide {
        self.side
    }
    pub(crate) const fn law(&self) -> &ScalarExteriorLaw {
        &self.law
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScalarInterfaceSide {
    domain: RawId,
    field: RawId,
    boundary: RawId,
    axis: usize,
    side: BoundarySide,
    trace: ScalarTermLineage,
    flux: ScalarTermLineage,
}

impl ScalarInterfaceSide {
    pub(crate) const fn trace(&self) -> ScalarTermLineage {
        self.trace
    }
    pub(crate) const fn field(&self) -> RawId {
        self.field
    }
    pub(crate) const fn domain(&self) -> RawId {
        self.domain
    }
    pub(crate) const fn boundary(&self) -> RawId {
        self.boundary
    }
    pub(crate) const fn axis(&self) -> usize {
        self.axis
    }
    pub(crate) const fn side(&self) -> BoundarySide {
        self.side
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScalarMaterialInterface {
    source: eqiora_realization::ConformingTraceSource,
    physical_support: Option<RawId>,
    sides: [ScalarInterfaceSide; 2],
}

impl ScalarMaterialInterface {
    pub(crate) const fn source(&self) -> eqiora_realization::ConformingTraceSource {
        self.source
    }
    pub(crate) const fn physical_support(&self) -> Option<RawId> {
        self.physical_support
    }
    pub(crate) const fn sides(&self) -> &[ScalarInterfaceSide; 2] {
        &self.sides
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScalarConservationRegion {
    domain: RawId,
    field: RawId,
    field_dimension: DimExponents,
    dimensions: usize,
    bounds: Vec<[f64; 2]>,
    balance_relation: RawId,
    balance_dimension: DimExponents,
    storage: Option<ScalarStorageMeaning>,
    flux: ScalarFluxMeaning,
    source: Option<VolumetricSourceMeaning>,
    exterior: BTreeMap<(usize, BoundarySide), ScalarExteriorBoundary>,
}

impl ScalarConservationRegion {
    pub(crate) const fn domain(&self) -> RawId {
        self.domain
    }
    pub(crate) const fn field(&self) -> RawId {
        self.field
    }
    pub(crate) const fn field_dimension(&self) -> DimExponents {
        self.field_dimension
    }
    pub(crate) const fn dimensions(&self) -> usize {
        self.dimensions
    }
    pub(crate) fn bounds(&self) -> &[[f64; 2]] {
        &self.bounds
    }
    pub(crate) const fn balance_relation(&self) -> RawId {
        self.balance_relation
    }
    pub(crate) const fn balance_dimension(&self) -> DimExponents {
        self.balance_dimension
    }
    pub(crate) const fn storage(&self) -> Option<&ScalarStorageMeaning> {
        self.storage.as_ref()
    }
    pub(crate) const fn flux(&self) -> &ScalarFluxMeaning {
        &self.flux
    }
    pub(crate) const fn source(&self) -> Option<&VolumetricSourceMeaning> {
        self.source.as_ref()
    }
    pub(crate) fn exterior(&self) -> impl ExactSizeIterator<Item = &ScalarExteriorBoundary> {
        self.exterior.values()
    }
    pub(crate) fn exterior_at(
        &self,
        axis: usize,
        side: BoundarySide,
    ) -> Option<&ScalarExteriorBoundary> {
        self.exterior.get(&(axis, side))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScalarConservationDescriptor {
    model: OntologyId<Model>,
    semantic_revision: u64,
    regions: Vec<ScalarConservationRegion>,
    interfaces: Vec<ScalarMaterialInterface>,
    parameters: Vec<Id<kinds::Parameter>>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScalarRegionSupport {
    pub(crate) domain: RawId,
    pub(crate) bounds: Vec<[f64; 2]>,
    pub(crate) boundaries: BTreeMap<(usize, BoundarySide), RawId>,
}

impl ScalarRegionSupport {
    pub(crate) fn new(
        domain: RawId,
        bounds: Vec<[f64; 2]>,
        boundaries: BTreeMap<(usize, BoundarySide), RawId>,
    ) -> Self {
        Self {
            domain,
            bounds,
            boundaries,
        }
    }
}

impl ScalarConservationDescriptor {
    pub(crate) const fn model(&self) -> OntologyId<Model> {
        self.model
    }
    pub(crate) const fn semantic_revision(&self) -> u64 {
        self.semantic_revision
    }
    pub(crate) fn regions(&self) -> impl ExactSizeIterator<Item = &ScalarConservationRegion> {
        self.regions.iter()
    }
    pub(crate) fn interfaces(&self) -> impl ExactSizeIterator<Item = &ScalarMaterialInterface> {
        self.interfaces.iter()
    }
    pub(crate) fn parameters(&self) -> &[Id<kinds::Parameter>] {
        &self.parameters
    }
}

mod balance;
mod boundary;
mod descriptor_support;
mod interface;
mod physical_interface;
mod recognize;
pub(crate) use recognize::cartesian_region_supports;
mod retained;
mod support;

use balance::*;
use boundary::*;
use descriptor_support::*;
use interface::*;
use support::*;

pub(crate) fn recognize_scalar_conservation(
    program: &KernelProgram,
) -> Result<ScalarConservationDescriptor, Diagnostic> {
    recognize::recognize_scalar_conservation(program)
}

pub(crate) fn recognize_scalar_conservation_on_supports(
    program: &KernelProgram,
    supports: Vec<ScalarRegionSupport>,
) -> Result<ScalarConservationDescriptor, Diagnostic> {
    recognize::recognize_scalar_conservation_on_supports(program, supports)
}

#[cfg(test)]
mod tests;
