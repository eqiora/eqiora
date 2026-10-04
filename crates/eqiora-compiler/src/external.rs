//! Client-neutral static inputs for one selected occurrence.

use eqiora_core::ValueLiteral;
use eqiora_schema::kernel::GeometryDigest;

/// One checked mathematical interval or exact Geometry support for an occurrence.
///
/// Geometry bindings retain revision membership and parent topology; mathematical
/// factors retain finite dimensioned bounds through the ordinary Domain owner.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ExternalSupportBinding {
    /// Dimensioned mathematical interval; no Geometry artifact is invented.
    CoordinateInterval {
        slot: String,
        bounds: eqiora_schema::kernel::AxisBounds,
    },
    /// An explicit finite exterior of one exactly bound parent region.
    CompleteExterior {
        slot: String,
        geometry: GeometryDigest,
        parent_slot: String,
        members: Vec<ExternalGeometryBoundaryMember>,
    },
    /// A full-dimensional named region.
    Region {
        /// Public Component support slot.
        slot: String,
        /// Exact Geometry artifact identity.
        geometry: GeometryDigest,
        /// Named full-dimensional entity set in that Geometry.
        entity_set: String,
        /// Exact ambient dimension.
        ambient_dimension: usize,
    },
    /// A named boundary of another bound region support.
    Boundary {
        /// Public Component support slot.
        slot: String,
        /// Exact Geometry artifact identity.
        geometry: GeometryDigest,
        /// Named boundary entity set in that Geometry.
        entity_set: String,
        /// Public Component slot naming the exact parent region.
        parent_slot: String,
        /// Derived only from exact primitive Geometry topology.
        embedding: Option<eqiora_schema::kernel::CartesianBoundaryEmbedding>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExternalGeometryBoundaryMember {
    pub(crate) entity_set: String,
    pub(crate) embedding: Option<eqiora_schema::kernel::CartesianBoundaryEmbedding>,
}

impl ExternalSupportBinding {
    /// Project one already validated full-dimensional selection.
    #[must_use]
    pub(crate) fn region(
        slot: impl Into<String>,
        geometry: GeometryDigest,
        entity_set: impl Into<String>,
        ambient_dimension: usize,
    ) -> Self {
        Self::Region {
            slot: slot.into(),
            geometry,
            entity_set: entity_set.into(),
            ambient_dimension,
        }
    }

    /// Project one already validated codimension-one selection and parent.
    #[must_use]
    pub(crate) fn boundary(
        slot: impl Into<String>,
        geometry: GeometryDigest,
        entity_set: impl Into<String>,
        parent_slot: impl Into<String>,
        embedding: Option<eqiora_schema::kernel::CartesianBoundaryEmbedding>,
    ) -> Self {
        Self::Boundary {
            slot: slot.into(),
            geometry,
            entity_set: entity_set.into(),
            parent_slot: parent_slot.into(),
            embedding,
        }
    }

    /// Bound Component support slot.
    #[must_use]
    pub(crate) fn slot(&self) -> &str {
        match self {
            Self::CoordinateInterval { slot, .. }
            | Self::Region { slot, .. }
            | Self::Boundary { slot, .. }
            | Self::CompleteExterior { slot, .. } => slot,
        }
    }

    pub(crate) fn allocated_support_count(&self) -> usize {
        match self {
            Self::CompleteExterior { members, .. } => members.len(),
            _ => 1,
        }
    }
}

/// One explicit coherent-SI value for a public Component Parameter.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExternalParameterBinding {
    parameter: String,
    value: ValueLiteral,
}

impl ExternalParameterBinding {
    /// Construct a named, explicitly dimensioned scalar value.
    #[must_use]
    pub(crate) fn new(parameter: impl Into<String>, value: ValueLiteral) -> Self {
        Self {
            parameter: parameter.into(),
            value,
        }
    }

    /// Public Component Parameter name.
    #[must_use]
    pub(crate) fn parameter(&self) -> &str {
        &self.parameter
    }

    /// Explicit coherent-SI scalar value and dimension.
    #[must_use]
    pub(crate) const fn value(&self) -> &ValueLiteral {
        &self.value
    }
}

/// Closed input selecting one local Component occurrence to materialize.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExternalComponentBinding {
    model: String,
    component: String,
    supports: Vec<ExternalSupportBinding>,
    parameters: Vec<ExternalParameterBinding>,
    pub(crate) clocks: Vec<(String, eqiora_schema::kernel::ClockDomainDef)>,
    pub(crate) properties:
        std::collections::BTreeMap<String, std::sync::Arc<eqiora_schema::kernel::PropertyRelease>>,
}

impl ExternalComponentBinding {
    /// Construct one bounded local Component occurrence.
    #[must_use]
    pub(crate) fn new(
        model: impl Into<String>,
        component: impl Into<String>,
        supports: Vec<ExternalSupportBinding>,
        parameters: Vec<ExternalParameterBinding>,
    ) -> Self {
        Self {
            model: model.into(),
            component: component.into(),
            supports,
            parameters,
            clocks: Vec::new(),
            properties: Default::default(),
        }
    }

    pub(crate) fn properties(
        &self,
    ) -> &std::collections::BTreeMap<String, std::sync::Arc<eqiora_schema::kernel::PropertyRelease>>
    {
        &self.properties
    }

    pub(crate) fn model(&self) -> &str {
        &self.model
    }

    pub(crate) fn component(&self) -> &str {
        &self.component
    }

    /// Exact checked support bindings.
    #[must_use]
    pub(crate) fn supports(&self) -> &[ExternalSupportBinding] {
        &self.supports
    }

    /// Explicit public Parameter bindings.
    #[must_use]
    pub(crate) fn parameters(&self) -> &[ExternalParameterBinding] {
        &self.parameters
    }
}

/// One named static argument, interpreted by the selected signature's category.
#[derive(Clone, Copy, Debug)]
pub enum StaticBindingValue<'a> {
    /// A bounded dimensioned mathematical coordinate factor.
    CoordinateInterval(eqiora_schema::kernel::AxisBounds),
    /// A closed typed initializer or an exact source property reference.
    Expression(&'a eqiora_lang::Expr),
    /// A checked value retaining its complete nominal type and exact payload.
    Value(&'a eqiora_core::ValueLiteral),
    /// An existing nominal clock; its identity is preserved.
    Clock(&'a eqiora_schema::kernel::ClockDomainDef),
    /// Explicit finite boundary selections forming the complete exterior of an exact parent.
    CompleteExterior {
        geometry: &'a eqiora_geometry::CanonicalGeometryV1,
        members: &'a [&'a eqiora_geometry::NamedEntitySet],
        parent: &'a eqiora_geometry::NamedEntitySet,
    },
    /// An authoritative Geometry selection and, for a boundary, its exact parent.
    GeometrySupport {
        geometry: &'a eqiora_geometry::CanonicalGeometryV1,
        selection: &'a eqiora_geometry::NamedEntitySet,
        parent: Option<&'a eqiora_geometry::NamedEntitySet>,
    },
}
