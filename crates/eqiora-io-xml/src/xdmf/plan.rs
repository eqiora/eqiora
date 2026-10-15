use std::collections::BTreeSet;

pub(super) use crate::xml::invalid_import;
use eqiora_core::Diagnostic;

/// Stable identity recorded by the L4 external-import manifest.
pub const XDMF_ADAPTER_ID: &str = "eqiora.xdmf";
/// Exact adapter implementation version.
pub const XDMF_ADAPTER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Independent syntax, source, decoded-array, and work budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XdmfImportLimits {
    /// Maximum complete XML metadata bytes.
    pub max_metadata_bytes: usize,
    /// Maximum nested element depth.
    pub max_depth: usize,
    /// Maximum element count.
    pub max_elements: usize,
    /// Maximum attributes on one element.
    pub max_attributes_per_element: usize,
    /// Maximum aggregate decoded text and attribute bytes.
    pub max_text_bytes: usize,
    /// Maximum HDF `DataItem` count.
    pub max_data_items: usize,
    /// Maximum declared array rank.
    pub max_array_rank: usize,
    /// Maximum scalar values in one resolved array.
    pub max_array_values: usize,
    /// Maximum complete bytes in one resolved external-source occurrence.
    pub max_source_bytes: usize,
    /// Maximum aggregate bytes across resolved source occurrences.
    pub max_total_source_bytes: usize,
    /// Maximum aggregate logical bytes across resolved scalar arrays.
    pub max_resolved_bytes: usize,
    /// Maximum parser work units across one metadata document: input bytes,
    /// XML events, attributes (including declarations), and decoded name,
    /// attribute, text and comment bytes.
    pub max_parser_work: usize,
    /// Maximum resolution work units across one response set.
    pub max_resolution_work: usize,
}

impl Default for XdmfImportLimits {
    fn default() -> Self {
        Self {
            max_metadata_bytes: 4 * 1024 * 1024,
            max_depth: 32,
            max_elements: 100_000,
            max_attributes_per_element: 32,
            max_text_bytes: 8 * 1024 * 1024,
            max_data_items: 16_384,
            max_array_rank: 8,
            max_array_values: 8_000_000,
            max_source_bytes: 256 * 1024 * 1024,
            max_total_source_bytes: 512 * 1024 * 1024,
            max_resolved_bytes: 512 * 1024 * 1024,
            max_parser_work: 64_000_000,
            max_resolution_work: 64_000_000,
        }
    }
}

impl XdmfImportLimits {
    pub(crate) fn validate(self) -> Result<Self, Diagnostic> {
        if [
            self.max_metadata_bytes,
            self.max_depth,
            self.max_elements,
            self.max_attributes_per_element,
            self.max_text_bytes,
            self.max_data_items,
            self.max_array_rank,
            self.max_array_values,
            self.max_source_bytes,
            self.max_total_source_bytes,
            self.max_resolved_bytes,
            self.max_parser_work,
            self.max_resolution_work,
        ]
        .contains(&0)
        {
            return Err(invalid_import("XDMF import limits must all be positive"));
        }
        Ok(self)
    }
}

/// Explicit structural selection; names never select content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XdmfSelection {
    grid: Vec<u32>,
    attributes: Vec<Vec<u32>>,
}

impl XdmfSelection {
    /// Select one grid and caller-ordered unique attributes by element-child path.
    ///
    /// # Errors
    /// Returns `EQ0810` for a root selector or a repeated selector.
    pub fn new(grid: Vec<u32>, attributes: Vec<Vec<u32>>) -> Result<Self, Diagnostic> {
        if grid.is_empty() || attributes.iter().any(Vec::is_empty) {
            return Err(invalid_import("XDMF selection paths must be non-root"));
        }
        let mut unique = BTreeSet::new();
        unique.insert(grid.clone());
        for attribute in &attributes {
            if !unique.insert(attribute.clone()) {
                return Err(invalid_import("XDMF selection paths must be unique"));
            }
        }
        Ok(Self { grid, attributes })
    }

    /// Selected Uniform Grid element path.
    #[must_use]
    pub fn grid(&self) -> &[u32] {
        &self.grid
    }

    /// Attribute element paths in explicit caller order.
    #[must_use]
    pub fn attributes(&self) -> &[Vec<u32>] {
        &self.attributes
    }
}

/// Canonical role of one requested array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XdmfArrayRole {
    /// Mesh coordinates; always first.
    Geometry,
    /// Mesh cell connectivity; always second.
    Topology,
    /// One selected field in caller selection order.
    Attribute,
}

/// Closed scalar grammar admitted from HDF references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XdmfScalarType {
    /// Unsigned 64-bit connectivity.
    U64,
    /// IEEE-754 binary64 coordinates or field values.
    F64,
}

/// One immutable HDF array request derived from metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XdmfArrayRequest {
    pub(crate) ordinal: usize,
    pub(crate) role: XdmfArrayRole,
    pub(crate) origin_selector: Vec<u32>,
    pub(crate) source_locator: String,
    pub(crate) dataset_path: String,
    pub(crate) scalar: XdmfScalarType,
    pub(crate) shape: Vec<u64>,
}

impl XdmfArrayRequest {
    /// Canonical Geometry, Topology, then selected-Attribute ordinal.
    #[must_use]
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }
    /// Array role.
    #[must_use]
    pub const fn role(&self) -> XdmfArrayRole {
        self.role
    }
    /// Owning Geometry, Topology, or selected Attribute element-child path.
    #[must_use]
    pub fn origin_selector(&self) -> &[u32] {
        &self.origin_selector
    }
    /// Display locator passed to, but never opened by, this adapter.
    #[must_use]
    pub fn source_locator(&self) -> &str {
        &self.source_locator
    }
    /// Absolute dataset selector passed to the caller-owned resolver.
    #[must_use]
    pub fn dataset_path(&self) -> &str {
        &self.dataset_path
    }
    /// Required scalar grammar.
    #[must_use]
    pub const fn scalar(&self) -> XdmfScalarType {
        self.scalar
    }
    /// Exact declared positive shape.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
}

/// Typed values returned by a caller-owned resolver.
#[derive(Debug, Clone, PartialEq)]
pub enum XdmfArrayValues {
    /// Unsigned connectivity values.
    U64(Vec<u64>),
    /// Coordinate or field values.
    F64(Vec<f64>),
}

/// Complete source bytes and values bound to one exact request identity.
#[derive(Debug, Clone, PartialEq)]
pub struct XdmfArrayResponse {
    request: XdmfArrayRequest,
    source_bytes: Vec<u8>,
    values: XdmfArrayValues,
}

impl XdmfArrayResponse {
    /// Bind a resolver result to an exact request. Admission remains deferred
    /// until the complete ordered response set is checked by the plan.
    #[must_use]
    pub fn new(request: &XdmfArrayRequest, source_bytes: Vec<u8>, values: XdmfArrayValues) -> Self {
        Self {
            request: request.clone(),
            source_bytes,
            values,
        }
    }

    /// Exact request identity copied into this response.
    #[must_use]
    pub const fn request(&self) -> &XdmfArrayRequest {
        &self.request
    }
    /// Complete logical source occurrence bytes.
    #[must_use]
    pub fn source_bytes(&self) -> &[u8] {
        &self.source_bytes
    }
    /// Typed resolved values.
    #[must_use]
    pub const fn values(&self) -> &XdmfArrayValues {
        &self.values
    }
}

/// Immutable metadata-derived plan with no I/O authority.
#[derive(Debug, Clone, PartialEq)]
pub struct XdmfImportPlan {
    pub(crate) metadata: Vec<u8>,
    pub(crate) selection: XdmfSelection,
    pub(crate) limits: XdmfImportLimits,
    pub(crate) grid_name: Option<String>,
    pub(crate) dimension: usize,
    pub(crate) geometry_kind: GeometryKind,
    pub(crate) cell_count: usize,
    pub(crate) fields: Vec<FieldPlan>,
    pub(crate) requests: Vec<XdmfArrayRequest>,
}

impl XdmfImportPlan {
    /// Complete metadata bytes parsed by this plan.
    #[must_use]
    pub fn metadata_bytes(&self) -> &[u8] {
        &self.metadata
    }
    /// Exact structural selection.
    #[must_use]
    pub const fn selection(&self) -> &XdmfSelection {
        &self.selection
    }
    /// Canonically ordered typed resolver requests.
    #[must_use]
    pub fn requests(&self) -> &[XdmfArrayRequest] {
        &self.requests
    }
    /// Resource policy applied to parsing and response admission.
    #[must_use]
    pub const fn limits(&self) -> XdmfImportLimits {
        self.limits
    }
    /// Optional selected Grid display name; never a selector.
    #[must_use]
    pub fn grid_name(&self) -> Option<&str> {
        self.grid_name.as_deref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GeometryKind {
    Xy,
    Xyz,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FieldPlan {
    pub(crate) name: Option<String>,
    pub(crate) origin_selector: Vec<u32>,
    pub(crate) association: eqiora_meshing::DiscreteFieldAssociation,
    pub(crate) shape: eqiora_meshing::DiscreteFieldShape,
}
