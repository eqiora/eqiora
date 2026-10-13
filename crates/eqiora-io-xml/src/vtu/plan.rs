use std::collections::BTreeSet;

pub(super) use crate::xml::invalid_import;
use eqiora_core::Diagnostic;
use eqiora_meshing::{DiscreteFieldAssociation, DiscreteFieldShape, MeshQualityGate};

use super::parse::parse_document;
use super::resolve::{VtuImport, accept_plan};

pub(crate) const PORTABLE_MAX_SELECTED_FIELDS: usize = 4_096;
pub(crate) const PORTABLE_MAX_SELECTOR_DEPTH: usize = 64;
const PORTABLE_MAX_SELECTOR_VALUES: usize = 262_144;

/// Stable identity for future L4 external-import provenance.
pub const VTU_ADAPTER_ID: &str = "eqiora.vtu";
/// Exact adapter implementation version.
pub const VTU_ADAPTER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Independent syntax, structure, decoded-value, and work budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VtuImportLimits {
    /// Maximum complete `.vtu` source bytes.
    pub max_source_bytes: usize,
    /// Maximum XML element nesting depth.
    pub max_depth: usize,
    /// Maximum XML element count.
    pub max_elements: usize,
    /// Maximum attributes on one XML element.
    pub max_attributes_per_element: usize,
    /// Maximum aggregate decoded text and attribute bytes.
    pub max_text_bytes: usize,
    /// Maximum declared points in the selected Piece.
    pub max_points: usize,
    /// Maximum declared cells in the selected Piece.
    pub max_cells: usize,
    /// Maximum scalar values in one DataArray.
    pub max_array_values: usize,
    /// Maximum caller-selected field count.
    pub max_selected_fields: usize,
    /// Maximum element count in one structural selector.
    pub max_selector_depth: usize,
    /// Maximum aggregate element count across all structural selectors.
    pub max_selector_values: usize,
    /// Maximum values retained in normalized geometry, topology, and fields.
    pub max_resolved_values: usize,
    /// Maximum bytes retained in normalized geometry, topology, and fields.
    pub max_resolved_bytes: usize,
    /// Maximum aggregate source-scan, parser, and numeric-token work units.
    pub max_parser_work: usize,
}

impl Default for VtuImportLimits {
    fn default() -> Self {
        Self {
            max_source_bytes: 64 * 1024 * 1024,
            max_depth: 24,
            max_elements: 100_000,
            max_attributes_per_element: 16,
            max_text_bytes: 64 * 1024 * 1024,
            max_points: 2_000_000,
            max_cells: 2_000_000,
            max_array_values: 8_000_000,
            max_selected_fields: 1_024,
            max_selector_depth: 24,
            max_selector_values: 24 * 1_025,
            max_resolved_values: 8_000_000,
            max_resolved_bytes: 64 * 1024 * 1024,
            max_parser_work: 256 * 1024 * 1024,
        }
    }
}

impl VtuImportLimits {
    pub(crate) fn validate(self) -> Result<Self, Diagnostic> {
        if [
            self.max_source_bytes,
            self.max_depth,
            self.max_elements,
            self.max_attributes_per_element,
            self.max_text_bytes,
            self.max_points,
            self.max_cells,
            self.max_array_values,
            self.max_selected_fields,
            self.max_selector_depth,
            self.max_selector_values,
            self.max_resolved_values,
            self.max_resolved_bytes,
            self.max_parser_work,
        ]
        .contains(&0)
        {
            return Err(invalid_import("VTU import limits must all be positive"));
        }
        Ok(self)
    }
}

/// Explicit structural selection. Display names never select content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VtuSelection {
    piece: Vec<u32>,
    fields: Vec<Vec<u32>>,
}

impl VtuSelection {
    /// Select the sole Piece and caller-ordered PointData/CellData arrays.
    ///
    /// # Errors
    /// Returns `EQ0810` for root, repeated, or non-portably large selectors.
    pub fn new(piece: Vec<u32>, fields: Vec<Vec<u32>>) -> Result<Self, Diagnostic> {
        if piece.is_empty() || fields.iter().any(Vec::is_empty) {
            return Err(invalid_import("VTU selection paths must be non-root"));
        }
        if fields.len() > PORTABLE_MAX_SELECTED_FIELDS {
            return Err(invalid_import(
                "VTU selected field count exceeds the portable limit",
            ));
        }
        if piece.len() > PORTABLE_MAX_SELECTOR_DEPTH
            || fields
                .iter()
                .any(|selector| selector.len() > PORTABLE_MAX_SELECTOR_DEPTH)
        {
            return Err(invalid_import(
                "VTU selection path depth exceeds the portable limit",
            ));
        }
        let selector_values = fields.iter().try_fold(piece.len(), |total, selector| {
            total.checked_add(selector.len())
        });
        if selector_values.is_none_or(|total| total > PORTABLE_MAX_SELECTOR_VALUES) {
            return Err(invalid_import(
                "VTU aggregate selector size exceeds the portable limit",
            ));
        }
        let mut unique = BTreeSet::new();
        unique.insert(piece.as_slice());
        for field in &fields {
            if !unique.insert(field.as_slice()) {
                return Err(invalid_import("VTU selection paths must be unique"));
            }
        }
        Ok(Self { piece, fields })
    }

    pub(crate) fn validate_against(&self, limits: VtuImportLimits) -> Result<(), Diagnostic> {
        if self.fields.len() > limits.max_selected_fields {
            return Err(invalid_import(
                "VTU selected field count exceeds the configured limit",
            ));
        }
        if self.piece.len() > limits.max_selector_depth
            || self
                .fields
                .iter()
                .any(|selector| selector.len() > limits.max_selector_depth)
        {
            return Err(invalid_import(
                "VTU selection path depth exceeds the configured limit",
            ));
        }
        let selector_values = self
            .fields
            .iter()
            .try_fold(self.piece.len(), |total, selector| {
                total.checked_add(selector.len())
            })
            .ok_or_else(|| invalid_import("VTU aggregate selector size overflows usize"))?;
        if selector_values > limits.max_selector_values {
            return Err(invalid_import(
                "VTU aggregate selector size exceeds the configured limit",
            ));
        }
        Ok(())
    }

    /// Selected Piece element path.
    #[must_use]
    pub fn piece(&self) -> &[u32] {
        &self.piece
    }

    /// Selected DataArray paths in caller order.
    #[must_use]
    pub fn fields(&self) -> &[Vec<u32>] {
        &self.fields
    }
}

/// Homogeneous affine simplex cell family admitted by this adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VtuCellKind {
    /// VTK cell type 5, three vertices, intrinsic dimension two.
    Triangle,
    /// VTK cell type 10, four vertices, intrinsic dimension three.
    Tetrahedron,
}

impl VtuCellKind {
    #[must_use]
    pub(crate) const fn dimension(self) -> usize {
        match self {
            Self::Triangle => 2,
            Self::Tetrahedron => 3,
        }
    }

    #[must_use]
    pub(crate) const fn arity(self) -> usize {
        self.dimension() + 1
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct FieldPlan {
    pub(crate) selector: Vec<u32>,
    pub(crate) name: Option<String>,
    pub(crate) association: DiscreteFieldAssociation,
    pub(crate) shape: DiscreteFieldShape,
    pub(crate) raw_shape: Vec<u64>,
    pub(crate) values: Vec<f64>,
}

/// Immutable, fully decoded plan for one bounded ASCII VTU Piece.
#[derive(Debug, PartialEq)]
pub struct VtuImportPlan {
    pub(crate) source: Vec<u8>,
    pub(crate) selection: VtuSelection,
    pub(crate) limits: VtuImportLimits,
    pub(crate) cell_kind: VtuCellKind,
    pub(crate) geometry_selector: Vec<u32>,
    pub(crate) topology_selector: Vec<u32>,
    pub(crate) geometry_shape: Vec<u64>,
    pub(crate) topology_shape: Vec<u64>,
    pub(crate) geometry: Vec<f64>,
    pub(crate) topology: Vec<u64>,
    pub(crate) fields: Vec<FieldPlan>,
}

impl VtuImportPlan {
    /// Parse and normalize one pure ASCII VTU source without external I/O.
    ///
    /// # Errors
    /// Returns `EQ0810` for malformed XML, unsupported VTU semantics,
    /// inconsistent arrays, invalid selection, or any resource-limit excess.
    pub fn parse(
        source: &[u8],
        selection: VtuSelection,
        limits: VtuImportLimits,
    ) -> Result<Self, Diagnostic> {
        parse_document(source, selection, limits.validate()?)
    }

    /// Complete VTU source bytes used by this immutable plan.
    #[must_use]
    pub fn source_bytes(&self) -> &[u8] {
        &self.source
    }

    /// Exact structural selection.
    #[must_use]
    pub const fn selection(&self) -> &VtuSelection {
        &self.selection
    }

    /// Homogeneous accepted simplex cell family.
    #[must_use]
    pub const fn cell_kind(&self) -> VtuCellKind {
        self.cell_kind
    }

    /// Points DataArray structural selector.
    #[must_use]
    pub fn geometry_selector(&self) -> &[u32] {
        &self.geometry_selector
    }

    /// Composite Cells element structural selector.
    #[must_use]
    pub fn topology_selector(&self) -> &[u32] {
        &self.topology_selector
    }

    /// Normalized geometry array shape `[points, intrinsic_dimension]`.
    #[must_use]
    pub fn geometry_shape(&self) -> &[u64] {
        &self.geometry_shape
    }

    /// Normalized topology array shape `[cells, simplex_arity]`.
    #[must_use]
    pub fn topology_shape(&self) -> &[u64] {
        &self.topology_shape
    }

    /// Normalized intrinsic coordinates in point-major order.
    #[must_use]
    pub fn normalized_geometry(&self) -> &[f64] {
        &self.geometry
    }

    /// Normalized zero-based connectivity in cell-major order.
    #[must_use]
    pub fn normalized_topology(&self) -> &[u64] {
        &self.topology
    }

    /// Resource policy applied to this source.
    #[must_use]
    pub const fn limits(&self) -> VtuImportLimits {
        self.limits
    }

    /// Reconstruct through shared mesh and field invariants.
    ///
    /// # Errors
    /// Returns `EQ0810` when shared topology, geometry, quality, or field
    /// invariants reject the normalized VTU content.
    pub fn accept(&self, quality_gate: MeshQualityGate) -> Result<VtuImport, Diagnostic> {
        accept_plan(self, quality_gate)
    }
}
