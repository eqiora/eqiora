use std::collections::BTreeSet;
use std::fmt;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_meshing::{DiscreteFieldAssociation, DiscreteFieldShape};

/// Independent temporal-frame, field, text, and output-byte budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XdmfTemporalExportLimits {
    /// Maximum frames in one Temporal Collection.
    pub max_frames: usize,
    /// Maximum Attributes in one frame.
    pub max_fields_per_frame: usize,
    /// Maximum UTF-8 bytes in one display name, locator, or dataset path.
    pub max_text_bytes: usize,
    /// Maximum complete rendered XDMF metadata bytes.
    pub max_metadata_bytes: usize,
}

impl Default for XdmfTemporalExportLimits {
    fn default() -> Self {
        Self {
            max_frames: 16_384,
            max_fields_per_frame: 16_384,
            max_text_bytes: 16 * 1024,
            max_metadata_bytes: 16 * 1024 * 1024,
        }
    }
}

impl XdmfTemporalExportLimits {
    fn validate(self) -> Result<Self, Diagnostic> {
        if [
            self.max_frames,
            self.max_fields_per_frame,
            self.max_text_bytes,
            self.max_metadata_bytes,
        ]
        .contains(&0)
        {
            return Err(invalid_export(
                "XDMF temporal export limits must all be positive",
            ));
        }
        Ok(self)
    }
}

/// One typed mesh-associated Attribute emitted for every temporal frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XdmfTemporalField {
    name: String,
    association: DiscreteFieldAssociation,
    shape: DiscreteFieldShape,
    dataset_path: String,
}

impl XdmfTemporalField {
    /// Construct one scalar or dimension-matching vector Attribute.
    ///
    /// The name is inspectable output metadata, not Eqiora Field identity.
    /// L4 derives it from exact typed identity before entering this adapter.
    ///
    /// # Errors
    /// Returns `EQ0811` for invalid XML text or a noncanonical HDF5 dataset
    /// path.
    pub fn new(
        name: impl Into<String>,
        association: DiscreteFieldAssociation,
        shape: DiscreteFieldShape,
        dataset_path: impl Into<String>,
    ) -> Result<Self, Diagnostic> {
        let name = name.into();
        let dataset_path = dataset_path.into();
        validate_text("XDMF temporal Attribute name", &name)?;
        validate_dataset_path(&dataset_path)?;
        Ok(Self {
            name,
            association,
            shape,
            dataset_path,
        })
    }

    /// Inspectable Attribute name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Mesh entity association.
    #[must_use]
    pub const fn association(&self) -> DiscreteFieldAssociation {
        self.association
    }

    /// Scalar or fixed-vector component shape.
    #[must_use]
    pub const fn shape(&self) -> DiscreteFieldShape {
        self.shape
    }

    /// Canonical absolute HDF5 dataset path.
    #[must_use]
    pub fn dataset_path(&self) -> &str {
        &self.dataset_path
    }
}

/// One explicitly sequenced Uniform Grid in a Temporal Collection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XdmfTemporalFrame {
    sequence: u64,
    time_s_bits: u64,
    dimension: usize,
    vertex_count: usize,
    cell_count: usize,
    geometry_path: String,
    topology_path: String,
    fields: Vec<XdmfTemporalField>,
}

impl XdmfTemporalFrame {
    /// Construct one affine Tri3 or Tet4 output frame.
    ///
    /// Sequence, rather than input declaration order, gives the total output
    /// order. L4 must replace a remesh source tip with its accepted target
    /// representation before constructing frames; this adapter never invents
    /// an epsilon time offset or emits two samples at one coordinate.
    ///
    /// # Errors
    /// Returns `EQ0811` for unsupported dimension, empty mesh, invalid time,
    /// noncanonical paths, duplicate fields, or a vector shape differing from
    /// the spatial dimension.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sequence: u64,
        time_s: f64,
        dimension: usize,
        vertex_count: usize,
        cell_count: usize,
        geometry_path: impl Into<String>,
        topology_path: impl Into<String>,
        mut fields: Vec<XdmfTemporalField>,
    ) -> Result<Self, Diagnostic> {
        if !matches!(dimension, 2 | 3) || vertex_count == 0 || cell_count == 0 {
            return Err(invalid_export(
                "XDMF temporal frame requires a nonempty affine Tri3 or Tet4 mesh",
            ));
        }
        if !time_s.is_finite() || time_s < 0.0 || (time_s == 0.0 && time_s.is_sign_negative()) {
            return Err(invalid_export(
                "XDMF temporal frame time must be finite, nonnegative, and canonical",
            ));
        }
        let geometry_path = geometry_path.into();
        let topology_path = topology_path.into();
        validate_dataset_path(&geometry_path)?;
        validate_dataset_path(&topology_path)?;
        if geometry_path == topology_path {
            return Err(invalid_export(
                "XDMF temporal Geometry and Topology paths must differ",
            ));
        }
        for field in &fields {
            if let DiscreteFieldShape::Vector { components } = field.shape
                && usize::try_from(components.get()) != Ok(dimension)
            {
                return Err(invalid_export(
                    "XDMF temporal vectors must match the frame dimension",
                ));
            }
        }
        fields.sort_by(|left, right| {
            left.name
                .cmp(&right.name)
                .then_with(|| {
                    association_order(left.association).cmp(&association_order(right.association))
                })
                .then_with(|| field_shape_order(left.shape).cmp(&field_shape_order(right.shape)))
                .then_with(|| left.dataset_path.cmp(&right.dataset_path))
        });
        if fields.windows(2).any(|pair| pair[0] == pair[1])
            || fields
                .iter()
                .map(|field| field.name())
                .collect::<BTreeSet<_>>()
                .len()
                != fields.len()
        {
            return Err(invalid_export(
                "XDMF temporal Attributes must have unique names and declarations",
            ));
        }
        if fields.iter().any(|field| {
            matches!(
                field.dataset_path(),
                path if path == geometry_path || path == topology_path
            )
        }) {
            return Err(invalid_export(
                "XDMF temporal Attribute paths must not alias Geometry or Topology",
            ));
        }
        Ok(Self {
            sequence,
            time_s_bits: time_s.to_bits(),
            dimension,
            vertex_count,
            cell_count,
            geometry_path,
            topology_path,
            fields,
        })
    }

    /// Explicit representation sequence.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Accepted coherent-SI time.
    #[must_use]
    pub const fn time_s(&self) -> f64 {
        f64::from_bits(self.time_s_bits)
    }

    /// Spatial dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Mesh vertex count.
    #[must_use]
    pub const fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Mesh top-cell count.
    #[must_use]
    pub const fn cell_count(&self) -> usize {
        self.cell_count
    }

    /// Geometry dataset path.
    #[must_use]
    pub fn geometry_path(&self) -> &str {
        &self.geometry_path
    }

    /// Topology dataset path.
    #[must_use]
    pub fn topology_path(&self) -> &str {
        &self.topology_path
    }

    /// Canonically ordered Attribute declarations.
    #[must_use]
    pub fn fields(&self) -> &[XdmfTemporalField] {
        &self.fields
    }
}

/// Pure deterministic XDMF 3 Temporal Collection metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XdmfTemporalExportPlan {
    source_locator: String,
    frames: Vec<XdmfTemporalFrame>,
    metadata: Vec<u8>,
}

impl XdmfTemporalExportPlan {
    /// Canonicalize and render one temporal sequence without opening its HDF5
    /// display locator.
    ///
    /// Frame declaration order is non-semantic: explicit contiguous sequence
    /// ordinals define output order. Times must increase strictly after L4's
    /// remesh replacement policy. Every frame must expose the
    /// same ordered Attribute name/association/component inventory, while mesh
    /// entity counts and dataset paths may change across remeshing.
    ///
    /// # Errors
    /// Returns `EQ0811` for a noncanonical sequence, decreasing time, field
    /// inventory drift, invalid text, or output-budget
    /// excess.
    pub fn new(
        source_locator: impl Into<String>,
        mut frames: Vec<XdmfTemporalFrame>,
        limits: XdmfTemporalExportLimits,
    ) -> Result<Self, Diagnostic> {
        let limits = limits.validate()?;
        let source_locator = source_locator.into();
        validate_text("XDMF temporal HDF5 display locator", &source_locator)?;
        require_at_most(
            source_locator.len(),
            limits.max_text_bytes,
            "XDMF temporal display locator bytes",
        )?;
        if frames.len() < 2 {
            return Err(invalid_export(
                "XDMF Temporal Collection requires at least two frames",
            ));
        }
        require_at_most(frames.len(), limits.max_frames, "XDMF temporal frame count")?;
        frames.sort_by_key(XdmfTemporalFrame::sequence);
        for (index, frame) in frames.iter().enumerate() {
            if usize::try_from(frame.sequence()).ok() != Some(index) {
                return Err(invalid_export(
                    "XDMF temporal frame sequence must be contiguous from zero",
                ));
            }
            require_at_most(
                frame.fields().len(),
                limits.max_fields_per_frame,
                "XDMF temporal Attributes per frame",
            )?;
            for text in std::iter::once(frame.geometry_path())
                .chain(std::iter::once(frame.topology_path()))
                .chain(frame.fields().iter().flat_map(|field| {
                    std::iter::once(field.name()).chain(std::iter::once(field.dataset_path()))
                }))
            {
                require_at_most(
                    text.len(),
                    limits.max_text_bytes,
                    "XDMF temporal name or path bytes",
                )?;
            }
        }
        if frames
            .windows(2)
            .any(|pair| pair[0].time_s() >= pair[1].time_s())
        {
            return Err(invalid_export(
                "XDMF temporal frame times must increase strictly after remesh replacement",
            ));
        }
        let signature = field_signature(&frames[0]);
        if frames
            .iter()
            .skip(1)
            .any(|frame| field_signature(frame) != signature)
        {
            return Err(invalid_export(
                "XDMF temporal frames must retain one exact Attribute inventory",
            ));
        }
        let metadata = render(&source_locator, &frames, limits.max_metadata_bytes)?;
        Ok(Self {
            source_locator,
            frames,
            metadata,
        })
    }

    /// Display locator rendered into every HDF DataItem and never opened here.
    #[must_use]
    pub fn source_locator(&self) -> &str {
        &self.source_locator
    }

    /// Canonically sequenced frames.
    #[must_use]
    pub fn frames(&self) -> &[XdmfTemporalFrame] {
        &self.frames
    }

    /// Complete deterministic UTF-8 XDMF metadata.
    #[must_use]
    pub fn metadata_bytes(&self) -> &[u8] {
        &self.metadata
    }
}

fn field_signature(
    frame: &XdmfTemporalFrame,
) -> Vec<(&str, DiscreteFieldAssociation, DiscreteFieldShape)> {
    frame
        .fields()
        .iter()
        .map(|field| (field.name(), field.association(), field.shape()))
        .collect()
}

const fn association_order(value: DiscreteFieldAssociation) -> u8 {
    match value {
        DiscreteFieldAssociation::Vertex => 0,
        DiscreteFieldAssociation::Cell => 1,
    }
}

fn field_shape_order(value: DiscreteFieldShape) -> (u8, u32) {
    match value {
        DiscreteFieldShape::Scalar => (0, 0),
        DiscreteFieldShape::Vector { components } => (1, components.get()),
    }
}

fn render(
    locator: &str,
    frames: &[XdmfTemporalFrame],
    limit: usize,
) -> Result<Vec<u8>, Diagnostic> {
    let mut counter = ByteCounter::new(limit);
    render_into(&mut counter, locator, frames).map_err(|_| {
        invalid_export("rendered XDMF temporal metadata exceeds its configured byte limit")
    })?;
    let required = counter.bytes;
    let mut output = String::new();
    output.try_reserve_exact(required).map_err(|error| {
        invalid_export(format!(
            "cannot reserve rendered XDMF temporal metadata: {error}",
        ))
    })?;
    render_into(&mut output, locator, frames)
        .map_err(|_| invalid_export("cannot render XDMF temporal metadata"))?;
    if output.len() != required {
        return Err(invalid_export(
            "XDMF temporal metadata size changed during rendering",
        ));
    }
    Ok(output.into_bytes())
}

fn render_into(
    output: &mut impl fmt::Write,
    locator: &str,
    frames: &[XdmfTemporalFrame],
) -> fmt::Result {
    output.write_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")?;
    output.write_str("<Xdmf Version=\"3.0\">\n  <Domain>\n")?;
    output.write_str(
        "    <Grid Name=\"eqiora-spatial-series\" GridType=\"Collection\" CollectionType=\"Temporal\">\n",
    )?;
    for frame in frames {
        writeln!(
            output,
            "      <Grid Name=\"frame-{:020}\" GridType=\"Uniform\">",
            frame.sequence()
        )?;
        writeln!(output, "        <Time Value=\"{}\"/>", frame.time_s())?;
        let topology = if frame.dimension() == 2 {
            "Triangle"
        } else {
            "Tetrahedron"
        };
        writeln!(
            output,
            "        <Topology TopologyType=\"{topology}\" NumberOfElements=\"{}\">",
            frame.cell_count()
        )?;
        write!(
            output,
            "          <DataItem Format=\"HDF\" DataType=\"UInt\" Precision=\"8\" Dimensions=\"{} {}\">",
            frame.cell_count(),
            frame.dimension() + 1
        )?;
        write_hdf_reference(output, locator, frame.topology_path())?;
        output.write_str("</DataItem>\n        </Topology>\n")?;
        let geometry = if frame.dimension() == 2 { "XY" } else { "XYZ" };
        writeln!(output, "        <Geometry GeometryType=\"{geometry}\">")?;
        write!(
            output,
            "          <DataItem Format=\"HDF\" DataType=\"Float\" Precision=\"8\" Dimensions=\"{} {}\">",
            frame.vertex_count(),
            frame.dimension()
        )?;
        write_hdf_reference(output, locator, frame.geometry_path())?;
        output.write_str("</DataItem>\n        </Geometry>\n")?;
        for field in frame.fields() {
            output.write_str("        <Attribute Name=\"")?;
            write_xml_attribute(output, field.name())?;
            let attribute_type = match field.shape() {
                DiscreteFieldShape::Scalar => "Scalar",
                DiscreteFieldShape::Vector { .. } => "Vector",
            };
            let center = match field.association() {
                DiscreteFieldAssociation::Vertex => "Node",
                DiscreteFieldAssociation::Cell => "Cell",
            };
            writeln!(
                output,
                "\" AttributeType=\"{attribute_type}\" Center=\"{center}\">"
            )?;
            let entities = match field.association() {
                DiscreteFieldAssociation::Vertex => frame.vertex_count(),
                DiscreteFieldAssociation::Cell => frame.cell_count(),
            };
            write!(
                output,
                "          <DataItem Format=\"HDF\" DataType=\"Float\" Precision=\"8\" Dimensions=\"{entities}"
            )?;
            if let DiscreteFieldShape::Vector { components } = field.shape() {
                write!(output, " {}", components.get())?;
            }
            output.write_str("\">")?;
            write_hdf_reference(output, locator, field.dataset_path())?;
            output.write_str("</DataItem>\n        </Attribute>\n")?;
        }
        output.write_str("      </Grid>\n")?;
    }
    output.write_str("    </Grid>\n  </Domain>\n</Xdmf>\n")
}

fn write_hdf_reference(output: &mut impl fmt::Write, locator: &str, path: &str) -> fmt::Result {
    write_xml_text(output, locator)?;
    output.write_char(':')?;
    write_xml_text(output, path)
}

fn write_xml_attribute(output: &mut impl fmt::Write, value: &str) -> fmt::Result {
    for character in value.chars() {
        match character {
            '&' => output.write_str("&amp;")?,
            '<' => output.write_str("&lt;")?,
            '>' => output.write_str("&gt;")?,
            '"' => output.write_str("&quot;")?,
            '\'' => output.write_str("&apos;")?,
            character => output.write_char(character)?,
        }
    }
    Ok(())
}

fn write_xml_text(output: &mut impl fmt::Write, value: &str) -> fmt::Result {
    for character in value.chars() {
        match character {
            '&' => output.write_str("&amp;")?,
            '<' => output.write_str("&lt;")?,
            '>' => output.write_str("&gt;")?,
            character => output.write_char(character)?,
        }
    }
    Ok(())
}

fn validate_text(label: &str, value: &str) -> Result<(), Diagnostic> {
    if value.is_empty() || !value.chars().all(is_xml_1_0_character) {
        Err(invalid_export(format!(
            "{label} must be nonempty valid XML 1.0 text",
        )))
    } else {
        Ok(())
    }
}

fn validate_dataset_path(path: &str) -> Result<(), Diagnostic> {
    validate_text("XDMF temporal dataset path", path)?;
    if path == "/" || !path.starts_with('/') || path.ends_with('/') {
        return Err(invalid_export(
            "XDMF temporal dataset path must be a non-root canonical absolute path",
        ));
    }
    if path
        .split('/')
        .skip(1)
        .any(|segment| segment.is_empty() || matches!(segment, "." | ".."))
    {
        return Err(invalid_export(
            "XDMF temporal dataset path contains a forbidden segment",
        ));
    }
    Ok(())
}

fn is_xml_1_0_character(character: char) -> bool {
    matches!(character, '\u{0009}' | '\u{000A}' | '\u{000D}')
        || ('\u{0020}'..='\u{D7FF}').contains(&character)
        || ('\u{E000}'..='\u{FFFD}').contains(&character)
        || ('\u{10000}'..='\u{10FFFF}').contains(&character)
}

fn require_at_most(actual: usize, limit: usize, label: &str) -> Result<(), Diagnostic> {
    if actual > limit {
        Err(invalid_export(format!(
            "{label} {actual} exceeds configured limit {limit}",
        )))
    } else {
        Ok(())
    }
}

fn invalid_export(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXTERNAL_DATA_EXPORT, message)
}

struct ByteCounter {
    bytes: usize,
    limit: usize,
}

impl ByteCounter {
    const fn new(limit: usize) -> Self {
        Self { bytes: 0, limit }
    }
}

impl fmt::Write for ByteCounter {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        self.bytes = self.bytes.checked_add(value.len()).ok_or(fmt::Error)?;
        if self.bytes > self.limit {
            return Err(fmt::Error);
        }
        Ok(())
    }
}
