use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_meshing::{MeshQualityGate, SimplicialMesh};

mod api;
mod binary;
mod budget;
mod records;

use api::{DecodedElementBlock, DecodedMsh};
pub use api::{Msh41Policy, import_msh41};
use budget::DecodedBudget;

const MSH_VERSION: &str = "4.1";

/// Decoded-work and input-size policy applied by the owned bounded decoder.
///
/// Every declared loop is charged to one of these semantic count budgets.
/// Binary counts must also fit the minimum records encodable in the remaining
/// input. Aggregate decoded-byte and work budgets conservatively cover
/// structural indexes, decoded coordinates and connectivity, lookup state,
/// canonical output, and simplex topology closure; declaration-sized importer
/// storage also uses fallible reservation. The byte budget is a deterministic
/// logical account, not a promise of exact allocator RSS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DecoderLimits {
    /// Maximum complete input size.
    pub max_bytes: usize,
    /// Maximum number of decoded geometric-entity records.
    pub max_entities: usize,
    /// Maximum total decoded physical and boundary references in `$Entities`.
    pub max_entity_references: usize,
    /// Maximum decoded node-block count.
    pub max_node_blocks: usize,
    /// Maximum decoded element-block count.
    pub max_element_blocks: usize,
    /// Maximum decoded node count.
    pub max_nodes: usize,
    /// Maximum decoded element count, including ignored lower-dimensional elements.
    pub max_elements: usize,
    /// Maximum lower-dimensional elements decoded but omitted from canonical cells.
    pub max_ignored_elements: usize,
    /// Maximum aggregate logical bytes for decoder and canonical mesh state.
    pub max_decoded_bytes: usize,
    /// Maximum aggregate decode, lookup, and topology-construction work units.
    pub max_decoded_work: usize,
}

impl Default for DecoderLimits {
    fn default() -> Self {
        Self {
            max_bytes: 16 * 1024 * 1024,
            max_entities: 1_000_000,
            max_entity_references: 8_000_000,
            max_node_blocks: 100_000,
            max_element_blocks: 100_000,
            max_nodes: 1_000_000,
            max_elements: 2_000_000,
            max_ignored_elements: 16_384,
            max_decoded_bytes: 256 * 1024 * 1024,
            max_decoded_work: 32_000_000,
        }
    }
}

impl DecoderLimits {
    fn validate(self) -> Result<Self, Diagnostic> {
        if [
            self.max_bytes,
            self.max_entities,
            self.max_entity_references,
            self.max_node_blocks,
            self.max_element_blocks,
            self.max_nodes,
            self.max_elements,
            self.max_ignored_elements,
            self.max_decoded_bytes,
            self.max_decoded_work,
        ]
        .contains(&0)
        {
            return Err(invalid_import(
                "Gmsh import resource limits must all be positive",
            ));
        }
        Ok(self)
    }
}

/// Strict importer for full-dimensional affine simplex meshes in MSH 4.1.
///
/// This type accepts bytes rather than paths so filesystem policy and source
/// provenance stay outside accepted mesh identity.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Decoder {
    dimension: usize,
    quality_gate: MeshQualityGate,
    limits: DecoderLimits,
}

impl Decoder {
    /// Construct an importer for XY triangles (`dimension = 2`) or XYZ
    /// tetrahedra (`dimension = 3`).
    ///
    /// # Errors
    /// Returns `EQ0808` for an unsupported dimension or zero resource limit.
    pub fn new(
        dimension: usize,
        quality_gate: MeshQualityGate,
        limits: DecoderLimits,
    ) -> Result<Self, Diagnostic> {
        if !matches!(dimension, 2 | 3) {
            return Err(invalid_import(
                "Gmsh simplex import supports spatial dimension two or three",
            ));
        }
        Ok(Self {
            dimension,
            quality_gate,
            limits: limits.validate()?,
        })
    }

    /// Parse, validate, and reconstruct one admitted MSH 4.1 mesh.
    ///
    /// # Errors
    /// Returns `EQ0808` when the bytes exceed the admitted syntax, semantics,
    /// or resource boundary. Accepted syntax may still return `EQ0803` when
    /// the shared mesh contract rejects topology, geometry, or quality.
    pub fn import_bytes(self, bytes: &[u8]) -> Result<SimplicialMesh, Diagnostic> {
        self.decode(bytes).and_then(|decoded| {
            SimplicialMesh::new(
                self.dimension,
                decoded.vertices,
                decoded.cells,
                self.quality_gate,
            )
        })
    }

    /// Parse an ASCII MSH 4.1 image while retaining its element-block entity
    /// tags and normalized connectivity.
    ///
    /// Generated-provider adapters use this closed provenance path to map CAD
    /// entities to Mesh entities without coordinate classification. Caller
    /// imports that do not authenticate entity meaning should use
    /// [`Self::import_bytes`] and remain unlabelled.
    pub fn import_ascii_bytes_with_entities(self, bytes: &[u8]) -> Result<DecodedMsh, Diagnostic> {
        if declared_encoding(bytes)? != InputEncoding::Ascii {
            return Err(invalid_import(
                "entity-provenance import requires ASCII MSH 4.1",
            ));
        }
        let decoded = self.decode(bytes)?;
        let mesh = SimplicialMesh::new(
            self.dimension,
            decoded.vertices,
            decoded.cells,
            self.quality_gate,
        )?;
        Ok(DecodedMsh {
            mesh,
            element_blocks: decoded.element_blocks,
        })
    }

    fn decode(self, bytes: &[u8]) -> Result<DecodedMesh, Diagnostic> {
        if bytes.len() > self.limits.max_bytes {
            return Err(invalid_import(
                "Gmsh input exceeds the configured byte limit",
            ));
        }
        let encoding = declared_encoding(bytes)?;
        let mut decoded_budget = DecodedBudget::new(self.limits);
        let decoded = match encoding {
            InputEncoding::Ascii => {
                let text = std::str::from_utf8(bytes)
                    .map_err(|_| invalid_import("ASCII MSH 4.1 input must be valid UTF-8"))?;
                DecodedMesh::parse_ascii(text, self.dimension, self.limits, &mut decoded_budget)?
            }
            InputEncoding::Binary { size_t_size } => binary::parse(
                bytes,
                self.dimension,
                self.limits,
                size_t_size,
                &mut decoded_budget,
            )?,
        };
        Ok(decoded)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputEncoding {
    Ascii,
    Binary { size_t_size: usize },
}

fn declared_encoding(bytes: &[u8]) -> Result<InputEncoding, Diagnostic> {
    let mut lines = bytes.splitn(3, |&byte| byte == b'\n');
    let section = strip_carriage_return(
        lines
            .next()
            .ok_or_else(|| invalid_import("MSH input has no format section"))?,
    );
    if section != b"$MeshFormat" {
        return Err(invalid_import("MSH input must begin with $MeshFormat"));
    }
    let header = strip_carriage_return(
        lines
            .next()
            .ok_or_else(|| invalid_import("MSH input has no format header"))?,
    );
    let header = std::str::from_utf8(header)
        .map_err(|_| invalid_import("MSH format header must be ASCII"))?;
    let [version, file_type, size_t_size] = exact_fields::<3>(Some(header), "$MeshFormat header")?;
    if version != MSH_VERSION {
        return Err(invalid_import("Gmsh import requires MSH version 4.1"));
    }
    let file_type = parse_usize(file_type, "$MeshFormat file type")?;
    let size_t_size = parse_usize(size_t_size, "$MeshFormat data size")?;
    match (file_type, size_t_size) {
        (0, 8) => Ok(InputEncoding::Ascii),
        (1, 4 | 8) => Ok(InputEncoding::Binary { size_t_size }),
        (0, _) => Err(invalid_import(
            "ASCII MSH 4.1 import requires the data-size declaration 8",
        )),
        (1, _) => Err(invalid_import(
            "binary MSH 4.1 import requires a four- or eight-byte size_t declaration",
        )),
        _ => Err(invalid_import(
            "MSH 4.1 file type must be zero for ASCII or one for binary",
        )),
    }
}

fn strip_carriage_return(line: &[u8]) -> &[u8] {
    line.strip_suffix(b"\r").unwrap_or(line)
}

#[derive(Debug)]
struct DecodedMesh {
    vertices: Vec<Vec<f64>>,
    cells: Vec<Vec<usize>>,
    element_blocks: Vec<DecodedElementBlock>,
}

#[derive(Debug)]
struct DecodedNodes {
    vertices: Vec<Vec<f64>>,
    vertex_by_tag: HashMap<u64, usize>,
}

impl DecodedMesh {
    fn parse_ascii(
        text: &str,
        dimension: usize,
        limits: DecoderLimits,
        budget: &mut DecodedBudget,
    ) -> Result<Self, Diagnostic> {
        let sections = Sections::parse(text, budget)?;
        sections.validate_admitted_order()?;
        parse_mesh_format(sections.required("MeshFormat")?)?;
        if let Some(entities) = sections.optional("Entities")? {
            parse_entities(entities, dimension, limits, budget)?;
        }
        let nodes = parse_nodes(sections.required("Nodes")?, dimension, limits, budget)?;
        let (cells, element_blocks) = parse_elements(
            sections.required("Elements")?,
            dimension,
            limits,
            &nodes.vertex_by_tag,
            budget,
        )?;
        Ok(Self {
            vertices: nodes.vertices,
            cells,
            element_blocks,
        })
    }
}

#[derive(Debug)]
struct Section<'a> {
    name: &'a str,
    lines: Vec<&'a str>,
}

#[derive(Debug)]
struct Sections<'a>(Vec<Section<'a>>);

impl<'a> Sections<'a> {
    fn parse(text: &'a str, budget: &mut DecodedBudget) -> Result<Self, Diagnostic> {
        let line_count = text.lines().count();
        budget.charge_ascii_lines(line_count)?;
        let mut lines = fallible_vec(line_count, "ASCII line-index storage")?;
        lines.extend(text.lines().map(str::trim));
        let mut sections = fallible_vec(4, "MSH section-index storage")?;
        let mut index = 0;
        while index < lines.len() {
            if lines[index].is_empty() {
                index = checked_next_index(index, "MSH line cursor")?;
                continue;
            }
            let start = lines[index];
            let Some(name) = start.strip_prefix('$') else {
                return Err(invalid_import("text outside a delimited MSH section"));
            };
            if name.is_empty() || name.starts_with("End") {
                return Err(invalid_import("invalid MSH section delimiter"));
            }
            index = checked_next_index(index, "MSH section cursor")?;
            let mut body = Vec::new();
            while index < lines.len() && lines[index].strip_prefix("$End") != Some(name) {
                if lines[index].starts_with('$') {
                    return Err(invalid_import("nested or mismatched MSH section delimiter"));
                }
                if !lines[index].is_empty() {
                    fallible_push(&mut body, lines[index], "MSH section-body storage")?;
                }
                index = checked_next_index(index, "MSH section-body cursor")?;
            }
            if index == lines.len() {
                return Err(invalid_import("unterminated MSH section"));
            }
            fallible_push(
                &mut sections,
                Section { name, lines: body },
                "MSH section-index storage",
            )?;
            index = checked_next_index(index, "MSH section-end cursor")?;
        }
        Ok(Self(sections))
    }

    fn required(&self, name: &str) -> Result<&[&'a str], Diagnostic> {
        self.optional(name)?
            .ok_or_else(|| invalid_import(format!("MSH input requires one ${name} section")))
    }

    fn optional(&self, name: &str) -> Result<Option<&[&'a str]>, Diagnostic> {
        let mut matching = self.0.iter().filter(|section| section.name == name);
        let first = matching.next();
        if matching.next().is_some() {
            return Err(invalid_import(format!(
                "MSH input contains duplicate ${name} sections"
            )));
        }
        Ok(first.map(|section| section.lines.as_slice()))
    }

    fn validate_admitted_order(&self) -> Result<(), Diagnostic> {
        let admitted = match self.0.as_slice() {
            [format, nodes, elements] => {
                format.name == "MeshFormat" && nodes.name == "Nodes" && elements.name == "Elements"
            }
            [format, entities, nodes, elements] => {
                format.name == "MeshFormat"
                    && entities.name == "Entities"
                    && nodes.name == "Nodes"
                    && elements.name == "Elements"
            }
            _ => false,
        };
        if admitted {
            Ok(())
        } else {
            Err(invalid_import(
                "MSH input must contain only one ordered $MeshFormat, optional $Entities, $Nodes, and $Elements section",
            ))
        }
    }
}

fn parse_mesh_format(lines: &[&str]) -> Result<(), Diagnostic> {
    if lines.len() != 1 {
        return Err(invalid_import(
            "$MeshFormat must contain exactly one header line",
        ));
    }
    let fields = exact_fields::<3>(Some(lines[0]), "$MeshFormat header")?;
    if fields != [MSH_VERSION, "0", "8"] {
        return Err(invalid_import(
            "Gmsh import requires the ASCII MSH 4.1 header `4.1 0 8`",
        ));
    }
    Ok(())
}

fn parse_entities(
    lines: &[&str],
    requested_dimension: usize,
    limits: DecoderLimits,
    budget: &mut DecodedBudget,
) -> Result<(), Diagnostic> {
    let header = exact_fields::<4>(lines.first().copied(), "$Entities header")?;
    let counts = [
        parse_usize(header[0], "$Entities point count")?,
        parse_usize(header[1], "$Entities curve count")?,
        parse_usize(header[2], "$Entities surface count")?,
        parse_usize(header[3], "$Entities volume count")?,
    ];
    if counts
        .iter()
        .enumerate()
        .any(|(dimension, &count)| dimension > requested_dimension && count != 0)
    {
        return Err(invalid_import(
            "$Entities contains geometry above the requested mesh dimension",
        ));
    }
    let total = checked_sum(&counts, "$Entities count")?;
    let record_lines = total
        .checked_add(1)
        .ok_or_else(|| invalid_import("$Entities record count overflows usize"))?;
    if total > limits.max_entities || lines.len() != record_lines {
        return Err(invalid_import(
            "$Entities exceeds its limit or disagrees with its declared count",
        ));
    }
    budget.charge_entities(&counts)?;
    let mut references = 0usize;
    let mut entity_tags = [
        fallible_set(counts[0], "$Entities point-tag storage")?,
        fallible_set(counts[1], "$Entities curve-tag storage")?,
        fallible_set(counts[2], "$Entities surface-tag storage")?,
        fallible_set(counts[3], "$Entities volume-tag storage")?,
    ];
    let mut line_index = 1_usize;
    for (dimension, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            let line = take_declared_line(lines, &mut line_index, "$Entities record")?;
            let mut tokens = TokenCursor::new(line);
            let tag = parse_i32(tokens.next("$Entities tag")?, "$Entities tag")?;
            if tag <= 0 || !entity_tags[dimension].insert(tag) {
                return Err(invalid_import(
                    "entity tags must be positive and unique per dimension",
                ));
            }
            let coordinate_count = if dimension == 0 { 3 } else { 6 };
            for _ in 0..coordinate_count {
                parse_finite(tokens.next("$Entities coordinate")?, "$Entities coordinate")?;
            }
            let physical_count = parse_usize(
                tokens.next("$Entities physical tag count")?,
                "$Entities physical tag count",
            )?;
            if physical_count != 0 {
                return Err(invalid_import(
                    "physical-group membership is outside the admitted import boundary",
                ));
            }
            if dimension > 0 {
                let boundary_count = parse_usize(
                    tokens.next("$Entities boundary count")?,
                    "$Entities boundary count",
                )?;
                references = references
                    .checked_add(boundary_count)
                    .ok_or_else(|| invalid_import("$Entities reference count overflows"))?;
                if references > limits.max_entity_references {
                    return Err(invalid_import(
                        "$Entities boundary references exceed the configured limit",
                    ));
                }
                budget.charge_entity_references(boundary_count)?;
                for _ in 0..boundary_count {
                    if parse_i32(
                        tokens.next("$Entities boundary tag")?,
                        "$Entities boundary tag",
                    )? == 0
                    {
                        return Err(invalid_import("entity boundary tags must be nonzero"));
                    }
                }
            }
            tokens.finish("$Entities record")?;
        }
    }
    Ok(())
}

fn parse_nodes(
    lines: &[&str],
    dimension: usize,
    limits: DecoderLimits,
    budget: &mut DecodedBudget,
) -> Result<DecodedNodes, Diagnostic> {
    let [blocks, total, minimum, maximum] =
        exact_fields::<4>(lines.first().copied(), "$Nodes header")?;
    let header = records::Header {
        blocks: parse_usize(blocks, "$Nodes block count")?,
        total: parse_usize(total, "$Nodes total count")?,
        minimum: parse_u64(minimum, "$Nodes minimum tag")?,
        maximum: parse_u64(maximum, "$Nodes maximum tag")?,
    };
    if header.blocks > lines.len() || header.total > lines.len() {
        return Err(invalid_import(
            "$Nodes declarations exceed the available records",
        ));
    }
    let block_count = header.blocks;
    let mut nodes = records::Nodes::new(header, dimension, limits, budget)?;
    let mut line_index = 1;
    for _ in 0..block_count {
        let block = exact_fields::<4>(
            Some(take_declared_line(
                lines,
                &mut line_index,
                "$Nodes block header",
            )?),
            "$Nodes block header",
        )?;
        let entity_dim = parse_i32(block[0], "$Nodes entity dimension")?;
        let entity_tag = parse_i32(block[1], "$Nodes entity tag")?;
        let parametric = parse_usize(block[2], "$Nodes parametric flag")?;
        let count = parse_usize(block[3], "$Nodes block count")?;
        nodes.block(entity_dim, entity_tag, parametric != 0, count)?;
        for _ in 0..count {
            let line = take_declared_line(lines, &mut line_index, "$Nodes node tag")?;
            let [value] = exact_fields::<1>(Some(line), "$Nodes node tag")?;
            nodes.tag(parse_u64(value, "$Nodes node tag")?)?;
        }
        for _ in 0..count {
            let line = take_declared_line(lines, &mut line_index, "$Nodes coordinate")?;
            let coordinate = exact_fields::<3>(Some(line), "$Nodes coordinate")?;
            nodes.coordinate([
                parse_finite(coordinate[0], "$Nodes x coordinate")?,
                parse_finite(coordinate[1], "$Nodes y coordinate")?,
                parse_finite(coordinate[2], "$Nodes z coordinate")?,
            ])?;
        }
    }
    if line_index != lines.len() {
        return Err(invalid_import("$Nodes records exceed the declared blocks"));
    }
    nodes.finish()
}

fn parse_elements(
    lines: &[&str],
    dimension: usize,
    limits: DecoderLimits,
    vertex_by_tag: &HashMap<u64, usize>,
    budget: &mut DecodedBudget,
) -> Result<(Vec<Vec<usize>>, Vec<DecodedElementBlock>), Diagnostic> {
    let [blocks, total, minimum, maximum] =
        exact_fields::<4>(lines.first().copied(), "$Elements header")?;
    let header = records::Header {
        blocks: parse_usize(blocks, "$Elements block count")?,
        total: parse_usize(total, "$Elements total count")?,
        minimum: parse_u64(minimum, "$Elements minimum tag")?,
        maximum: parse_u64(maximum, "$Elements maximum tag")?,
    };
    if header.blocks > lines.len() || header.total > lines.len() {
        return Err(invalid_import(
            "$Elements declarations exceed the available records",
        ));
    }
    let block_count = header.blocks;
    let mut elements = records::Elements::new(header, dimension, limits, budget, true)?;
    let mut line_index = 1;
    for _ in 0..block_count {
        let header = exact_fields::<4>(
            Some(take_declared_line(
                lines,
                &mut line_index,
                "$Elements block header",
            )?),
            "$Elements block header",
        )?;
        let entity_dim = parse_i32(header[0], "$Elements entity dimension")?;
        let entity_tag = parse_i32(header[1], "$Elements entity tag")?;
        let element_type = u32::try_from(parse_usize(header[2], "$Elements element type")?)
            .map_err(|_| invalid_import("MSH element type exceeds u32"))?;
        let count = parse_usize(header[3], "$Elements block count")?;
        let mut block = elements.block(entity_dim, entity_tag, element_type, count, budget)?;
        for _ in 0..count {
            let line = take_declared_line(lines, &mut line_index, "$Elements record")?;
            let mut tokens = TokenCursor::new(line);
            let tag = parse_u64(
                tokens.next("$Elements element tag")?,
                "$Elements element tag",
            )?;
            let mut nodes = [0; 4];
            let nodes = &mut nodes[..=block.dimension];
            for node in &mut *nodes {
                *node = parse_u64(tokens.next("$Elements node tag")?, "$Elements node tag")?;
            }
            tokens.finish("$Elements record")?;
            elements.element(&mut block, tag, nodes, vertex_by_tag)?;
        }
        elements.finish_block(block);
    }
    if line_index != lines.len() {
        return Err(invalid_import(
            "$Elements records exceed the declared blocks",
        ));
    }
    elements.finish()
}

const fn linear_simplex_type(dimension: usize) -> u32 {
    match dimension {
        0 => 15,
        1 => 1,
        2 => 2,
        3 => 4,
        _ => 0,
    }
}

fn exact_fields<'a, const COUNT: usize>(
    line: Option<&'a str>,
    context: &str,
) -> Result<[&'a str; COUNT], Diagnostic> {
    let line = line.ok_or_else(|| invalid_import(format!("missing {context}")))?;
    let mut tokens = TokenCursor::new(line);
    let mut values = [""; COUNT];
    for value in &mut values {
        *value = tokens.next(context)?;
    }
    tokens.finish(context)?;
    Ok(values)
}

struct TokenCursor<'a> {
    tokens: std::str::SplitAsciiWhitespace<'a>,
}

impl<'a> TokenCursor<'a> {
    fn new(line: &'a str) -> Self {
        Self {
            tokens: line.split_ascii_whitespace(),
        }
    }

    fn next(&mut self, context: &str) -> Result<&'a str, Diagnostic> {
        self.tokens
            .next()
            .ok_or_else(|| invalid_import(format!("missing token in {context}")))
    }

    fn finish(mut self, context: &str) -> Result<(), Diagnostic> {
        if self.tokens.next().is_some() {
            Err(invalid_import(format!(
                "unexpected trailing token in {context}",
            )))
        } else {
            Ok(())
        }
    }
}

fn parse_usize(value: &str, context: &str) -> Result<usize, Diagnostic> {
    value
        .parse()
        .map_err(|_| invalid_import(format!("invalid non-negative integer in {context}")))
}

fn parse_u64(value: &str, context: &str) -> Result<u64, Diagnostic> {
    value
        .parse()
        .map_err(|_| invalid_import(format!("invalid unsigned tag in {context}")))
}

fn parse_i32(value: &str, context: &str) -> Result<i32, Diagnostic> {
    value
        .parse()
        .map_err(|_| invalid_import(format!("invalid signed integer in {context}")))
}

fn parse_finite(value: &str, context: &str) -> Result<f64, Diagnostic> {
    let value = value
        .parse::<f64>()
        .map_err(|_| invalid_import(format!("invalid floating-point value in {context}")))?;
    if !value.is_finite() {
        return Err(invalid_import(format!("non-finite value in {context}")));
    }
    Ok(value)
}

fn decoded_coordinate(dimension: usize, x: f64, y: f64, z: f64) -> Result<Vec<f64>, Diagnostic> {
    match dimension {
        2 if z == 0.0 => fallible_vec_from_slice(&[x, y], "decoded vertex coordinate"),
        2 => Err(invalid_import(
            "two-dimensional import requires every node in the XY plane",
        )),
        3 => fallible_vec_from_slice(&[x, y, z], "decoded vertex coordinate"),
        _ => Err(invalid_import(
            "Gmsh simplex import supports spatial dimension two or three",
        )),
    }
}

fn checked_sum(values: &[usize], context: &str) -> Result<usize, Diagnostic> {
    values.iter().try_fold(0usize, |sum, &value| {
        sum.checked_add(value)
            .ok_or_else(|| invalid_import(format!("{context} overflows usize")))
    })
}

fn checked_next_index(index: usize, context: &str) -> Result<usize, Diagnostic> {
    index
        .checked_add(1)
        .ok_or_else(|| invalid_import(format!("{context} overflows usize")))
}

fn take_declared_line<'a>(
    lines: &[&'a str],
    cursor: &mut usize,
    context: &str,
) -> Result<&'a str, Diagnostic> {
    let line = lines
        .get(*cursor)
        .copied()
        .ok_or_else(|| invalid_import(format!("missing {context}")))?;
    *cursor = checked_next_index(*cursor, context)?;
    Ok(line)
}

fn fallible_vec<T>(capacity: usize, context: &str) -> Result<Vec<T>, Diagnostic> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| invalid_import(format!("{context} exceeds available decoded memory")))?;
    Ok(values)
}

fn fallible_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    context: &str,
) -> Result<(), Diagnostic> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| invalid_import(format!("{context} exceeds available decoded memory")))
}

fn fallible_push<T>(values: &mut Vec<T>, value: T, context: &str) -> Result<(), Diagnostic> {
    if values.len() == values.capacity() {
        values
            .try_reserve_exact(1)
            .map_err(|_| invalid_import(format!("{context} exceeds available decoded memory")))?;
    }
    values.push(value);
    Ok(())
}

fn fallible_vec_from_slice<T: Copy>(values: &[T], context: &str) -> Result<Vec<T>, Diagnostic> {
    let mut decoded = fallible_vec(values.len(), context)?;
    decoded.extend_from_slice(values);
    Ok(decoded)
}

fn fallible_set<T: Eq + Hash>(capacity: usize, context: &str) -> Result<HashSet<T>, Diagnostic> {
    let mut values = HashSet::new();
    values
        .try_reserve(capacity)
        .map_err(|_| invalid_import(format!("{context} exceeds available decoded memory")))?;
    Ok(values)
}

fn fallible_map<K: Eq + Hash, V>(
    capacity: usize,
    context: &str,
) -> Result<HashMap<K, V>, Diagnostic> {
    let mut values = HashMap::new();
    values
        .try_reserve(capacity)
        .map_err(|_| invalid_import(format!("{context} exceeds available decoded memory")))?;
    Ok(values)
}

fn invalid_import(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_MESH_IMPORT, message)
}

#[cfg(test)]
mod tests;
