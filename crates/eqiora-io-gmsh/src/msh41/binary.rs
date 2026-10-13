use std::collections::HashMap;

use eqiora_core::Diagnostic;

use super::{
    DecodedBudget, DecodedMesh, DecodedNodes, DecoderLimits, MSH_VERSION, exact_fields,
    fallible_set, invalid_import, parse_usize, records, strip_carriage_return,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

pub(super) fn parse(
    bytes: &[u8],
    dimension: usize,
    limits: DecoderLimits,
    declared_size_t_size: usize,
    budget: &mut DecodedBudget,
) -> Result<DecodedMesh, Diagnostic> {
    let mut cursor = BinaryCursor::new(bytes);
    cursor.expect_line(b"$MeshFormat", "$MeshFormat section")?;
    let header = cursor.read_ascii_line("$MeshFormat header")?;
    let header = std::str::from_utf8(header)
        .map_err(|_| invalid_import("binary MSH format header must be ASCII"))?;
    let header = exact_fields::<3>(Some(header), "$MeshFormat header")?;
    if header[0] != MSH_VERSION
        || header[1] != "1"
        || parse_usize(header[2], "$MeshFormat data size")? != declared_size_t_size
    {
        return Err(invalid_import(
            "binary Gmsh import requires the MSH 4.1 header with file type one",
        ));
    }
    cursor.size_t_size = declared_size_t_size;
    cursor.endian = cursor.read_endian_marker()?;
    cursor.expect_line_break("binary endian marker")?;
    cursor.expect_line(b"$EndMeshFormat", "$EndMeshFormat delimiter")?;

    let entities = cursor.consume_line(b"$Entities")?;
    if entities {
        parse_entities(&mut cursor, dimension, limits, budget)?;
        cursor.end_section(b"$EndEntities", "$Entities section")?;
    }

    cursor.expect_line(b"$Nodes", "$Nodes section")?;
    let nodes = parse_nodes(&mut cursor, dimension, limits, budget)?;
    cursor.end_section(b"$EndNodes", "$Nodes section")?;

    cursor.expect_line(b"$Elements", "$Elements section")?;
    let cells = parse_elements(&mut cursor, dimension, limits, &nodes.vertex_by_tag, budget)?;
    cursor.end_section(b"$EndElements", "$Elements section")?;
    cursor.expect_end()?;

    Ok(DecodedMesh {
        vertices: nodes.vertices,
        cells,
        element_blocks: Vec::new(),
    })
}

fn parse_entities(
    cursor: &mut BinaryCursor<'_>,
    dimension: usize,
    limits: DecoderLimits,
    budget: &mut DecodedBudget,
) -> Result<(), Diagnostic> {
    let mut counts = [0_usize; 4];
    for count in &mut counts {
        *count = cursor.read_usize("$Entities count")?;
    }
    if counts
        .iter()
        .enumerate()
        .any(|(entity_dimension, &count)| entity_dimension > dimension && count != 0)
    {
        return Err(invalid_import(
            "$Entities contains geometry above the requested mesh dimension",
        ));
    }
    let total = counts.iter().try_fold(0_usize, |sum, &count| {
        sum.checked_add(count)
            .ok_or_else(|| invalid_import("$Entities count overflows usize"))
    })?;
    if total > limits.max_entities {
        return Err(invalid_import(
            "$Entities exceeds the configured entity limit",
        ));
    }
    budget.charge_entities(&counts)?;

    cursor.ensure_minimum_remaining(
        &[
            (counts[0], cursor.entity_record_size(0)?),
            (counts[1], cursor.entity_record_size(1)?),
            (counts[2], cursor.entity_record_size(2)?),
            (counts[3], cursor.entity_record_size(3)?),
        ],
        "$Entities declarations",
    )?;

    let mut tags = [
        fallible_set(counts[0], "$Entities point-tag storage")?,
        fallible_set(counts[1], "$Entities curve-tag storage")?,
        fallible_set(counts[2], "$Entities surface-tag storage")?,
        fallible_set(counts[3], "$Entities volume-tag storage")?,
    ];
    let mut references = 0_usize;
    for (entity_dimension, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            let tag = cursor.read_i32("$Entities tag")?;
            if tag <= 0 || !tags[entity_dimension].insert(tag) {
                return Err(invalid_import(
                    "entity tags must be positive and unique per dimension",
                ));
            }
            let coordinate_count = if entity_dimension == 0 { 3 } else { 6 };
            for _ in 0..coordinate_count {
                cursor.read_finite("$Entities coordinate")?;
            }
            if cursor.read_usize("$Entities physical tag count")? != 0 {
                return Err(invalid_import(
                    "physical-group membership is outside the admitted import boundary",
                ));
            }
            if entity_dimension > 0 {
                let count = cursor.read_usize("$Entities boundary count")?;
                references = references
                    .checked_add(count)
                    .ok_or_else(|| invalid_import("$Entities reference count overflows usize"))?;
                if references > limits.max_entity_references {
                    return Err(invalid_import(
                        "$Entities exceeds the configured boundary-reference limit",
                    ));
                }
                budget.charge_entity_references(count)?;
                cursor.ensure_minimum_remaining(
                    &[(count, 4)],
                    "$Entities boundary-reference declaration",
                )?;
                for _ in 0..count {
                    if cursor.read_i32("$Entities boundary tag")? == 0 {
                        return Err(invalid_import("entity boundary tags must be nonzero"));
                    }
                }
            }
        }
    }
    Ok(())
}

fn parse_nodes(
    cursor: &mut BinaryCursor<'_>,
    dimension: usize,
    limits: DecoderLimits,
    budget: &mut DecodedBudget,
) -> Result<DecodedNodes, Diagnostic> {
    let header = records::Header {
        blocks: cursor.read_usize("$Nodes block count")?,
        total: cursor.read_usize("$Nodes total count")?,
        minimum: cursor.read_size_t("$Nodes minimum tag")?,
        maximum: cursor.read_size_t("$Nodes maximum tag")?,
    };
    cursor.ensure_minimum_remaining(
        &[
            (header.blocks, cursor.node_block_header_size()?),
            (header.total, cursor.node_record_size()?),
        ],
        "$Nodes declarations",
    )?;
    let block_count = header.blocks;
    let mut nodes = records::Nodes::new(header, dimension, limits, budget)?;
    for _ in 0..block_count {
        let entity_dim = cursor.read_i32("$Nodes entity dimension")?;
        let entity_tag = cursor.read_i32("$Nodes entity tag")?;
        let parametric = cursor.read_i32("$Nodes parametric flag")?;
        let count = cursor.read_usize("$Nodes block count")?;
        nodes.block(entity_dim, entity_tag, parametric != 0, count)?;
        cursor.ensure_minimum_remaining(
            &[(count, cursor.node_record_size()?)],
            "$Nodes block declaration",
        )?;
        for _ in 0..count {
            nodes.tag(cursor.read_size_t("$Nodes node tag")?)?;
        }
        for _ in 0..count {
            nodes.coordinate([
                cursor.read_finite("$Nodes x coordinate")?,
                cursor.read_finite("$Nodes y coordinate")?,
                cursor.read_finite("$Nodes z coordinate")?,
            ])?;
        }
    }
    nodes.finish()
}

fn parse_elements(
    cursor: &mut BinaryCursor<'_>,
    dimension: usize,
    limits: DecoderLimits,
    vertex_by_tag: &HashMap<u64, usize>,
    budget: &mut DecodedBudget,
) -> Result<Vec<Vec<usize>>, Diagnostic> {
    let header = records::Header {
        blocks: cursor.read_usize("$Elements block count")?,
        total: cursor.read_usize("$Elements total count")?,
        minimum: cursor.read_size_t("$Elements minimum tag")?,
        maximum: cursor.read_size_t("$Elements maximum tag")?,
    };
    cursor.ensure_minimum_remaining(
        &[
            (header.blocks, cursor.element_block_header_size()?),
            (header.total, cursor.minimum_element_record_size()?),
        ],
        "$Elements declarations",
    )?;
    let block_count = header.blocks;
    let mut elements = records::Elements::new(header, dimension, limits, budget, false)?;
    for _ in 0..block_count {
        let entity_dim = cursor.read_i32("$Elements entity dimension")?;
        let entity_tag = cursor.read_i32("$Elements entity tag")?;
        let element_type = u32::try_from(cursor.read_i32("$Elements element type")?)
            .map_err(|_| invalid_import("MSH element type must be positive"))?;
        let count = cursor.read_usize("$Elements block count")?;
        let mut block = elements.block(entity_dim, entity_tag, element_type, count, budget)?;
        cursor.ensure_minimum_remaining(
            &[(count, cursor.element_record_size(block.dimension)?)],
            "$Elements block declaration",
        )?;
        for _ in 0..count {
            let tag = cursor.read_size_t("$Elements element tag")?;
            let mut nodes = [0; 4];
            let nodes = &mut nodes[..=block.dimension];
            for node in &mut *nodes {
                *node = cursor.read_size_t("$Elements node tag")?;
            }
            elements.element(&mut block, tag, nodes, vertex_by_tag)?;
        }
        elements.finish_block(block);
    }
    elements.finish().map(|(cells, _)| cells)
}

struct BinaryCursor<'a> {
    bytes: &'a [u8],
    position: usize,
    endian: Endian,
    size_t_size: usize,
}

impl<'a> BinaryCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            position: 0,
            endian: Endian::Little,
            size_t_size: 0,
        }
    }

    fn read_endian_marker(&mut self) -> Result<Endian, Diagnostic> {
        let marker = self.take(4, "binary endian marker")?;
        match marker {
            [1, 0, 0, 0] => Ok(Endian::Little),
            [0, 0, 0, 1] => Ok(Endian::Big),
            _ => Err(invalid_import(
                "binary MSH endian marker must encode the integer one",
            )),
        }
    }

    fn encoded_size_t_size(&self) -> Result<usize, Diagnostic> {
        match self.size_t_size {
            size @ (4 | 8) => Ok(size),
            _ => Err(invalid_import(
                "binary MSH size_t width was not admitted by the format header",
            )),
        }
    }

    fn entity_record_size(&self, dimension: usize) -> Result<usize, Diagnostic> {
        let coordinate_bytes = if dimension == 0 { 3 * 8 } else { 6 * 8 };
        let count_fields = if dimension == 0 { 1 } else { 2 };
        let count_bytes = count_fields * self.encoded_size_t_size()?;
        4_usize
            .checked_add(coordinate_bytes)
            .and_then(|size| size.checked_add(count_bytes))
            .ok_or_else(|| invalid_import("$Entities minimum record size overflows usize"))
    }

    fn node_block_header_size(&self) -> Result<usize, Diagnostic> {
        self.encoded_size_t_size()?
            .checked_add(3 * 4)
            .ok_or_else(|| invalid_import("$Nodes block-header size overflows usize"))
    }

    fn node_record_size(&self) -> Result<usize, Diagnostic> {
        self.encoded_size_t_size()?
            .checked_add(3 * 8)
            .ok_or_else(|| invalid_import("$Nodes minimum record size overflows usize"))
    }

    fn element_block_header_size(&self) -> Result<usize, Diagnostic> {
        self.encoded_size_t_size()?
            .checked_add(3 * 4)
            .ok_or_else(|| invalid_import("$Elements block-header size overflows usize"))
    }

    fn minimum_element_record_size(&self) -> Result<usize, Diagnostic> {
        self.element_record_size(0)
    }

    fn element_record_size(&self, dimension: usize) -> Result<usize, Diagnostic> {
        let width = self.encoded_size_t_size()?;
        dimension
            .checked_add(2)
            .and_then(|fields| fields.checked_mul(width))
            .ok_or_else(|| invalid_import("$Elements minimum record size overflows usize"))
    }

    fn ensure_minimum_remaining(
        &self,
        terms: &[(usize, usize)],
        context: &str,
    ) -> Result<(), Diagnostic> {
        let required = terms.iter().try_fold(0_usize, |sum, &(count, width)| {
            count
                .checked_mul(width)
                .and_then(|bytes| sum.checked_add(bytes))
                .ok_or_else(|| invalid_import(format!("{context} byte budget overflows usize")))
        })?;
        let remaining = self.bytes.len().saturating_sub(self.position);
        if required > remaining {
            Err(invalid_import(format!(
                "{context} exceed the minimum records encodable in the remaining input",
            )))
        } else {
            Ok(())
        }
    }

    fn read_i32(&mut self, context: &str) -> Result<i32, Diagnostic> {
        let bytes: [u8; 4] = self
            .take(4, context)?
            .try_into()
            .expect("the bounded cursor returned four bytes");
        Ok(match self.endian {
            Endian::Little => i32::from_le_bytes(bytes),
            Endian::Big => i32::from_be_bytes(bytes),
        })
    }

    fn read_size_t(&mut self, context: &str) -> Result<u64, Diagnostic> {
        match self.size_t_size {
            4 => {
                let bytes: [u8; 4] = self
                    .take(4, context)?
                    .try_into()
                    .expect("the bounded cursor returned four bytes");
                Ok(match self.endian {
                    Endian::Little => u32::from_le_bytes(bytes),
                    Endian::Big => u32::from_be_bytes(bytes),
                }
                .into())
            }
            8 => {
                let bytes: [u8; 8] = self
                    .take(8, context)?
                    .try_into()
                    .expect("the bounded cursor returned eight bytes");
                Ok(match self.endian {
                    Endian::Little => u64::from_le_bytes(bytes),
                    Endian::Big => u64::from_be_bytes(bytes),
                })
            }
            _ => Err(invalid_import(
                "binary MSH size_t width was not admitted by the format header",
            )),
        }
    }

    fn read_usize(&mut self, context: &str) -> Result<usize, Diagnostic> {
        usize::try_from(self.read_size_t(context)?)
            .map_err(|_| invalid_import(format!("{context} exceeds local usize")))
    }

    fn read_finite(&mut self, context: &str) -> Result<f64, Diagnostic> {
        let bytes: [u8; 8] = self
            .take(8, context)?
            .try_into()
            .expect("the bounded cursor returned eight bytes");
        let value = match self.endian {
            Endian::Little => f64::from_le_bytes(bytes),
            Endian::Big => f64::from_be_bytes(bytes),
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err(invalid_import(format!("non-finite value in {context}")))
        }
    }

    fn take(&mut self, count: usize, context: &str) -> Result<&'a [u8], Diagnostic> {
        let end = self
            .position
            .checked_add(count)
            .ok_or_else(|| invalid_import(format!("{context} byte range overflows usize")))?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| invalid_import(format!("truncated binary data in {context}")))?;
        self.position = end;
        Ok(bytes)
    }

    fn read_ascii_line(&mut self, context: &str) -> Result<&'a [u8], Diagnostic> {
        let tail = self
            .bytes
            .get(self.position..)
            .ok_or_else(|| invalid_import(format!("missing {context}")))?;
        let length = tail
            .iter()
            .position(|&byte| byte == b'\n')
            .ok_or_else(|| invalid_import(format!("unterminated {context}")))?;
        let line = strip_carriage_return(&tail[..length]);
        self.position = self
            .position
            .checked_add(length + 1)
            .ok_or_else(|| invalid_import(format!("{context} position overflows usize")))?;
        Ok(line)
    }

    fn expect_line(&mut self, expected: &[u8], context: &str) -> Result<(), Diagnostic> {
        if self.read_ascii_line(context)? == expected {
            Ok(())
        } else {
            Err(invalid_import(format!("invalid {context}")))
        }
    }

    fn consume_line(&mut self, expected: &[u8]) -> Result<bool, Diagnostic> {
        let original = self.position;
        let line = self.read_ascii_line("MSH section delimiter")?;
        if line == expected {
            Ok(true)
        } else {
            self.position = original;
            Ok(false)
        }
    }

    fn expect_line_break(&mut self, context: &str) -> Result<(), Diagnostic> {
        if self.bytes.get(self.position) == Some(&b'\n') {
            self.position += 1;
            return Ok(());
        }
        if self
            .bytes
            .get(self.position..self.position.saturating_add(2))
            == Some(b"\r\n")
        {
            self.position += 2;
            return Ok(());
        }
        Err(invalid_import(format!(
            "{context} must be followed by a line break",
        )))
    }

    fn end_section(&mut self, delimiter: &[u8], context: &str) -> Result<(), Diagnostic> {
        self.expect_line_break(context)?;
        self.expect_line(delimiter, context)
    }

    fn expect_end(&self) -> Result<(), Diagnostic> {
        if self.bytes[self.position..]
            .iter()
            .all(|byte| byte.is_ascii_whitespace())
        {
            Ok(())
        } else {
            Err(invalid_import(
                "binary MSH input contains an unsupported or trailing section",
            ))
        }
    }
}
