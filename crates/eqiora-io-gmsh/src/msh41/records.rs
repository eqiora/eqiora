//! Encoding-independent admission and accumulation of MSH node/element records.

use std::collections::{HashMap, HashSet};

use eqiora_core::Diagnostic;

use super::{
    DecodedBudget, DecodedElementBlock, DecodedNodes, DecoderLimits, decoded_coordinate,
    fallible_map, fallible_reserve, fallible_set, fallible_vec, invalid_import,
    linear_simplex_type,
};

pub(super) struct Header {
    pub(super) blocks: usize,
    pub(super) total: usize,
    pub(super) minimum: u64,
    pub(super) maximum: u64,
}

impl Header {
    fn validate(&self, blocks: usize, records: usize, section: &str) -> Result<(), Diagnostic> {
        if self.blocks == 0 || self.total == 0 || self.blocks > blocks || self.total > records {
            return Err(invalid_import(format!(
                "{section} count is zero or exceeds its resource limit"
            )));
        }
        Ok(())
    }

    fn add_block(&self, total: &mut usize, count: usize, section: &str) -> Result<(), Diagnostic> {
        *total = total
            .checked_add(count)
            .ok_or_else(|| invalid_import(format!("{section} block counts overflow usize")))?;
        if *total > self.total {
            return Err(invalid_import(format!(
                "{section} block counts exceed the declared bounded total"
            )));
        }
        Ok(())
    }
}

#[derive(Default)]
struct TagBounds(Option<(u64, u64)>);

impl TagBounds {
    fn observe(&mut self, tag: u64) {
        self.0 = Some(match self.0 {
            Some((minimum, maximum)) => (minimum.min(tag), maximum.max(tag)),
            None => (tag, tag),
        });
    }

    fn agrees(&self, header: &Header) -> bool {
        self.0 == Some((header.minimum, header.maximum))
    }
}

fn entity_dimension(raw: i32, tag: i32, dimension: usize) -> Result<usize, Diagnostic> {
    if raw < 0 || raw as usize > dimension || tag <= 0 {
        return Err(invalid_import(
            "blocks require an admitted entity dimension and positive tag",
        ));
    }
    Ok(raw as usize)
}

pub(super) struct Nodes {
    header: Header,
    dimension: usize,
    checked_total: usize,
    bounds: TagBounds,
    decoded: DecodedNodes,
}

impl Nodes {
    pub(super) fn new(
        header: Header,
        dimension: usize,
        limits: DecoderLimits,
        budget: &mut DecodedBudget,
    ) -> Result<Self, Diagnostic> {
        header.validate(limits.max_node_blocks, limits.max_nodes, "$Nodes")?;
        budget.charge_nodes(header.blocks, header.total, dimension)?;
        let decoded = DecodedNodes {
            vertices: fallible_vec(header.total, "decoded vertex storage")?,
            vertex_by_tag: fallible_map(header.total, "$Nodes tag-to-vertex storage")?,
        };
        Ok(Self {
            header,
            dimension,
            checked_total: 0,
            bounds: TagBounds::default(),
            decoded,
        })
    }

    pub(super) fn block(
        &mut self,
        dimension: i32,
        tag: i32,
        parametric: bool,
        count: usize,
    ) -> Result<(), Diagnostic> {
        entity_dimension(dimension, tag, self.dimension)?;
        if parametric {
            return Err(invalid_import(
                "node blocks require no parametric coordinates",
            ));
        }
        self.header
            .add_block(&mut self.checked_total, count, "$Nodes")
    }

    pub(super) fn tag(&mut self, tag: u64) -> Result<(), Diagnostic> {
        let index = self.decoded.vertex_by_tag.len();
        if tag == 0 || self.decoded.vertex_by_tag.insert(tag, index).is_some() {
            return Err(invalid_import("node tags must be positive and unique"));
        }
        self.bounds.observe(tag);
        Ok(())
    }

    pub(super) fn coordinate(&mut self, xyz: [f64; 3]) -> Result<(), Diagnostic> {
        self.decoded
            .vertices
            .push(decoded_coordinate(self.dimension, xyz[0], xyz[1], xyz[2])?);
        Ok(())
    }

    pub(super) fn finish(self) -> Result<DecodedNodes, Diagnostic> {
        if self.checked_total != self.header.total
            || self.decoded.vertices.len() != self.header.total
            || self.decoded.vertex_by_tag.len() != self.header.total
            || !self.bounds.agrees(&self.header)
        {
            return Err(invalid_import(
                "$Nodes records disagree with declared totals or tag bounds",
            ));
        }
        Ok(self.decoded)
    }
}

pub(super) struct Elements {
    header: Header,
    dimension: usize,
    checked_total: usize,
    top_dimensional: usize,
    bounds: TagBounds,
    tags: HashSet<u64>,
    cells: Vec<Vec<usize>>,
    // Only the ASCII provenance entry point consumes block assignments.
    blocks: Option<Vec<DecodedElementBlock>>,
}

impl Elements {
    pub(super) fn new(
        header: Header,
        dimension: usize,
        limits: DecoderLimits,
        budget: &mut DecodedBudget,
        retain_blocks: bool,
    ) -> Result<Self, Diagnostic> {
        header.validate(limits.max_element_blocks, limits.max_elements, "$Elements")?;
        budget.charge_element_blocks(header.blocks)?;
        let tags = fallible_set(header.total, "$Elements unique-tag storage")?;
        let blocks = if retain_blocks {
            Some(fallible_vec(
                header.blocks,
                "$Elements provenance-block storage",
            )?)
        } else {
            None
        };
        Ok(Self {
            header,
            dimension,
            checked_total: 0,
            top_dimensional: 0,
            bounds: TagBounds::default(),
            tags,
            cells: Vec::new(),
            blocks,
        })
    }

    pub(super) fn block(
        &mut self,
        dimension: i32,
        tag: i32,
        element_type: u32,
        count: usize,
        budget: &mut DecodedBudget,
    ) -> Result<DecodedElementBlock, Diagnostic> {
        let dimension = entity_dimension(dimension, tag, self.dimension)?;
        if element_type != linear_simplex_type(dimension) {
            return Err(invalid_import(
                "element blocks must use the linear simplex type of their entity dimension",
            ));
        }
        self.header
            .add_block(&mut self.checked_total, count, "$Elements")?;
        budget.charge_elements(count, dimension, self.dimension)?;
        if dimension == self.dimension {
            self.top_dimensional = self
                .top_dimensional
                .checked_add(count)
                .ok_or_else(|| invalid_import("top-dimensional element count overflows usize"))?;
            fallible_reserve(
                &mut self.cells,
                count,
                "decoded top-dimensional cell storage",
            )?;
        }
        Ok(DecodedElementBlock {
            dimension,
            entity_tag: tag as u32,
            elements: if self.blocks.is_some() {
                fallible_vec(count, "$Elements provenance storage")?
            } else {
                Vec::new()
            },
        })
    }

    pub(super) fn element(
        &mut self,
        block: &mut DecodedElementBlock,
        tag: u64,
        nodes: &[u64],
        vertex_by_tag: &HashMap<u64, usize>,
    ) -> Result<(), Diagnostic> {
        if nodes.len() != block.dimension + 1 || nodes.len() > 4 {
            return Err(invalid_import(
                "element connectivity disagrees with its simplex dimension",
            ));
        }
        if tag == 0 || !self.tags.insert(tag) {
            return Err(invalid_import("element tags must be positive and unique"));
        }
        self.bounds.observe(tag);
        let mut vertices = [0; 4];
        for (node, vertex) in nodes.iter().zip(&mut vertices) {
            if *node == 0 {
                return Err(invalid_import("element node tags must be positive"));
            }
            *vertex = *vertex_by_tag
                .get(node)
                .ok_or_else(|| invalid_import("MSH element references an unknown node tag"))?;
        }
        let copy = || {
            let mut element = fallible_vec(nodes.len(), "decoded simplex-connectivity storage")?;
            element.extend_from_slice(&vertices[..nodes.len()]);
            Ok::<_, Diagnostic>(element)
        };
        if block.dimension == self.dimension {
            self.cells.push(copy()?);
        }
        if self.blocks.is_some() {
            block.elements.push(copy()?);
        }
        Ok(())
    }

    pub(super) fn finish_block(&mut self, block: DecodedElementBlock) {
        if let Some(blocks) = &mut self.blocks {
            blocks.push(block);
        }
    }

    pub(super) fn finish(self) -> Result<(Vec<Vec<usize>>, Vec<DecodedElementBlock>), Diagnostic> {
        if self.checked_total != self.header.total
            || self.tags.len() != self.header.total
            || !self.bounds.agrees(&self.header)
            || self.cells.len() != self.top_dimensional
        {
            return Err(invalid_import(
                "$Elements records disagree with declared totals or tag bounds",
            ));
        }
        if self.cells.is_empty() {
            return Err(invalid_import(
                "MSH input contains no admitted top-dimensional simplex cells",
            ));
        }
        Ok((self.cells, self.blocks.unwrap_or_default()))
    }
}
