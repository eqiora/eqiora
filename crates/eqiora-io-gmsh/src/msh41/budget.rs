use std::mem::size_of;

use eqiora_core::Diagnostic;

use super::{DecoderLimits, invalid_import};

// These charges bound simultaneous structural indexes, owned coordinate and
// connectivity decoding, tag lookup, canonical reconstruction, and the upper
// closure expansion of `SimplicialMesh`. They intentionally remain
// conservative across the ASCII and binary decoder layouts. Hash entries are
// charged at four words and topology entries at thirty-two words, covering
// keys, duplicated closure vectors, tree bookkeeping, boundary state, and
// construction temporaries for the admitted two- and three-dimensional
// simplex families.
const WORD: usize = size_of::<usize>();
const VECTOR_HEADER_BYTES: usize = 3 * WORD;
const HASH_ENTRY_BYTES: usize = 4 * WORD;
const BLOCK_BYTES: usize = 24 * WORD;
const TOPOLOGY_ENTITY_BYTES: usize = 32 * WORD;
// One line can coexist in the initial `Vec<&str>` (two words), one exact-growth
// section body (two words), and the amortized section/delimiter index. Six
// words per source line is conservative for both empty and nonempty lines.
const ASCII_LINE_INDEX_BYTES: usize = 6 * WORD;
const ASCII_SECTION_INDEX_BYTES: usize = 4 * 5 * WORD;

/// Aggregate resource account consumed by the owned bounded decoder.
pub(super) struct DecodedBudget {
    remaining_bytes: usize,
    remaining_work: usize,
    max_ignored_elements: usize,
    ignored_elements: usize,
}

impl DecodedBudget {
    pub(super) const fn new(limits: DecoderLimits) -> Self {
        Self {
            remaining_bytes: limits.max_decoded_bytes,
            remaining_work: limits.max_decoded_work,
            max_ignored_elements: limits.max_ignored_elements,
            ignored_elements: 0,
        }
    }

    pub(super) fn charge_ascii_lines(&mut self, count: usize) -> Result<(), Diagnostic> {
        let line_bytes = count
            .checked_mul(ASCII_LINE_INDEX_BYTES)
            .ok_or_else(|| invalid_import("ASCII line-index charge overflows usize"))?;
        let bytes = line_bytes
            .checked_add(ASCII_SECTION_INDEX_BYTES)
            .ok_or_else(|| invalid_import("ASCII section-index charge overflows usize"))?;
        let work = count
            .checked_mul(4)
            .ok_or_else(|| invalid_import("ASCII line-index work overflows usize"))?;
        self.charge(bytes, work, "ASCII structural index")
    }

    pub(super) fn charge_entities(&mut self, counts: &[usize; 4]) -> Result<(), Diagnostic> {
        let total = checked_sum(counts.iter().copied(), "$Entities decoded count")?;
        let bytes_per_entity = HASH_ENTRY_BYTES
            .checked_add(20 * WORD)
            .ok_or_else(|| invalid_import("$Entities decoded-byte charge overflows usize"))?;
        self.charge_product(total, bytes_per_entity, 20, "$Entities")
    }

    pub(super) fn charge_entity_references(&mut self, count: usize) -> Result<(), Diagnostic> {
        // One word per reference conservatively covers validation scratch and
        // capacity rounding even though admitted entity references are not
        // retained in the canonical mesh.
        self.charge_product(count, WORD, 3, "$Entities boundary references")
    }

    pub(super) fn charge_nodes(
        &mut self,
        block_count: usize,
        node_count: usize,
        dimension: usize,
    ) -> Result<(), Diagnostic> {
        self.charge_product(block_count, BLOCK_BYTES, 8, "$Nodes blocks")?;

        let coordinate_bytes = dimension
            .checked_mul(size_of::<f64>())
            .ok_or_else(|| invalid_import("$Nodes coordinate-byte charge overflows usize"))?;
        let bytes_per_node = size_of::<u64>()
            .checked_add(HASH_ENTRY_BYTES)
            .and_then(|bytes| bytes.checked_add(3 * size_of::<f64>()))
            .and_then(|bytes| bytes.checked_add(HASH_ENTRY_BYTES))
            .and_then(|bytes| bytes.checked_add(VECTOR_HEADER_BYTES))
            .and_then(|bytes| bytes.checked_add(coordinate_bytes))
            .and_then(|bytes| bytes.checked_add(size_of::<u64>()))
            .and_then(|bytes| bytes.checked_add(HASH_ENTRY_BYTES))
            .and_then(|bytes| bytes.checked_add(TOPOLOGY_ENTITY_BYTES))
            .ok_or_else(|| invalid_import("$Nodes decoded-byte charge overflows usize"))?;
        let work_per_node = dimension
            .checked_mul(2)
            .and_then(|work| work.checked_add(12))
            .ok_or_else(|| invalid_import("$Nodes decoded-work charge overflows usize"))?;
        self.charge_product(node_count, bytes_per_node, work_per_node, "$Nodes")
    }

    pub(super) fn charge_element_inventory(
        &mut self,
        block_count: usize,
        element_count: usize,
    ) -> Result<(), Diagnostic> {
        self.charge_product(block_count, BLOCK_BYTES, 8, "$Elements blocks")?;
        // The complete tag index is reserved at the section header. Fund its
        // storage and initialization now, not later as element blocks arrive.
        // These charges are removed from each record below; valid input keeps
        // the same aggregate logical byte/work cost.
        self.charge_product(
            element_count,
            size_of::<u64>() + HASH_ENTRY_BYTES,
            2,
            "$Elements tag inventory",
        )
    }

    pub(super) fn charge_elements(
        &mut self,
        count: usize,
        entity_dimension: usize,
        requested_dimension: usize,
    ) -> Result<(), Diagnostic> {
        let top_dimensional = entity_dimension == requested_dimension;
        if !top_dimensional {
            self.ignored_elements = self
                .ignored_elements
                .checked_add(count)
                .ok_or_else(|| invalid_import("ignored-element count overflows usize"))?;
            if self.ignored_elements > self.max_ignored_elements {
                return Err(invalid_import(
                    "$Elements exceeds the configured ignored lower-dimensional element limit",
                ));
            }
        }

        let arity = entity_dimension
            .checked_add(1)
            .ok_or_else(|| invalid_import("element arity overflows usize"))?;
        let connectivity_bytes = arity
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| invalid_import("element connectivity-byte charge overflows usize"))?;
        let mut bytes_per_element = (4 * WORD)
            .checked_add(connectivity_bytes)
            .and_then(|bytes| bytes.checked_add(HASH_ENTRY_BYTES))
            .ok_or_else(|| invalid_import("$Elements decoded-byte charge overflows usize"))?;
        let mut work_per_element = entity_dimension
            .checked_add(2)
            .and_then(|fields| fields.checked_mul(2))
            .and_then(|work| work.checked_add(6))
            .ok_or_else(|| invalid_import("$Elements decoded-work charge overflows usize"))?;

        if top_dimensional {
            let canonical_connectivity_bytes = arity
                .checked_mul(size_of::<usize>())
                .and_then(|bytes| bytes.checked_add(VECTOR_HEADER_BYTES))
                .ok_or_else(|| {
                    invalid_import("canonical cell decoded-byte charge overflows usize")
                })?;
            let topology_occurrences: usize = match requested_dimension {
                // Unique entities can only reduce these per-cell closure bounds.
                2 => 4,  // three edges and the cell
                3 => 11, // six edges, four faces, and the cell
                _ => return Err(invalid_import("unsupported decoded-budget dimension")),
            };
            let topology_bytes = topology_occurrences
                .checked_mul(TOPOLOGY_ENTITY_BYTES)
                .ok_or_else(|| invalid_import("topology decoded-byte charge overflows usize"))?;
            bytes_per_element = bytes_per_element
                .checked_add(canonical_connectivity_bytes)
                .and_then(|bytes| bytes.checked_add(topology_bytes))
                .ok_or_else(|| invalid_import("canonical decoded-byte charge overflows usize"))?;
            work_per_element = topology_occurrences
                .checked_mul(8)
                .and_then(|work| work.checked_add(work_per_element))
                .ok_or_else(|| invalid_import("canonical decoded-work charge overflows usize"))?;
        }
        self.charge_product(count, bytes_per_element, work_per_element, "$Elements")
    }

    fn charge_product(
        &mut self,
        count: usize,
        bytes_per_item: usize,
        work_per_item: usize,
        context: &str,
    ) -> Result<(), Diagnostic> {
        let bytes = count
            .checked_mul(bytes_per_item)
            .ok_or_else(|| invalid_import(format!("{context} decoded-byte charge overflows")))?;
        let work = count
            .checked_mul(work_per_item)
            .ok_or_else(|| invalid_import(format!("{context} decoded-work charge overflows")))?;
        self.charge(bytes, work, context)
    }

    fn charge(&mut self, bytes: usize, work: usize, context: &str) -> Result<(), Diagnostic> {
        let remaining_bytes = self.remaining_bytes.checked_sub(bytes).ok_or_else(|| {
            invalid_import(format!(
                "{context} exceeds the configured aggregate decoded-byte budget",
            ))
        })?;
        let remaining_work = self.remaining_work.checked_sub(work).ok_or_else(|| {
            invalid_import(format!(
                "{context} exceeds the configured aggregate decoded-work budget",
            ))
        })?;
        self.remaining_bytes = remaining_bytes;
        self.remaining_work = remaining_work;
        Ok(())
    }
}

fn checked_sum(
    values: impl IntoIterator<Item = usize>,
    context: &str,
) -> Result<usize, Diagnostic> {
    values.into_iter().try_fold(0_usize, |sum, value| {
        sum.checked_add(value)
            .ok_or_else(|| invalid_import(format!("{context} overflows usize")))
    })
}
