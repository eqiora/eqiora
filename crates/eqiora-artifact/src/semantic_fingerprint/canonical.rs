//! Exact graph labelling and bounded canonical byte encoding.

use super::*;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Signature {
    intrinsic: Vec<u8>,
    outgoing: Vec<(Vec<u8>, usize)>,
    incoming: Vec<(Vec<u8>, usize)>,
}

pub(super) struct Canonicalizer<'a> {
    graph: &'a ProjectionGraph,
    limits: SemanticFingerprintLimits,
    search_states: usize,
    refinement_work: usize,
    serialization_work: usize,
    best: Option<Vec<u8>>,
}

impl<'a> Canonicalizer<'a> {
    pub(super) const fn new(graph: &'a ProjectionGraph, limits: SemanticFingerprintLimits) -> Self {
        Self {
            graph,
            limits,
            search_states: 0,
            refinement_work: 0,
            serialization_work: 0,
            best: None,
        }
    }

    pub(super) fn canonicalize(mut self) -> Result<Vec<u8>, Diagnostic> {
        let mut groups = BTreeMap::<Vec<u8>, Vec<usize>>::new();
        for (index, vertex) in self.graph.vertices.iter().enumerate() {
            groups
                .entry(vertex.intrinsic.clone())
                .or_default()
                .push(index);
        }
        let partition = groups.into_values().collect::<Vec<_>>();
        let partition = self.refine(partition)?;
        self.search(partition, 0)?;
        self.best
            .take()
            .ok_or_else(|| fingerprint_error("canonical-label search produced no discrete leaf"))
    }

    fn search(&mut self, partition: Vec<Vec<usize>>, depth: usize) -> Result<(), Diagnostic> {
        self.search_states = self
            .search_states
            .checked_add(1)
            .ok_or_else(|| fingerprint_error("canonical-label search-state count overflows"))?;
        if self.search_states > self.limits.max_search_states {
            return Err(fingerprint_error(format!(
                "exact semantic graph canonicalization exceeds the {} search-state limit",
                self.limits.max_search_states
            )));
        }
        let Some(cell_index) = partition.iter().position(|cell| cell.len() > 1) else {
            let remaining = self
                .limits
                .max_serialization_work
                .checked_sub(self.serialization_work)
                .ok_or_else(|| fingerprint_error("canonical serialization work overflows"))?;
            if remaining == 0 {
                return Err(fingerprint_error(format!(
                    "exact semantic graph canonicalization exceeds the {} serialization-byte-work limit",
                    self.limits.max_serialization_work
                )));
            }
            let canonical = self.serialize(&partition, remaining)?;
            self.serialization_work = self
                .serialization_work
                .checked_add(canonical.len())
                .ok_or_else(|| fingerprint_error("canonical serialization work overflows"))?;
            if self
                .best
                .as_ref()
                .is_none_or(|current| canonical < *current)
            {
                self.best = Some(canonical);
            }
            return Ok(());
        };
        if depth >= self.limits.max_individualization_depth {
            return Err(fingerprint_error(format!(
                "exact semantic graph canonicalization exceeds the {} individualization-depth limit",
                self.limits.max_individualization_depth
            )));
        }

        let candidates = partition[cell_index].clone();
        for candidate in candidates {
            let mut branch = partition.clone();
            let remainder = branch[cell_index]
                .iter()
                .copied()
                .filter(|vertex| *vertex != candidate)
                .collect::<Vec<_>>();
            branch.splice(cell_index..=cell_index, [vec![candidate], remainder]);
            let branch = self.refine(branch)?;
            self.search(branch, depth + 1)?;
        }
        Ok(())
    }

    fn refine(&mut self, mut partition: Vec<Vec<usize>>) -> Result<Vec<Vec<usize>>, Diagnostic> {
        loop {
            self.account_refinement_work()?;
            let mut cell_of = vec![0_usize; self.graph.vertices.len()];
            for (cell, vertices) in partition.iter().enumerate() {
                for &vertex in vertices {
                    cell_of[vertex] = cell;
                }
            }
            let mut groups = BTreeMap::<(usize, Signature), Vec<usize>>::new();
            for (cell, vertices) in partition.iter().enumerate() {
                for &vertex in vertices {
                    let value = &self.graph.vertices[vertex];
                    let mut outgoing = value
                        .outgoing
                        .iter()
                        .map(|reference| (reference.label.clone(), cell_of[reference.target]))
                        .collect::<Vec<_>>();
                    outgoing.sort_unstable();
                    let mut incoming = value
                        .incoming
                        .iter()
                        .map(|reference| (reference.label.clone(), cell_of[reference.target]))
                        .collect::<Vec<_>>();
                    incoming.sort_unstable();
                    let signature = Signature {
                        intrinsic: value.intrinsic.clone(),
                        outgoing,
                        incoming,
                    };
                    groups.entry((cell, signature)).or_default().push(vertex);
                }
            }
            let refined = groups.into_values().collect::<Vec<_>>();
            if refined.len() == partition.len() {
                return Ok(refined);
            }
            partition = refined;
        }
    }

    fn account_refinement_work(&mut self) -> Result<(), Diagnostic> {
        let round = self
            .graph
            .vertices
            .len()
            .checked_add(self.graph.reference_count.saturating_mul(2))
            .ok_or_else(|| fingerprint_error("canonical refinement work overflows usize"))?;
        self.refinement_work = self
            .refinement_work
            .checked_add(round)
            .ok_or_else(|| fingerprint_error("canonical refinement work overflows usize"))?;
        if self.refinement_work > self.limits.max_refinement_work {
            return Err(fingerprint_error(format!(
                "exact semantic graph canonicalization exceeds the {} refinement-work limit",
                self.limits.max_refinement_work
            )));
        }
        Ok(())
    }

    fn serialize(
        &self,
        partition: &[Vec<usize>],
        remaining_work: usize,
    ) -> Result<Vec<u8>, Diagnostic> {
        let order = partition
            .iter()
            .map(|cell| {
                if cell.len() == 1 {
                    Ok(cell[0])
                } else {
                    Err(fingerprint_error(
                        "canonical semantic partition is unexpectedly non-discrete",
                    ))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut ordinal = vec![0_usize; order.len()];
        for (position, &vertex) in order.iter().enumerate() {
            ordinal[vertex] = position;
        }
        let mut encoder = Encoder::new(self.limits.max_canonical_bytes.min(remaining_work));
        encoder.raw(PROJECTION_MAGIC)?;
        encoder.u16(SemanticFingerprintGeneration::V21.code())?;
        encoder.len(order.len())?;
        for vertex in order {
            let value = &self.graph.vertices[vertex];
            encoder.bytes(&value.intrinsic)?;
            let mut references = value
                .outgoing
                .iter()
                .map(|reference| (reference.label.as_slice(), ordinal[reference.target]))
                .collect::<Vec<_>>();
            references.sort_unstable();
            encoder.len(references.len())?;
            for (label, target) in references {
                encoder.bytes(label)?;
                encoder.usize(target)?;
            }
        }
        encoder.finish()
    }
}

pub(super) struct Encoder {
    bytes: Vec<u8>,
    limit: usize,
}

impl Encoder {
    pub(super) const fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
        }
    }

    fn reserve(&mut self, count: usize) -> Result<(), Diagnostic> {
        let required = self
            .bytes
            .len()
            .checked_add(count)
            .ok_or_else(|| fingerprint_error("canonical projection length overflows usize"))?;
        if required > self.limit {
            return Err(fingerprint_error(format!(
                "canonical semantic projection exceeds the {} byte limit",
                self.limit
            )));
        }
        self.bytes
            .try_reserve_exact(count)
            .map_err(|_| fingerprint_error("cannot reserve canonical semantic projection bytes"))
    }

    pub(super) fn raw(&mut self, value: &[u8]) -> Result<(), Diagnostic> {
        self.reserve(value.len())?;
        self.bytes.extend_from_slice(value);
        Ok(())
    }

    pub(super) fn bytes(&mut self, value: &[u8]) -> Result<(), Diagnostic> {
        self.len(value.len())?;
        self.raw(value)
    }

    pub(super) fn bool(&mut self, value: bool) -> Result<(), Diagnostic> {
        self.u8(u8::from(value))
    }

    pub(super) fn u8(&mut self, value: u8) -> Result<(), Diagnostic> {
        self.raw(&[value])
    }

    pub(super) fn u16(&mut self, value: u16) -> Result<(), Diagnostic> {
        self.raw(&value.to_be_bytes())
    }

    pub(super) fn u32(&mut self, value: u32) -> Result<(), Diagnostic> {
        self.raw(&value.to_be_bytes())
    }

    pub(super) fn i32(&mut self, value: i32) -> Result<(), Diagnostic> {
        self.raw(&value.to_be_bytes())
    }

    pub(super) fn u64(&mut self, value: u64) -> Result<(), Diagnostic> {
        self.raw(&value.to_be_bytes())
    }

    pub(super) fn usize(&mut self, value: usize) -> Result<(), Diagnostic> {
        let value = u64::try_from(value)
            .map_err(|_| fingerprint_error("canonical usize value exceeds u64"))?;
        self.u64(value)
    }

    pub(super) fn len(&mut self, value: usize) -> Result<(), Diagnostic> {
        let value = u32::try_from(value)
            .map_err(|_| fingerprint_error("canonical collection length exceeds u32"))?;
        self.u32(value)
    }

    pub(super) fn finish(self) -> Result<Vec<u8>, Diagnostic> {
        if self.bytes.len() > self.limit {
            Err(fingerprint_error(
                "canonical semantic projection exceeds its byte limit",
            ))
        } else {
            Ok(self.bytes)
        }
    }
}
