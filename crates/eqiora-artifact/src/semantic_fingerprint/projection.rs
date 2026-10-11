//! Closed attributed graph construction before exact canonical labelling.

use super::*;

#[derive(Clone)]
pub(super) struct Reference {
    pub(super) label: Vec<u8>,
    pub(super) target: usize,
}

pub(super) struct Vertex {
    pub(super) intrinsic: Vec<u8>,
    pub(super) outgoing: Vec<Reference>,
    pub(super) incoming: Vec<Reference>,
}

pub(super) struct ProjectionGraph {
    pub(super) vertices: Vec<Vertex>,
    pub(super) reference_count: usize,
}

impl ProjectionGraph {
    pub(super) fn from_program(
        program: &KernelProgram,
        limits: SemanticFingerprintLimits,
    ) -> Result<Self, Diagnostic> {
        let node_count = program.nodes().len();
        if node_count == 0 || node_count > limits.max_nodes {
            return Err(fingerprint_error(format!(
                "structural semantic projection requires 1..={} nodes, found {}",
                limits.max_nodes, node_count
            )));
        }
        if u32::try_from(node_count).is_err() {
            return Err(fingerprint_error(
                "structural semantic projection node count exceeds u32",
            ));
        }
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(node_count)
            .map_err(|_| fingerprint_error("cannot reserve semantic projection node view"))?;
        nodes.extend(program.nodes());

        let ids = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.id(), index))
            .collect::<BTreeMap<_, _>>();
        let mut budget = ConstructionBudget::new(limits);
        let mut vertices = Vec::new();
        vertices
            .try_reserve_exact(node_count)
            .map_err(|_| fingerprint_error("cannot reserve semantic projection vertices"))?;
        for node in nodes {
            let mut references = Vec::new();
            let intrinsic = encode_node(
                node,
                program.typed_value(node.id()),
                program.boundary().contains(&node.id()),
                &ids,
                &mut references,
                &mut budget,
            )?;
            budget.account_bytes(intrinsic.len())?;
            vertices.push(Vertex {
                intrinsic,
                outgoing: references,
                incoming: Vec::new(),
            });
        }

        for edge in program.edges() {
            let source = lookup(&ids, edge.from(), "edge source")?;
            let target = lookup(&ids, edge.to(), "edge target")?;
            vertices[source].outgoing.push(Reference {
                label: edge_label(edge.kind())?,
                target,
            });
            budget.account_reference()?;
        }

        let reference_count = vertices.iter().map(|vertex| vertex.outgoing.len()).sum();
        if reference_count > limits.max_references {
            return Err(fingerprint_error(format!(
                "structural semantic projection has {reference_count} references, exceeding the {} reference limit",
                limits.max_references
            )));
        }
        for source in 0..vertices.len() {
            vertices[source].outgoing.sort_by(|left, right| {
                left.label
                    .cmp(&right.label)
                    .then(left.target.cmp(&right.target))
            });
            for reference in vertices[source].outgoing.clone() {
                vertices[reference.target].incoming.push(Reference {
                    label: reference.label,
                    target: source,
                });
            }
        }
        for vertex in &mut vertices {
            vertex.incoming.sort_by(|left, right| {
                left.label
                    .cmp(&right.label)
                    .then(left.target.cmp(&right.target))
            });
        }
        Ok(Self {
            vertices,
            reference_count,
        })
    }
}

pub(super) struct ConstructionBudget {
    pub(super) limits: SemanticFingerprintLimits,
    references: usize,
    expression_nodes: usize,
    bytes: usize,
}

impl ConstructionBudget {
    const fn new(limits: SemanticFingerprintLimits) -> Self {
        Self {
            limits,
            references: 0,
            expression_nodes: 0,
            bytes: 0,
        }
    }

    pub(super) fn account_reference(&mut self) -> Result<(), Diagnostic> {
        self.references = self
            .references
            .checked_add(1)
            .ok_or_else(|| fingerprint_error("semantic reference count overflows usize"))?;
        if self.references > self.limits.max_references {
            return Err(fingerprint_error(format!(
                "semantic reference count exceeds the {} reference limit",
                self.limits.max_references
            )));
        }
        Ok(())
    }

    pub(super) fn account_expression_nodes(&mut self, count: usize) -> Result<(), Diagnostic> {
        self.expression_nodes = self
            .expression_nodes
            .checked_add(count)
            .ok_or_else(|| fingerprint_error("expression-node count overflows usize"))?;
        if self.expression_nodes > self.limits.max_expression_nodes {
            return Err(fingerprint_error(format!(
                "semantic projection expression-node count exceeds the {} node limit",
                self.limits.max_expression_nodes
            )));
        }
        Ok(())
    }

    pub(super) fn account_bytes(&mut self, count: usize) -> Result<(), Diagnostic> {
        self.bytes = self
            .bytes
            .checked_add(count)
            .ok_or_else(|| fingerprint_error("semantic projection byte count overflows usize"))?;
        if self.bytes > self.limits.max_canonical_bytes {
            return Err(fingerprint_error(format!(
                "semantic projection exceeds the {} byte limit",
                self.limits.max_canonical_bytes
            )));
        }
        Ok(())
    }
}
