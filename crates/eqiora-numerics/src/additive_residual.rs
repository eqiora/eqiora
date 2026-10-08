//! Bounded, admission-only projection of additive residual structure.

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, GraphPath, RawId};
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode};

const MAX_ADDITIVE_DEPTH: usize = 16;
const MAX_ADDITIVE_LEAVES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdditiveSign {
    Positive,
    Negative,
}

impl AdditiveSign {
    const fn negated(self) -> Self {
        match self {
            Self::Positive => Self::Negative,
            Self::Negative => Self::Positive,
        }
    }

    pub(crate) fn is_opposite(self, other: Self) -> bool {
        self != other
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdditiveStep {
    AddLeft,
    AddRight,
    SubLeft,
    SubRight,
    NegOperand,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SignedOpaqueLeaf {
    value: ExprId,
    sign: AdditiveSign,
    provenance: Vec<AdditiveStep>,
}

impl SignedOpaqueLeaf {
    pub(crate) const fn value(&self) -> ExprId {
        self.value
    }

    pub(crate) const fn sign(&self) -> AdditiveSign {
        self.sign
    }

    pub(crate) fn describe(&self) -> String {
        format!(
            "{} expression node {} at {:?}",
            match self.sign {
                AdditiveSign::Positive => "+",
                AdditiveSign::Negative => "-",
            },
            self.value.index(),
            self.provenance
        )
    }
}

/// Private proof view used only while admitting a bounded capability.
///
/// It never rewrites the executable expression DAG. `Add`, `Sub`, and `Neg`
/// are traversed as signed structure; every other node remains an opaque leaf.
pub(crate) struct AdditiveResidualView {
    owner: RawId,
    leaves: Vec<SignedOpaqueLeaf>,
}

impl AdditiveResidualView {
    pub(crate) fn derive(
        expression: &ExprDag,
        root: ExprId,
        owner: RawId,
    ) -> Result<Self, Diagnostic> {
        Self::derive_preserving(expression, root, owner, &|_| false)
    }

    /// Retain admitted coefficient subtrees at their source boundary.
    pub(crate) fn derive_preserving(
        expression: &ExprDag,
        root: ExprId,
        owner: RawId,
        preserve: &impl Fn(ExprId) -> bool,
    ) -> Result<Self, Diagnostic> {
        let mut leaves = Vec::new();
        let mut provenance = Vec::new();
        flatten(
            expression,
            root,
            AdditiveSign::Positive,
            0,
            &mut provenance,
            &mut leaves,
            owner,
            preserve,
        )?;
        Ok(Self { owner, leaves })
    }

    pub(crate) fn leaves(&self) -> &[SignedOpaqueLeaf] {
        &self.leaves
    }

    pub(crate) fn mismatch(&self, expectation: &str) -> Diagnostic {
        admission_error(
            self.owner,
            format!(
                "{expectation}; unmatched signed leaves: [{}]",
                self.signed_leaves()
            ),
        )
    }

    pub(crate) fn signed_leaves(&self) -> String {
        self.leaves
            .iter()
            .map(SignedOpaqueLeaf::describe)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[allow(clippy::too_many_arguments)]
fn flatten(
    expression: &ExprDag,
    value: ExprId,
    sign: AdditiveSign,
    depth: usize,
    provenance: &mut Vec<AdditiveStep>,
    leaves: &mut Vec<SignedOpaqueLeaf>,
    owner: RawId,
    preserve: &impl Fn(ExprId) -> bool,
) -> Result<(), Diagnostic> {
    if depth > MAX_ADDITIVE_DEPTH {
        return Err(admission_error(
            owner,
            format!("additive residual exceeds maximum depth {MAX_ADDITIVE_DEPTH}"),
        ));
    }
    match expression.node(value) {
        Some(ExprNode::Add(left, right)) if !preserve(value) => {
            descend(
                expression,
                *left,
                sign,
                depth,
                AdditiveStep::AddLeft,
                provenance,
                leaves,
                owner,
                preserve,
            )?;
            descend(
                expression,
                *right,
                sign,
                depth,
                AdditiveStep::AddRight,
                provenance,
                leaves,
                owner,
                preserve,
            )
        }
        Some(ExprNode::Sub(left, right)) if !preserve(value) => {
            descend(
                expression,
                *left,
                sign,
                depth,
                AdditiveStep::SubLeft,
                provenance,
                leaves,
                owner,
                preserve,
            )?;
            descend(
                expression,
                *right,
                sign.negated(),
                depth,
                AdditiveStep::SubRight,
                provenance,
                leaves,
                owner,
                preserve,
            )
        }
        Some(ExprNode::Neg(operand)) if !preserve(value) => descend(
            expression,
            *operand,
            sign.negated(),
            depth,
            AdditiveStep::NegOperand,
            provenance,
            leaves,
            owner,
            preserve,
        ),
        Some(_) => {
            if leaves.len() == MAX_ADDITIVE_LEAVES {
                return Err(admission_error(
                    owner,
                    format!("additive residual exceeds maximum leaf count {MAX_ADDITIVE_LEAVES}"),
                ));
            }
            leaves.push(SignedOpaqueLeaf {
                value,
                sign,
                provenance: provenance.clone(),
            });
            Ok(())
        }
        None => Err(admission_error(
            owner,
            format!(
                "additive residual references missing expression node {}",
                value.index()
            ),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn descend(
    expression: &ExprDag,
    value: ExprId,
    sign: AdditiveSign,
    depth: usize,
    step: AdditiveStep,
    provenance: &mut Vec<AdditiveStep>,
    leaves: &mut Vec<SignedOpaqueLeaf>,
    owner: RawId,
    preserve: &impl Fn(ExprId) -> bool,
) -> Result<(), Diagnostic> {
    provenance.push(step);
    let result = flatten(
        expression,
        value,
        sign,
        depth + 1,
        provenance,
        leaves,
        owner,
        preserve,
    );
    provenance.pop();
    result
}

fn admission_error(owner: RawId, message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_SPATIAL_LOWERING, message).with_graph_path(GraphPath::new([
        owner.kind().graph().name().to_owned(),
        format!("{:?}", owner.kind()),
        owner.to_string(),
    ]))
}
