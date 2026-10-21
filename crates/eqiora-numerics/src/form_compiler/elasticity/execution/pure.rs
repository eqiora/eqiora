//! Canonical polynomial operators lower into the closed elasticity instruction tape.
use super::*;
use eqiora_core::DimExponents;
use eqiora_ir::{CalculusNode, ExactRational, OperatorApplicationProof, PureOperatorDefinition};

impl ProgramCompiler<'_> {
    pub(super) fn emit_standard_operator(
        &mut self,
        typed: &TypedResidual<RawId>,
        node: ExprId,
        axes: &[usize],
        standard: StandardPureOperator,
    ) -> Result<u16, Diagnostic> {
        let proof = OperatorApplicationProof::classify(typed, node, standard)
            .map_err(|error| tape_error(format!("pure-operator replay failed: {error}")))?
            .ok_or_else(|| tape_error("typed source changed its standard pure operator"))?;
        let definition = standard_definition(standard)?;
        if proof.definition_digest() != definition.digest() {
            return Err(tape_error("pure-operator digest is stale"));
        }
        self.emit_operator(
            &definition,
            axes,
            FormalOperand::Typed(proof.operand()),
            Some(node),
        )
    }

    pub(super) fn emit_operator(
        &mut self,
        definition: &PureOperatorDefinition,
        result_axes: &[usize],
        formal: FormalOperand,
        source_node: Option<ExprId>,
    ) -> Result<u16, Diagnostic> {
        let coordinates = result_axes
            .iter()
            .map(|axis| {
                u32::try_from(*axis).map_err(|_| {
                    tape_error("pure-operator result coordinate exceeds the portable range")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let digest = definition.digest();
        let mut values = Vec::with_capacity(definition.nodes().len());
        for node in definition.nodes() {
            let provenance = self.source_provenance(source_node, Some(digest));
            let value = match node {
                CalculusNode::Rational { value, dimension } => {
                    if *dimension != DimExponents::DIMENSIONLESS {
                        return Err(tape_error(
                            "class operator requires dimensionless constants",
                        ));
                    }
                    self.constant(
                        value
                            .to_f64()
                            .map_err(|error| tape_error(error.to_string()))?,
                        provenance,
                    )?
                }
                CalculusNode::BoundInput(_)
                | CalculusNode::Differentiated { .. }
                | CalculusNode::Require { .. }
                | CalculusNode::Boolean(_)
                | CalculusNode::Compare(..)
                | CalculusNode::Not(_)
                | CalculusNode::And(..)
                | CalculusNode::Or(..)
                | CalculusNode::Select { .. }
                | CalculusNode::Complex { .. }
                | CalculusNode::UnaryMath(..)
                | CalculusNode::Div(..) => {
                    return Err(tape_error(
                        "class operator requires unguarded polynomial calculus",
                    ));
                }
                CalculusNode::FormalComponent { formal: 0, axes } => {
                    let component = axes
                        .iter()
                        .map(|axis| {
                            axis.resolve(&coordinates)
                                .map(|value| value as usize)
                                .map_err(|_| tape_error("pure-operator result axis is invalid"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    self.emit_formal(formal, &component, digest)?
                }
                CalculusNode::FormalComponent { .. } => {
                    return Err(tape_error("pure-operator formal slot is outside the class"));
                }
                CalculusNode::KroneckerDelta(left, right) => {
                    let left = left
                        .resolve(&coordinates)
                        .map_err(|_| tape_error("pure-operator delta axis is invalid"))?;
                    let right = right
                        .resolve(&coordinates)
                        .map_err(|_| tape_error("pure-operator delta axis is invalid"))?;
                    self.constant(if left == right { 1.0 } else { 0.0 }, provenance)?
                }
                CalculusNode::Neg(value) => self.push(
                    LocalFormInstruction::Neg(calculus_operand(&values, *value)?),
                    provenance,
                )?,
                CalculusNode::Add(left, right) => self.push(
                    LocalFormInstruction::Add(
                        calculus_operand(&values, *left)?,
                        calculus_operand(&values, *right)?,
                    ),
                    provenance,
                )?,
                CalculusNode::Mul(left, right) => self.push(
                    LocalFormInstruction::Mul(
                        calculus_operand(&values, *left)?,
                        calculus_operand(&values, *right)?,
                    ),
                    provenance,
                )?,
                CalculusNode::Demand { value, .. } => {
                    // This tape evaluates every emitted instruction in order,
                    // so the prerequisite is already retained and validated.
                    calculus_operand(&values, *value)?
                }
            };
            values.push(value);
        }
        values
            .get(usize::try_from(definition.root().index()).map_err(|_| tape_error("root"))?)
            .copied()
            .ok_or_else(|| tape_error("pure-operator root is invalid"))
    }

    fn emit_formal(
        &mut self,
        formal: FormalOperand,
        axes: &[usize],
        digest: OperatorDefinitionDigest,
    ) -> Result<u16, Diagnostic> {
        match formal {
            FormalOperand::Typed(node) => self.emit_typed(node, axes, Some(digest)),
            FormalOperand::StateGradient if axes.len() == ROOT_COUNT => self.read(
                LocalFormInput2d::StateJet {
                    component: to_u8(axes[0])?,
                    axis: to_u8(axes[1])?,
                },
                self.source_provenance(None, Some(digest)),
            ),
            FormalOperand::StateDivergence if axes.is_empty() => {
                self.emit_state_divergence(self.source_provenance(None, Some(digest)))
            }
            _ => Err(tape_error(
                "pure-operator formal component shape is invalid",
            )),
        }
    }
}

pub(super) fn standard_definition(
    standard: StandardPureOperator,
) -> Result<PureOperatorDefinition, Diagnostic> {
    match standard {
        StandardPureOperator::SymmetricPart => PureOperatorDefinition::symmetric_part(),
        StandardPureOperator::IsotropicLift => PureOperatorDefinition::isotropic_lift(),
    }
    .map_err(|error| tape_error(format!("standard pure operator is invalid: {error}")))
}

pub(super) fn definition_source_scale(
    definition: &PureOperatorDefinition,
) -> Result<Option<ExactRational>, Diagnostic> {
    let mut rationals = Vec::new();
    for node in definition.nodes() {
        match node {
            CalculusNode::Rational { value, dimension } => {
                if *dimension != DimExponents::DIMENSIONLESS {
                    return Err(tape_error(
                        "class operator requires dimensionless constants",
                    ));
                }
                rationals.push(value.clone());
            }
            CalculusNode::FormalComponent { .. }
            | CalculusNode::KroneckerDelta(..)
            | CalculusNode::Neg(_)
            | CalculusNode::Add(..)
            | CalculusNode::Mul(..)
            | CalculusNode::Demand { .. } => {}
            CalculusNode::BoundInput(_)
            | CalculusNode::Differentiated { .. }
            | CalculusNode::Require { .. }
            | CalculusNode::Boolean(_)
            | CalculusNode::Compare(..)
            | CalculusNode::Not(_)
            | CalculusNode::And(..)
            | CalculusNode::Or(..)
            | CalculusNode::Select { .. }
            | CalculusNode::Complex { .. }
            | CalculusNode::UnaryMath(..)
            | CalculusNode::Div(..) => {
                return Err(tape_error(
                    "class operator requires unguarded polynomial calculus",
                ));
            }
        }
    }
    let value = match rationals.as_slice() {
        [] => return Ok(None),
        [value] => value,
        _ => {
            return Err(tape_error(
                "class operator has ambiguous exact normalization",
            ));
        }
    };
    ExactRational::integer(1)
        .checked_div(value.clone())
        .map(Some)
        .map_err(|error| tape_error(format!("pure-operator scale is invalid: {error}")))
}

fn calculus_operand(values: &[u16], operand: eqiora_ir::CalculusNodeId) -> Result<u16, Diagnostic> {
    values
        .get(usize::try_from(operand.index()).map_err(|_| tape_error("calculus operand"))?)
        .copied()
        .ok_or_else(|| tape_error("pure calculus contains a forward reference"))
}
