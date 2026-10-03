//! Closed scalar instruction tape for the private Cartesian Q1 elasticity form.

mod evaluate;

use evaluate::{closed_inputs, instruction_tangent, instruction_value, reverse};

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id, RawId};
use eqiora_ir::{
    CalculusNode, ExactRational, OperatorApplicationProof, OperatorDefinitionDigest,
    PureOperatorDefinition, StandardPureOperator,
};
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{ExprId, ExprNode, SymbolRef};

use super::{
    DIVERGENCE_BY_PARTS, ElasticityDerivationSource, SOURCE_PAIRING, TEST_PAIRING, tape_error,
};
use crate::form_compiler::vocabulary::PrimalGalerkinCorrespondence;

const INPUT_COUNT: usize = 12;
const ROOT_COUNT: usize = 2;
const MAX_INSTRUCTIONS: usize = 512;
const MAX_DEPENDENCIES: usize = 1_024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LocalFormInput2d {
    Material(u8),
    StateJet { component: u8, axis: u8 },
    ShapeGradient(u8),
    ShapeValue,
    BodyForce(u8),
    QuadratureScale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LocalFormInstruction {
    Read(u16),
    ConstantBits(u64),
    Neg(u16),
    Add(u16, u16),
    Mul(u16, u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LocalFormInstructionProvenance {
    pub(super) rule_id: Option<&'static str>,
    pub(super) relation: Option<RawId>,
    pub(super) source_node: Option<ExprId>,
    pub(super) operator_definition: Option<OperatorDefinitionDigest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NormalizedLocalFormInstructionProvenance {
    pub(super) rule_id: Option<&'static str>,
    pub(super) operator_definition: Option<OperatorDefinitionDigest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NormalizedLocalFormProgram2d {
    pub(super) inputs: Vec<LocalFormInput2d>,
    pub(super) instructions: Vec<LocalFormInstruction>,
    pub(super) roots: Vec<u16>,
    pub(super) provenance: Vec<NormalizedLocalFormInstructionProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LocalFormProgram2d {
    pub(super) inputs: Vec<LocalFormInput2d>,
    pub(super) instructions: Vec<LocalFormInstruction>,
    pub(super) roots: Vec<u16>,
    pub(super) provenance: Vec<LocalFormInstructionProvenance>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct LocalFormExecution {
    roots: Vec<f64>,
    root_tangents: Option<Vec<f64>>,
    input_cotangents: Option<Vec<f64>>,
}

impl LocalFormProgram2d {
    pub(super) fn validate(&self) -> Result<(), Diagnostic> {
        if self.inputs.as_slice() != closed_inputs() {
            return Err(tape_error("local-form input inventory or order is stale"));
        }
        if self.instructions.is_empty() || self.instructions.len() > MAX_INSTRUCTIONS {
            return Err(tape_error(
                "local-form instruction count is outside its frozen bound",
            ));
        }
        if self.provenance.len() != self.instructions.len() {
            return Err(tape_error(
                "local-form provenance is not instruction-aligned",
            ));
        }
        if self.roots.len() != ROOT_COUNT
            || self.roots[0] >= self.roots[1]
            || self
                .roots
                .iter()
                .any(|root| usize::from(*root) >= self.instructions.len())
        {
            return Err(tape_error("local-form roots do not retain component order"));
        }

        let symmetric = standard_definition(StandardPureOperator::SymmetricPart)?.digest();
        let isotropic = standard_definition(StandardPureOperator::IsotropicLift)?.digest();
        let mut dependencies = 0_usize;
        let mut reads = [false; INPUT_COUNT];
        let mut semantic_provenance = None;
        for (index, (instruction, provenance)) in
            self.instructions.iter().zip(&self.provenance).enumerate()
        {
            let prior = |operand: u16| usize::from(operand) < index;
            match *instruction {
                LocalFormInstruction::Read(input) if usize::from(input) < INPUT_COUNT => {
                    reads[usize::from(input)] = true;
                }
                LocalFormInstruction::ConstantBits(bits) if f64::from_bits(bits).is_finite() => {}
                LocalFormInstruction::Neg(value) if prior(value) => dependencies += 1,
                LocalFormInstruction::Add(left, right) | LocalFormInstruction::Mul(left, right)
                    if prior(left) && prior(right) =>
                {
                    dependencies += 2;
                }
                _ => {
                    return Err(tape_error(
                        "local-form instruction has an invalid reference",
                    ));
                }
            }
            if dependencies > MAX_DEPENDENCIES {
                return Err(tape_error(
                    "local-form dependency count exceeds its frozen bound",
                ));
            }
            if provenance.rule_id.is_some_and(|rule| {
                !matches!(rule, TEST_PAIRING | DIVERGENCE_BY_PARTS | SOURCE_PAIRING)
            }) {
                return Err(tape_error(
                    "local-form instruction has unknown weak-rule provenance",
                ));
            }
            if provenance
                .operator_definition
                .is_some_and(|digest| digest != symmetric && digest != isotropic)
            {
                return Err(tape_error(
                    "local-form instruction has unknown pure-operator provenance",
                ));
            }
            let row_semantic = match (provenance.relation, provenance.source_node) {
                (None, None) => false,
                (Some(_), Some(_)) => true,
                _ => return Err(tape_error("local-form Semantic provenance is incomplete")),
            };
            match semantic_provenance {
                Some(expected) if expected != row_semantic => {
                    return Err(tape_error(
                        "local-form program mixes witness and derived provenance",
                    ));
                }
                Some(_) => {}
                None => semantic_provenance = Some(row_semantic),
            }
        }
        if reads.contains(&false) {
            return Err(tape_error(
                "local-form program leaves a dense input role unread",
            ));
        }
        Ok(())
    }

    pub(super) fn normalized(&self) -> NormalizedLocalFormProgram2d {
        NormalizedLocalFormProgram2d {
            inputs: self.inputs.clone(),
            instructions: self.instructions.clone(),
            roots: self.roots.clone(),
            provenance: self
                .provenance
                .iter()
                .map(|provenance| NormalizedLocalFormInstructionProvenance {
                    rule_id: provenance.rule_id,
                    operator_definition: provenance.operator_definition,
                })
                .collect(),
        }
    }

    pub(super) fn execute<'a>(
        &'a self,
        inputs: &'a [f64],
        input_tangent: Option<&'a [f64]>,
        root_cotangent: Option<&'a [f64]>,
    ) -> Result<LocalFormExecution, Diagnostic> {
        self.validate()?;
        if inputs.len() != INPUT_COUNT
            || input_tangent.is_some_and(|values| values.len() != INPUT_COUNT)
            || root_cotangent.is_some_and(|values| values.len() != ROOT_COUNT)
        {
            return Err(tape_error("local-form execution input shape is invalid"));
        }
        if inputs
            .iter()
            .chain(input_tangent.into_iter().flatten())
            .chain(root_cotangent.into_iter().flatten())
            .any(|value| !value.is_finite())
        {
            return Err(tape_error("local-form execution input is non-finite"));
        }

        let mut values = Vec::with_capacity(self.instructions.len());
        let mut tangents = input_tangent.map(|_| Vec::with_capacity(self.instructions.len()));
        for instruction in &self.instructions {
            let value = instruction_value(*instruction, inputs, &values);
            if !value.is_finite() {
                return Err(tape_error("local-form primal intermediate is non-finite"));
            }
            values.push(value);
            if let (Some(input_tangent), Some(tangents)) = (input_tangent, tangents.as_mut()) {
                let tangent = instruction_tangent(*instruction, input_tangent, &values, tangents);
                if !tangent.is_finite() {
                    return Err(tape_error("local-form tangent intermediate is non-finite"));
                }
                tangents.push(tangent);
            }
        }

        let roots = self
            .roots
            .iter()
            .map(|root| values[usize::from(*root)])
            .collect::<Vec<_>>();
        let root_tangents = tangents.as_ref().map(|tangents| {
            self.roots
                .iter()
                .map(|root| tangents[usize::from(*root)])
                .collect::<Vec<_>>()
        });
        let input_cotangents = root_cotangent
            .map(|cotangent| reverse(self, &values, cotangent))
            .transpose()?;
        if roots
            .iter()
            .chain(root_tangents.iter().flatten())
            .chain(input_cotangents.iter().flatten())
            .any(|value| !value.is_finite())
        {
            return Err(tape_error("local-form execution output is non-finite"));
        }
        Ok(LocalFormExecution {
            roots,
            root_tangents,
            input_cotangents,
        })
    }
}

impl LocalFormExecution {
    pub(super) fn roots(&self) -> &[f64] {
        &self.roots
    }

    pub(super) fn root_tangents(&self) -> Option<&[f64]> {
        self.root_tangents.as_deref()
    }

    pub(super) fn input_cotangents(&self) -> Option<&[f64]> {
        self.input_cotangents.as_deref()
    }
}

pub(super) fn recognize_volume(
    typed: &TypedResidual<RawId>,
    _owner: RawId,
    displacement: RawId,
) -> Result<super::VolumeNodes, Diagnostic> {
    let expression = typed.expression();
    let [root] = expression.roots() else {
        return Err(tape_error("elasticity balance must have one typed root"));
    };
    let Some(ExprNode::Sub(operator, load_gradient)) = expression.node(*root) else {
        return Err(tape_error(
            "elasticity balance root is not the frozen subtraction",
        ));
    };
    let Some(ExprNode::Neg(divergence)) = expression.node(*operator) else {
        return Err(tape_error(
            "elasticity balance does not begin with negative divergence",
        ));
    };
    let Some(ExprNode::Divergence(stress)) = expression.node(*divergence) else {
        return Err(tape_error("elasticity balance has no stress divergence"));
    };
    let Some(ExprNode::Gradient(load)) = expression.node(*load_gradient) else {
        return Err(tape_error(
            "elasticity balance has no conservative load gradient",
        ));
    };
    if !matches!(
        expression.node(*load),
        Some(ExprNode::Symbol(SymbolRef::Field(_)))
    ) || !contains_displacement(expression, *stress, displacement)
    {
        return Err(tape_error(
            "elasticity balance field identities differ from the admitted typed source",
        ));
    }
    Ok(super::VolumeNodes {
        root: *root,
        divergence: *divergence,
        stress: *stress,
        load_gradient: *load_gradient,
    })
}

fn contains_displacement(
    expression: &eqiora_schema::kernel::ExprDag,
    root: ExprId,
    displacement: RawId,
) -> bool {
    let mut pending = vec![root];
    while let Some(node) = pending.pop() {
        match expression.node(node) {
            Some(ExprNode::Symbol(SymbolRef::Field(field))) if field.erase() == displacement => {
                return true;
            }
            Some(ExprNode::Neg(value))
            | Some(ExprNode::Gradient(value))
            | Some(ExprNode::Divergence(value))
            | Some(ExprNode::SymmetricPart(value))
            | Some(ExprNode::IsotropicLift(value)) => pending.push(*value),
            Some(ExprNode::Add(left, right))
            | Some(ExprNode::Sub(left, right))
            | Some(ExprNode::Mul(left, right)) => {
                pending.push(*right);
                pending.push(*left);
            }
            _ => {}
        }
    }
    false
}

pub(super) fn compile_derived_local_form_program_2d(
    source: ElasticityDerivationSource<'_>,
    certificate: &PrimalGalerkinCorrespondence,
) -> Result<LocalFormProgram2d, Diagnostic> {
    source.validate(certificate)?;
    let stress = source.volume.stress;
    let mut compiler = ProgramCompiler::derived(source, certificate);
    let program = compiler.compile_derived(stress)?;
    program.validate()?;
    let witness = ProgramCompiler::witness().compile_witness()?;
    if program.normalized() != witness.normalized() {
        return Err(tape_error(
            "typed elasticity program differs from its frozen normalized witness",
        ));
    }
    Ok(program)
}

pub(super) fn compile_witness_local_form_program_2d() -> Result<LocalFormProgram2d, Diagnostic> {
    let program = ProgramCompiler::witness().compile_witness()?;
    program.validate()?;
    Ok(program)
}

struct ProgramCompiler<'a> {
    typed: Option<&'a TypedResidual<RawId>>,
    displacement: Option<RawId>,
    materials: Option<[Id<kinds::Parameter>; 2]>,
    relation: Option<RawId>,
    certificate: Option<&'a PrimalGalerkinCorrespondence>,
    instructions: Vec<LocalFormInstruction>,
    provenance: Vec<LocalFormInstructionProvenance>,
}

#[derive(Debug, Clone, Copy)]
enum FormalOperand {
    Typed(ExprId),
    StateGradient,
    StateDivergence,
}

#[derive(Debug, Clone, Copy)]
enum StressSource {
    Derived(ExprId),
    Witness,
}

impl<'a> ProgramCompiler<'a> {
    fn derived(
        source: ElasticityDerivationSource<'a>,
        certificate: &'a PrimalGalerkinCorrespondence,
    ) -> Self {
        Self {
            typed: Some(source.typed_balance),
            displacement: Some(source.displacement),
            materials: Some(source.material_parameters),
            relation: Some(source.balance_relation),
            certificate: Some(certificate),
            instructions: Vec::new(),
            provenance: Vec::new(),
        }
    }

    fn witness() -> Self {
        Self {
            typed: None,
            displacement: None,
            materials: None,
            relation: None,
            certificate: None,
            instructions: Vec::new(),
            provenance: Vec::new(),
        }
    }

    fn compile_derived(&mut self, stress: ExprId) -> Result<LocalFormProgram2d, Diagnostic> {
        let mut roots = Vec::with_capacity(ROOT_COUNT);
        for component in 0..ROOT_COUNT {
            roots.push(self.emit_weak_root(component, StressSource::Derived(stress))?);
        }
        self.finish(roots)
    }

    fn compile_witness(&mut self) -> Result<LocalFormProgram2d, Diagnostic> {
        let mut roots = Vec::with_capacity(ROOT_COUNT);
        for component in 0..ROOT_COUNT {
            roots.push(self.emit_weak_root(component, StressSource::Witness)?);
        }
        self.finish(roots)
    }

    fn emit_weak_root(
        &mut self,
        component: usize,
        stress: StressSource,
    ) -> Result<u16, Diagnostic> {
        let mut contraction = None;
        for axis in 0..ROOT_COUNT {
            let stress = match stress {
                StressSource::Derived(root) => self.emit_typed(root, &[component, axis], None)?,
                StressSource::Witness => self.emit_witness_stress(&[component, axis])?,
            };
            let gradient = self.read(
                LocalFormInput2d::ShapeGradient(to_u8(axis)?),
                self.weak_provenance(DIVERGENCE_BY_PARTS)?,
            )?;
            let product = self.push(
                LocalFormInstruction::Mul(stress, gradient),
                self.weak_provenance(DIVERGENCE_BY_PARTS)?,
            )?;
            contraction = Some(match contraction {
                None => product,
                Some(left) => self.push(
                    LocalFormInstruction::Add(left, product),
                    self.weak_provenance(DIVERGENCE_BY_PARTS)?,
                )?,
            });
        }
        let shape = self.read(
            LocalFormInput2d::ShapeValue,
            self.weak_provenance(SOURCE_PAIRING)?,
        )?;
        let force = self.read(
            LocalFormInput2d::BodyForce(to_u8(component)?),
            self.weak_provenance(SOURCE_PAIRING)?,
        )?;
        let load = self.push(
            LocalFormInstruction::Mul(shape, force),
            self.weak_provenance(SOURCE_PAIRING)?,
        )?;
        let negative_load = self.push(
            LocalFormInstruction::Neg(load),
            self.weak_provenance(SOURCE_PAIRING)?,
        )?;
        let residual = self.push(
            LocalFormInstruction::Add(contraction.expect("two axes"), negative_load),
            self.weak_provenance(TEST_PAIRING)?,
        )?;
        let scale = self.read(
            LocalFormInput2d::QuadratureScale,
            self.weak_provenance(TEST_PAIRING)?,
        )?;
        self.push(
            LocalFormInstruction::Mul(scale, residual),
            self.weak_provenance(TEST_PAIRING)?,
        )
    }

    fn emit_witness_stress(&mut self, axes: &[usize]) -> Result<u16, Diagnostic> {
        let standards = [
            StandardPureOperator::SymmetricPart,
            StandardPureOperator::IsotropicLift,
        ];
        let materials = closed_inputs()
            .iter()
            .copied()
            .filter(|input| matches!(input, LocalFormInput2d::Material(_)));
        let mut sum = None;
        for (material, standard) in materials.zip(standards) {
            let definition = standard_definition(standard)?;
            let source_scale = definition_source_scale(&definition)?;
            let mut coefficient = if let Some(scale) = source_scale {
                self.constant(scale.as_f64(), self.source_provenance(None, None))?
            } else {
                self.read(material, self.source_provenance(None, None))?
            };
            if source_scale.is_some() {
                let material = self.read(material, self.source_provenance(None, None))?;
                coefficient = self.push(
                    LocalFormInstruction::Mul(coefficient, material),
                    self.source_provenance(None, None),
                )?;
            }
            let formal = match definition.formals() {
                [value] if value.spatial_rank() == Some(2) => FormalOperand::StateGradient,
                [value] if value.is_invariant_scalar() => FormalOperand::StateDivergence,
                _ => return Err(tape_error("class descriptor has an invalid formal shape")),
            };
            let operator = self.emit_operator(&definition, axes, formal, None)?;
            let term = self.push(
                LocalFormInstruction::Mul(coefficient, operator),
                self.source_provenance(None, None),
            )?;
            sum = Some(match sum {
                None => term,
                Some(left) => self.push(
                    LocalFormInstruction::Add(left, term),
                    self.source_provenance(None, None),
                )?,
            });
        }
        sum.ok_or_else(|| tape_error("class descriptor has no material/operator terms"))
    }

    fn emit_typed(
        &mut self,
        node: ExprId,
        axes: &[usize],
        active_operator: Option<OperatorDefinitionDigest>,
    ) -> Result<u16, Diagnostic> {
        let typed = self
            .typed
            .ok_or_else(|| tape_error("typed source is absent"))?;
        let expression = typed.expression();
        let value = expression
            .node(node)
            .ok_or_else(|| tape_error("typed source node is missing"))?
            .clone();
        let provenance = self.source_provenance(Some(node), active_operator);
        match value {
            ExprNode::Constant(quantity) if axes.is_empty() => self.constant(
                quantity
                    .real_scalar_value()
                    .ok_or_else(|| {
                        tape_error("elasticity execution requires real scalar constants")
                    })?
                    .value(),
                provenance,
            ),
            ExprNode::Symbol(SymbolRef::Parameter(parameter)) if axes.is_empty() => {
                let material = self
                    .materials
                    .ok_or_else(|| tape_error("material identities are absent"))?
                    .iter()
                    .position(|candidate| *candidate == parameter)
                    .ok_or_else(|| tape_error("typed source reads a foreign Parameter"))?;
                self.read(LocalFormInput2d::Material(to_u8(material)?), provenance)
            }
            ExprNode::Neg(value) => {
                let value = self.emit_typed(value, axes, active_operator)?;
                self.push(LocalFormInstruction::Neg(value), provenance)
            }
            ExprNode::Add(left, right) => {
                let left = self.emit_typed(left, axes, active_operator)?;
                let right = self.emit_typed(right, axes, active_operator)?;
                self.push(LocalFormInstruction::Add(left, right), provenance)
            }
            ExprNode::Sub(left, right) => {
                let left = self.emit_typed(left, axes, active_operator)?;
                let right = self.emit_typed(right, axes, active_operator)?;
                let negative = self.push(LocalFormInstruction::Neg(right), provenance)?;
                self.push(LocalFormInstruction::Add(left, negative), provenance)
            }
            ExprNode::Mul(left, right) => {
                let left_axes = self.operand_axes(left, axes)?;
                let right_axes = self.operand_axes(right, axes)?;
                let left = self.emit_typed(left, &left_axes, active_operator)?;
                let right = self.emit_typed(right, &right_axes, active_operator)?;
                self.push(LocalFormInstruction::Mul(left, right), provenance)
            }
            ExprNode::Gradient(argument)
                if self.is_displacement(argument) && axes.len() == ROOT_COUNT =>
            {
                self.read(
                    LocalFormInput2d::StateJet {
                        component: to_u8(axes[0])?,
                        axis: to_u8(axes[1])?,
                    },
                    provenance,
                )
            }
            ExprNode::Divergence(argument) if self.is_displacement(argument) && axes.is_empty() => {
                self.emit_state_divergence(provenance)
            }
            ExprNode::SymmetricPart(_) => {
                self.emit_standard_operator(typed, node, axes, StandardPureOperator::SymmetricPart)
            }
            ExprNode::IsotropicLift(_) => {
                self.emit_standard_operator(typed, node, axes, StandardPureOperator::IsotropicLift)
            }
            _ => Err(tape_error(
                "typed stress contains a node outside the frozen local-form subset",
            )),
        }
    }

    fn emit_standard_operator(
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

    fn emit_operator(
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
                    self.constant(value.as_f64(), provenance)?
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
                | CalculusNode::UnaryMath(..) => {
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

    fn emit_state_divergence(
        &mut self,
        provenance: LocalFormInstructionProvenance,
    ) -> Result<u16, Diagnostic> {
        let first = self.read(
            LocalFormInput2d::StateJet {
                component: 0,
                axis: 0,
            },
            provenance,
        )?;
        let second = self.read(
            LocalFormInput2d::StateJet {
                component: 1,
                axis: 1,
            },
            provenance,
        )?;
        self.push(LocalFormInstruction::Add(first, second), provenance)
    }

    fn operand_axes(&self, node: ExprId, axes: &[usize]) -> Result<Vec<usize>, Diagnostic> {
        let typed = self
            .typed
            .ok_or_else(|| tape_error("typed source is absent"))?;
        let node_type = typed
            .node_type(node)
            .ok_or_else(|| tape_error("typed source sidecar is stale"))?;
        Ok(if node_type.shape().is_scalar() {
            Vec::new()
        } else {
            axes.to_vec()
        })
    }

    fn is_displacement(&self, node: ExprId) -> bool {
        let Some(typed) = self.typed else {
            return false;
        };
        matches!(
            typed.expression().node(node),
            Some(ExprNode::Symbol(SymbolRef::Field(field)))
                if Some(field.erase()) == self.displacement
        )
    }
    fn read(
        &mut self,
        input: LocalFormInput2d,
        provenance: LocalFormInstructionProvenance,
    ) -> Result<u16, Diagnostic> {
        let input = closed_inputs()
            .iter()
            .position(|candidate| *candidate == input)
            .ok_or_else(|| tape_error("compiler requested an unknown dense input role"))?;
        self.push(LocalFormInstruction::Read(to_u16(input)?), provenance)
    }
    fn constant(
        &mut self,
        value: f64,
        provenance: LocalFormInstructionProvenance,
    ) -> Result<u16, Diagnostic> {
        if !value.is_finite() {
            return Err(tape_error("compiler produced a non-finite constant"));
        }
        self.push(
            LocalFormInstruction::ConstantBits(value.to_bits()),
            provenance,
        )
    }
    fn push(
        &mut self,
        instruction: LocalFormInstruction,
        provenance: LocalFormInstructionProvenance,
    ) -> Result<u16, Diagnostic> {
        if self.instructions.len() >= MAX_INSTRUCTIONS {
            return Err(tape_error("local-form instruction bound is exceeded"));
        }
        let index = to_u16(self.instructions.len())?;
        self.instructions.push(instruction);
        self.provenance.push(provenance);
        Ok(index)
    }
    fn source_provenance(
        &self,
        source_node: Option<ExprId>,
        operator_definition: Option<OperatorDefinitionDigest>,
    ) -> LocalFormInstructionProvenance {
        let source_node = source_node.or_else(|| {
            self.certificate.and_then(|certificate| {
                certificate
                    .entries
                    .iter()
                    .find(|entry| {
                        entry.rule_id == SOURCE_PAIRING && Some(entry.relation) == self.relation
                    })
                    .map(|entry| entry.source_node)
            })
        });
        LocalFormInstructionProvenance {
            rule_id: None,
            relation: self.relation,
            source_node: self.relation.and(source_node),
            operator_definition,
        }
    }
    fn weak_provenance(
        &self,
        rule_id: &'static str,
    ) -> Result<LocalFormInstructionProvenance, Diagnostic> {
        let Some(certificate) = self.certificate else {
            return Ok(LocalFormInstructionProvenance {
                rule_id: Some(rule_id),
                relation: None,
                source_node: None,
                operator_definition: None,
            });
        };
        let relation = self
            .relation
            .ok_or_else(|| tape_error("derived compiler has no Relation identity"))?;
        let mut matching = certificate
            .entries
            .iter()
            .filter(|entry| entry.rule_id == rule_id && entry.relation == relation);
        let entry = matching
            .next()
            .ok_or_else(|| tape_error("certificate has no matching weak-rule provenance"))?;
        if matching.next().is_some() {
            return Err(tape_error("certificate has duplicate weak-rule provenance"));
        }
        Ok(LocalFormInstructionProvenance {
            rule_id: Some(entry.rule_id),
            relation: Some(entry.relation),
            source_node: Some(entry.source_node),
            operator_definition: None,
        })
    }
    fn finish(&mut self, roots: Vec<u16>) -> Result<LocalFormProgram2d, Diagnostic> {
        Ok(LocalFormProgram2d {
            inputs: closed_inputs().to_vec(),
            instructions: std::mem::take(&mut self.instructions),
            roots,
            provenance: std::mem::take(&mut self.provenance),
        })
    }
}

fn standard_definition(
    standard: StandardPureOperator,
) -> Result<PureOperatorDefinition, Diagnostic> {
    match standard {
        StandardPureOperator::SymmetricPart => PureOperatorDefinition::symmetric_part(),
        StandardPureOperator::IsotropicLift => PureOperatorDefinition::isotropic_lift(),
    }
    .map_err(|error| tape_error(format!("standard pure operator is invalid: {error}")))
}

fn definition_source_scale(
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
                rationals.push(*value);
            }
            CalculusNode::FormalComponent { .. }
            | CalculusNode::KroneckerDelta(..)
            | CalculusNode::Neg(_)
            | CalculusNode::Add(..)
            | CalculusNode::Mul(..) => {}
            CalculusNode::BoundInput(_)
            | CalculusNode::Differentiated { .. }
            | CalculusNode::Require { .. }
            | CalculusNode::Boolean(_)
            | CalculusNode::Compare(..)
            | CalculusNode::Not(_)
            | CalculusNode::And(..)
            | CalculusNode::Or(..)
            | CalculusNode::Select { .. }
            | CalculusNode::UnaryMath(..) => {
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
    ExactRational::new(
        i64::try_from(value.denominator()).map_err(|_| tape_error("rational denominator"))?,
        value.numerator(),
    )
    .map(Some)
    .map_err(|error| tape_error(format!("pure-operator scale is invalid: {error}")))
}

fn calculus_operand(values: &[u16], operand: eqiora_ir::CalculusNodeId) -> Result<u16, Diagnostic> {
    values
        .get(usize::try_from(operand.index()).map_err(|_| tape_error("calculus operand"))?)
        .copied()
        .ok_or_else(|| tape_error("pure calculus contains a forward reference"))
}

fn to_u8(value: usize) -> Result<u8, Diagnostic> {
    u8::try_from(value).map_err(|_| tape_error("local-form coordinate exceeds u8"))
}

fn to_u16(value: usize) -> Result<u16, Diagnostic> {
    u16::try_from(value).map_err(|_| tape_error("local-form index exceeds u16"))
}
