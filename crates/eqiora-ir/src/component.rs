mod complex;
mod finite;

use std::collections::HashMap;

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, DynQuantity, GraphPath, ValueShape};
use eqiora_schema::kernel::typing::{ExpressionType, TypedResidual};
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef};

use crate::scalar::{
    ScalarInputIrBuilder, ScalarInputOperatorIr, ScalarInputSlot, ScalarInputValueId,
};
use crate::{
    OperatorExpansionExt, PureOperatorDefinition, ScalarCalculusNode, StandardPureOperator,
};

/// Real coordinate within a mathematical scalar, independent of storage precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ScalarPart {
    /// Real component (the only coordinate of a real scalar).
    Real,
    /// Imaginary component of a complex scalar.
    Imaginary,
}

/// One scalar coordinate of a shaped Semantic Model symbol.
///
/// The empty component index denotes a scalar. Non-scalar indices are
/// lexicographic row-major coordinates with the last axis varying fastest.
/// Real and imaginary parts retain that same symbol and channel coordinate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScalarSymbolCoordinate {
    symbol: SymbolRef,
    component_index: Box<[u32]>,
    part: ScalarPart,
}

impl ScalarSymbolCoordinate {
    /// Enumerate a numeric value's coordinates in channel-major, real/imaginary order.
    ///
    /// # Errors
    /// Rejects discrete values or unrepresentable component cardinalities.
    pub fn for_value(
        symbol: SymbolRef,
        value_type: &eqiora_core::ValueType,
    ) -> Result<Vec<Self>, Diagnostic> {
        let parts: &[ScalarPart] = match value_type.scalar_domain() {
            eqiora_core::ScalarDomain::Real => &[ScalarPart::Real],
            eqiora_core::ScalarDomain::Complex => &[ScalarPart::Real, ScalarPart::Imaginary],
            _ => {
                return Err(invalid_component_ir(
                    "scalar coordinates require a numeric value",
                ));
            }
        };
        let count = value_type
            .shape()
            .component_count()
            .ok_or_else(|| invalid_component_ir("coordinate count overflow"))?;
        let mut coordinates = Vec::new();
        coordinates
            .try_reserve_exact(
                count
                    .checked_mul(parts.len())
                    .ok_or_else(|| invalid_component_ir("coordinate count overflow"))?,
            )
            .map_err(|_| invalid_component_ir("cannot allocate scalar coordinates"))?;
        for flat in 0..count {
            let component_index = row_major_index(value_type.shape(), flat)?;
            coordinates.extend(parts.iter().map(|&part| Self {
                symbol,
                component_index: component_index.clone(),
                part,
            }));
        }
        Ok(coordinates)
    }

    /// Semantic symbol before Operator lowering.
    #[must_use]
    pub const fn symbol(&self) -> SymbolRef {
        self.symbol
    }

    /// Whether this is the imaginary coordinate; false denotes the real coordinate.
    /// This is separate from the channel index.
    #[must_use]
    pub const fn is_imaginary(&self) -> bool {
        matches!(self.part, ScalarPart::Imaginary)
    }

    /// Exact row-major component multi-index; empty for a scalar.
    #[must_use]
    pub const fn component_index(&self) -> &[u32] {
        &self.component_index
    }
}

/// One deterministic scalar residual row lowered from a shaped root.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentScalarRow {
    root_index: usize,
    component_index: Box<[u32]>,
    part: ScalarPart,
    symbols: Vec<ScalarSymbolCoordinate>,
    ir: ScalarInputOperatorIr,
}

impl ComponentScalarRow {
    /// Original Relation root index.
    #[must_use]
    pub const fn root_index(&self) -> usize {
        self.root_index
    }

    /// Whether this is the imaginary residual coordinate of the original root.
    #[must_use]
    pub const fn is_imaginary(&self) -> bool {
        matches!(self.part, ScalarPart::Imaginary)
    }

    /// Row-major component multi-index within the original shaped root.
    #[must_use]
    pub const fn component_index(&self) -> &[u32] {
        &self.component_index
    }

    /// Dense shaped-symbol coordinates expected by [`Self::evaluate`].
    #[must_use]
    pub fn symbols(&self) -> &[ScalarSymbolCoordinate] {
        &self.symbols
    }

    /// Typed IR-local reads corresponding one-for-one with [`Self::symbols`].
    ///
    /// These dense slots are evaluator plumbing. Their source coordinates
    /// retain real Semantic identities, but the slots are not Parameters or
    /// any other Semantic symbol kind.
    #[must_use]
    pub fn input_slots(&self) -> &[ScalarInputSlot] {
        self.ir.slots()
    }

    /// Structurally bind a real affine row in exact symbol/channel/part coordinates.
    /// This uses the ordinary scalar SSA affine proof, without numerical probing.
    ///
    /// # Errors
    /// Rejects nonlinear dependence, duplicate coordinates, missing or nonfinite
    /// bindings, selected coordinates bound as constants, and invalid arithmetic.
    pub fn bind_affine(
        &self,
        selected: &[ScalarSymbolCoordinate],
        bindings: &[(ScalarSymbolCoordinate, f64)],
    ) -> Result<crate::BoundAffineScalarIr<ScalarSymbolCoordinate>, Diagnostic> {
        self.ir.bind_affine(selected, bindings)
    }

    /// Evaluate this scalar row using dense inputs matching [`Self::symbols`].
    ///
    /// # Errors
    /// Returns the ordinary scalar Operator IR diagnostics for invalid input
    /// cardinality or non-finite arithmetic.
    pub fn evaluate(&self, inputs: &[f64]) -> Result<f64, Diagnostic> {
        let values = self.ir.evaluate(inputs)?;
        values
            .first()
            .copied()
            .ok_or_else(|| invalid_component_ir("component scalar row produced no residual value"))
    }
}

/// Operator-lowering proof that shaped residuals mean componentwise zero.
///
/// Semantic expression nodes remain shaped. This lowered form alone expands
/// roots into deterministic scalar rows; it does not add component-selection
/// nodes to canonical meaning.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentScalarization {
    rows: Vec<ComponentScalarRow>,
}

impl ComponentScalarization {
    /// Scalarize one fully typed pointwise residual.
    ///
    /// Rows are ordered by Relation root, row-major component index, then
    /// real/imaginary part. A real scalar root contributes one row with an empty
    /// index; a complex root contributes two. Spatial operators fail closed
    /// because their scalarization belongs to a discretized
    /// Operator lowering, not this pointwise contract.
    ///
    /// # Errors
    /// Returns `EQ0701` for shape cardinality mismatch, unrepresentable row
    /// count, inconsistent repeated-symbol shape, a non-pointwise expression,
    /// or an invalid exact component coordinate.
    pub fn lower<I: Clone + Eq>(residual: &TypedResidual<I>) -> Result<Self, Diagnostic> {
        if residual.node_types().iter().any(|value| {
            !matches!(
                value.value_type.scalar_domain(),
                eqiora_core::ScalarDomain::Real | eqiora_core::ScalarDomain::Complex
            )
        }) {
            return Err(invalid_component_ir(
                "component scalarization requires real or complex mathematical values",
            ));
        }
        let expression = residual.expression();
        let mut rows = Vec::new();
        let mut finite_products = 0usize;
        for (root_index, root) in expression.roots().iter().copied().enumerate() {
            let root_node_index = node_index(root, expression.nodes().len())?;
            let root_shape = residual.node_types()[root_node_index].shape();
            let component_count = root_shape.component_count().ok_or_else(|| {
                invalid_component_ir("component scalarization row count exceeds local usize")
            })?;

            for flat_index in 0..component_count {
                let component_index = row_major_index(root_shape, flat_index)?;
                let parts: &[ScalarPart] = if residual.node_types()[root_node_index]
                    .value_type
                    .scalar_domain()
                    == eqiora_core::ScalarDomain::Complex
                {
                    &[ScalarPart::Real, ScalarPart::Imaginary]
                } else {
                    &[ScalarPart::Real]
                };
                for &part in parts {
                    let (ir, symbols) = component_single_root(
                        expression,
                        residual.node_types(),
                        root,
                        &component_index,
                        part,
                        &mut finite_products,
                    )?;
                    let aligned = ir.slots().iter().zip(&symbols).enumerate().all(
                        |(index, (slot, coordinate))| {
                            usize::try_from(slot.ordinal()) == Ok(index)
                                && slot.source() == coordinate
                        },
                    );
                    if ir.slots().len() != symbols.len() || !aligned {
                        return Err(invalid_component_ir(
                            "component input coordinates do not match typed local slots",
                        ));
                    }
                    rows.push(ComponentScalarRow {
                        root_index,
                        component_index: component_index.clone(),
                        part,
                        symbols,
                        ir,
                    });
                }
            }
        }
        Ok(Self { rows })
    }

    /// Deterministic scalar residual rows.
    #[must_use]
    pub fn rows(&self) -> &[ComponentScalarRow] {
        &self.rows
    }

    /// Evaluate every row through one shaped-coordinate resolver.
    ///
    /// # Errors
    /// Returns `EQ0702` when a coordinate is missing, or the underlying scalar
    /// Operator IR diagnostic for non-finite arithmetic.
    pub fn evaluate(
        &self,
        mut resolve: impl FnMut(&ScalarSymbolCoordinate) -> Option<f64>,
    ) -> Result<Vec<f64>, Diagnostic> {
        self.rows
            .iter()
            .map(|row| {
                let inputs = row
                    .symbols()
                    .iter()
                    .map(|coordinate| {
                        resolve(coordinate).ok_or_else(|| {
                            Diagnostic::error(
                                codes::OPERATOR_INPUT_MISMATCH,
                                format!(
                                    "missing component input for {:?}{:?}",
                                    coordinate.symbol(),
                                    coordinate.component_index()
                                ),
                            )
                            .with_graph_path(GraphPath::new(["operator-ir", "component-inputs"]))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                row.evaluate(&inputs)
            })
            .collect()
    }
}

fn component_single_root<I: Clone + Eq>(
    expression: &ExprDag,
    node_types: &[ExpressionType<I>],
    root: ExprId,
    component_index: &[u32],
    part: ScalarPart,
    finite_products: &mut usize,
) -> Result<(ScalarInputOperatorIr, Vec<ScalarSymbolCoordinate>), Diagnostic> {
    let mut lowering = ComponentDagLowering {
        finite_products,
        expression,
        node_types,
        builder: ScalarInputIrBuilder::new(),
        remapped: HashMap::new(),
        inputs: Vec::new(),
        input_nodes: HashMap::new(),
        symbol_shapes: HashMap::new(),
    };
    let root = lowering.lower_part(root, component_index, part)?;
    Ok((lowering.builder.finish([root])?, lowering.inputs))
}

struct ComponentDagLowering<'a, I> {
    finite_products: &'a mut usize,
    expression: &'a ExprDag,
    node_types: &'a [ExpressionType<I>],
    builder: ScalarInputIrBuilder,
    remapped: HashMap<(usize, Box<[u32]>, ScalarPart), ScalarInputValueId>,
    inputs: Vec<ScalarSymbolCoordinate>,
    input_nodes: HashMap<ScalarSymbolCoordinate, ScalarInputValueId>,
    symbol_shapes: HashMap<SymbolRef, ValueShape>,
}

impl<I: Clone + Eq> ComponentDagLowering<'_, I> {
    fn lower(
        &mut self,
        value: ExprId,
        component: &[u32],
    ) -> Result<ScalarInputValueId, Diagnostic> {
        self.lower_part(value, component, ScalarPart::Real)
    }

    fn lower_part(
        &mut self,
        value: ExprId,
        component: &[u32],
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let index = node_index(value, self.expression.nodes().len())?;
        let node_type = self
            .node_types
            .get(index)
            .ok_or_else(|| invalid_component_ir("component node has no inferred type"))?;
        validate_component(node_type.shape(), component)?;
        let key = (index, component.into(), part);
        if let Some(mapped) = self.remapped.get(&key) {
            return Ok(*mapped);
        }

        if part == ScalarPart::Imaginary
            && node_type.value_type.scalar_domain() == eqiora_core::ScalarDomain::Real
        {
            // Embedding does not erase demanded domain checks in the real operand.
            let real = self.lower_part(value, component, ScalarPart::Real)?;
            let zero = self.builder.sub(real, real)?;
            self.remapped.insert(key, zero);
            return Ok(zero);
        }
        let node = self.expression.nodes()[index].clone();
        if let Some(mapped) = self.lower_complex(&node, component, part)? {
            self.remapped.insert(key, mapped);
            return Ok(mapped);
        }
        let mapped = match node {
            ExprNode::FiniteUnary(operation, operand) => {
                self.lower_finite_unary(operation, operand, component, part)?
            }
            ExprNode::FiniteBinary(operation, left, right) => {
                self.lower_finite_binary(operation, left, right, component, part)?
            }
            ExprNode::Constant(constant) => {
                let flat = constant
                    .value_type()
                    .shape()
                    .extents()
                    .iter()
                    .zip(component)
                    .fold(0_usize, |offset, (extent, coordinate)| {
                        offset * extent.get() as usize + *coordinate as usize
                    });
                let pair = constant.component(flat).ok_or_else(|| {
                    invalid_component_ir("literal component is outside its exact shape")
                })?;
                self.builder.constant(eqiora_core::DynQuantity::new(
                    if part == ScalarPart::Real {
                        pair.0
                    } else {
                        pair.1
                    },
                    constant.value_type().dimension(),
                ))?
            }
            ExprNode::Array { elements } => {
                let (channel, inner) = component.split_first().ok_or_else(|| {
                    invalid_component_ir("array component needs an outer channel")
                })?;
                let element = elements.get(*channel as usize).copied().ok_or_else(|| {
                    invalid_component_ir("array channel is outside its exact extent")
                })?;
                self.lower_part(element, inner, part)?
            }
            ExprNode::Index { value, index } => {
                let coordinates = std::iter::once(index)
                    .chain(component.iter().copied())
                    .collect::<Vec<_>>();
                self.lower_part(value, &coordinates, part)?
            }
            ExprNode::Symbol(symbol) => self.input(symbol, node_type.shape(), component, part)?,
            ExprNode::Neg(operand) => {
                let operand = self.lower_shaped_part(operand, component, part)?;
                self.builder.neg(operand)?
            }
            ExprNode::Add(left, right) => {
                let left = self.lower_shaped_part(left, component, part)?;
                let right = self.lower_shaped_part(right, component, part)?;
                self.builder.add(left, right)?
            }
            ExprNode::Sub(left, right) => {
                let left = self.lower_shaped_part(left, component, part)?;
                let right = self.lower_shaped_part(right, component, part)?;
                self.builder.sub(left, right)?
            }
            ExprNode::Mul(left, right) => {
                let left = self.lower_shaped_part(left, component, part)?;
                let right = self.lower_shaped_part(right, component, part)?;
                self.builder.mul(left, right)?
            }
            ExprNode::Div(numerator, denominator) => {
                let numerator = self.lower_shaped(numerator, component)?;
                let denominator = self.lower_shaped(denominator, component)?;
                self.builder.div(numerator, denominator)?
            }
            ExprNode::PowI(base, exponent) => {
                let base = self.lower_shaped(base, component)?;
                self.builder.powi(base, exponent)?
            }
            ExprNode::SymmetricPart(operand) => self.lower_pure_operator(
                StandardPureOperator::SymmetricPart,
                operand,
                component,
                node_type,
            )?,
            ExprNode::IsotropicLift(operand) => self.lower_pure_operator(
                StandardPureOperator::IsotropicLift,
                operand,
                component,
                node_type,
            )?,
            ExprNode::PureOperatorApplication(application) => {
                let definition = self
                    .expression
                    .definition(application.definition())
                    .cloned()
                    .ok_or_else(|| {
                        invalid_component_ir(
                            "pure operator application has no exact expression-local definition",
                        )
                    })?;
                self.lower_pure_definition(
                    &definition,
                    application.arguments(),
                    component,
                    node_type,
                )?
            }
            _ => {
                return Err(invalid_component_ir(
                    "component scalarization requires pointwise algebra or tensor structure",
                ));
            }
        };
        self.remapped.insert(key, mapped);
        Ok(mapped)
    }

    fn lower_pure_operator(
        &mut self,
        operator: StandardPureOperator,
        operand: ExprId,
        component: &[u32],
        expected_result: &ExpressionType<I>,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let definition = match operator {
            StandardPureOperator::SymmetricPart => PureOperatorDefinition::symmetric_part(),
            StandardPureOperator::IsotropicLift => PureOperatorDefinition::isotropic_lift(),
        }
        .map_err(|error| invalid_component_ir(format!("invalid pure operator: {error}")))?;
        self.lower_pure_definition(&definition, &[operand], component, expected_result)
    }

    fn lower_pure_definition(
        &mut self,
        definition: &PureOperatorDefinition,
        arguments: &[ExprId],
        component: &[u32],
        expected_result: &ExpressionType<I>,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let argument_types = arguments
            .iter()
            .map(|argument| {
                let index = node_index(*argument, self.expression.nodes().len())?;
                self.node_types.get(index).cloned().ok_or_else(|| {
                    invalid_component_ir("pure operator argument has no inferred type")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expansion = definition.instantiate(&argument_types).map_err(|error| {
            invalid_component_ir(format!("pure operator typing failed: {error}"))
        })?;
        if expansion.result_type() != expected_result {
            return Err(invalid_component_ir(
                "pure operator expansion differs from the inferred Kernel type",
            ));
        }
        let calculus = expansion.component(component).map_err(|error| {
            invalid_component_ir(format!("pure operator component expansion failed: {error}"))
        })?;
        let mut remapped = vec![None; calculus.nodes().len()];
        self.lower_calculus_component(&calculus, calculus.root(), arguments, &mut remapped)
    }

    fn lower_calculus_component(
        &mut self,
        calculus: &crate::ScalarCalculus<I>,
        value: crate::CalculusNodeId,
        arguments: &[ExprId],
        remapped: &mut [Option<ScalarInputValueId>],
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let index = usize::try_from(value.index())
            .ok()
            .filter(|index| *index < calculus.nodes().len())
            .ok_or_else(|| {
                invalid_component_ir("pure calculus contains an invalid value reference")
            })?;
        if let Some(mapped) = remapped[index] {
            return Ok(mapped);
        }
        let node = calculus.nodes()[index].clone();
        let mapped = match node {
            ScalarCalculusNode::Rational { value, dimension } => self
                .builder
                .constant(DynQuantity::new(value.as_f64(), dimension))?,
            ScalarCalculusNode::FormalComponent(atom) => {
                let operand = arguments
                    .get(usize::from(atom.formal()))
                    .copied()
                    .ok_or_else(|| {
                        invalid_component_ir("pure calculus referenced an unexpected formal")
                    })?;
                self.lower(operand, atom.component())?
            }
            ScalarCalculusNode::Neg(value) => {
                let value = self.lower_calculus_component(calculus, value, arguments, remapped)?;
                self.builder.neg(value)?
            }
            ScalarCalculusNode::Add(left, right) => {
                let left = self.lower_calculus_component(calculus, left, arguments, remapped)?;
                let right = self.lower_calculus_component(calculus, right, arguments, remapped)?;
                self.builder.add(left, right)?
            }
            ScalarCalculusNode::Mul(left, right) => {
                let left = self.lower_calculus_component(calculus, left, arguments, remapped)?;
                let right = self.lower_calculus_component(calculus, right, arguments, remapped)?;
                self.builder.mul(left, right)?
            }
        };
        remapped[index] = Some(mapped);
        Ok(mapped)
    }

    fn lower_shaped(
        &mut self,
        operand: ExprId,
        result_component: &[u32],
    ) -> Result<ScalarInputValueId, Diagnostic> {
        self.lower_shaped_part(operand, result_component, ScalarPart::Real)
    }

    fn lower_shaped_part(
        &mut self,
        operand: ExprId,
        result_component: &[u32],
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let operand_index = node_index(operand, self.expression.nodes().len())?;
        let operand_shape = self
            .node_types
            .get(operand_index)
            .ok_or_else(|| invalid_component_ir("component operand has no inferred type"))?
            .shape();
        if operand_shape.is_scalar() {
            self.lower_part(operand, &[], part)
        } else {
            self.lower_part(operand, result_component, part)
        }
    }

    fn input(
        &mut self,
        symbol: SymbolRef,
        shape: &ValueShape,
        component: &[u32],
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        match self.symbol_shapes.entry(symbol) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(shape.clone());
            }
            std::collections::hash_map::Entry::Occupied(entry) if entry.get() != shape => {
                return Err(invalid_component_ir(format!(
                    "repeated symbol {symbol:?} has inconsistent exact shapes"
                )));
            }
            std::collections::hash_map::Entry::Occupied(_) => {}
        }
        let coordinate = ScalarSymbolCoordinate {
            symbol,
            component_index: component.into(),
            part,
        };
        if let Some(mapped) = self.input_nodes.get(&coordinate) {
            return Ok(*mapped);
        }

        let ordinal = u32::try_from(self.inputs.len())
            .map_err(|_| invalid_component_ir("component input slot exceeds portable u32"))?;
        let mapped = self
            .builder
            .input(ScalarInputSlot::new(ordinal, coordinate.clone()))?;
        self.inputs.push(coordinate.clone());
        self.input_nodes.insert(coordinate, mapped);
        Ok(mapped)
    }
}

fn validate_component(shape: &ValueShape, component: &[u32]) -> Result<(), Diagnostic> {
    if shape.rank() != component.len()
        || shape
            .extents()
            .iter()
            .zip(component)
            .any(|(extent, coordinate)| *coordinate >= extent.get())
    {
        Err(invalid_component_ir(
            "component coordinate is outside the exact value shape",
        ))
    } else {
        Ok(())
    }
}

fn node_index(id: ExprId, upper_bound: usize) -> Result<usize, Diagnostic> {
    usize::try_from(id.index())
        .ok()
        .filter(|index| *index < upper_bound)
        .ok_or_else(|| invalid_component_ir("component scalarization ExprId is out of bounds"))
}

fn row_major_index(shape: &ValueShape, mut flat: usize) -> Result<Box<[u32]>, Diagnostic> {
    let mut index = vec![0_u32; shape.rank()];
    for (axis, extent) in shape.extents().iter().enumerate().rev() {
        let extent = usize::try_from(extent.get())
            .map_err(|_| invalid_component_ir("shape extent exceeds local usize"))?;
        index[axis] = u32::try_from(flat % extent)
            .map_err(|_| invalid_component_ir("component index exceeds portable u32"))?;
        flat /= extent;
    }
    if flat != 0 {
        return Err(invalid_component_ir(
            "flat component index exceeds the exact shape",
        ));
    }
    Ok(index.into_boxed_slice())
}

fn invalid_component_ir(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_OPERATOR_IR, message)
        .with_graph_path(GraphPath::new(["operator-ir", "component-scalarization"]))
}

#[cfg(test)]
mod tests;
