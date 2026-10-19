#[cfg(test)]
mod affine_tests;
mod complex;
mod coordinate;
use coordinate::ScalarPart;
pub use coordinate::ScalarSymbolCoordinate;
mod linearization;
pub use linearization::ComponentLinearization;
mod finite;
#[cfg(test)]
mod guard_tests;
mod map_factor;
mod pullback;
mod pure;
mod spatial;

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

    /// Borrow the row's scalar operation structure for algebraic coefficient proofs.
    /// This does not evaluate inputs or remove the executable row's domain checks.
    #[must_use]
    pub fn expression_view(&self) -> crate::ScalarExpressionView<'_> {
        self.ir.expression_view()
    }

    /// Conservative polynomial degree in inputs with the supplied degree bounds.
    /// Input order is [`Self::symbols`]. `None` denotes an unproved degree;
    /// bound numeric values never erase dependence. This shares the structural
    /// degree rules used to admit polynomial pencils.
    #[must_use]
    pub fn polynomial_degree(&self, inputs: &[Option<usize>]) -> Option<usize> {
        self.ir.polynomial_degree(inputs)
    }

    /// Prove a row Jacobian constant with respect to every other source coordinate.
    /// This uses the scalar SSA proof, retaining real/imaginary component identity.
    /// Unselected inputs remain symbolic, so state-dependent mass cannot be frozen
    /// accidentally by evaluation at a particular point.
    ///
    /// # Errors
    /// Rejects duplicate selections, nonlinear dependence, variable coefficients,
    /// unsupported instructions, and nonfinite coefficient arithmetic.
    pub fn constant_coordinate_jacobian(
        &self,
        selected: &[ScalarSymbolCoordinate],
    ) -> Result<crate::ConstantSymbolJacobian, Diagnostic> {
        self.ir.constant_coordinate_jacobian(selected)
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

    /// Bind an affine row and its directional coefficient/offset derivative.
    /// Selected coordinates stay symbolic; every other coordinate supplies its
    /// value and optional tangent. `None` holds that input fixed; `Some(0.)`
    /// remains active and must have a defined derivative. The same arithmetic propagates both,
    /// without finite differences or probing selected coordinates.
    ///
    /// # Errors
    /// Rejects the ordinary affine binding failures, nonfinite directions and
    /// undefined derivatives, including a bound predicate at its boundary.
    pub fn bind_affine_direction(
        &self,
        selected: &[ScalarSymbolCoordinate],
        bindings: &[(ScalarSymbolCoordinate, f64, Option<f64>)],
    ) -> Result<
        (
            crate::BoundAffineScalarIr<ScalarSymbolCoordinate>,
            crate::BoundAffineScalarIr<ScalarSymbolCoordinate>,
        ),
        Diagnostic,
    > {
        self.ir.bind_affine_direction(selected, bindings)
    }

    /// Prove and bind a homogeneous polynomial pencil in real spectral coordinates.
    ///
    /// Each key lists spectral powers in the supplied coordinate order; each
    /// value is the corresponding homogeneous mode map. Missing keys are zero.
    /// Structural degree admission precedes coefficient evaluation. Coefficients
    /// are propagated algebraically, never inferred from sampled values. Complex
    /// analyticity, physical dimensions and eigenproblem classes remain with the
    /// mathematical owner. `term_budget` limits retained coefficient resources,
    /// not the dimension or mathematical degree of admitted finite spaces.
    pub fn bind_polynomial_pencil(
        &self,
        selected: &[ScalarSymbolCoordinate],
        spectral: &[ScalarSymbolCoordinate],
        maximum_degree: u32,
        term_budget: usize,
        bindings: &[(ScalarSymbolCoordinate, f64)],
    ) -> Result<
        std::collections::BTreeMap<Vec<u32>, crate::BoundAffineScalarIr<ScalarSymbolCoordinate>>,
        Diagnostic,
    > {
        self.ir
            .bind_polynomial_pencil(selected, spectral, maximum_degree, term_budget, bindings)
    }

    /// Bind first-order products in the exact real component-coordinate order.
    /// Complex symbols retain two coordinates per component. The derivative is
    /// real-linear: `conj`, `real`, `imag`, and `abs2` do not require holomorphy.
    /// The VJP uses the Euclidean pairing on these real coordinates, equivalent
    /// to `Re(sum(conj(a) * b))` on complex components and the ordinary product
    /// on real components. An `abs2(z)` cotangent therefore includes the factor
    /// two from its real differential; it is not a Wirtinger partial.
    ///
    /// Numerical regularity checks are replayed by the shared primal/JVP/VJP evaluator.
    /// # Errors
    /// Rejects wrong cardinality, non-finite points and unsupported numerical instructions.
    pub fn linearize(
        &self,
        inputs: &[f64],
        roles: &[crate::DifferentiationRole],
    ) -> Result<crate::ScalarLinearization<'_>, Diagnostic> {
        self.ir.linearize(inputs, roles)
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
        Self::lower_selected(residual, residual.expression().roots())
    }

    /// Lower selected retained operands using their original typed environment.
    /// Unselected roots do not contribute execution or derivative inputs.
    /// # Errors
    /// Rejects foreign roots and the same unsupported values or resource limits
    /// as full component scalarization.
    pub fn lower_selected<I: Clone + Eq>(
        residual: &TypedResidual<I>,
        roots: &[ExprId],
    ) -> Result<Self, Diagnostic> {
        Self::lower_roots(residual, roots, SpatialContext::Pointwise)
    }

    /// Lower selected spatial operands while retaining exact Field derivative inputs.
    /// Pointwise algebra shares the ordinary component IR. Field gradients retain
    /// separate physical directions, component indices and real/imaginary parts;
    /// the consumer must bind them through its admitted spatial basis. This does
    /// not evaluate a Field gradient from point samples or create shadow Fields.
    /// Traces use the operand reconstructed at the boundary point. Only normal
    /// contractions require an exact oriented normal from the spatial realization;
    /// consumers without normal contractions may pass `None`.
    ///
    /// # Errors
    /// Rejects unsupported differential expressions, Field pullbacks without a
    /// reconstruction, and the ordinary typed component lowering failures.
    pub fn lower_spatial_selected<I: Clone + Eq>(
        residual: &TypedResidual<I>,
        roots: &[ExprId],
        normal: Option<&[f64]>,
    ) -> Result<Self, Diagnostic> {
        Self::lower_roots(residual, roots, SpatialContext::Reconstructed { normal })
    }

    fn lower_roots<I: Clone + Eq>(
        residual: &TypedResidual<I>,
        roots: &[ExprId],
        spatial: SpatialContext<'_>,
    ) -> Result<Self, Diagnostic> {
        let expression = residual.expression();
        let mut rows = Vec::new();
        let mut finite_products = 0usize;
        for (root_index, root) in roots.iter().copied().enumerate() {
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
                        spatial,
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
    spatial: SpatialContext<'_>,
) -> Result<(ScalarInputOperatorIr, Vec<ScalarSymbolCoordinate>), Diagnostic> {
    let mut lowering = ComponentDagLowering::new(expression, node_types, finite_products, spatial);
    let root = lowering.lower_part(root, component_index, part)?;
    Ok((lowering.builder.finish([root])?, lowering.inputs))
}

#[derive(Clone, Copy)]
enum SpatialContext<'a> {
    Pointwise,
    Reconstructed { normal: Option<&'a [f64]> },
}

struct ComponentDagLowering<'a, I> {
    finite_products: &'a mut usize,
    expression: &'a ExprDag,
    node_types: &'a [ExpressionType<I>],
    builder: ScalarInputIrBuilder,
    remapped: HashMap<(usize, Box<[u32]>, ScalarPart), ScalarInputValueId>,
    map_entries: HashMap<ExprId, ScalarInputValueId>,
    inputs: Vec<ScalarSymbolCoordinate>,
    input_nodes: HashMap<ScalarSymbolCoordinate, ScalarInputValueId>,
    symbol_shapes: HashMap<SymbolRef, ValueShape>,
    coordinate_bindings: HashMap<SymbolRef, ScalarInputValueId>,
    pullback_depth: usize,
    spatial: SpatialContext<'a>,
}

impl<'a, I: Clone + Eq> ComponentDagLowering<'a, I> {
    fn new(
        expression: &'a ExprDag,
        node_types: &'a [ExpressionType<I>],
        finite_products: &'a mut usize,
        spatial: SpatialContext<'a>,
    ) -> Self {
        Self {
            finite_products,
            expression,
            node_types,
            builder: ScalarInputIrBuilder::new(),
            remapped: HashMap::new(),
            map_entries: HashMap::new(),
            inputs: Vec::new(),
            input_nodes: HashMap::new(),
            symbol_shapes: HashMap::new(),
            coordinate_bindings: HashMap::new(),
            pullback_depth: 0,
            spatial,
        }
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
        // Only demanded operands belong to this projection. Other retained roots
        // may carry exact discrete values that must never enter floating-point IR.
        if !matches!(
            node_type.value_type.scalar_domain(),
            eqiora_core::ScalarDomain::Real | eqiora_core::ScalarDomain::Complex
        ) {
            return Err(invalid_component_ir(
                "component scalarization requires real or complex mathematical values",
            ));
        }
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
            ExprNode::Pullback { value, at, .. } => self.lower_pullback(value, &at, part)?,
            ExprNode::CoordinateMapFactor { .. } if !self.coordinate_bindings.is_empty() => {
                self.lower_before_substitution(value, component, part)?
            }
            ExprNode::CoordinateMapFactor { factor, source, at } => {
                self.lower_map_factor(factor, &source, &at, node_type.dimension())?
            }
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
            ExprNode::Gradient(_) | ExprNode::CoordinatePartial { .. }
                if matches!(self.spatial, SpatialContext::Reconstructed { .. }) =>
            {
                self.lower_field_gradient(value, component, part)?
            }
            ExprNode::Divergence(operand)
                if matches!(self.spatial, SpatialContext::Reconstructed { .. }) =>
            {
                self.lower_field_divergence(operand, component, part, node_type.dimension())?
            }
            ExprNode::Trace { value, .. } => {
                if !matches!(self.spatial, SpatialContext::Reconstructed { .. }) {
                    return Err(invalid_component_ir(
                        "spatial trace requires a reconstructed boundary value",
                    ));
                }
                self.lower_part(value, component, part)?
            }
            ExprNode::NormalComponent { value, .. } => {
                self.lower_normal_component(value, component, part)?
            }
            ExprNode::Symbol(symbol) => {
                if let Some(bound) = self.coordinate_bindings.get(&symbol) {
                    *bound
                } else {
                    if self.pullback_depth > 0
                        && node_type.support.is_some()
                        && !matches!(symbol, SymbolRef::Coordinate { .. })
                    {
                        return Err(invalid_component_ir(
                            "spatial Field pullback requires an admitted point reconstruction",
                        ));
                    }
                    self.input(
                        symbol,
                        node_type.shape(),
                        node_type.dimension(),
                        component,
                        part,
                        &[],
                    )?
                }
            }
            ExprNode::Require { condition, value } => {
                let condition = self.lower_condition(condition)?;
                let value = self.lower_shaped_part(value, component, part)?;
                self.builder.require(condition, value)?
            }
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
                part,
            )?,
            ExprNode::IsotropicLift(operand) => self.lower_pure_operator(
                StandardPureOperator::IsotropicLift,
                operand,
                component,
                node_type,
                part,
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
                    part,
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

    fn lower_condition(&mut self, id: ExprId) -> Result<ScalarInputValueId, Diagnostic> {
        let key = (
            id.index() as usize,
            Box::<[u32]>::default(),
            ScalarPart::Real,
        );
        if let Some(value) = self.remapped.get(&key) {
            return Ok(*value);
        }
        let value = match self.expression.node(id).cloned() {
            Some(ExprNode::Compare(op, left, right)) => {
                for id in [left, right] {
                    let ty = &self.node_types[id.index() as usize].value_type;
                    if ty.scalar_domain() != eqiora_core::ScalarDomain::Real
                        || !ty.shape().is_scalar()
                        || ty.array_rank() != 0
                    {
                        return Err(invalid_component_ir(
                            "component domain comparisons require real scalar operands",
                        ));
                    }
                }
                let left = self.lower_part(left, &[], ScalarPart::Real)?;
                let right = self.lower_part(right, &[], ScalarPart::Real)?;
                self.builder.compare(op, left, right)?
            }
            Some(ExprNode::And(left, right)) => {
                let left = self.lower_condition(left)?;
                let right = self.lower_condition(right)?;
                self.builder.and(left, right)?
            }
            _ => {
                return Err(invalid_component_ir(
                    "component domain guards require real scalar comparisons and conjunctions",
                ));
            }
        };
        self.remapped.insert(key, value);
        Ok(value)
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
        dimension: eqiora_core::DimExponents,
        component: &[u32],
        part: ScalarPart,
        spatial_derivative_axes: &[usize],
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
            spatial_derivative_axes: spatial_derivative_axes.into(),
        };
        if let Some(mapped) = self.input_nodes.get(&coordinate) {
            return Ok(*mapped);
        }

        let ordinal = u32::try_from(self.inputs.len())
            .map_err(|_| invalid_component_ir("component input slot exceeds portable u32"))?;
        let mapped =
            self.builder
                .input(ScalarInputSlot::new(ordinal, coordinate.clone(), dimension))?;
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
