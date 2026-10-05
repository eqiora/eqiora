//! Method-neutral evaluation tape for scalar spatial data.

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, GraphPath, Id, RawId, Scalar};
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef, UnaryMathFunction};
use eqiora_sem::KernelProgram;

mod lowering;
mod tangent;
pub(crate) use lowering::{Coefficient, lower};
use num_complex::ComplexFloat;
pub(crate) use tangent::sqrt_derivative_root;

// Physical point samplers cannot interpret an abstract product factor's axis
// as an axis of their ambient point, even when the units happen to agree.
pub(crate) fn physical_coordinate(
    program: &KernelProgram,
    support: Id<kinds::Domain>,
    factor: Id<kinds::Domain>,
) -> bool {
    use eqiora_graph::EdgeKind;
    use eqiora_schema::kernel::{DomainKind, KernelNode};
    match program.node(support.erase()) {
        Some(KernelNode::Domain(domain)) => match domain.kind() {
            DomainKind::CartesianBox { .. } | DomainKind::GeometryRegion { .. } => {
                support == factor
            }
            DomainKind::CartesianBoundary { .. } | DomainKind::GeometryBoundary { .. } => {
                program.edges().iter().any(|edge| {
                    edge.from() == support.erase()
                        && edge.kind() == EdgeKind::BoundaryOf
                        && edge.to() == factor.erase()
                })
            }
            _ => false,
        },
        _ => false,
    }
}

/// Inspectable scalar expression over one physical coordinate space.
///
/// This is a lowered semantic object, not a Rust callback: distinct numerical
/// realizations can evaluate the same immutable tape without owning model
/// meaning. The tape contains only scalar arithmetic, indexed physical
/// coordinates, and dimensionally validated unary mathematics.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarSpatialExpression<S: Scalar> {
    coordinate_dimension: usize,
    instructions: Vec<Instruction<S>>,
    root: usize,
    coordinate_dependent: bool,
    parameter_fields: Vec<Id<kinds::Parameter>>,
    parameter_values: Vec<S>,
}

#[derive(Debug, Clone, PartialEq)]
enum Instruction<S: Scalar> {
    Constant(S),
    Parameter(usize),
    Coordinate(usize),
    Neg(usize),
    Conjugate(usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    PowI(usize, i32),
    Sin(usize),
    Sqrt(usize),
}

impl<S: Scalar + ComplexFloat<Real = f64> + From<f64>> ScalarSpatialExpression<S> {
    /// Exact physical coordinate dimension admitted by this tape.
    #[must_use]
    pub const fn coordinate_dimension(&self) -> usize {
        self.coordinate_dimension
    }

    /// Evaluate in coherent SI coordinates.
    ///
    /// # Errors
    /// Returns `EQ0702` for coordinate shape mismatch and `EQ0505` for a
    /// non-finite coordinate or intermediate value.
    pub fn evaluate(&self, coordinates: &[f64]) -> Result<S, Diagnostic> {
        self.validate_coordinates(coordinates)?;
        let values = self.primal_values(coordinates)?;
        Ok(values[self.root])
    }

    fn primal_values(&self, coordinates: &[f64]) -> Result<Vec<S>, Diagnostic> {
        let mut values: Vec<S> = Vec::with_capacity(self.instructions.len());
        for instruction in &self.instructions {
            let value = match *instruction {
                Instruction::Constant(value) => value,
                Instruction::Parameter(parameter) => self.parameter_values[parameter],
                Instruction::Coordinate(axis) => <S as From<f64>>::from(coordinates[axis]),
                Instruction::Neg(value) => -values[value],
                Instruction::Conjugate(value) => values[value].conj(),
                Instruction::Add(left, right) => values[left] + values[right],
                Instruction::Sub(left, right) => values[left] - values[right],
                Instruction::Mul(left, right) => values[left] * values[right],
                Instruction::Div(left, right) => values[left] / values[right],
                Instruction::PowI(base, exponent) => values[base].powi(exponent),
                Instruction::Sin(value) => values[value].sin(),
                Instruction::Sqrt(value) => {
                    if S::DOMAIN == eqiora_core::ScalarDomain::Real {
                        <S as From<f64>>::from(real_sqrt(values[value].re())?)
                    } else {
                        values[value].sqrt()
                    }
                }
            };
            if !value.is_finite() {
                return Err(nonfinite(
                    "scalar spatial expression produced a non-finite primal value",
                ));
            }
            values.push(value);
        }
        Ok(values)
    }
}

impl<S: Scalar + ComplexFloat<Real = f64> + From<f64>> ScalarSpatialExpression<S> {
    /// Canonical Parameters retained by this tape, in dense tangent order.
    ///
    /// The order is deterministic first occurrence in the lowered expression.
    #[must_use]
    pub fn parameter_fields(&self) -> &[Id<kinds::Parameter>] {
        &self.parameter_fields
    }

    /// Complete Parameter values at this tape's evaluation point.
    #[must_use]
    pub fn parameter_values(&self) -> &[S] {
        &self.parameter_values
    }

    /// Return the same immutable tape at one complete enclosing Parameter point.
    ///
    /// `parameter_fields` may contain Parameters used by sibling expressions,
    /// but it must contain every Parameter used by this tape exactly once.
    /// The returned expression owns its point; this expression remains
    /// unchanged.
    ///
    /// # Errors
    /// Returns a shape/identity diagnostic for mismatched, duplicate, or
    /// incomplete coordinates, and a non-finite diagnostic for invalid values.
    pub(crate) fn bind_parameter_point(
        &self,
        parameter_fields: &[Id<kinds::Parameter>],
        parameter_values: &[S],
    ) -> Result<Self, Diagnostic> {
        if parameter_fields.len() != parameter_values.len() {
            return Err(input_mismatch(format!(
                "Parameter point contains {} identities and {} values",
                parameter_fields.len(),
                parameter_values.len()
            )));
        }
        if parameter_fields
            .iter()
            .enumerate()
            .any(|(index, field)| parameter_fields[..index].contains(field))
        {
            return Err(input_mismatch("Parameter point identities must be unique"));
        }
        if parameter_values.iter().any(|value| !value.is_finite()) {
            return Err(nonfinite("Parameter point contains a non-finite value"));
        }

        let bound_values = self
            .parameter_fields
            .iter()
            .map(|field| {
                parameter_fields
                    .iter()
                    .position(|candidate| candidate == field)
                    .map(|index| parameter_values[index])
                    .ok_or_else(|| {
                        input_mismatch(
                            "Parameter point is missing an identity required by the spatial expression",
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut bound = self.clone();
        bound.parameter_values = bound_values;
        Ok(bound)
    }
}

impl ScalarSpatialExpression<f64> {
    /// Evaluate the primal value and one Parameter JVP in a single tape pass.
    ///
    /// `parameter_tangent` uses [`Self::parameter_fields`] order. Coordinates
    /// are held fixed: spatial differentiation is a separate operator action.
    ///
    /// # Errors
    /// Returns `EQ0702` for a coordinate/tangent shape mismatch and `EQ0505`
    /// for non-finite input or an invalid/non-finite intermediate value.
    pub fn evaluate_parameter_jvp(
        &self,
        coordinates: &[f64],
        parameter_tangent: &[f64],
    ) -> Result<(f64, f64), Diagnostic> {
        self.evaluate_jvp(
            coordinates,
            &vec![0.0; self.coordinate_dimension],
            parameter_tangent,
        )
    }

    /// Evaluate the primal value and a combined coordinate/Parameter JVP.
    ///
    /// Coordinate tangents are physical mesh-motion velocities. Parameter
    /// tangents retain [`Self::parameter_fields`] order. Supplying both forms
    /// one total derivative of the same immutable expression tape.
    ///
    /// # Errors
    /// Returns `EQ0702` for a coordinate/tangent shape mismatch and `EQ0505`
    /// for non-finite input or an invalid/non-finite intermediate value.
    pub fn evaluate_jvp(
        &self,
        coordinates: &[f64],
        coordinate_tangent: &[f64],
        parameter_tangent: &[f64],
    ) -> Result<(f64, f64), Diagnostic> {
        self.evaluate_tangent(coordinates, coordinate_tangent, parameter_tangent)
    }

    /// Evaluate the primal value and one Parameter VJP in a reverse tape pass.
    ///
    /// The returned cotangent uses [`Self::parameter_fields`] order.
    /// Coordinates are held fixed: use [`Self::evaluate_vjp`] when coordinate
    /// cotangents are also required.
    ///
    /// # Errors
    /// Returns `EQ0702` for coordinate shape mismatch and `EQ0505` for a
    /// non-finite coordinate, cotangent, or intermediate value.
    pub fn evaluate_parameter_vjp(
        &self,
        coordinates: &[f64],
        output_cotangent: f64,
    ) -> Result<(f64, Vec<f64>), Diagnostic> {
        let (primal, _, parameter_cotangent) = self.evaluate_vjp(coordinates, output_cotangent)?;
        Ok((primal, parameter_cotangent))
    }

    /// Evaluate the primal value and coordinate/Parameter VJPs.
    ///
    /// The first returned cotangent uses physical coordinate-axis order; the
    /// second uses [`Self::parameter_fields`] order. Both are pullbacks of the
    /// same immutable expression tape at its bound Parameter values.
    ///
    /// # Errors
    /// Returns `EQ0702` for coordinate shape mismatch and `EQ0505` for a
    /// non-finite coordinate, cotangent, or intermediate value.
    pub fn evaluate_vjp(
        &self,
        coordinates: &[f64],
        output_cotangent: f64,
    ) -> Result<(f64, Vec<f64>, Vec<f64>), Diagnostic> {
        self.validate_coordinates(coordinates)?;
        if !output_cotangent.is_finite() {
            return Err(nonfinite(
                "spatial expression output cotangent is non-finite",
            ));
        }

        let values = self.primal_values(coordinates)?;
        let mut adjoints = vec![0.0; self.instructions.len()];
        adjoints[self.root] = output_cotangent;
        let mut coordinate_cotangent = vec![0.0; self.coordinate_dimension];
        let mut parameter_cotangent = vec![0.0; self.parameter_fields.len()];

        for index in (0..self.instructions.len()).rev() {
            let adjoint = adjoints[index];
            match self.instructions[index] {
                Instruction::Constant(_) => {}
                Instruction::Parameter(parameter) => {
                    parameter_cotangent[parameter] += adjoint;
                }
                Instruction::Coordinate(axis) => {
                    coordinate_cotangent[axis] += adjoint;
                }
                Instruction::Neg(value) => adjoints[value] -= adjoint,
                Instruction::Conjugate(value) => adjoints[value] += adjoint,
                Instruction::Add(left, right) => {
                    adjoints[left] += adjoint;
                    adjoints[right] += adjoint;
                }
                Instruction::Sub(left, right) => {
                    adjoints[left] += adjoint;
                    adjoints[right] -= adjoint;
                }
                Instruction::Mul(left, right) => {
                    adjoints[left] += adjoint * values[right];
                    adjoints[right] += adjoint * values[left];
                }
                Instruction::Div(left, right) => {
                    adjoints[left] += adjoint / values[right];
                    adjoints[right] -= adjoint * values[left] / values[right].powi(2);
                }
                Instruction::PowI(base, exponent) if exponent != 0 => {
                    adjoints[base] += adjoint
                        * f64::from(exponent)
                        * tangent::preceding_power(values[base], exponent);
                }
                Instruction::PowI(_, _) => {}
                Instruction::Sin(value) => {
                    adjoints[value] += adjoint * f64::cos(values[value]);
                }
                Instruction::Sqrt(value) => {
                    if values[index] == 0.0 {
                        return Err(nonfinite("square-root derivative is undefined at zero"));
                    }
                    adjoints[value] += adjoint / (2.0 * values[index]);
                }
            }
        }

        if adjoints
            .iter()
            .chain(&coordinate_cotangent)
            .chain(&parameter_cotangent)
            .any(|value| !value.is_finite())
        {
            return Err(nonfinite(
                "scalar spatial expression produced a non-finite cotangent",
            ));
        }

        Ok((values[self.root], coordinate_cotangent, parameter_cotangent))
    }
}

impl<S: Scalar + ComplexFloat<Real = f64> + From<f64>> ScalarSpatialExpression<S> {
    fn validate_coordinates(&self, coordinates: &[f64]) -> Result<(), Diagnostic> {
        if coordinates.len() != self.coordinate_dimension {
            return Err(input_mismatch(format!(
                "spatial expression expects {} coordinates, received {}",
                self.coordinate_dimension,
                coordinates.len()
            )));
        }
        if coordinates.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(nonfinite("spatial coordinate is non-finite"));
        }
        Ok(())
    }
}

impl<S: Scalar + ComplexFloat<Real = f64> + From<f64>> ScalarSpatialExpression<S> {
    /// Whether evaluation depends on the physical coordinate.
    #[must_use]
    pub const fn is_coordinate_dependent(&self) -> bool {
        self.coordinate_dependent
    }

    /// Evaluate a spatially constant tape once.
    #[must_use]
    pub fn constant_value(&self) -> Option<S> {
        (!self.coordinate_dependent)
            .then(|| self.evaluate(&vec![0.0; self.coordinate_dimension]).ok())
            .flatten()
    }

    /// Whether two coefficients retain the same exact lowered relation.
    ///
    /// Equality includes the expression tape, canonical Parameter identities,
    /// and their revision-local values. It intentionally does not infer
    /// equality from coincident numerical values; algebraic equivalence is a
    /// separate compiler concern.
    #[must_use]
    pub(crate) fn is_same_coefficient_as(&self, other: &Self) -> bool {
        self == other
    }
}

impl ScalarSpatialExpression<f64> {
    /// Exact affine coordinate gradient when the lowered tape is affine.
    ///
    /// Parameter values are already point-bound constants. Returning
    /// `None` is a structural rejection, not a sample-based linearity guess.
    pub(crate) fn affine_gradient(&self) -> Option<Vec<f64>> {
        let zero = || vec![0.0; self.coordinate_dimension];
        let mut forms: Vec<(f64, Vec<f64>)> = Vec::with_capacity(self.instructions.len());
        for instruction in &self.instructions {
            let form = match *instruction {
                Instruction::Constant(value) => (value, zero()),
                Instruction::Parameter(parameter) => (self.parameter_values[parameter], zero()),
                Instruction::Coordinate(axis) => {
                    let mut gradient = zero();
                    gradient[axis] = 1.0;
                    (0.0, gradient)
                }
                Instruction::Neg(value) => scale_affine(&forms[value], -1.0),
                Instruction::Conjugate(value) => forms[value].clone(),
                Instruction::Add(left, right) => add_affine(&forms[left], &forms[right], 1.0),
                Instruction::Sub(left, right) => add_affine(&forms[left], &forms[right], -1.0),
                Instruction::Mul(left, right) => {
                    let left_spatial = affine_is_spatial(&forms[left]);
                    let right_spatial = affine_is_spatial(&forms[right]);
                    match (left_spatial, right_spatial) {
                        (true, true) => return None,
                        (true, false) => scale_affine(&forms[left], forms[right].0),
                        (false, true) => scale_affine(&forms[right], forms[left].0),
                        (false, false) => (forms[left].0 * forms[right].0, zero()),
                    }
                }
                Instruction::Div(left, right)
                    if !affine_is_spatial(&forms[right]) && forms[right].0 != 0.0 =>
                {
                    scale_affine(&forms[left], forms[right].0.recip())
                }
                Instruction::PowI(_, 0) => (1.0, zero()),
                Instruction::PowI(base, 1) => forms[base].clone(),
                Instruction::PowI(base, exponent) if !affine_is_spatial(&forms[base]) => {
                    (forms[base].0.powi(exponent), zero())
                }
                Instruction::Sin(value) if !affine_is_spatial(&forms[value]) => {
                    (forms[value].0.sin(), zero())
                }
                Instruction::Sqrt(value) if !affine_is_spatial(&forms[value]) => {
                    (real_sqrt(forms[value].0).ok()?, zero())
                }
                _ => return None,
            };
            if !form.0.is_finite() || form.1.iter().any(|value| !value.is_finite()) {
                return None;
            }
            forms.push(form);
        }
        forms.get(self.root).map(|(_, gradient)| gradient.clone())
    }
}

impl<S: Scalar + ComplexFloat<Real = f64> + From<f64>> ScalarSpatialExpression<S> {
    pub(crate) fn constant(coordinate_dimension: usize, value: S) -> Self {
        Self {
            coordinate_dimension,
            instructions: vec![Instruction::Constant(value)],
            root: 0,
            coordinate_dependent: false,
            parameter_fields: Vec::new(),
            parameter_values: Vec::new(),
        }
    }

    pub(crate) fn multiply(self, right: Self) -> Self {
        assert_eq!(self.coordinate_dimension, right.coordinate_dimension);
        let mut parameter_fields = self.parameter_fields.clone();
        let mut parameter_values = self.parameter_values.clone();
        let right_parameter_remap = right
            .parameter_fields
            .iter()
            .zip(&right.parameter_values)
            .map(|(field, value)| {
                parameter_fields
                    .iter()
                    .position(|existing| existing == field)
                    .unwrap_or_else(|| {
                        let index = parameter_fields.len();
                        parameter_fields.push(*field);
                        parameter_values.push(*value);
                        index
                    })
            })
            .collect::<Vec<_>>();
        let right_offset = self.instructions.len();
        let mut instructions = self.instructions;
        instructions.extend(right.instructions.into_iter().map(|instruction| {
            remap_instruction(instruction, right_offset, &right_parameter_remap)
        }));
        let root = instructions.len();
        instructions.push(Instruction::Mul(self.root, right_offset + right.root));
        Self {
            coordinate_dimension: self.coordinate_dimension,
            instructions,
            root,
            coordinate_dependent: self.coordinate_dependent || right.coordinate_dependent,
            parameter_fields,
            parameter_values,
        }
    }
}

fn affine_is_spatial((_, gradient): &(f64, Vec<f64>)) -> bool {
    gradient.iter().any(|value| *value != 0.0)
}

fn scale_affine((constant, gradient): &(f64, Vec<f64>), scale: f64) -> (f64, Vec<f64>) {
    (
        scale * constant,
        gradient.iter().map(|value| scale * value).collect(),
    )
}

fn add_affine(
    (left_constant, left_gradient): &(f64, Vec<f64>),
    (right_constant, right_gradient): &(f64, Vec<f64>),
    right_scale: f64,
) -> (f64, Vec<f64>) {
    (
        left_constant + right_scale * right_constant,
        left_gradient
            .iter()
            .zip(right_gradient)
            .map(|(left, right)| left + right_scale * right)
            .collect(),
    )
}

fn remap_instruction<S: Scalar>(
    instruction: Instruction<S>,
    node_offset: usize,
    parameter_remap: &[usize],
) -> Instruction<S> {
    match instruction {
        Instruction::Constant(value) => Instruction::Constant(value),
        Instruction::Parameter(parameter) => Instruction::Parameter(parameter_remap[parameter]),
        Instruction::Coordinate(axis) => Instruction::Coordinate(axis),
        Instruction::Neg(value) => Instruction::Neg(node_offset + value),
        Instruction::Conjugate(value) => Instruction::Conjugate(node_offset + value),
        Instruction::Add(left, right) => Instruction::Add(node_offset + left, node_offset + right),
        Instruction::Sub(left, right) => Instruction::Sub(node_offset + left, node_offset + right),
        Instruction::Mul(left, right) => Instruction::Mul(node_offset + left, node_offset + right),
        Instruction::Div(left, right) => Instruction::Div(node_offset + left, node_offset + right),
        Instruction::PowI(base, exponent) => Instruction::PowI(node_offset + base, exponent),
        Instruction::Sin(value) => Instruction::Sin(node_offset + value),
        Instruction::Sqrt(value) => Instruction::Sqrt(node_offset + value),
    }
}

fn required_nodes(
    expression: &ExprDag,
    root: ExprId,
    owner: RawId,
) -> Result<Vec<bool>, Diagnostic> {
    let mut required = vec![false; expression.nodes().len()];
    let mut pending = vec![root];
    while let Some(value) = pending.pop() {
        let index = usize::try_from(value.index())
            .map_err(|_| invalid(owner, "source expression index exceeds usize"))?;
        let Some(slot) = required.get_mut(index) else {
            return Err(invalid(
                owner,
                "source expression references a missing node",
            ));
        };
        if *slot {
            continue;
        }
        *slot = true;
        let node = expression
            .node(value)
            .ok_or_else(|| invalid(owner, "source expression node is missing"))?;
        match node {
            ExprNode::Neg(value) | ExprNode::PowI(value, _) | ExprNode::UnaryMath(_, value) => {
                pending.push(*value);
            }
            ExprNode::Add(left, right)
            | ExprNode::Sub(left, right)
            | ExprNode::Mul(left, right)
            | ExprNode::Div(left, right)
            | ExprNode::Complex {
                real: left,
                imag: right,
            } => {
                pending.push(*left);
                pending.push(*right);
            }
            _ => {}
        }
    }
    Ok(required)
}

fn remapped(remap: &[Option<usize>], value: ExprId, owner: RawId) -> Result<usize, Diagnostic> {
    usize::try_from(value.index())
        .ok()
        .and_then(|index| remap.get(index).copied().flatten())
        .ok_or_else(|| invalid(owner, "source expression is not topologically ordered"))
}

fn invalid(owner: RawId, message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_SPATIAL_LOWERING, message).with_graph_path(GraphPath::new([
        owner.kind().graph().name().to_owned(),
        format!("{:?}", owner.kind()),
        owner.to_string(),
    ]))
}

fn real_sqrt(value: f64) -> Result<f64, Diagnostic> {
    if value < 0.0 {
        return Err(nonfinite(
            "real square root requires a nonnegative argument",
        ));
    }
    Ok(value.sqrt())
}

fn nonfinite(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NONFINITE_EVALUATION, message)
}

fn input_mismatch(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::OPERATOR_INPUT_MISMATCH, message)
}

#[cfg(test)]
mod tests;
