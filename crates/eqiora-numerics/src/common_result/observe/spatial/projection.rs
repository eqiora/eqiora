//! Spatial sampling delegates scalar arithmetic and differentiation to Operator IR.

use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, DynQuantity, RawId};
use eqiora_ir::{LinearizedRelation, RelationTangent, ScalarOperatorIr};
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{ExprDagBuilder, ExprId, ExprNode, ObservableDef, SymbolRef};
use eqiora_sem::KernelProgram;

use super::PointField;
use super::invalid;

pub(super) fn evaluate(
    program: &KernelProgram,
    observable: &ObservableDef,
    typed: &TypedResidual<RawId>,
    coordinates: &[f64],
    normal: Option<(usize, f64)>,
    fields: &BTreeMap<RawId, PointField>,
    derivative: bool,
) -> Result<f64, Diagnostic> {
    let mut projection = Projection {
        program,
        typed,
        coordinates,
        normal,
        fields,
        builder: ExprDagBuilder::new(),
        samples: Vec::new(),
        memo: BTreeMap::new(),
        remaining: 1_000_000,
    };
    let root = projection.scalar(observable.expression().roots()[0], 0)?;
    let expression = projection.builder.finish([root])?;
    let operator = ScalarOperatorIr::lower(&expression)?;
    let values = operator
        .symbols()
        .iter()
        .map(|symbol| match symbol {
            SymbolRef::Parameter(id) => program
                .value(id.erase())
                .map(|value| value.value())
                .ok_or_else(|| invalid("Observable Parameter is unavailable")),
            _ => Err(invalid("spatial Observable contains an unadmitted symbol")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if derivative {
        let ids = projection
            .samples
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        let tangents = projection
            .samples
            .iter()
            .map(|(_, tangent)| *tangent)
            .collect::<Vec<_>>();
        let linearization = operator.linearize_samples(&values, &ids)?;
        let mut action = [0.0];
        linearization.jvp(RelationTangent::Unknown(&tangents), &mut action)?;
        Ok(action[0])
    } else {
        let values = operator.evaluate(&values)?;
        values
            .first()
            .copied()
            .ok_or_else(|| invalid("Observable point evaluation has no root"))
    }
}

struct Projection<'a> {
    program: &'a KernelProgram,
    typed: &'a TypedResidual<RawId>,
    coordinates: &'a [f64],
    normal: Option<(usize, f64)>,
    fields: &'a BTreeMap<RawId, PointField>,
    builder: ExprDagBuilder,
    samples: Vec<(ExprId, f64)>,
    memo: BTreeMap<(ExprId, Vec<u32>), ExprId>,
    remaining: usize,
}

impl Projection<'_> {
    fn scalar(&mut self, id: ExprId, depth: usize) -> Result<ExprId, Diagnostic> {
        self.component(id, &[], depth)
    }

    fn component(
        &mut self,
        id: ExprId,
        coordinate: &[u32],
        depth: usize,
    ) -> Result<ExprId, Diagnostic> {
        if depth > 256 || self.remaining == 0 {
            return Err(invalid(
                "Observable spatial component projection exceeds its work bound",
            ));
        }
        let key = (id, coordinate.to_vec());
        if let Some(value) = self.memo.get(&key) {
            return Ok(*value);
        }
        self.remaining -= 1;
        let ty = self.typed.node_type(id).expect("typed node exists");
        if ty.value_type.array_rank() != 0
            || coordinate.len() != ty.shape().rank()
            || coordinate
                .iter()
                .zip(ty.shape().extents())
                .any(|(index, extent)| *index >= extent.get())
        {
            return Err(invalid(
                "Observable component coordinates differ from the exact value type",
            ));
        }
        let node = self.typed.expression().node(id).expect("typed node exists");
        let value = match node {
            ExprNode::Constant(value) => self.constant_component(value, coordinate)?,
            ExprNode::Symbol(SymbolRef::Parameter(parameter)) => {
                if coordinate.is_empty() {
                    self.builder.symbol(SymbolRef::Parameter(*parameter))?
                } else {
                    // State tangents hold Parameters fixed at this exact Model value.
                    let value = self
                        .program
                        .typed_value(parameter.erase())
                        .ok_or_else(|| invalid("Observable Parameter value is unavailable"))?;
                    self.constant_component(value, coordinate)?
                }
            }
            ExprNode::Symbol(SymbolRef::Field(field)) => {
                let sample = self
                    .fields
                    .get(&field.erase())
                    .ok_or_else(|| invalid("Observable Field sample is outside this Result"))?;
                let component =
                    flat_component(sample.value.value_type().shape().extents(), coordinate);
                let real = sample
                    .value
                    .component(component)
                    .ok_or_else(|| invalid("Observable Field component is unavailable"))?
                    .0;
                let value = self
                    .builder
                    .constant(DynQuantity::new(real, ty.dimension()))?;
                self.samples.push((value, sample.tangent[0][component]));
                value
            }
            ExprNode::CoordinatePartial { value, wrt } => {
                let Some(ExprNode::Symbol(SymbolRef::Field(field))) =
                    self.typed.expression().node(*value)
                else {
                    return Err(invalid(
                        "coordinate partial sampling requires an admitted scalar Field",
                    ));
                };
                let Some(ExprNode::Symbol(SymbolRef::Coordinate {
                    support,
                    factor,
                    axis,
                })) = self.typed.expression().node(*wrt)
                else {
                    return Err(invalid("coordinate partial selector is unavailable"));
                };
                if !coordinate.is_empty()
                    || !crate::spatial_expression::physical_coordinate(
                        self.program,
                        *support,
                        *factor,
                    )
                {
                    return Err(invalid(
                        "coordinate partial is outside this physical scalar Field realization",
                    ));
                }
                let sample = self
                    .fields
                    .get(&field.erase())
                    .ok_or_else(|| invalid("coordinate partial Field is outside this Result"))?;
                let gradient = sample
                    .gradient
                    .get(*axis)
                    .ok_or_else(|| invalid("coordinate partial axis is unavailable"))?;
                let value = self
                    .builder
                    .constant(DynQuantity::new(*gradient, ty.dimension()))?;
                self.samples
                    .push((value, sample.gradient_tangent[0][*axis]));
                value
            }
            ExprNode::Gradient(field) => {
                let ExprNode::Symbol(SymbolRef::Field(field)) =
                    self.typed.expression().node(*field).expect("typed operand")
                else {
                    return Err(invalid(
                        "Observable gradient requires an admitted Field operand",
                    ));
                };
                let sample = self
                    .fields
                    .get(&field.erase())
                    .ok_or_else(|| invalid("Observable gradient Field is outside this Result"))?;
                let axis = flat_component(ty.shape().extents(), coordinate);
                let value = self
                    .builder
                    .constant(DynQuantity::new(sample.gradient[axis], ty.dimension()))?;
                self.samples.push((value, sample.gradient_tangent[0][axis]));
                value
            }
            ExprNode::Divergence(value) => {
                let Some(ExprNode::Symbol(SymbolRef::Field(field))) =
                    self.typed.expression().node(*value)
                else {
                    return Err(invalid(
                        "Observable divergence requires an admitted vector Field",
                    ));
                };
                let sample = self
                    .fields
                    .get(&field.erase())
                    .ok_or_else(|| invalid("Observable divergence Field is outside this Result"))?;
                let [extent] = sample.value.value_type().shape().extents() else {
                    return Err(invalid(
                        "Observable first-gradient divergence requires a vector Field",
                    ));
                };
                let dimension = self.coordinates.len();
                if extent.get() as usize != dimension || !coordinate.is_empty() {
                    return Err(invalid(
                        "Observable divergence differs from its spatial component extent",
                    ));
                }
                let mut divergence = 0.0;
                let mut tangent = 0.0;
                for axis in 0..dimension {
                    divergence += sample.gradient[axis * dimension + axis];
                    tangent += sample.gradient_tangent[0][axis * dimension + axis];
                }
                let value = self
                    .builder
                    .constant(DynQuantity::new(divergence, ty.dimension()))?;
                self.samples.push((value, tangent));
                value
            }
            ExprNode::Symbol(SymbolRef::Coordinate {
                support,
                factor,
                axis,
            }) => {
                if !crate::spatial_expression::physical_coordinate(self.program, *support, *factor)
                {
                    return Err(invalid(
                        "Observable coordinate is not an ambient physical coordinate",
                    ));
                }
                let value = self
                    .coordinates
                    .get(*axis)
                    .ok_or_else(|| invalid("Observable coordinate axis is unavailable"))?;
                self.builder
                    .constant(DynQuantity::new(*value, ty.dimension()))?
            }
            ExprNode::PureOperatorApplication(application) => {
                let definition = self
                    .typed
                    .expression()
                    .definition(application.definition())
                    .ok_or_else(|| invalid("Observable pure operator definition is unavailable"))?;
                self.pure(definition, application.arguments(), coordinate, depth)?
            }
            ExprNode::SymmetricPart(value) => self.pure(
                &eqiora_schema::kernel::pure_operator::PureOperatorDefinition::symmetric_part()
                    .map_err(|error| invalid(error.to_string()))?,
                &[*value],
                coordinate,
                depth,
            )?,
            ExprNode::Trace { value, .. } => self.component(*value, coordinate, depth + 1)?,
            ExprNode::NormalComponent { value, .. } => {
                let (axis, sign) = self
                    .normal
                    .ok_or_else(|| invalid("normal Observable requires an oriented boundary"))?;
                let mut indices = coordinate.to_vec();
                indices.push(axis as u32);
                let value = self.component(*value, &indices, depth + 1)?;
                if sign < 0.0 {
                    self.builder.neg(value)?
                } else {
                    value
                }
            }
            ExprNode::Neg(value) => {
                let value = self.component(*value, coordinate, depth + 1)?;
                self.builder.neg(value)?
            }
            ExprNode::Add(a, b)
            | ExprNode::Sub(a, b)
            | ExprNode::Mul(a, b)
            | ExprNode::Div(a, b) => {
                let operand_coordinate = |id| {
                    if self
                        .typed
                        .node_type(id)
                        .expect("typed operand")
                        .shape()
                        .is_scalar()
                    {
                        &[][..]
                    } else {
                        coordinate
                    }
                };
                let ac = operand_coordinate(*a);
                let bc = operand_coordinate(*b);
                let a = self.component(*a, ac, depth + 1)?;
                let b = self.component(*b, bc, depth + 1)?;
                match node {
                    ExprNode::Add(..) => self.builder.add(a, b)?,
                    ExprNode::Sub(..) => self.builder.sub(a, b)?,
                    ExprNode::Mul(..) => self.builder.mul(a, b)?,
                    _ => self.builder.div(a, b)?,
                }
            }
            ExprNode::PowI(value, exponent) => {
                let value = self.scalar(*value, depth + 1)?;
                self.builder.powi(value, *exponent)?
            }
            ExprNode::UnaryMath(function, value) => {
                let value = self.scalar(*value, depth + 1)?;
                self.builder.unary_math(*function, value)?
            }
            _ => {
                return Err(invalid(
                    "Observable spatial expression dependence is outside the admitted component projection",
                ));
            }
        };
        self.memo.insert(key, value);
        Ok(value)
    }

    fn pure(
        &mut self,
        definition: &eqiora_schema::kernel::pure_operator::PureOperatorDefinition,
        operands: &[ExprId],
        coordinate: &[u32],
        depth: usize,
    ) -> Result<ExprId, Diagnostic> {
        let types = operands
            .iter()
            .map(|argument| {
                self.typed
                    .node_type(*argument)
                    .cloned()
                    .ok_or_else(|| invalid("Observable pure operator argument type is unavailable"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let instance = definition
            .instantiate(&types)
            .map_err(|error| invalid(error.to_string()))?;
        let mut arguments = Vec::new();
        for (argument, ty) in operands.iter().zip(&types) {
            let count = ty
                .shape()
                .component_count()
                .filter(|count| *count <= self.remaining)
                .ok_or_else(|| invalid("Observable component arguments exceed the work bound"))?;
            let mut values = Vec::with_capacity(count);
            for flat in 0..count {
                let mut rest = flat;
                let mut indices = vec![0; ty.shape().rank()];
                for (index, extent) in indices.iter_mut().zip(ty.shape().extents()).rev() {
                    *index = (rest % extent.get() as usize) as u32;
                    rest /= extent.get() as usize;
                }
                values.push(self.component(*argument, &indices, depth + 1)?);
            }
            arguments.push(values);
        }
        self.remaining = self
            .remaining
            .checked_sub(definition.nodes().len())
            .ok_or_else(|| invalid("Observable pure expansion exceeds the work bound"))?;
        self.builder
            .project_operator_component(&instance, &arguments, coordinate, 1_000_000)
    }

    fn constant_component(
        &mut self,
        value: &eqiora_core::ValueLiteral,
        coordinate: &[u32],
    ) -> Result<ExprId, Diagnostic> {
        if coordinate.is_empty() {
            return self.builder.constant(value.clone());
        }
        let flat = value
            .value_type()
            .shape()
            .extents()
            .iter()
            .zip(coordinate)
            .fold(0usize, |offset, (extent, index)| {
                offset * extent.get() as usize + *index as usize
            });
        let (real, imaginary) = value
            .component(flat)
            .ok_or_else(|| invalid("Observable constant component is unavailable"))?;
        if imaginary != 0.0 {
            return Err(invalid(
                "Observable spatial profile requires real components",
            ));
        }
        self.builder
            .constant(DynQuantity::new(real, value.value_type().dimension()))
    }
}

fn flat_component(extents: &[std::num::NonZeroU32], coordinate: &[u32]) -> usize {
    extents
        .iter()
        .zip(coordinate)
        .fold(0, |offset, (extent, index)| {
            offset * extent.get() as usize + *index as usize
        })
}
