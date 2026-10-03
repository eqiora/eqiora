//! Spatial sampling delegates scalar arithmetic and differentiation to Operator IR.

use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, DimExponents, DynQuantity, RawId};
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
        typed,
        coordinates,
        normal,
        fields,
        builder: ExprDagBuilder::new(),
        samples: Vec::new(),
        memo: vec![None; typed.expression().nodes().len()],
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
    typed: &'a TypedResidual<RawId>,
    coordinates: &'a [f64],
    normal: Option<(usize, f64)>,
    fields: &'a BTreeMap<RawId, PointField>,
    builder: ExprDagBuilder,
    samples: Vec<(ExprId, f64)>,
    memo: Vec<Option<ExprId>>,
}

impl Projection<'_> {
    fn scalar(&mut self, id: ExprId, depth: usize) -> Result<ExprId, Diagnostic> {
        if depth > 256 {
            return Err(invalid(
                "Observable spatial expression exceeds the bounded projection depth",
            ));
        }
        if let Some(value) = self.memo[id.index() as usize] {
            return Ok(value);
        }
        let node = self
            .typed
            .expression()
            .node(id)
            .expect("typed expression node exists");
        let value = match node {
            ExprNode::Constant(value) => self.builder.constant(value.clone())?,
            ExprNode::Symbol(SymbolRef::Parameter(id)) => {
                self.builder.symbol(SymbolRef::Parameter(*id))?
            }
            ExprNode::Symbol(SymbolRef::Field(field)) => {
                let sample = self
                    .fields
                    .get(&field.erase())
                    .ok_or_else(|| invalid("Observable Field sample is outside this Result"))?;
                let value = self.builder.constant(sample.value.clone())?;
                self.samples.push((value, sample.tangent));
                value
            }
            ExprNode::SpatialCoordinate(axis) => {
                let coordinate = self
                    .coordinates
                    .get(*axis)
                    .ok_or_else(|| invalid("Observable coordinate axis is unavailable"))?;
                self.builder.constant(DynQuantity::new(
                    *coordinate,
                    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("length dimension"),
                ))?
            }
            ExprNode::PureOperatorApplication(application) => {
                let definition = self
                    .typed
                    .expression()
                    .definition(application.definition())
                    .ok_or_else(|| invalid("Observable pure operator definition is unavailable"))?;
                let types = application
                    .arguments()
                    .iter()
                    .map(|argument| {
                        self.typed.node_type(*argument).cloned().ok_or_else(|| {
                            invalid("Observable pure operator argument type is unavailable")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let instance = definition
                    .instantiate(&types)
                    .map_err(|error| invalid(error.to_string()))?;
                let arguments = application
                    .arguments()
                    .iter()
                    .map(|argument| self.scalar(*argument, depth + 1))
                    .collect::<Result<Vec<_>, _>>()?;
                self.builder
                    .project_scalar_operator(&instance, &arguments, 1_000_000)?
            }
            ExprNode::Trace(value) => self.scalar(*value, depth + 1)?,
            ExprNode::NormalComponent(value) => self.normal_component(*value, depth + 1)?,
            ExprNode::Neg(value) => {
                let value = self.scalar(*value, depth + 1)?;
                self.builder.neg(value)?
            }
            ExprNode::Add(a, b)
            | ExprNode::Sub(a, b)
            | ExprNode::Mul(a, b)
            | ExprNode::Div(a, b) => {
                let (a, b) = (self.scalar(*a, depth + 1)?, self.scalar(*b, depth + 1)?);
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
                    "Observable spatial expression dependence is outside the admitted scalar projection",
                ));
            }
        };
        self.memo[id.index() as usize] = Some(value);
        Ok(value)
    }

    fn normal_component(&mut self, id: ExprId, depth: usize) -> Result<ExprId, Diagnostic> {
        if depth > 256 {
            return Err(invalid(
                "Observable normal expression exceeds the bounded projection depth",
            ));
        }
        let (axis, sign) = self
            .normal
            .ok_or_else(|| invalid("normal Observable requires an oriented boundary"))?;
        let node = self
            .typed
            .expression()
            .node(id)
            .expect("typed expression node exists");
        match node {
            ExprNode::Gradient(field) => {
                let ExprNode::Symbol(SymbolRef::Field(field)) = self
                    .typed
                    .expression()
                    .node(*field)
                    .expect("typed gradient operand exists")
                else {
                    return Err(invalid(
                        "Observable normal gradient currently requires an admitted scalar Field operand",
                    ));
                };
                let sample = self
                    .fields
                    .get(&field.erase())
                    .ok_or_else(|| invalid("Observable gradient Field is outside this Result"))?;
                let dimension = self
                    .typed
                    .node_type(id)
                    .expect("typed gradient exists")
                    .dimension();
                let value = self
                    .builder
                    .constant(DynQuantity::new(sign * sample.gradient[axis], dimension))?;
                self.samples
                    .push((value, sign * sample.gradient_tangent[axis]));
                Ok(value)
            }
            ExprNode::Constant(value) => {
                if value.component_count() != self.coordinates.len() {
                    return Err(invalid(
                        "Observable constant flux requires one Cartesian vector",
                    ));
                }
                let (component, imaginary) = value
                    .component(axis)
                    .ok_or_else(|| invalid("Observable constant flux axis is unavailable"))?;
                if imaginary != 0.0 {
                    return Err(invalid("Observable spatial profile requires real flux"));
                }
                self.builder.constant(DynQuantity::new(
                    sign * component,
                    value.value_type().dimension(),
                ))
            }
            ExprNode::Neg(value) => {
                let value = self.normal_component(*value, depth + 1)?;
                self.builder.neg(value)
            }
            ExprNode::Mul(a, b) => {
                let a_scalar = self
                    .typed
                    .node_type(*a)
                    .expect("typed operand")
                    .value_type
                    .shape()
                    .component_count()
                    == Some(1);
                let (a, b) = if a_scalar {
                    (
                        self.scalar(*a, depth + 1)?,
                        self.normal_component(*b, depth + 1)?,
                    )
                } else {
                    (
                        self.normal_component(*a, depth + 1)?,
                        self.scalar(*b, depth + 1)?,
                    )
                };
                self.builder.mul(a, b)
            }
            ExprNode::Add(a, b) | ExprNode::Sub(a, b) => {
                let (a, b) = (
                    self.normal_component(*a, depth + 1)?,
                    self.normal_component(*b, depth + 1)?,
                );
                if matches!(node, ExprNode::Add(..)) {
                    self.builder.add(a, b)
                } else {
                    self.builder.sub(a, b)
                }
            }
            _ => Err(invalid(
                "Observable normal projection is outside the admitted constant-vector/scalar-gradient profile",
            )),
        }
    }
}
