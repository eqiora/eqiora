//! Bind field jets to the existing exact local calculus; no spatial discretization.
use std::collections::HashMap;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id, RawId, ScalarDomain, ValueShape};
use eqiora_schema::kernel::pure_operator::{
    CalculusBuilder, CalculusNode, CalculusNodeId, ExactRational, MAX_FORMALS,
    PureOperatorDefinition, PureOperatorError, PureValueClass,
};
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{ExprId, ExprNode, SymbolRef};
use num_rational::BigRational;
use num_traits::ToPrimitive;

/// Inputs retain actual Model identities, never synthetic Parameters for jets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Input {
    Value(SymbolRef, Vec<u32>),
    Gradient(Id<kinds::Field>, Vec<u32>),
    Direction { input: usize, order: u8 },
}

pub(super) struct LocalVariation {
    pub definition: PureOperatorDefinition,
    pub inputs: Vec<(Input, DimExponents)>,
}

fn reject(message: impl std::fmt::Display) -> Diagnostic {
    Diagnostic::error(
        eqiora_core::diagnostic::codes::LANGUAGE_TYPE_ERROR,
        format!("functional variation local density: {message}"),
    )
}

fn class(dimension: DimExponents) -> Result<PureValueClass, Diagnostic> {
    PureValueClass::invariant_scalar()
        .with_dimension(dimension)
        .with_scalar_domain(ScalarDomain::Real)
        .map_err(reject)
}

fn coordinates(shape: &ValueShape) -> Result<Vec<Vec<u32>>, Diagnostic> {
    let count = shape
        .component_count()
        .filter(|n| *n <= MAX_FORMALS)
        .ok_or_else(|| reject("component inventory exceeds the local input bound"))?;
    Ok((0..count)
        .map(|mut flat| {
            let mut indices = vec![0; shape.rank()];
            for (index, extent) in indices.iter_mut().zip(shape.extents()).rev() {
                *index = (flat % extent.get() as usize) as u32;
                flat /= extent.get() as usize;
            }
            indices
        })
        .collect())
}

fn add_input(
    inputs: &mut Vec<(Input, DimExponents)>,
    input: Input,
    dimension: DimExponents,
) -> Result<(), Diagnostic> {
    if let Some((_, actual)) = inputs.iter().find(|(candidate, _)| candidate == &input) {
        if *actual != dimension {
            return Err(reject("one input has inconsistent dimensions"));
        }
        return Ok(());
    }
    if inputs.len() >= MAX_FORMALS {
        return Err(reject("local input bound exceeded"));
    }
    inputs.push((input, dimension));
    Ok(())
}

fn vector_field(
    typed: &TypedResidual<RawId>,
    value: ExprId,
) -> Result<(Id<kinds::Field>, u32), Diagnostic> {
    let Some(ExprNode::Symbol(SymbolRef::Field(field))) = typed.expression().node(value) else {
        return Err(reject("divergence must belong to an exact vector Field"));
    };
    let shape = typed
        .node_type(value)
        .ok_or_else(|| reject("untyped divergence operand"))?
        .shape();
    let [extent] = shape.extents() else {
        return Err(reject("first-gradient divergence requires a vector Field"));
    };
    Ok((*field, extent.get()))
}

/// The caller owns the fixed measure, held-variable inventory and admissible directions.
/// This transform preserves the density dimension for each ordered directional product.
pub(super) fn derive(
    typed: &TypedResidual<RawId>,
    wrt: Id<kinds::Field>,
    order: u8,
) -> Result<LocalVariation, Diagnostic> {
    if !matches!(order, 1 | 2) {
        return Err(reject("only first and second products are admitted"));
    }
    let mut inputs = Vec::new();
    for (index, node) in typed.expression().nodes().iter().enumerate() {
        let id = typed
            .expression()
            .node_id(index as u32)
            .expect("existing node");
        let ty = typed
            .node_type(id)
            .ok_or_else(|| reject("untyped density node"))?;
        if ty.value_type.scalar_domain() != ScalarDomain::Real || ty.value_type.array_rank() != 0 {
            return Err(reject("density requires real full-coordinate values"));
        }
        match node {
            ExprNode::Symbol(
                symbol @ (SymbolRef::Field(_)
                | SymbolRef::Parameter(_)
                | SymbolRef::Coordinate { .. }),
            ) => {
                for coordinate in coordinates(ty.shape())? {
                    add_input(
                        &mut inputs,
                        Input::Value(*symbol, coordinate),
                        ty.dimension(),
                    )?;
                }
            }
            ExprNode::Gradient(value) => {
                let Some(ExprNode::Symbol(SymbolRef::Field(field))) =
                    typed.expression().node(*value)
                else {
                    return Err(reject("first gradient must belong to an exact Field"));
                };
                for coordinate in coordinates(ty.shape())? {
                    add_input(
                        &mut inputs,
                        Input::Gradient(*field, coordinate),
                        ty.dimension(),
                    )?;
                }
            }
            ExprNode::Divergence(value) => {
                let (field, extent) = vector_field(typed, *value)?;
                for axis in 0..extent {
                    add_input(
                        &mut inputs,
                        Input::Gradient(field, vec![axis, axis]),
                        ty.dimension(),
                    )?;
                }
            }
            _ => {}
        }
    }
    let selected = inputs.iter().enumerate().filter_map(|(index, (input, _))| {
        matches!(input, Input::Value(SymbolRef::Field(field), _) | Input::Gradient(field, _) if *field == wrt)
            .then_some(index)
    }).collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(reject("selected Field has no admitted local jet"));
    }
    for level in 1..=order {
        for index in &selected {
            let dimension = inputs[*index].1;
            add_input(
                &mut inputs,
                Input::Direction {
                    input: *index,
                    order: level,
                },
                dimension,
            )?;
        }
    }
    let [source] = typed.expression().roots() else {
        return Err(reject("energy density must have one root"));
    };
    let ty = typed
        .node_type(*source)
        .ok_or_else(|| reject("untyped density root"))?;
    if !ty.shape().is_scalar() {
        return Err(reject("energy density must be scalar"));
    }
    let builder = CalculusBuilder::new(
        inputs
            .iter()
            .map(|(_, dimension)| class(*dimension))
            .collect::<Result<Vec<_>, _>>()?,
        class(ty.dimension())?,
    )
    .map_err(reject)?;
    let mut projection = Projection {
        typed,
        inputs: &inputs,
        builder,
        memo: HashMap::new(),
        remaining: 65536,
    };
    let mut root = projection.component(*source, &[], 0)?;
    for level in 1..=order {
        let directions = selected
            .iter()
            .map(|index| {
                let direction = projection.input(&Input::Direction {
                    input: *index,
                    order: level,
                })?;
                Ok((*index as u16, direction))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        root = projection.builder.jvp(root, &directions).map_err(reject)?;
    }
    let definition = projection.builder.finish(root).map_err(reject)?;
    Ok(LocalVariation { definition, inputs })
}

struct Projection<'a> {
    typed: &'a TypedResidual<RawId>,
    inputs: &'a [(Input, DimExponents)],
    builder: CalculusBuilder,
    memo: HashMap<(ExprId, Vec<u32>), CalculusNodeId>,
    remaining: usize,
}

impl Projection<'_> {
    fn push(&mut self, node: CalculusNode) -> Result<CalculusNodeId, Diagnostic> {
        self.builder.push(node).map_err(reject)
    }

    fn input(&mut self, input: &Input) -> Result<CalculusNodeId, Diagnostic> {
        let formal = self
            .inputs
            .iter()
            .position(|(candidate, _)| candidate == input)
            .ok_or_else(|| reject("missing exact input binding"))?;
        self.push(CalculusNode::FormalComponent {
            formal: formal as u16,
            axes: Box::new([]),
        })
    }

    fn component(
        &mut self,
        id: ExprId,
        coordinate: &[u32],
        depth: usize,
    ) -> Result<CalculusNodeId, Diagnostic> {
        if depth > 128 || self.remaining == 0 {
            return Err(reject("density expansion bound exceeded"));
        }
        let key = (id, coordinate.to_vec());
        if let Some(value) = self.memo.get(&key) {
            return Ok(*value);
        }
        self.remaining -= 1;
        let ty = self
            .typed
            .node_type(id)
            .ok_or_else(|| reject("missing component type"))?;
        if coordinate.len() != ty.shape().rank()
            || coordinate
                .iter()
                .zip(ty.shape().extents())
                .any(|(i, n)| *i >= n.get())
        {
            return Err(reject("component coordinate differs from its exact shape"));
        }
        let node = self
            .typed
            .expression()
            .node(id)
            .ok_or_else(|| reject("missing expression"))?;
        let value = match node {
            ExprNode::Symbol(symbol) => self.input(&Input::Value(*symbol, coordinate.to_vec()))?,
            ExprNode::Gradient(value) => {
                let Some(ExprNode::Symbol(SymbolRef::Field(field))) =
                    self.typed.expression().node(*value)
                else {
                    return Err(reject("gradient does not bind a Field"));
                };
                self.input(&Input::Gradient(*field, coordinate.to_vec()))?
            }
            ExprNode::Trace { value, .. } => {
                if !matches!(
                    self.typed.expression().node(*value),
                    Some(ExprNode::Symbol(SymbolRef::Field(_)))
                ) {
                    return Err(reject(
                        "surface energy currently requires a trace of an exact Field",
                    ));
                }
                self.component(*value, coordinate, depth + 1)?
            }
            ExprNode::Divergence(value) => {
                let (field, extent) = vector_field(self.typed, *value)?;
                let mut sum = self.push(CalculusNode::Rational {
                    value: ExactRational::integer(0),
                    dimension: ty.dimension(),
                })?;
                for axis in 0..extent {
                    let diagonal = self.input(&Input::Gradient(field, vec![axis, axis]))?;
                    sum = self.push(CalculusNode::Add(sum, diagonal))?;
                }
                sum
            }
            ExprNode::Constant(value) => {
                let offset = coordinate
                    .iter()
                    .zip(ty.shape().extents())
                    .fold(0usize, |offset, (i, n)| {
                        offset * n.get() as usize + *i as usize
                    });
                let (real, imaginary) = value
                    .component(offset)
                    .ok_or_else(|| reject("missing literal component"))?;
                if imaginary != 0.0 {
                    return Err(reject("complex density is not admitted"));
                }
                self.push(CalculusNode::Rational {
                    value: rational(real)?,
                    dimension: ty.dimension(),
                })?
            }
            ExprNode::Neg(value) => {
                let value = self.component(*value, coordinate, depth + 1)?;
                self.push(CalculusNode::Neg(value))?
            }
            ExprNode::Add(a, b) | ExprNode::Sub(a, b) | ExprNode::Mul(a, b) => {
                let ac = if self
                    .typed
                    .node_type(*a)
                    .expect("typed operand")
                    .shape()
                    .is_scalar()
                {
                    &[][..]
                } else {
                    coordinate
                };
                let bc = if self
                    .typed
                    .node_type(*b)
                    .expect("typed operand")
                    .shape()
                    .is_scalar()
                {
                    &[][..]
                } else {
                    coordinate
                };
                let left = self.component(*a, ac, depth + 1)?;
                let right = self.component(*b, bc, depth + 1)?;
                match node {
                    ExprNode::Add(..) => self.push(CalculusNode::Add(left, right))?,
                    ExprNode::Mul(..) => self.push(CalculusNode::Mul(left, right))?,
                    _ => {
                        let right = self.push(CalculusNode::Neg(right))?;
                        self.push(CalculusNode::Add(left, right))?
                    }
                }
            }
            ExprNode::Div(a, b) => {
                let Some(ExprNode::Constant(value)) = self.typed.expression().node(*b) else {
                    return Err(reject(
                        "polynomial density division requires a constant scalar denominator",
                    ));
                };
                let scalar = value
                    .real_scalar_value()
                    .ok_or_else(|| reject("denominator must be real scalar"))?;
                let fraction = BigRational::from_float(scalar.value())
                    .filter(|value| *value != BigRational::from_integer(0.into()))
                    .ok_or_else(|| reject("zero or nonfinite density denominator"))?
                    .recip();
                let inverse = exact(&fraction)?;
                let dimension = scalar
                    .dim()
                    .pow(-1, 1)
                    .ok_or_else(|| reject("reciprocal dimension overflows"))?;
                let left = self.component(*a, coordinate, depth + 1)?;
                let right = self.push(CalculusNode::Rational {
                    value: inverse,
                    dimension,
                })?;
                self.push(CalculusNode::Mul(left, right))?
            }
            ExprNode::PowI(base, exponent) if (0..=16).contains(exponent) => {
                let base = self.component(*base, &[], depth + 1)?;
                let mut result = self.push(CalculusNode::Rational {
                    value: ExactRational::integer(1),
                    dimension: DimExponents::DIMENSIONLESS,
                })?;
                for _ in 0..*exponent {
                    result = self.push(CalculusNode::Mul(result, base))?;
                }
                result
            }
            ExprNode::PureOperatorApplication(application) => {
                let definition = self
                    .typed
                    .expression()
                    .definition(application.definition())
                    .ok_or_else(|| reject("missing pure definition"))?;
                self.pure(definition, application.arguments(), coordinate, depth + 1)?
            }
            ExprNode::SymmetricPart(value) => self.pure(
                &PureOperatorDefinition::symmetric_part().map_err(reject)?,
                &[*value],
                coordinate,
                depth + 1,
            )?,
            _ => {
                return Err(reject(
                    "expression is outside the bounded polynomial first-gradient profile",
                ));
            }
        };
        self.memo.insert(key, value);
        Ok(value)
    }

    fn pure(
        &mut self,
        definition: &PureOperatorDefinition,
        arguments: &[ExprId],
        coordinate: &[u32],
        depth: usize,
    ) -> Result<CalculusNodeId, Diagnostic> {
        let types = arguments
            .iter()
            .map(|id| {
                self.typed
                    .node_type(*id)
                    .cloned()
                    .ok_or_else(|| reject("untyped pure input"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        definition.instantiate(&types).map_err(reject)?;
        self.remaining = self
            .remaining
            .checked_sub(definition.nodes().len())
            .ok_or_else(|| reject("pure density expansion bound exceeded"))?;
        let mut mapped = Vec::new();
        for node in definition.nodes() {
            let get = |id: CalculusNodeId| mapped[id.index() as usize];
            let value = match node {
                CalculusNode::FormalComponent { formal, axes } => {
                    let indices = axes
                        .iter()
                        .map(|axis| axis.resolve(coordinate))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(reject)?;
                    self.component(arguments[usize::from(*formal)], &indices, depth + 1)?
                }
                CalculusNode::Rational { value, dimension } => {
                    self.push(CalculusNode::Rational {
                        value: *value,
                        dimension: *dimension,
                    })?
                }
                CalculusNode::KroneckerDelta(a, b) => {
                    let same = a.resolve(coordinate).map_err(reject)?
                        == b.resolve(coordinate).map_err(reject)?;
                    self.push(CalculusNode::Rational {
                        value: ExactRational::integer(i64::from(same)),
                        dimension: DimExponents::DIMENSIONLESS,
                    })?
                }
                CalculusNode::Neg(a) => self.push(CalculusNode::Neg(get(*a)))?,
                CalculusNode::Add(a, b) => self.push(CalculusNode::Add(get(*a), get(*b)))?,
                CalculusNode::Mul(a, b) => self.push(CalculusNode::Mul(get(*a), get(*b)))?,
                _ => {
                    return Err(reject(
                        "pure density must be an undifferentiated polynomial",
                    ));
                }
            };
            mapped.push(value);
        }
        Ok(mapped[definition.root().index() as usize])
    }
}

fn rational(value: f64) -> Result<ExactRational, Diagnostic> {
    exact(&BigRational::from_float(value).ok_or_else(|| reject("nonfinite density literal"))?)
}

fn exact(value: &BigRational) -> Result<ExactRational, Diagnostic> {
    ExactRational::from_canonical_parts(
        value
            .numer()
            .to_i64()
            .ok_or_else(|| reject("literal numerator exceeds exact calculus bounds"))?,
        value
            .denom()
            .to_u64()
            .ok_or_else(|| reject("literal denominator exceeds exact calculus bounds"))?,
    )
    .map_err(|error: PureOperatorError| reject(error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DynQuantity, ValueType};
    use eqiora_schema::kernel::pure_operator::ExactPolynomial;
    use eqiora_schema::kernel::{
        ExprDagBuilder,
        typing::{ExpressionType, RootContract, SpatialSupport},
    };

    fn polynomial(derived: &LocalVariation) -> ExactPolynomial<usize> {
        let mut values: Vec<ExactPolynomial<usize>> = Vec::new();
        for node in derived.definition.nodes() {
            let get = |id: CalculusNodeId| &values[id.index() as usize];
            values.push(match node {
                CalculusNode::FormalComponent { formal, .. } => {
                    ExactPolynomial::atom(usize::from(*formal))
                }
                CalculusNode::Rational { value, .. } => ExactPolynomial::constant(*value),
                CalculusNode::Add(a, b) => get(*a).checked_add(get(*b)).unwrap(),
                CalculusNode::Mul(a, b) => get(*a).checked_mul(get(*b)).unwrap(),
                CalculusNode::Neg(a) => get(*a).checked_neg().unwrap(),
                CalculusNode::BoundInput(value) | CalculusNode::Differentiated { value, .. } => {
                    get(*value).clone()
                }
                _ => panic!("non-polynomial output"),
            });
        }
        values[derived.definition.root().index() as usize].clone()
    }

    fn product(coefficient: i64, a: usize, b: usize) -> ExactPolynomial<usize> {
        ExactPolynomial::constant(ExactRational::integer(coefficient))
            .checked_mul(&ExactPolynomial::atom(a))
            .unwrap()
            .checked_mul(&ExactPolynomial::atom(b))
            .unwrap()
    }

    #[test]
    fn local_gradient_energy_retains_ordered_directions_and_density_units() {
        let field = Id::<kinds::Field>::from_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAX".parse().unwrap());
        let domain =
            Id::<kinds::Domain>::from_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAW".parse().unwrap()).erase();
        let density = DimExponents::from_integers([1, 1, -2, 0, 0, 0, 0]).unwrap();
        let stiffness = DimExponents::from_integers([1, 3, -2, 0, 0, 0, 0]).unwrap();
        let mut dag = ExprDagBuilder::new();
        let c = dag.symbol(SymbolRef::Field(field)).unwrap();
        let gradient = dag.gradient(c).unwrap();
        let square = dag
            .pure_operator(
                &PureOperatorDefinition::contract(1, 1, 1, &[(0, 0)]).unwrap(),
                [gradient, gradient],
            )
            .unwrap();
        let a = dag.constant(DynQuantity::new(2.0, density)).unwrap();
        let k = dag.constant(DynQuantity::new(3.0, stiffness)).unwrap();
        let cc = dag.mul(c, c).unwrap();
        let bulk = dag.mul(a, cc).unwrap();
        let interfacial = dag.mul(k, square).unwrap();
        let energy = dag.add(bulk, interfacial).unwrap();
        let two = dag
            .constant(DynQuantity::new(2.0, DimExponents::DIMENSIONLESS))
            .unwrap();
        let energy = dag.div(energy, two).unwrap();
        let support = SpatialSupport::Volume {
            domain,
            dimensions: 1,
        };
        let typed = TypedResidual::infer(
            dag.finish([energy]).unwrap(),
            Some(support.clone()),
            |_| None,
            RootContract::Observable,
            |symbol| {
                if symbol == SymbolRef::Field(field) {
                    Ok(ExpressionType::new(
                        ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap(),
                        Some(support.clone()),
                    ))
                } else {
                    Err(())
                }
            },
        )
        .unwrap();
        for order in [1, 2] {
            let derived = derive(&typed, field, order).unwrap();
            let slot = |input: Input| {
                derived
                    .inputs
                    .iter()
                    .position(|(candidate, _)| *candidate == input)
                    .unwrap()
            };
            let c = slot(Input::Value(SymbolRef::Field(field), vec![]));
            let g = slot(Input::Gradient(field, vec![0]));
            let eta = slot(Input::Direction { input: c, order: 1 });
            let eta_x = slot(Input::Direction { input: g, order: 1 });
            // Independently, psi=(2*c^2+3*c_x^2)/2 gives
            // d psi=2*c*eta+3*c_x*eta_x and
            // d2 psi=2*eta*zeta+3*eta_x*zeta_x.
            let expected = if order == 1 {
                product(2, c, eta)
                    .checked_add(&product(3, g, eta_x))
                    .unwrap()
            } else {
                let zeta = slot(Input::Direction { input: c, order: 2 });
                let zeta_x = slot(Input::Direction { input: g, order: 2 });
                product(2, eta, zeta)
                    .checked_add(&product(3, eta_x, zeta_x))
                    .unwrap()
            };
            assert_eq!(polynomial(&derived), expected);
            assert_eq!(derived.definition.result_rule().dimension(), Some(density));
            assert_eq!(
                derived
                    .inputs
                    .iter()
                    .filter(|(input, _)| matches!(input, Input::Direction { .. }))
                    .count(),
                2 * usize::from(order)
            );
            assert!(
                derived
                    .definition
                    .nodes()
                    .iter()
                    .any(|node| matches!(node, CalculusNode::Differentiated { .. }))
            );
        }
        assert!(derive(&typed, field, 3).is_err());
    }

    #[test]
    fn quadratic_elastic_energy_retains_full_coordinate_shear_factors() {
        let field = Id::<kinds::Field>::from_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAX".parse().unwrap());
        let domain =
            Id::<kinds::Domain>::from_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAW".parse().unwrap()).erase();
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let density = DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).unwrap();
        let mut dag = ExprDagBuilder::new();
        let u = dag.symbol(SymbolRef::Field(field)).unwrap();
        let gradient = dag.gradient(u).unwrap();
        let strain = dag.symmetric_part(gradient).unwrap();
        let square = dag
            .pure_operator(
                &PureOperatorDefinition::contract(2, 2, 2, &[(0, 0), (1, 1)]).unwrap(),
                [strain, strain],
            )
            .unwrap();
        let mu = dag.constant(DynQuantity::new(3.0, density)).unwrap();
        let shear_energy = dag.mul(mu, square).unwrap();
        let divergence = dag.divergence(u).unwrap();
        let dilation_square = dag.mul(divergence, divergence).unwrap();
        let half_lambda = dag.constant(DynQuantity::new(1.0, density)).unwrap();
        let dilation_energy = dag.mul(half_lambda, dilation_square).unwrap();
        let energy = dag.add(shear_energy, dilation_energy).unwrap();
        let support = SpatialSupport::Volume {
            domain,
            dimensions: 2,
        };
        let typed = TypedResidual::infer(
            dag.finish([energy]).unwrap(),
            Some(support.clone()),
            |_| None,
            RootContract::Observable,
            |symbol| {
                if symbol != SymbolRef::Field(field) {
                    return Err(());
                }
                Ok(ExpressionType::new(
                    ValueType::shaped(
                        ScalarDomain::Real,
                        length,
                        ValueShape::new([2]).unwrap(),
                        eqiora_core::ValueFrame::SpatialCartesian,
                    )
                    .unwrap(),
                    Some(support.clone()),
                ))
            },
        )
        .unwrap();
        for order in [1, 2] {
            let derived = derive(&typed, field, order).unwrap();
            let slot = |input: Input| {
                derived
                    .inputs
                    .iter()
                    .position(|(candidate, _)| *candidate == input)
                    .unwrap()
            };
            let g = [[0, 0], [0, 1], [1, 0], [1, 1]]
                .map(|indices| slot(Input::Gradient(field, indices.to_vec())));
            let eta = g.map(|input| slot(Input::Direction { input, order: 1 }));
            let left = if order == 1 { g } else { eta };
            let right = if order == 1 {
                eta
            } else {
                g.map(|input| slot(Input::Direction { input, order: 2 }))
            };
            // Independently expand psi = 3*(u_x^2+v_y^2+(u_y+v_x)^2/2).
            // d psi = 6*u_x*eta_u,x + 6*v_y*eta_v,y
            //       + 3*(u_y+v_x)*(eta_u,y+eta_v,x).
            // d2 replaces each original gradient by the other direction;
            // both off-diagonal entries occur, with no engineering-shear rescaling.
            let mut expected = product(6, left[0], right[0])
                .checked_add(&product(6, left[3], right[3]))
                .unwrap();
            for i in [1, 2] {
                for j in [1, 2] {
                    expected = expected
                        .checked_add(&product(3, left[i], right[j]))
                        .unwrap();
                }
            }
            // lambda=2 adds (u_x+v_y)^2 to psi: its first/second variation
            // is 2*(left_u,x+left_v,y)*(right_u,x+right_v,y).
            for i in [0, 3] {
                for j in [0, 3] {
                    expected = expected
                        .checked_add(&product(2, left[i], right[j]))
                        .unwrap();
                }
            }
            assert_eq!(polynomial(&derived), expected);
            assert_eq!(derived.definition.result_rule().dimension(), Some(density));
        }
    }

    #[test]
    fn curl_energy_variations_match_independent_antisymmetric_gradient_pairs() {
        let field = Id::<kinds::Field>::from_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAX".parse().unwrap());
        let domain =
            Id::<kinds::Domain>::from_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAW".parse().unwrap()).erase();
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let density = DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).unwrap();
        for dimensions in [2, 3] {
            let mut dag = ExprDagBuilder::new();
            let u = dag.symbol(SymbolRef::Field(field)).unwrap();
            let gradient = dag.gradient(u).unwrap();
            let curl = dag
                .pure_operator(
                    &PureOperatorDefinition::curl_from_gradient(dimensions, 1).unwrap(),
                    [gradient],
                )
                .unwrap();
            let square = if dimensions == 2 {
                dag.mul(curl, curl).unwrap()
            } else {
                dag.pure_operator(
                    &PureOperatorDefinition::contract(3, 1, 1, &[(0, 0)]).unwrap(),
                    [curl, curl],
                )
                .unwrap()
            };
            let stiffness = dag.constant(DynQuantity::new(1.5, density)).unwrap();
            let energy = dag.mul(stiffness, square).unwrap();
            let support = SpatialSupport::Volume {
                domain,
                dimensions: dimensions as usize,
            };
            let typed = TypedResidual::infer(
                dag.finish([energy]).unwrap(),
                Some(support.clone()),
                |_| None,
                RootContract::Observable,
                |symbol| {
                    if symbol != SymbolRef::Field(field) {
                        return Err(());
                    }
                    Ok(ExpressionType::new(
                        ValueType::shaped(
                            ScalarDomain::Real,
                            length,
                            ValueShape::new([dimensions]).unwrap(),
                            eqiora_core::ValueFrame::SpatialCartesian,
                        )
                        .unwrap(),
                        Some(support.clone()),
                    ))
                },
            )
            .unwrap();
            // Independently, psi = 3/2 sum_{i<j}(u_j,i-u_i,j)^2.
            // Its first variation is 3 sum (u_j,i-u_i,j)(eta_j,i-eta_i,j);
            // the second replaces u by zeta. Each mixed term is negative,
            // diagonal gradients contribute zero, and no conjugation occurs.
            for order in [1, 2] {
                let derived = derive(&typed, field, order).unwrap();
                let slot = |input: Input| {
                    derived
                        .inputs
                        .iter()
                        .position(|(candidate, _)| *candidate == input)
                        .unwrap()
                };
                let mut expected = ExactPolynomial::constant(ExactRational::integer(0));
                for i in 0..dimensions {
                    for j in i + 1..dimensions {
                        let g = [[j, i], [i, j]]
                            .map(|indices| slot(Input::Gradient(field, indices.to_vec())));
                        let eta = g.map(|input| slot(Input::Direction { input, order: 1 }));
                        let other = if order == 1 {
                            g
                        } else {
                            g.map(|input| slot(Input::Direction { input, order: 2 }))
                        };
                        for (a, b, sign) in [(0, 0, 3), (1, 1, 3), (0, 1, -3), (1, 0, -3)] {
                            expected = expected
                                .checked_add(&product(sign, other[a], eta[b]))
                                .unwrap();
                        }
                    }
                }
                assert_eq!(polynomial(&derived), expected);
                assert_eq!(derived.definition.result_rule().dimension(), Some(density));
            }
        }
    }
}
