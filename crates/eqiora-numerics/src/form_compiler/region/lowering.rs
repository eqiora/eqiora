use eqiora_core::Diagnostic;
use eqiora_schema::kernel::{ExprId, ExprNode, SymbolRef};

use super::flux::FluxTerm;
use super::{Context, Data, Pairing, Row, Term, invalid};

#[derive(Clone, Copy)]
enum Position {
    Strong,
    Flux,
    Isotropic,
}

pub(super) fn lower(
    context: &Context<'_>,
    id: ExprId,
    coefficient: Data,
    row: &mut Row,
    depth: usize,
) -> Result<(), Diagnostic> {
    expression(context, id, coefficient, row, Position::Strong, depth)
}

pub(super) fn boundary_flux(
    context: &Context<'_>,
    id: ExprId,
    row: &Row,
) -> Result<Vec<FluxTerm>, Diagnostic> {
    let mut boundary = row.clone();
    boundary.terms.clear();
    boundary.dyadics.clear();
    boundary.flux.clear();
    expression(
        context,
        id,
        Data::constant(context.dimension, 1.0),
        &mut boundary,
        Position::Flux,
        0,
    )?;
    if !boundary.dyadics.is_empty() {
        return Err(invalid(
            "explicit nonlinear boundary flux laws are not yet admitted",
        ));
    }
    Ok(boundary.flux)
}

fn expression(
    context: &Context<'_>,
    id: ExprId,
    coefficient: Data,
    row: &mut Row,
    position: Position,
    depth: usize,
) -> Result<(), Diagnostic> {
    if depth > 128 {
        return Err(invalid("region expression nesting exceeds 128"));
    }
    let node = context.dag.node(id).expect("validated DAG");
    let negative = |value: Data| value.multiply(Data::constant(context.dimension, -1.0));
    let recurse = |id, coefficient, row: &mut Row| {
        expression(context, id, coefficient, row, position, depth + 1)
    };
    match node {
        ExprNode::Add(a, b) => {
            recurse(*a, coefficient.clone(), row)?;
            recurse(*b, coefficient, row)
        }
        ExprNode::Sub(a, b) => {
            recurse(*a, coefficient.clone(), row)?;
            recurse(*b, negative(coefficient), row)
        }
        ExprNode::Neg(a) => recurse(*a, negative(coefficient), row),
        ExprNode::Mul(a, b) => {
            if let Ok(data) = context.data(*a, depth + 1) {
                recurse(*b, coefficient.multiply(data), row)
            } else if let Ok(data) = context.data(*b, depth + 1) {
                recurse(*a, coefficient.multiply(data), row)
            } else {
                Err(invalid(
                    "region product has nonlinear unknown-dependent coefficients",
                ))
            }
        }
        ExprNode::Div(a, b) => recurse(*a, coefficient.divide(context.data(*b, depth + 1)?), row),
        ExprNode::Constant(value) if value.is_zero() => {
            if coefficient.spatial() {
                return Err(invalid(
                    "discarded zero requires a coordinate-independent multiplier",
                ));
            }
            coefficient.evaluate(&vec![0.0; context.dimension])?;
            Ok(())
        }
        ExprNode::Divergence(value) if matches!(position, Position::Strong) => {
            if coefficient.spatial() {
                return Err(invalid(
                    "spatial multiplier outside divergence requires product-rule lowering",
                ));
            }
            if let Some(ExprNode::Symbol(SymbolRef::Field(field))) = context.dag.node(*value) {
                row.terms.push(Term {
                    positive_diffusion: false,
                    trial: field.erase(),
                    derivative: false,
                    pairing: Pairing::TestValueTrialDivergence,
                    coefficient,
                });
                Ok(())
            } else {
                expression(
                    context,
                    *value,
                    negative(coefficient),
                    row,
                    Position::Flux,
                    depth + 1,
                )
            }
        }
        ExprNode::Gradient(value) if matches!(position, Position::Strong) => {
            if let Ok(potential) = context.data(*value, depth + 1) {
                if !potential.spatial() {
                    potential.evaluate(&vec![0.0; context.dimension])?;
                }
                if row.forcing.len() != context.dimension {
                    return Err(invalid("gradient forcing requires a spatial vector row"));
                }
                for (axis, forcing) in row.forcing.iter_mut().enumerate() {
                    // The tape owner demands every primal intermediate. A finite
                    // derivative must not hide an overflowing or undefined potential.
                    let derivative = potential
                        .clone()
                        .multiply(Data::constant(context.dimension, 0.0))
                        .add(potential.coordinate_derivative(axis, context.dimension)?);
                    *forcing = forcing
                        .clone()
                        .add(negative(coefficient.clone().multiply(derivative)));
                }
                return Ok(());
            }
            if coefficient.spatial() {
                return Err(invalid(
                    "spatial multiplier outside gradient requires product-rule lowering",
                ));
            }
            expression(
                context,
                *value,
                negative(coefficient),
                row,
                Position::Isotropic,
                depth + 1,
            )
        }
        ExprNode::PureOperatorApplication(application) if matches!(position, Position::Flux) => {
            let definition = eqiora_ir::PureOperatorDefinition::dyadic_product()
                .map_err(|_| invalid("invalid dyadic operator definition"))?;
            let arguments = application.arguments();
            let field = arguments
                .first()
                .and_then(|id| match context.dag.node(*id) {
                    Some(ExprNode::Symbol(SymbolRef::Field(field))) => Some(field.erase()),
                    _ => None,
                });
            if application.definition() != definition.digest()
                || arguments.len() != 2
                || field.is_none()
                || arguments.get(1).and_then(|id| match context.dag.node(*id) {
                    Some(ExprNode::Symbol(SymbolRef::Field(field))) => Some(field.erase()),
                    _ => None,
                }) != field
                || context.coefficients.contains_key(&field.unwrap())
            {
                return Err(invalid(
                    "dyadic flux requires the exact operator and identical trial Fields",
                ));
            }
            row.dyadics.push(super::nonlinear::DyadicTerm {
                field: field.unwrap(),
                coefficient,
                split: false,
            });
            Ok(())
        }
        ExprNode::Gradient(value) if matches!(position, Position::Flux) => {
            trial(context, *value, coefficient, row, Pairing::Gradient)
        }
        ExprNode::SymmetricPart(value) if matches!(position, Position::Flux) => {
            let Some(ExprNode::Gradient(field)) = context.dag.node(*value) else {
                return Err(invalid("symmetric weak term requires a Field gradient"));
            };
            trial(
                context,
                *field,
                coefficient,
                row,
                Pairing::SymmetricGradient,
            )
        }
        ExprNode::IsotropicLift(value) if matches!(position, Position::Flux) => expression(
            context,
            *value,
            coefficient,
            row,
            Position::Isotropic,
            depth + 1,
        ),
        ExprNode::Divergence(value) if matches!(position, Position::Isotropic) => {
            trial(context, *value, coefficient, row, Pairing::Divergence)
        }
        ExprNode::Symbol(SymbolRef::Field(field))
            if !context.coefficients.contains_key(&field.erase()) =>
        {
            let pairing = match position {
                Position::Strong => Pairing::Value,
                Position::Isotropic => Pairing::TestDivergenceTrialValue,
                Position::Flux => return Err(invalid("unsupported bare Field flux")),
            };
            trial(context, id, coefficient, row, pairing)
        }
        ExprNode::Symbol(SymbolRef::Derivative(_)) if matches!(position, Position::Strong) => {
            trial(context, id, coefficient, row, Pairing::Value)
        }
        _ => {
            let data = context.data(id, depth + 1)?;
            match position {
                Position::Strong if row.value_type.shape().is_scalar() => {
                    row.forcing[0] = row.forcing[0]
                        .clone()
                        .add(negative(coefficient.multiply(data)));
                    Ok(())
                }
                // A uniform isotropic stress has zero strong divergence/gradient.
                Position::Isotropic if !data.spatial() && !coefficient.spatial() => {
                    // Coordinate independence proves this value is uniform, not
                    // defined. Check the exact immutable Parameter point before
                    // erasing its gradient; role dependencies remain retained.
                    let value = coefficient.multiply(data);
                    value.evaluate(&vec![0.0; context.dimension])?;
                    row.flux.push(FluxTerm::Isotropic(value));
                    Ok(())
                }
                _ => Err(invalid("unsupported spatial forcing in region weak form")),
            }
        }
    }
}

fn trial(
    context: &Context<'_>,
    id: ExprId,
    coefficient: Data,
    row: &mut Row,
    pairing: Pairing,
) -> Result<(), Diagnostic> {
    let (field, derivative) = match context.dag.node(id) {
        Some(ExprNode::Symbol(SymbolRef::Field(field)))
            if !context.coefficients.contains_key(&field.erase()) =>
        {
            (field.erase(), false)
        }
        Some(ExprNode::Symbol(SymbolRef::Derivative(field))) if pairing == Pairing::Value => {
            (field.erase(), true)
        }
        _ => {
            return Err(invalid(
                "region differential operator requires one exact trial Field",
            ));
        }
    };
    let term = Term {
        positive_diffusion: false,
        trial: field,
        derivative,
        pairing,
        coefficient,
    };
    if matches!(
        pairing,
        Pairing::Gradient
            | Pairing::SymmetricGradient
            | Pairing::Divergence
            | Pairing::TestDivergenceTrialValue
    ) {
        row.flux.push(FluxTerm::Trial(term.clone()));
    }
    row.terms.push(term);
    Ok(())
}
