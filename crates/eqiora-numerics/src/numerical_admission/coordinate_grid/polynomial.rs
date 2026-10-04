//! Structural polynomial classification; execution remains with numerical assembly.
use super::*;
use eqiora_core::RawId;
use eqiora_schema::kernel::pure_operator::{ExactPolynomial, ExactPolynomialError, ExactRational};
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Atom {
    Coordinate,
    Parameter(RawId),
    Field(RawId),
    Partial(RawId),
}
pub(super) type Polynomial = ExactPolynomial<Atom>;

pub(super) fn normalize(
    domain: Id<kinds::Domain>,
    dag: &ExprDag,
    root: ExprId,
) -> Result<Polynomial, Diagnostic> {
    if dag.nodes().len() > 4096 {
        return Err(invalid(
            "coordinate conservation expression exceeds 4096 nodes",
        ));
    }
    let mut remaining = 262_144usize;
    let mut charge = |value: &Polynomial| {
        let work = 1 + value
            .terms()
            .map(|(atoms, _)| 1 + atoms.len())
            .sum::<usize>();
        remaining = remaining
            .checked_sub(work)
            .ok_or(ExactPolynomialError::Limit)?;
        Ok::<_, ExactPolynomialError>(())
    };
    let coordinate = |node: &ExprNode| matches!(node, ExprNode::Symbol(SymbolRef::Coordinate {support, factor, axis:0}) if *support == domain && *factor == domain);
    let constant = |value: f64| {
        ExactRational::from_binary64(value)
            .map(Polynomial::constant)
            .ok_or_else(|| {
                invalid("coordinate conservation coefficient exceeds exact rational bounds")
            })
    };
    let mut reachable = vec![false; dag.nodes().len()];
    reachable[root.index() as usize] = true;
    for index in (0..=root.index() as usize).rev() {
        if reachable[index] {
            let mut mark = |id: ExprId| {
                reachable[id.index() as usize] = true;
            };
            match &dag.nodes()[index] {
                ExprNode::Neg(value) | ExprNode::PowI(value, _) => mark(*value),
                ExprNode::Add(a, b) | ExprNode::Sub(a, b) | ExprNode::Mul(a, b) => {
                    mark(*a);
                    mark(*b);
                }
                ExprNode::PureOperatorApplication(application) => {
                    for argument in application.arguments() {
                        mark(*argument);
                    }
                }
                // Coordinates and Field derivatives are exact semantic atoms here.
                ExprNode::Constant(_)
                | ExprNode::Symbol(_)
                | ExprNode::CoordinatePartial { .. } => {}
                _ => {
                    return Err(invalid(
                        "coordinate conservation contains an unsupported expression",
                    ));
                }
            }
        }
    }
    let mut values: Vec<Option<Polynomial>> = vec![None; dag.nodes().len()];
    for index in 0..=root.index() as usize {
        if !reachable[index] {
            continue;
        }
        let get = |id: ExprId| {
            values[id.index() as usize]
                .as_ref()
                .expect("topological expression")
        };
        let node = &dag.nodes()[index];
        let value = match node {
            ExprNode::Constant(value) => constant(
                value
                    .real_scalar_value()
                    .ok_or_else(|| invalid("coordinate conservation requires real scalars"))?
                    .value(),
            )?,
            ExprNode::Symbol(SymbolRef::Parameter(id)) => {
                Polynomial::atom(Atom::Parameter(id.erase()))
            }
            ExprNode::Symbol(SymbolRef::Field(id)) => Polynomial::atom(Atom::Field(id.erase())),
            _ if coordinate(node) => Polynomial::atom(Atom::Coordinate),
            ExprNode::CoordinatePartial { value, wrt } => {
                match (dag.node(*value), dag.node(*wrt)) {
                    (Some(ExprNode::Symbol(SymbolRef::Field(id))), Some(wrt))
                        if coordinate(wrt) =>
                    {
                        Polynomial::atom(Atom::Partial(id.erase()))
                    }
                    _ => {
                        return Err(invalid(
                            "coordinate conservation requires an exact first Field derivative",
                        ));
                    }
                }
            }
            ExprNode::Neg(value) => get(*value).checked_neg().map_err(error)?,
            ExprNode::Add(a, b) => get(*a).checked_add(get(*b)).map_err(error)?,
            ExprNode::Sub(a, b) => get(*a)
                .checked_add(&get(*b).checked_neg().map_err(error)?)
                .map_err(error)?,
            ExprNode::Mul(a, b) => get(*a).checked_mul(get(*b)).map_err(error)?,
            ExprNode::PowI(base, exponent) if (0..=8).contains(exponent) => {
                let mut value = Polynomial::constant(ExactRational::integer(1));
                for _ in 0..*exponent {
                    value = value.checked_mul(get(*base)).map_err(error)?;
                    charge(&value).map_err(error)?;
                }
                value
            }
            ExprNode::PureOperatorApplication(application) => {
                let definition = dag
                    .definition(application.definition())
                    .expect("retained definition");
                let arguments = application
                    .arguments()
                    .iter()
                    .map(|id| get(*id))
                    .collect::<Vec<_>>();
                Polynomial::substitute(definition, &arguments, &mut charge)
                    .map_err(error)?
                    .ok_or_else(|| {
                        invalid("coordinate conservation requires a scalar polynomial operator")
                    })?
            }
            _ => {
                return Err(invalid(
                    "coordinate conservation contains an unsupported expression",
                ));
            }
        };
        charge(&value).map_err(error)?;
        values[index] = Some(value);
    }
    Ok(values[root.index() as usize].take().expect("selected root"))
}

pub(super) fn error(value: ExactPolynomialError) -> Diagnostic {
    invalid(format!(
        "coordinate conservation exact polynomial projection failed: {value:?}"
    ))
}
pub(super) fn coefficient(value: &Polynomial, atoms: &[Atom]) -> ExactRational {
    let mut atoms = atoms.to_vec();
    atoms.sort();
    value
        .terms()
        .find(|(key, _)| *key == atoms.as_slice())
        .map_or(ExactRational::integer(0), |(_, coefficient)| coefficient)
}
pub(super) fn constant(program: &KernelProgram, value: &Polynomial) -> Option<f64> {
    let mut terms = value.terms();
    let Some((atoms, coefficient)) = terms.next() else {
        return Some(0.0);
    };
    if terms.next().is_some() {
        return None;
    }
    fixed_coefficient(program, atoms, coefficient)
}

pub(super) fn fixed_coefficient(
    program: &KernelProgram,
    atoms: &[Atom],
    coefficient: ExactRational,
) -> Option<f64> {
    let scale = match atoms {
        [] => 1.0,
        [Atom::Parameter(id)] => program.typed_value(*id)?.real_scalar_value()?.value(),
        _ => return None,
    };
    let value = coefficient.as_f64() * scale;
    value.is_finite().then_some(value)
}
