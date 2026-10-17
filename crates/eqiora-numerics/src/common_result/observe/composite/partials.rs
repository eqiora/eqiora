//! Chain reduced covectors through the same typed component arithmetic as values.
use super::super::partials::Covector;
use super::*;
use eqiora_ir::{RelationCotangent, ScalarSymbolCoordinate};

pub(super) struct PartialContext<'a> {
    pub(super) parameters: &'a [ScalarSymbolCoordinate],
    pub(super) accepted: HashMap<Id<kinds::Observable>, Vec<Covector>>,
}

impl<'a> PartialContext<'a> {
    pub(super) fn new(parameters: &'a [ScalarSymbolCoordinate]) -> Self {
        Self {
            parameters,
            accepted: HashMap::new(),
        }
    }
}

impl Context<'_> {
    pub(super) fn finite_partials(
        &self,
        definition: &ObservableDef,
    ) -> Result<Vec<Covector>, Diagnostic> {
        let context = self.partials.as_ref().expect("requested partials");
        let typed = self
            .program
            .typed_observable(definition.id())
            .map_err(|errors| errors.into_iter().next().expect("failed typing"))?;
        let operator = ComponentScalarization::lower(&typed)?;
        let mut bound = HashMap::new();
        for coordinate in operator.rows().iter().flat_map(|row| row.symbols()) {
            if bound.contains_key(coordinate) {
                continue;
            }
            let symbol = coordinate.symbol();
            let value = self
                .resolve(symbol)
                .ok_or_else(|| invalid("composite Observable input is unavailable"))?;
            let role = match symbol {
                SymbolRef::Parameter(_) if context.parameters.contains(coordinate) => {
                    DifferentiationRole::Parameter
                }
                SymbolRef::Parameter(_) | SymbolRef::Time => DifferentiationRole::Frozen,
                SymbolRef::Observable(_) => DifferentiationRole::Unknown,
                _ => {
                    return Err(invalid(
                        "spatial point partials require reduced Observable inputs or Parameters",
                    ));
                }
            };
            bound.insert(coordinate.clone(), (coordinate.read_value(&value)?, role));
        }
        let action = operator.linearize(|coordinate| bound.get(coordinate).copied())?;
        let mut rows = vec![Covector::zero(context.parameters.len()); action.residual_dimension()];
        let mut weights = vec![0.; rows.len()];
        let mut unknown = vec![0.; action.unknown_dimension()];
        let mut parameter = vec![0.; action.parameter_dimension()];
        for (row, output) in rows.iter_mut().enumerate() {
            weights[row] = 1.;
            action.vjp(
                &weights,
                RelationCotangent::Both {
                    unknown: &mut unknown,
                    parameter: &mut parameter,
                },
            )?;
            weights[row] = 0.;
            for (coordinate, &weight) in action.unknown_coordinates().iter().zip(&unknown) {
                let SymbolRef::Observable(id) = coordinate.symbol() else {
                    unreachable!("reduced dependency")
                };
                let dependency = context.accepted.get(&id).ok_or_else(|| {
                    invalid("Observable dependency has no spatial point partials")
                })?;
                let value = &self.accepted[&id].0;
                let index =
                    super::super::projection::coordinate_index(coordinate, value.value_type())?;
                output.add_scaled(&dependency[index], weight)?;
            }
            for (coordinate, &weight) in action.parameter_coordinates().iter().zip(&parameter) {
                let index = context
                    .parameters
                    .iter()
                    .position(|candidate| candidate == coordinate)
                    .expect("selected Parameter coordinate");
                output.parameters[index] += weight;
            }
            if output.parameters.iter().any(|value| !value.is_finite()) {
                return Err(invalid(
                    "composite Observable Parameter covector is non-finite",
                ));
            }
        }
        Ok(rows)
    }
}
