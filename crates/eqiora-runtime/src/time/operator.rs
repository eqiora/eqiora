//! Time projection over the existing scalar and finite-component IR profiles.
use super::*;
use eqiora_ir::{ComponentScalarization, ScalarSymbolCoordinate};
use eqiora_schema::kernel::typing::TypedResidual;

#[derive(Debug, Clone, PartialEq)]
enum Profile {
    Scalar(ScalarOperatorIr),
    Components(ComponentScalarization),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct TimeOperator {
    profile: Profile,
    symbols: Vec<ScalarSymbolCoordinate>,
}

impl TimeOperator {
    pub(super) fn lower(typed: &TypedResidual<eqiora_core::RawId>) -> Result<Self, Diagnostic> {
        let scalar = typed.node_types().iter().all(|ty| {
            ty.shape().is_scalar()
                && ty.value_type.scalar_domain() != eqiora_core::ScalarDomain::Complex
        });
        let (profile, symbols) = if scalar {
            let operator = ScalarOperatorIr::lower_typed_scalar(typed)?;
            let real = eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                eqiora_core::DimExponents::DIMENSIONLESS,
            )
            .expect("real scalar type");
            let symbols = operator
                .symbols()
                .iter()
                .map(|&symbol| {
                    ScalarSymbolCoordinate::for_value(symbol, &real)
                        .map(|mut values| values.remove(0))
                })
                .collect::<Result<Vec<_>, _>>()?;
            (Profile::Scalar(operator), symbols)
        } else {
            let operator = ComponentScalarization::lower(typed)?;
            let mut symbols = Vec::new();
            for row in operator.rows() {
                for symbol in row.symbols() {
                    if !symbols.contains(symbol) {
                        symbols.push(symbol.clone());
                    }
                }
            }
            (Profile::Components(operator), symbols)
        };
        Ok(Self { profile, symbols })
    }

    pub(super) fn symbols(&self) -> &[ScalarSymbolCoordinate] {
        &self.symbols
    }
    pub(super) fn residual_count(&self) -> usize {
        match &self.profile {
            Profile::Scalar(operator) => operator.residual_count(),
            Profile::Components(operator) => operator.rows().len(),
        }
    }
    pub(super) fn derivative_coefficients(
        &self,
        relation: Id<kinds::Relation>,
        rates: &[ScalarSymbolCoordinate],
    ) -> Result<Vec<f64>, Diagnostic> {
        match &self.profile {
            Profile::Scalar(operator) => operator
                .constant_symbol_jacobian(
                    &rates
                        .iter()
                        .map(ScalarSymbolCoordinate::symbol)
                        .collect::<Vec<_>>(),
                )
                .map(|proof| proof.coefficients().to_vec())
                .map_err(|failure| derivative_structure_error(relation, failure)),
            Profile::Components(operator) => {
                let mut values = Vec::new();
                for row in operator.rows() {
                    values
                        .extend_from_slice(row.constant_coordinate_jacobian(rates)?.coefficients());
                }
                Ok(values)
            }
        }
    }
    /// Prove homogeneity and constant state coefficients with bound Parameters.
    /// Time is selected, never frozen at a probe point; its coefficient must vanish.
    pub(super) fn require_homogeneous_constant(
        &self,
        relation: Id<kinds::Relation>,
        selected: &[ScalarSymbolCoordinate],
        bindings: &[(ScalarSymbolCoordinate, f64)],
        time_column: Option<usize>,
    ) -> Result<(), Diagnostic> {
        let check = |coefficients: &[f64], offsets: &[f64]| {
            if offsets.iter().any(|&offset| offset != 0.)
                || time_column.is_some_and(|column| {
                    coefficients
                        .chunks_exact(selected.len())
                        .any(|row| row[column] != 0.)
                })
            {
                Err(invalid_time(
                    relation,
                    "constant generator requires homogeneous autonomous equations",
                ))
            } else {
                Ok(())
            }
        };
        match &self.profile {
            Profile::Scalar(operator) => {
                let selected = selected
                    .iter()
                    .map(ScalarSymbolCoordinate::symbol)
                    .collect::<Vec<_>>();
                let bindings = bindings
                    .iter()
                    .map(|(c, v)| (c.symbol(), *v))
                    .collect::<Vec<_>>();
                let affine = operator
                    .bind_affine(&selected, &bindings)
                    .map_err(|error| {
                        invalid_time(
                            relation,
                            format!(
                                "constant generator requires a structural affine proof: {error:?}"
                            ),
                        )
                    })?;
                check(affine.coefficients(), affine.offsets())
            }
            Profile::Components(operator) => {
                for row in operator.rows() {
                    let affine = row.bind_affine(selected, bindings)?;
                    check(affine.coefficients(), affine.offsets())?;
                }
                Ok(())
            }
        }
    }
    pub(super) fn evaluate(&self, inputs: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        match &self.profile {
            Profile::Scalar(operator) => operator.evaluate(inputs),
            Profile::Components(operator) => operator.evaluate(|source| {
                self.symbols
                    .iter()
                    .position(|symbol| symbol == source)
                    .and_then(|index| inputs.get(index).copied())
            }),
        }
    }
    pub(super) fn linearize<'a>(
        &'a self,
        inputs: &[f64],
        roles: &[DifferentiationRole],
    ) -> Result<Box<dyn LinearizedRelation<f64> + 'a>, Diagnostic> {
        match &self.profile {
            Profile::Scalar(operator) => Ok(Box::new(operator.linearize(inputs, roles)?)),
            Profile::Components(operator) => Ok(Box::new(operator.linearize(|source| {
                let index = self.symbols.iter().position(|symbol| symbol == source)?;
                Some((*inputs.get(index)?, *roles.get(index)?))
            })?)),
        }
    }
}
