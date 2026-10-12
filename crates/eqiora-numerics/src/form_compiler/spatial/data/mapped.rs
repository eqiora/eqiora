//! Fixed-time affine map factors retain their exact Parameter binding.
use super::*;
use eqiora_core::{DimExponents, Id, ScalarDomain, ValueLiteral, ValueType, entity::kinds};
use eqiora_ir::ScalarOperatorIr;
use eqiora_schema::kernel::typing::TypedResidual;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct MapFactorData {
    typed: Arc<TypedResidual<RawId>>,
    id: ExprId,
    time_s: Option<f64>,
    parameters: BTreeMap<RawId, ValueLiteral>,
    value: f64,
}

impl MapFactorData {
    pub(super) fn new<S: Coefficient>(
        context: &Context<'_, S>,
        id: ExprId,
    ) -> Result<Self, Diagnostic> {
        let typed = super::super::super::scalar::typed_relation(context.program, context.owner)?;
        if typed.expression() != context.dag {
            return Err(super::super::invalid(
                "map coefficient requires its exact typed relation expression",
            ));
        }
        let mut parameters = BTreeMap::new();
        let value = evaluate(&typed, id, context.time_s, &mut |parameter| {
            let value = context.program.typed_value(parameter.erase())?.clone();
            parameters.insert(parameter.erase(), value.clone());
            Some(value)
        })?;
        Ok(Self {
            typed: Arc::new(typed),
            id,
            time_s: context.time_s,
            parameters,
            value,
        })
    }

    pub(super) fn value(&self) -> f64 {
        self.value
    }

    pub(super) fn bind_parameter_point<S: Coefficient>(
        &self,
        fields: &[Id<kinds::Parameter>],
        values: &[S],
    ) -> Result<Self, Diagnostic> {
        // Reuse the coefficient tape's complete point validation, including
        // duplicate identities and finite values, even for an empty map point.
        ScalarSpatialExpression::constant(1, <S as From<f64>>::from(0.0))
            .bind_parameter_point(fields, values)?;
        let mut bound = self.clone();
        for (parameter, literal) in &mut bound.parameters {
            let index = fields
                .iter()
                .position(|field| field.erase() == *parameter)
                .ok_or_else(|| {
                    super::super::invalid("map coefficient Parameter point omits an exact identity")
                })?;
            if values[index].im() != 0.0 {
                return Err(super::super::invalid(
                    "affine map Parameter must remain real",
                ));
            }
            *literal = ValueLiteral::from_real(literal.value_type().clone(), values[index].re())
                .map_err(|_| super::super::invalid("invalid real map Parameter value"))?;
        }
        bound.value = evaluate(&bound.typed, bound.id, bound.time_s, &mut |parameter| {
            bound.parameters.get(&parameter.erase()).cloned()
        })?;
        Ok(bound)
    }
}

fn evaluate(
    typed: &TypedResidual<RawId>,
    id: ExprId,
    time_s: Option<f64>,
    parameter: &mut impl FnMut(Id<kinds::Parameter>) -> Option<ValueLiteral>,
) -> Result<f64, Diagnostic> {
    let Some(ExprNode::CoordinateMapFactor { factor, source, .. }) = typed.expression().node(id)
    else {
        return Err(super::super::invalid(
            "coefficient is not an exact coordinate map factor",
        ));
    };
    let time = time_s
        .map(|time| {
            ValueLiteral::from_real(
                ValueType::scalar(
                    ScalarDomain::Real,
                    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap(),
                )
                .expect("real physical Time type"),
                time,
            )
        })
        .transpose()
        .map_err(|_| super::super::invalid("map Time requires a finite real value"))?;
    let rows =
        ScalarOperatorIr::bind_affine_coordinate_map(typed, id, &mut |symbol| match symbol {
            SymbolRef::Time => time.clone(),
            SymbolRef::Parameter(id) => parameter(id),
            _ => None,
        })?;
    let entries = rows
        .iter()
        .flat_map(|row| row.coefficients().iter().copied())
        .collect::<Vec<_>>();
    ScalarOperatorIr::coordinate_map_factor(&entries, source.len(), *factor)
}
