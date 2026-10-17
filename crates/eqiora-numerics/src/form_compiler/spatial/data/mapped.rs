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

    pub(super) fn constant_in_direction(
        &self,
        direction: Direction<'_>,
    ) -> Result<bool, Diagnostic> {
        for (parameter, value) in &self.parameters {
            if !matches!(
                value.value_type().scalar_domain(),
                ScalarDomain::Real | ScalarDomain::Complex
            ) {
                continue;
            }
            let parameter = parameter.downcast().ok_or_else(|| {
                super::super::invalid("map coefficient identity is not a Parameter")
            })?;
            for coordinate in ScalarSymbolCoordinate::for_value(
                SymbolRef::Parameter(parameter),
                value.value_type(),
            )? {
                if direction(&coordinate) != 0.0 {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    pub(super) fn value(&self) -> f64 {
        self.value
    }

    pub(super) fn bind_parameter_point(
        &self,
        point: &crate::parameter_point::ParameterPoint,
    ) -> Result<Self, Diagnostic> {
        let mut bound = self.clone();
        for (parameter, literal) in &mut bound.parameters {
            let replacement = point
                .get(parameter.downcast().ok_or_else(|| {
                    super::super::invalid("map coefficient identity is not a Parameter")
                })?)
                .ok_or_else(|| {
                    super::super::invalid("map coefficient Parameter point omits an exact identity")
                })?;
            if replacement.value_type() != literal.value_type() {
                return Err(super::super::invalid(
                    "map coefficient Parameter changes its exact type",
                ));
            }
            *literal = replacement.clone();
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
