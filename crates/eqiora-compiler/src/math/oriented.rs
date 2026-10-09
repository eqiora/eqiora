//! Oriented Cartesian operations compose retained gradient/normal nodes and exact pure maps.
use eqiora_schema::kernel::{
    pure_operator::PureOperatorDefinition,
    typing::{self, ExpressionType, SpatialSupport},
};

#[derive(Clone, Copy)]
pub(crate) enum Operation {
    Curl,
    TangentialTrace,
}

impl Operation {
    pub(crate) fn named(name: &str) -> Option<Self> {
        match name {
            "curl" => Some(Self::Curl),
            "tangential_trace" => Some(Self::TangentialTrace),
            _ => None,
        }
    }

    pub(crate) fn definition<I: Clone + Eq + std::fmt::Debug>(
        self,
        operand: &ExpressionType<I>,
    ) -> Result<(PureOperatorDefinition, ExpressionType<I>), String> {
        let dimensions = operand
            .support
            .as_ref()
            .and_then(SpatialSupport::ambient_dimensions)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or("oriented operator requires an exact physical Cartesian support")?;
        let (definition, argument) = match self {
            Self::Curl => (
                PureOperatorDefinition::curl_from_gradient(
                    dimensions,
                    u16::try_from(operand.shape().rank())
                        .map_err(|_| "unsupported curl operand rank")?,
                ),
                typing::gradient(operand).map_err(|error| error.to_string())?,
            ),
            Self::TangentialTrace => (
                PureOperatorDefinition::tangential_lift(dimensions),
                operand.clone(),
            ),
        };
        let definition = definition.map_err(|error| error.to_string())?;
        definition
            .instantiate(std::slice::from_ref(&argument))
            .map_err(|error| error.to_string())?;
        Ok((definition, argument))
    }

    pub(crate) fn result_type<I: Clone + Eq + std::fmt::Debug>(
        self,
        operand: &ExpressionType<I>,
        support: Option<&SpatialSupport<I>>,
    ) -> Result<ExpressionType<I>, String> {
        let (definition, argument) = self.definition(operand)?;
        let result = definition
            .instantiate(&[argument])
            .map_err(|error| error.to_string())?
            .result_type()
            .clone();
        match self {
            Self::Curl => Ok(result),
            Self::TangentialTrace => {
                typing::normal(&result, support).map_err(|error| error.to_string())
            }
        }
    }
}
