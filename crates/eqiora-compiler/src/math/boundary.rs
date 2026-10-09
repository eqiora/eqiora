//! Boundary selection for source trace operators; Domain owners supply exact supports.
use eqiora_lang::{CallArguments, Expr, ExprKind};
use eqiora_schema::kernel::typing::{self, ExpressionType, SpatialSupport};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operation {
    Trace,
    Normal,
    Tangential,
}

impl Operation {
    pub(crate) fn named(name: &str) -> Option<Self> {
        match name {
            "trace" => Some(Self::Trace),
            "normal" => Some(Self::Normal),
            "tangential_trace" => Some(Self::Tangential),
            _ => None,
        }
    }

    pub(crate) fn result_type<I: Clone + Eq + std::fmt::Debug>(
        self,
        operand: &ExpressionType<I>,
        target: Option<&SpatialSupport<I>>,
        from: Option<&SpatialSupport<I>>,
    ) -> Result<ExpressionType<I>, String> {
        if let Some(from) = from {
            match (from, target) {
                (
                    SpatialSupport::Volume { domain, dimensions },
                    Some(SpatialSupport::Boundary {
                        parent,
                        dimensions: target_dimensions,
                        ..
                    }),
                ) if domain == parent && dimensions == target_dimensions => {}
                _ => {
                    return Err("from must name the selected boundary's exact parent volume".into());
                }
            }
        }
        match self {
            Self::Trace => typing::trace(operand, target).map_err(|error| error.to_string()),
            Self::Normal => typing::normal(operand, target).map_err(|error| error.to_string()),
            Self::Tangential => {
                super::oriented::Operation::TangentialTrace.result_type(operand, target)
            }
        }
    }
}

pub(crate) struct Arguments<'a> {
    pub(crate) value: &'a Expr,
    pub(crate) on: Option<&'a str>,
    pub(crate) from: Option<&'a str>,
}

pub(crate) fn source(arguments: &CallArguments) -> Result<Arguments<'_>, &'static str> {
    let (values, options) = arguments.parts();
    let [value] = values else {
        return Err("boundary operator requires one expression");
    };
    let mut on = None;
    let mut from = None;
    for option in options {
        let ExprKind::Name(name) = option.value().kind() else {
            return Err("boundary on/from selectors require exact support names");
        };
        let selected = match option.name() {
            "on" => &mut on,
            "from" => &mut from,
            _ => return Err("boundary operator accepts only on and from selectors"),
        };
        if selected.replace(name.as_str()).is_some() {
            return Err("boundary operator has a duplicate support selector");
        }
    }
    Ok(Arguments { value, on, from })
}
