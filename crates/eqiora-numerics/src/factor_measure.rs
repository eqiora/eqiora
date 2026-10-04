//! Exact factor identity and physical quadrature mapping shared by solve and observation.
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Id, RawId, entity::kinds};
use eqiora_meshing::QuadraturePoint;
use eqiora_schema::kernel::{
    AxisBounds, DomainKind, KernelNode, ObservableMeasure, typing::SpatialSupport,
};
use eqiora_sem::KernelProgram;

pub(crate) type Axis = ((RawId, usize), AxisBounds);

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(eqiora_core::diagnostic::codes::INVALID_REALIZATION, message)
}

pub(crate) fn axes(
    program: &KernelProgram,
    domain: Id<kinds::Domain>,
) -> Result<Vec<Axis>, Diagnostic> {
    let support = program
        .spatial_support(domain)
        .ok_or_else(|| invalid("factor integral support is outside the Model"))?;
    let factors = match support {
        SpatialSupport::Coordinates { factors, .. } => {
            factors.iter().map(|(id, _, _)| *id).collect()
        }
        SpatialSupport::Volume { domain, .. } => vec![*domain],
        _ => {
            return Err(invalid(
                "factor quadrature requires bounded Cartesian factors",
            ));
        }
    };
    let mut axes = Vec::new();
    for factor in factors {
        let Some(KernelNode::Domain(definition)) = program.node(factor) else {
            return Err(invalid("coordinate factor Domain is unavailable"));
        };
        let bounds = match definition.kind() {
            DomainKind::CoordinateInterval { bounds } => std::slice::from_ref(bounds),
            _ => program.resolved_cartesian_bounds(definition.id())?,
        };
        axes.extend(
            bounds
                .iter()
                .enumerate()
                .map(|(axis, bounds)| ((factor, axis), *bounds)),
        );
    }
    Ok(axes)
}

pub(crate) fn mapped_sample(
    axes: &[Axis],
    sample: &QuadraturePoint,
    measure: ObservableMeasure,
) -> Result<(Vec<DynQuantity>, DynQuantity), Diagnostic> {
    if sample.coordinates.len() != axes.len()
        || axes.is_empty()
        || measure == ObservableMeasure::Boundary
    {
        return Err(invalid(
            "factor volume quadrature requires its exact coordinate axes",
        ));
    }
    if measure == ObservableMeasure::SphericalVolume
        && (axes.len() != 1 || axes[0].1.lower().value() != 0.0)
    {
        return Err(invalid(
            "spherical volume quadrature requires a radial interval from zero",
        ));
    }
    let mut coordinates = Vec::with_capacity(axes.len());
    let mut weight = sample.weight;
    let mut dimension = DimExponents::DIMENSIONLESS;
    for ((_, bounds), coordinate) in axes.iter().zip(&sample.coordinates) {
        let lower = bounds.lower().value();
        let half_width = (bounds.upper().value() - lower) * 0.5;
        let value = lower + (coordinate + 1.0) * half_width;
        let unit = bounds.lower().dim();
        coordinates.push(DynQuantity::new(value, unit));
        weight *= half_width;
        dimension = dimension
            .mul(unit)
            .ok_or_else(|| invalid("factor measure dimension overflow"))?;
        if measure == ObservableMeasure::SphericalVolume {
            weight *= 4.0 * std::f64::consts::PI * value * value;
            dimension = dimension
                .mul(
                    unit.pow(2, 1)
                        .ok_or_else(|| invalid("radial measure dimension overflow"))?,
                )
                .ok_or_else(|| invalid("radial measure dimension overflow"))?;
        }
    }
    if !weight.is_finite() || coordinates.iter().any(|value| !value.value().is_finite()) {
        return Err(invalid("mapped factor quadrature must be finite"));
    }
    Ok((coordinates, DynQuantity::new(weight, dimension)))
}
