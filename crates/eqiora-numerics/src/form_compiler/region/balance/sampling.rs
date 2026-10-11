use super::*;
use crate::form_compiler::bilinear::Basis;
use eqiora_realization::SpaceFamily;

pub(super) struct Sample {
    pub(super) value: Vec<f64>,
    pub(super) gradient: Vec<f64>,
}
impl Sample {
    pub(super) fn basis(&self) -> Basis<'_> {
        Basis {
            value: &self.value,
            gradient: &self.gradient,
        }
    }
}
pub(super) struct Samples {
    pub(super) previous: Sample,
    pub(super) current: Sample,
    pub(super) difference: Sample,
}

pub(super) fn sample_fields(
    form: &BoundRegionForm<f64>,
    geometry: &AffineGeometryMap,
    reference_point: &[f64],
    previous: &BTreeMap<RawId, Vec<f64>>,
    current: &BTreeMap<RawId, Vec<f64>>,
) -> Result<BTreeMap<RawId, Samples>, Diagnostic> {
    let mut layouts = form
        .fields
        .iter()
        .map(|layout| (layout.field, layout))
        .collect::<BTreeMap<_, _>>();
    for (&state, &rate) in &form.eliminations {
        layouts.insert(
            state,
            form.fields
                .iter()
                .find(|layout| layout.field == rate)
                .ok_or_else(|| invalid("balance state is missing its exact rate layout"))?,
        );
    }
    if previous.keys().copied().collect::<Vec<_>>() != layouts.keys().copied().collect::<Vec<_>>()
        || current.keys().copied().collect::<Vec<_>>()
            != layouts.keys().copied().collect::<Vec<_>>()
    {
        return Err(invalid(
            "balance requires exact previous/current physical Field coverage",
        ));
    }
    let inverse = geometry.inverse_jacobian()?;
    let dimension = form.form.dimension;
    let mut result = BTreeMap::new();
    for (field, layout) in layouts {
        if !matches!(
            layout.space.family(),
            SpaceFamily::ContinuousLagrange { .. } | SpaceFamily::SimplexP1Bubble
        ) {
            return Err(invalid(
                "balance sampling requires scalar nodal bases with physical component replicas",
            ));
        }
        let basis = super::super::basis(layout.space, form.reference)?;
        let table = basis.tabulate(reference_point)?;
        let gradients = (0..table.values().len())
            .map(|node| {
                crate::affine_fem::physical_gradient(
                    table.gradient(node).expect("nodal gradient"),
                    &inverse,
                    dimension,
                )
            })
            .collect::<Vec<_>>();
        let sample = |values: &[f64]| -> Result<Sample, Diagnostic> {
            if values.len() != table.values().len() * layout.components
                || values.iter().any(|value| !value.is_finite())
            {
                return Err(invalid(
                    "balance physical coefficients have wrong shape or are nonfinite",
                ));
            }
            let mut value = vec![0.; layout.components];
            let mut gradient = vec![0.; layout.components * dimension];
            for node in 0..table.values().len() {
                for component in 0..layout.components {
                    let coefficient = values[node * layout.components + component];
                    value[component] += table.values()[node] * coefficient;
                    for axis in 0..dimension {
                        gradient[component * dimension + axis] +=
                            gradients[node][axis] * coefficient;
                    }
                }
            }
            Ok(Sample { value, gradient })
        };
        let old = sample(&previous[&field])?;
        let new = sample(&current[&field])?;
        let delta = Sample {
            value: new
                .value
                .iter()
                .zip(&old.value)
                .map(|(a, b)| a - b)
                .collect(),
            gradient: new
                .gradient
                .iter()
                .zip(&old.gradient)
                .map(|(a, b)| a - b)
                .collect(),
        };
        result.insert(
            field,
            Samples {
                previous: old,
                current: new,
                difference: delta,
            },
        );
    }
    let dt = form.time_step().expect("checked balance time");
    for (&state, &rate) in &form.eliminations {
        for ((old, new), velocity) in previous[&state]
            .iter()
            .zip(&current[&state])
            .zip(&current[&rate])
        {
            let increment = dt * velocity;
            // Three arithmetic operations, including the multiplication and
            // a previously rounded BE update; the bound scales with operands.
            let tolerance = 8. * f64::EPSILON * (old.abs() + new.abs() + increment.abs());
            if (new - old - increment).abs() > tolerance {
                return Err(invalid(
                    "volume balance history violates its exact backward-Euler kinematics",
                ));
            }
        }
    }
    Ok(result)
}
