//! Transport the existing coordinate oracles through the public typed boundary.

use eqiora::api::{
    DifferentiableCotangent, DifferentiableProgramIdentity, DifferentiableTangent,
    DifferentiableValueSignature, DifferentiableValues,
};

pub(crate) fn tangent(
    identity: &DifferentiableProgramIdentity,
    coordinates: &[f64],
) -> Result<DifferentiableTangent, eqiora::Diagnostic> {
    let leaves = components(identity.inputs(), coordinates);
    DifferentiableTangent::from_components(
        identity.inputs(),
        &leaves.iter().map(Vec::as_slice).collect::<Vec<_>>(),
    )
}

pub(crate) fn cotangent(
    identity: &DifferentiableProgramIdentity,
    coordinates: &[f64],
) -> Result<DifferentiableCotangent, eqiora::Diagnostic> {
    let leaves = components(identity.outputs(), coordinates);
    DifferentiableCotangent::from_components(
        identity.outputs(),
        &leaves.iter().map(Vec::as_slice).collect::<Vec<_>>(),
    )
}

fn components(
    signatures: &[DifferentiableValueSignature],
    coordinates: &[f64],
) -> Vec<Vec<(f64, f64)>> {
    let mut offset = 0;
    let leaves = signatures
        .iter()
        .map(|signature| {
            let width = if signature.mathematical_type().value_type.scalar_domain()
                == eqiora::ScalarDomain::Complex
            {
                2
            } else {
                1
            };
            let count = signature.shape().iter().product::<usize>();
            let end = offset + count * width;
            let leaf = coordinates[offset..end]
                .chunks_exact(width)
                .map(|pair| (pair[0], if width == 2 { pair[1] } else { 0.0 }))
                .collect();
            offset = end;
            leaf
        })
        .collect();
    assert_eq!(offset, coordinates.len(), "oracle coordinate inventory");
    leaves
}

pub(crate) trait FixtureCoordinates {
    fn fixture_coordinates(&self) -> Vec<f64>;
}

impl FixtureCoordinates for DifferentiableValues {
    fn fixture_coordinates(&self) -> Vec<f64> {
        project(self.signatures(), |index| {
            self.leaf(index).unwrap().components()
        })
    }
}

impl FixtureCoordinates for DifferentiableTangent {
    fn fixture_coordinates(&self) -> Vec<f64> {
        project(self.base_signatures(), |index| {
            self.components(index).unwrap()
        })
    }
}

impl FixtureCoordinates for DifferentiableCotangent {
    fn fixture_coordinates(&self) -> Vec<f64> {
        project(self.base_signatures(), |index| {
            self.components(index).unwrap()
        })
    }
}

fn project<I: Iterator<Item = (f64, f64)>>(
    signatures: &[DifferentiableValueSignature],
    components: impl Fn(usize) -> I,
) -> Vec<f64> {
    let mut coordinates = Vec::new();
    for (index, signature) in signatures.iter().enumerate() {
        for (re, im) in components(index) {
            coordinates.push(re);
            if signature.mathematical_type().value_type.scalar_domain()
                == eqiora::ScalarDomain::Complex
            {
                coordinates.push(im);
            }
        }
    }
    coordinates
}

/// Bind unchanged numeric map oracles to the exact shared/mapped base products.
pub(crate) fn mapped_jvp(
    view: &eqiora::api::EvaluationMapProducts<'_>,
    shared: &[f64],
    mapped: &[f64],
    seeds: usize,
) -> Result<eqiora::api::EvaluationMapJvp, eqiora::Diagnostic> {
    let plan = view.map().plan();
    let inputs = plan.program_identity().inputs();
    let shared_signatures = plan
        .shared_inputs()
        .iter()
        .map(|id| {
            inputs
                .iter()
                .find(|s| s.entity() == id.erase())
                .unwrap()
                .clone()
        })
        .collect::<Vec<_>>();
    let mapped_signatures = inputs
        .iter()
        .filter(|s| {
            !plan
                .shared_inputs()
                .iter()
                .any(|id| id.erase() == s.entity())
        })
        .cloned()
        .collect::<Vec<_>>();
    let shared = mapped_rows(
        &shared_signatures,
        shared,
        seeds,
        DifferentiableTangent::from_components,
    )?;
    let mapped = mapped_rows(
        &mapped_signatures,
        mapped,
        view.shape().iter().product(),
        DifferentiableTangent::from_components,
    )?;
    view.jvp(&shared, &mapped)
}

pub(crate) fn mapped_vjp(
    view: &eqiora::api::EvaluationMapProducts<'_>,
    values: &[f64],
) -> Result<eqiora::api::EvaluationMapVjp, eqiora::Diagnostic> {
    let values = mapped_rows(
        view.map().plan().program_identity().outputs(),
        values,
        view.shape().iter().product(),
        DifferentiableCotangent::from_components,
    )?;
    view.vjp(&values)
}

fn mapped_rows<T>(
    signatures: &[DifferentiableValueSignature],
    values: &[f64],
    count: usize,
    bind: impl Fn(&[DifferentiableValueSignature], &[&[(f64, f64)]]) -> Result<T, eqiora::Diagnostic>,
) -> Result<Vec<T>, eqiora::Diagnostic> {
    let width = if count == 0 {
        assert!(values.is_empty());
        0
    } else {
        assert_eq!(values.len() % count, 0);
        values.len() / count
    };
    (0..count)
        .map(|i| {
            let leaves = components(signatures, &values[i * width..(i + 1) * width]);
            bind(
                signatures,
                &leaves.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            )
        })
        .collect()
}
