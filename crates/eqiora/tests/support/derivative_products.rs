//! Transport the existing coordinate oracles through the public typed boundary.

use eqiora::api::{
    DifferentiableCotangent, DifferentiableProgramIdentity, DifferentiableTangent,
    DifferentiableValueSignature,
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
