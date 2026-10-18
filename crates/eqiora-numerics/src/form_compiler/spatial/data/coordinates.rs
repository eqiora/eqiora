//! Real-coordinate images of a retained affine coefficient action.
use super::*;
use num_complex::Complex64;

impl Data<Complex64> {
    /// Preserve the complete real-linear action instead of imposing C-linearity.
    /// Scalar multipliers act on each image; they are not themselves doubled.
    pub(in crate::form_compiler) fn coordinate_images(
        &self,
        dimension: usize,
    ) -> Result<[Self; 2], Diagnostic> {
        Ok(self
            .affine_images(dimension)?
            .unwrap_or_else(|| self.scalar_images(dimension)))
    }

    fn scalar_images(&self, dimension: usize) -> [Self; 2] {
        [
            self.clone(),
            self.clone()
                .multiply(Self::constant(dimension, Complex64::new(0., 1.))),
        ]
    }

    fn affine_images(&self, dimension: usize) -> Result<Option<[Self; 2]>, Diagnostic> {
        let images = |data: &Self| data.affine_images(dimension);
        Ok(match self.0.as_ref() {
            Node::Affine(data) => data
                .coordinate_images()
                .map(|images| images.map(|data| Self(Arc::new(Node::Affine(data))))),
            Node::Add(a, b) => match (images(a)?, images(b)?) {
                (None, None) => None,
                (a_images, b_images) => {
                    let [ar, ai] = a_images.unwrap_or_else(|| a.scalar_images(dimension));
                    let [br, bi] = b_images.unwrap_or_else(|| b.scalar_images(dimension));
                    Some([ar.add(br), ai.add(bi)])
                }
            },
            Node::Mul(a, b) => match (images(a)?, images(b)?) {
                (None, None) => None,
                (Some(values), None) => Some(values.map(|value| value.multiply(b.clone()))),
                (None, Some(values)) => Some(values.map(|value| a.clone().multiply(value))),
                (Some(_), Some(_)) => {
                    return Err(super::super::invalid(
                        "a spatial coefficient product contains two retained trial actions",
                    ));
                }
            },
            Node::Div(a, b) => {
                if images(b)?.is_some() {
                    return Err(super::super::invalid(
                        "a spatial coefficient denominator contains a retained trial action",
                    ));
                }
                images(a)?.map(|values| values.map(|value| value.divide(b.clone())))
            }
            Node::Chart(data, chart) => {
                images(data)?.map(|values| values.map(|value| value.on_uniform_chart(chart)))
            }
            Node::Derivative(primal, tangent) => match (images(primal)?, images(tangent)?) {
                (None, None) => None,
                (primal_images, tangent_images) => {
                    let [pr, pi] = primal_images.unwrap_or_else(|| primal.scalar_images(dimension));
                    let [tr, ti] =
                        tangent_images.unwrap_or_else(|| tangent.scalar_images(dimension));
                    Some([
                        Self(Arc::new(Node::Derivative(pr, tr))),
                        Self(Arc::new(Node::Derivative(pi, ti))),
                    ])
                }
            },
            _ => None,
        })
    }
}
