//! Method-neutral isotropic linear-elasticity primitives.

use crate::continuum_kinematics::symmetric_gradient_bilinear_entry;

/// One dimension-admitted isotropic small-strain constitutive state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct IsotropicElasticityMaterial<const D: usize> {
    shear_modulus: f64,
    first_lame_parameter: f64,
}

impl<const D: usize> IsotropicElasticityMaterial<D> {
    pub(crate) fn new(shear_modulus: f64, first_lame_parameter: f64) -> Option<Self> {
        if !(1..=3).contains(&D) {
            return None;
        }
        let volumetric_modulus = first_lame_parameter + 2.0 * shear_modulus / D as f64;
        (shear_modulus.is_finite()
            && shear_modulus > 0.0
            && first_lame_parameter.is_finite()
            && volumetric_modulus.is_finite()
            && volumetric_modulus > 0.0)
            .then_some(Self {
                shear_modulus,
                first_lame_parameter,
            })
    }

    pub(crate) const fn shear_modulus(self) -> f64 {
        self.shear_modulus
    }

    pub(crate) const fn first_lame_parameter(self) -> f64 {
        self.first_lame_parameter
    }

    pub(crate) fn stiffness_entry(
        self,
        row_gradient: &[f64],
        row_component: usize,
        column_gradient: &[f64],
        column_component: usize,
    ) -> f64 {
        self.shear_modulus
            * symmetric_gradient_bilinear_entry(
                row_gradient,
                row_component,
                column_gradient,
                column_component,
            )
            + self.first_lame_parameter
                * row_gradient[row_component]
                * column_gradient[column_component]
    }

    pub(crate) fn stress(self, strain: &[[f64; D]; D]) -> [[f64; D]; D] {
        let trace = tensor_trace(strain);
        std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                2.0 * self.shear_modulus * strain[row][column]
                    + if row == column {
                        self.first_lame_parameter * trace
                    } else {
                        0.0
                    }
            })
        })
    }

    pub(crate) fn strain_energy_density(self, strain: &[[f64; D]; D]) -> f64 {
        let squared_norm = strain
            .iter()
            .flatten()
            .map(|value| value * value)
            .sum::<f64>();
        let trace = tensor_trace(strain);
        self.shear_modulus * squared_norm + 0.5 * self.first_lame_parameter * trace * trace
    }
}

fn tensor_trace<const D: usize>(tensor: &[[f64; D]; D]) -> f64 {
    (0..D).map(|axis| tensor[axis][axis]).sum()
}

#[cfg(test)]
mod tests {
    use super::IsotropicElasticityMaterial;
    use crate::continuum_kinematics::symmetric_gradient;

    #[test]
    fn coercivity_uses_the_admitted_spatial_dimension_and_rejects_overflow() {
        assert!(IsotropicElasticityMaterial::<1>::new(2.0, -3.0).is_some());
        assert!(IsotropicElasticityMaterial::<2>::new(2.0, -3.0).is_none());
        assert!(IsotropicElasticityMaterial::<3>::new(2.0, -3.0).is_none());
        assert!(IsotropicElasticityMaterial::<4>::new(2.0, 1.0).is_none());
        assert!(IsotropicElasticityMaterial::<2>::new(f64::MAX, f64::MAX).is_none());
    }

    #[test]
    fn isotropic_state_uses_symmetric_strain_in_every_dimension() {
        let gradient = [[1.0, 2.0, 3.0], [5.0, 7.0, 11.0], [13.0, 17.0, 19.0]];
        let strain = symmetric_gradient(&gradient);
        let material = IsotropicElasticityMaterial::<3>::new(2.0, 3.0).unwrap();
        let stress = material.stress(&strain);

        assert_eq!(
            strain,
            [[1.0, 3.5, 8.0], [3.5, 7.0, 14.0], [8.0, 14.0, 19.0]]
        );
        assert_eq!(stress[0][1], 14.0);
        assert_eq!(stress[1][0], stress[0][1]);
        assert_eq!(stress[2][2], 4.0 * 19.0 + 3.0 * 27.0);
        assert_eq!(
            material.strain_energy_density(&strain),
            2.0 * strain
                .iter()
                .flatten()
                .map(|value| value * value)
                .sum::<f64>()
                + 1.5 * 27.0_f64.powi(2)
        );
    }
}
