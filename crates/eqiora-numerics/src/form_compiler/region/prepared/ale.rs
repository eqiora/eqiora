//! Apply an endpoint skew/GCL numerical pullback to an authored transport row.
use super::*;
use crate::form_compiler::bilinear::Pairing;
use crate::form_compiler::region::integration::{
    IntegralField, IntegralTerm, integrate_with_geometry_direction,
};
use eqiora_meshing::FixedTopologyGeometryAction;

impl BoundRegionForm<f64> {
    /// The caller selects the transported Fields from its authenticated ALE Plan.
    /// Storage and convection must share the same coefficient. Mesh velocity and
    /// its correction come only from the sealed consecutive geometry action.
    pub(crate) fn prepare_gcl_cell<const D: usize>(
        &self,
        selected_fields: &[RawId],
        motion: &FixedTopologyGeometryAction<D>,
        cell: usize,
        direction: Option<&AffineGeometryLinearization>,
        quadrature: &QuadratureRule,
    ) -> Result<CellActions<f64>, Diagnostic> {
        if D != self.form.dimension
            || self
                .time_step()
                .is_some_and(|step| step != motion.time_step())
        {
            return Err(invalid(
                "ALE pullback differs from its region dimension or time step",
            ));
        }
        let geometry = motion
            .cell(cell)
            .ok_or_else(|| invalid("ALE pullback has a foreign cell"))?;
        let map = geometry.current_map();
        let (mut primal, mut tangent) = self.prepare_cell_actions(
            map,
            map,
            direction.map(|direction| (direction, direction)),
            quadrature,
        )?;
        let fields = self
            .fields
            .iter()
            .map(IntegralField::from)
            .collect::<Vec<_>>();
        let inverse = map.inverse_jacobian()?;
        let inverse = inverse.as_slice();
        let inverse_tangent = direction
            .map(|direction| direction.inverse_jacobian_tangent())
            .transpose()?;
        let divergence_tangent =
            if let (Some(direction), Some(inverse_tangent)) = (direction, &inverse_tangent) {
                (0..D)
                    .flat_map(|i| {
                        (0..D).map(move |j| {
                            direction.jacobian_tangent()[i * D + j] / motion.time_step()
                                * inverse[j * D + i]
                                + geometry.reference_velocity_gradient()[i * D + j]
                                    * inverse_tangent[j * D + i]
                        })
                    })
                    .sum::<f64>()
            } else {
                0.
            };
        let mut selected = std::collections::BTreeSet::new();
        for &field in selected_fields {
            if !selected.insert(field) {
                return Err(invalid("ALE pullback repeats a transported Field"));
            }
            let row = self
                .fields
                .iter()
                .position(|layout| layout.field == field)
                .ok_or_else(|| invalid("ALE pullback has a foreign transported Field"))?;
            let source = &self.form.rows[row];
            let storage = source
                .terms
                .iter()
                .filter(|term| term.derivative)
                .collect::<Vec<_>>();
            let [storage] = storage.as_slice() else {
                return Err(invalid(
                    "ALE pullback requires one transported storage term",
                ));
            };
            let [transport] = source.dyadics.as_slice() else {
                return Err(invalid("ALE pullback requires one transported dyadic term"));
            };
            if storage.trial != field
                || storage.pairing != Pairing::Value
                || transport.field != field
                || !transport.split
                || self.eliminations.contains_key(&field)
                || !transport
                    .coefficient
                    .clone()
                    .multiply(super::super::Data::constant(D, -1.))
                    .same_coefficient(&storage.coefficient)
            {
                return Err(invalid(
                    "ALE skew pullback requires matching storage and constrained dyadic transport",
                ));
            }
            let mut terms = (0..D)
                .map(|axis| IntegralTerm {
                    test_coordinate: 0,
                    trial_coordinate: 0,
                    row,
                    column: row,
                    pairing: Pairing::SkewGradient(axis),
                    trial_scale: self.fields[row].scale,
                })
                .collect::<Vec<_>>();
            terms.push(IntegralTerm {
                test_coordinate: 0,
                trial_coordinate: 0,
                row,
                column: row,
                pairing: Pairing::Value,
                trial_scale: self.fields[row].scale,
            });
            let (correction, correction_tangent) = integrate_with_geometry_direction(
                None,
                self.reference,
                &fields,
                &terms,
                (map, direction),
                quadrature,
                |reference, physical, physical_tangent, coefficients, _, _, tangents, _, _| {
                    let (density, density_tangent) = super::super::evaluate::evaluate_data(
                        &storage.coefficient,
                        physical,
                        direction.map(|_| physical_tangent),
                    )?;
                    let weight = 0.5 * self.row_multipliers[row];
                    let velocity = geometry.mesh_velocity(reference)?;
                    for axis in 0..D {
                        coefficients[axis] = weight * density * velocity[axis];
                        if let Some(tangent) = tangents.get_mut(axis) {
                            *tangent = weight
                                * (density_tangent * velocity[axis]
                                    + density * physical_tangent[axis] / motion.time_step());
                        }
                    }
                    coefficients[D] = weight * density * geometry.current_velocity_divergence();
                    if let Some(tangent) = tangents.get_mut(D) {
                        *tangent = weight
                            * (density_tangent * geometry.current_velocity_divergence()
                                + density * divergence_tangent);
                    }
                    Ok(())
                },
            )?;
            primal.add_affine(&correction)?;
            if let (Some(tangent), Some(correction)) = (&mut tangent, correction_tangent) {
                tangent.add_affine(&correction)?;
            }
        }
        Ok((primal, tangent))
    }
}

impl<S: Coefficient> PreparedRegionCell<S> {
    fn add_affine(&mut self, correction: &LocalContribution<S>) -> Result<(), Diagnostic> {
        if self.affine.rows() != correction.rows() || self.affine.columns() != correction.columns()
        {
            return Err(invalid(
                "numerical pullback differs from its local equation inventory",
            ));
        }
        self.independent_coordinates = false;
        *self.affine = LocalContribution::new(
            self.affine.rows(),
            self.affine.columns(),
            self.affine
                .matrix()
                .iter()
                .zip(correction.matrix())
                .map(|(a, b)| *a + *b)
                .collect(),
            self.affine
                .rhs()
                .iter()
                .zip(correction.rhs())
                .map(|(a, b)| *a + *b)
                .collect(),
        )?;
        Ok(())
    }
}
