//! Previous physical inventory projected through the same scalar coordinates.
use super::*;

impl<S: Coefficient> BoundRegionForm<S> {
    pub(super) fn history_matrix<P: Projection<S>>(
        &self,
        field: RawId,
        geometry: &AffineGeometryMap,
        direction: Option<&AffineGeometryLinearization>,
        quadrature: &QuadratureRule,
    ) -> Result<(usize, super::super::integration::IntegralActions<P::Scalar>), Diagnostic> {
        let fields = super::super::coordinates::fields::<S, P>(&self.fields)?;
        let mut integrals = Vec::new();
        let mut data = Vec::new();
        let mut column = None;
        for (row, source) in self.form.rows.iter().enumerate() {
            for term in &source.terms {
                if term.trial != field {
                    continue;
                }
                let eliminated = self.eliminations.get(&term.trial);
                let scale = match (eliminated, term.derivative) {
                    (Some(_), false) => -1.,
                    (None, true) => self.time_step().expect("bound derivative step").recip(),
                    _ => continue,
                };
                let trial = eliminated.copied().unwrap_or(term.trial);
                let index = self
                    .fields
                    .iter()
                    .position(|layout| layout.field == trial)
                    .ok_or_else(|| invalid("history lacks an algebraic binding"))?;
                if column.is_some_and(|column| column != index) {
                    return Err(invalid("history has conflicting algebraic bindings"));
                }
                column = Some(index);
                for (trial_coordinate, source) in P::columns(
                    &term.coefficient,
                    self.form.dimension,
                    fields[index].scalar_coordinates,
                )?
                .into_iter()
                .enumerate()
                {
                    for test_coordinate in 0..fields[row].scalar_coordinates {
                        integrals.push(super::super::integration::IntegralTerm {
                            test_coordinate,
                            trial_coordinate,
                            row,
                            column: index,
                            pairing: term.pairing,
                            trial_scale: scale,
                        });
                        data.push((row, source.clone(), test_coordinate));
                    }
                }
            }
        }
        let column =
            column.ok_or_else(|| invalid("consumed history lacks a mathematical action"))?;
        let (matrix, tangent) = super::super::integration::integrate_with_geometry_direction(
            self.reference,
            &fields,
            &integrals,
            geometry,
            direction,
            quadrature,
            |_, physical, physical_tangent, coefficients, _, _, coefficient_tangents, _, _| {
                for (index, (value, (row, coefficient, coordinate))) in
                    coefficients.iter_mut().zip(&data).enumerate()
                {
                    let (primal, tangent) = super::super::evaluate::evaluate_data(
                        coefficient,
                        physical,
                        direction.map(|_| physical_tangent),
                    )?;
                    let scale = <S as From<f64>>::from(self.row_multipliers[*row]);
                    *value = P::component(scale * primal, *coordinate);
                    if let Some(output) = coefficient_tangents.get_mut(index) {
                        *output = P::component(scale * tangent, *coordinate);
                    }
                }
                Ok(())
            },
        )?;
        Ok((column, (matrix, tangent)))
    }
}
