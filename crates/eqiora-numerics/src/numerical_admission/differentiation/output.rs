//! Ordered output blocks preserve indexed Field projections and dense reduced actions.
use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(in crate::numerical_admission) enum Output {
    Empty { unknowns: usize, parameters: usize },
    Field(FieldCoefficientLinearization),
    Dense(DenseOutputLinearization),
    Sequence { parts: Vec<Self>, values: Vec<f64> },
}

impl Output {
    pub(in crate::numerical_admission) fn sequence(
        mut parts: Vec<Self>,
    ) -> Result<Self, Diagnostic> {
        let first = parts
            .first()
            .ok_or_else(|| invalid("output selection is empty"))?;
        if parts.iter().any(|part| {
            part.unknown_dimension() != first.unknown_dimension()
                || part.parameter_dimension() != first.parameter_dimension()
        }) {
            return Err(invalid("output blocks have inconsistent input coordinates"));
        }
        if parts.len() == 1 {
            return Ok(parts.pop().expect("one output"));
        }
        let values = parts
            .iter()
            .flat_map(|part| part.values().iter().copied())
            .collect();
        Ok(Self::Sequence { parts, values })
    }

    /// Compose a physical-coordinate selection into the stored partial actions.
    /// Field projections stay indexed; only already-dense Observable rows expand.
    pub(in crate::numerical_admission) fn project_unknowns(
        self,
        dimension: usize,
        coordinates: &[(usize, f64)],
    ) -> Result<Self, Diagnostic> {
        if coordinates.len() != self.unknown_dimension()
            || coordinates
                .iter()
                .any(|&(index, weight)| index >= dimension || !weight.is_finite())
        {
            return Err(invalid("output projection has a foreign unknown layout"));
        }
        match self {
            Self::Empty { parameters, .. } => Ok(Self::Empty {
                unknowns: dimension,
                parameters,
            }),
            Self::Field(value) => Ok(Self::Field(value.project_unknowns(dimension, coordinates)?)),
            Self::Dense(value) => {
                let size = value
                    .output_dimension()
                    .checked_mul(dimension)
                    .ok_or_else(|| invalid("projected output Jacobian shape overflows"))?;
                let mut jacobian = vec![0.; size];
                for row in 0..value.output_dimension() {
                    for (column, &(joint, weight)) in coordinates.iter().enumerate() {
                        jacobian[row * dimension + joint] +=
                            weight * value.unknown_jacobian()[row * coordinates.len() + column];
                    }
                }
                Ok(Self::Dense(DenseOutputLinearization::new(
                    value.values().to_vec(),
                    jacobian,
                    value.parameter_jacobian().to_vec(),
                )?))
            }
            Self::Sequence { parts, values } => Ok(Self::Sequence {
                parts: parts
                    .into_iter()
                    .map(|part| part.project_unknowns(dimension, coordinates))
                    .collect::<Result<Vec<_>, _>>()?,
                values,
            }),
        }
    }

    /// Materialize only an already assembled state action, retaining sparse Field rows.
    pub(in crate::numerical_admission) fn state_jacobian(
        &self,
    ) -> Result<Option<eqiora_assembly::CsrMatrix<f64>>, Diagnostic> {
        if self.output_dimension() == 0 {
            return Ok(None);
        }
        let mut rows = Vec::new();
        self.state_rows(&mut rows);
        let mut offsets = vec![0];
        let mut columns = Vec::new();
        let mut values = Vec::new();
        for row in rows {
            for (column, value) in row {
                if value != 0. {
                    columns.push(column);
                    values.push(value);
                }
            }
            offsets.push(values.len());
        }
        eqiora_assembly::CsrMatrix::from_sorted_csr(
            self.output_dimension(),
            self.unknown_dimension(),
            offsets,
            columns,
            values,
        )
        .map(Some)
    }

    fn state_rows(&self, rows: &mut Vec<Vec<(usize, f64)>>) {
        match self {
            Self::Empty { .. } => {}
            Self::Field(value) => rows.extend(
                value
                    .state_entries()
                    .map(|entry| entry.into_iter().collect()),
            ),
            Self::Dense(value) => {
                for row in 0..value.output_dimension() {
                    let start = row * value.unknown_dimension();
                    rows.push(
                        value.unknown_jacobian()[start..start + value.unknown_dimension()]
                            .iter()
                            .copied()
                            .enumerate()
                            .collect(),
                    );
                }
            }
            Self::Sequence { parts, .. } => {
                for part in parts {
                    part.state_rows(rows);
                }
            }
        }
    }

    pub(super) fn values(&self) -> &[f64] {
        match self {
            Self::Empty { .. } => &[],
            Self::Field(value) => value.values(),
            Self::Dense(value) => value.values(),
            Self::Sequence { values, .. } => values,
        }
    }
}

impl LinearizedOutput<f64> for Output {
    fn unknown_dimension(&self) -> usize {
        match self {
            Self::Empty { unknowns, .. } => *unknowns,
            Self::Field(value) => value.unknown_dimension(),
            Self::Dense(value) => value.unknown_dimension(),
            Self::Sequence { parts, .. } => parts[0].unknown_dimension(),
        }
    }
    fn parameter_dimension(&self) -> usize {
        match self {
            Self::Empty { parameters, .. } => *parameters,
            Self::Field(value) => value.parameter_dimension(),
            Self::Dense(value) => value.parameter_dimension(),
            Self::Sequence { parts, .. } => parts[0].parameter_dimension(),
        }
    }
    fn output_dimension(&self) -> usize {
        self.values().len()
    }
    fn primal(&self, output: &mut [f64]) -> Result<(), Diagnostic> {
        if output.len() != self.output_dimension() {
            return Err(invalid("output primal shape mismatch"));
        }
        output.copy_from_slice(self.values());
        Ok(())
    }
    fn jvp(
        &self,
        unknown: &[f64],
        parameter: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if output.len() != self.output_dimension()
            || unknown.len() != self.unknown_dimension()
            || parameter.len() != self.parameter_dimension()
            || unknown
                .iter()
                .chain(parameter)
                .any(|value| !value.is_finite())
        {
            return Err(invalid("output JVP shape or value mismatch"));
        }
        match self {
            Self::Empty { .. } => Ok(()),
            Self::Field(value) => value.jvp(unknown, parameter, output),
            Self::Dense(value) => value.jvp(unknown, parameter, output),
            Self::Sequence { parts, .. } => {
                let mut rest = output;
                for part in parts {
                    let (target, tail) = rest.split_at_mut(part.output_dimension());
                    part.jvp(unknown, parameter, target)?;
                    rest = tail;
                }
                Ok(())
            }
        }
    }
    fn vjp(
        &self,
        output: &[f64],
        unknown: &mut [f64],
        parameter: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if output.len() != self.output_dimension()
            || unknown.len() != self.unknown_dimension()
            || parameter.len() != self.parameter_dimension()
        {
            return Err(invalid("output VJP shape mismatch"));
        }
        match self {
            Self::Empty { .. } => {
                unknown.fill(0.);
                parameter.fill(0.);
                Ok(())
            }
            Self::Field(value) => value.vjp(output, unknown, parameter),
            Self::Dense(value) => value.vjp(output, unknown, parameter),
            Self::Sequence { parts, .. } => {
                unknown.fill(0.);
                parameter.fill(0.);
                let mut local_unknown = vec![0.; unknown.len()];
                let mut local_parameter = vec![0.; parameter.len()];
                let mut rest = output;
                for part in parts {
                    let (source, tail) = rest.split_at(part.output_dimension());
                    part.vjp(source, &mut local_unknown, &mut local_parameter)?;
                    for (target, local) in unknown
                        .iter_mut()
                        .chain(parameter.iter_mut())
                        .zip(local_unknown.iter().chain(&local_parameter))
                    {
                        *target += local;
                        if !target.is_finite() {
                            return Err(invalid("output VJP is non-finite"));
                        }
                    }
                    rest = tail;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_output_projection_preserves_direct_inputs_and_dual_actions() {
        // Physical u=x0, z=x3, eliminated trace g=2p. Observable u²+z+3p.
        let field = FieldCoefficientLinearization::new(
            vec![1., 4., 2.],
            2,
            1,
            vec![Some(0), Some(1), None],
            vec![0., 0., 2.],
        )
        .unwrap();
        let observable = DenseOutputLinearization::new(vec![8.], vec![2., 1.], vec![3.]).unwrap();
        let source =
            Output::sequence(vec![Output::Field(field), Output::Dense(observable)]).unwrap();
        let output = source
            .clone()
            .project_unknowns(4, &[(0, 1.), (3, 1.)])
            .unwrap();
        assert_eq!(output.values(), &[1., 4., 2., 8.]);
        let mut tangent = [0.; 4];
        output
            .jvp(&[5., 101., -73., 7.], &[11.], &mut tangent)
            .unwrap();
        assert_eq!(tangent, [5., 7., 22., 50.]);
        let mut unknown = [0.; 4];
        let mut parameter = [0.];
        output
            .vjp(&[2., 3., 5., 7.], &mut unknown, &mut parameter)
            .unwrap();
        assert_eq!(unknown, [16., 0., 0., 10.]);
        assert_eq!(parameter, [31.]);
        assert!(source.clone().project_unknowns(4, &[(0, 1.)]).is_err());
        assert!(
            source
                .clone()
                .project_unknowns(4, &[(0, 1.), (4, 1.)])
                .is_err()
        );
        let interpolated = source
            .clone()
            .project_unknowns(4, &[(0, 0.25), (3, 1.)])
            .unwrap();
        interpolated
            .jvp(&[5., 101., -73., 7.], &[11.], &mut tangent)
            .unwrap();
        assert_eq!(tangent, [1.25, 7., 22., 42.5]);
        interpolated
            .vjp(&[2., 3., 5., 7.], &mut unknown, &mut parameter)
            .unwrap();
        assert_eq!(unknown, [4., 0., 0., 10.]);
        assert_eq!(parameter, [31.]);
        let history = source
            .clone()
            .project_unknowns(2, &[(0, 0.75), (1, 0.)])
            .unwrap()
            .state_jacobian()
            .unwrap()
            .unwrap();
        use eqiora_solver::LinearOperator;
        history.apply(&[5., 7.], &mut tangent).unwrap();
        assert_eq!(tangent, [3.75, 0., 0., 7.5]);
        // A repeated input selection sums both Observable chain contributions.
        let shared = source.project_unknowns(1, &[(0, 1.), (0, 1.)]).unwrap();
        shared.jvp(&[5.], &[11.], &mut tangent).unwrap();
        assert_eq!(tangent, [5., 5., 22., 48.]);
    }
    #[test]
    fn unobserved_step_has_empty_outputs_but_retains_input_layout() {
        let output = Output::Empty {
            unknowns: 3,
            parameters: 2,
        };
        assert!(output.state_jacobian().unwrap().is_none());
        output.jvp(&[1., 2., 3.], &[4., 5.], &mut []).unwrap();
        assert!(output.jvp(&[1., 2.], &[4., 5.], &mut []).is_err());
        let mut unknown = [1.; 3];
        let mut parameter = [1.; 2];
        output.vjp(&[], &mut unknown, &mut parameter).unwrap();
        assert_eq!(unknown, [0.; 3]);
        assert_eq!(parameter, [0.; 2]);
    }
}
