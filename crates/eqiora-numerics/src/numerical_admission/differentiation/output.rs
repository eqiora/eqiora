//! Ordered output blocks preserve indexed Field projections and dense reduced actions.
use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(in crate::numerical_admission) enum Output {
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

    pub(super) fn values(&self) -> &[f64] {
        match self {
            Self::Field(value) => value.values(),
            Self::Dense(value) => value.values(),
            Self::Sequence { values, .. } => values,
        }
    }
}

impl LinearizedOutput<f64> for Output {
    fn unknown_dimension(&self) -> usize {
        match self {
            Self::Field(value) => value.unknown_dimension(),
            Self::Dense(value) => value.unknown_dimension(),
            Self::Sequence { parts, .. } => parts[0].unknown_dimension(),
        }
    }
    fn parameter_dimension(&self) -> usize {
        match self {
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
        if output.len() != self.output_dimension() {
            return Err(invalid("output JVP shape mismatch"));
        }
        match self {
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
