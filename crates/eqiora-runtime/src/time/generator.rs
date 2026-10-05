//! Structurally admitted constant generators reuse the ordinary physical rate action.
use super::*;

impl FirstOrderProgram {
    /// Prove a homogeneous autonomous linear system and return its row-major generator.
    ///
    /// Parameters retain their bound Model values. Time remains symbolic during
    /// affine admission, and every constant offset and time coefficient must be
    /// zero. Only after this proof are columns evaluated through the common rate
    /// action, including its full-rank mass solve. Coordinate order is exactly
    /// [`Self::state_coordinates`]; no sampled values establish linearity.
    ///
    /// # Errors
    /// Rejects nonlinear or time-dependent coefficients, forcing, unbound inputs,
    /// singular mass systems, and nonfinite coefficient evaluation.
    pub fn constant_generator(&self) -> Result<Vec<f64>, Diagnostic> {
        let mut selected = Vec::new();
        let mut bindings = Vec::new();
        let mut time_column = None;
        for (coordinate, binding) in self.operator.symbols().iter().zip(&self.bindings) {
            match binding {
                TimeBinding::State(_) => selected.push(coordinate.clone()),
                TimeBinding::Time => {
                    time_column = Some(selected.len());
                    selected.push(coordinate.clone());
                }
                TimeBinding::DerivativeZero => bindings.push((coordinate.clone(), 0.)),
                TimeBinding::Parameter(index) => {
                    bindings.push((coordinate.clone(), self.parameter_values[*index]))
                }
            }
        }
        self.operator.require_homogeneous_constant(
            self.relation,
            &selected,
            &bindings,
            time_column,
        )?;
        // Initial values do not participate in an operator identity. The zero
        // vector is used only to construct the existing algebraic rate action.
        let n = self.dimension();
        let problem = TimeProblem::new(
            self,
            self.equation_class(),
            InitialConditionPolicy::Provided,
            vec![0.; n],
        )?;
        let size = n
            .checked_mul(n)
            .ok_or_else(|| invalid_time(self.relation, "constant generator size overflows"))?;
        let mut result = vec![0.; size];
        let mut basis = vec![0.; n];
        for column in 0..n {
            basis[column] = 1.;
            let image = problem.rate(0., &basis)?;
            for row in 0..n {
                result[row * n + column] = image[row];
            }
            basis[column] = 0.;
        }
        Ok(result)
    }
}
