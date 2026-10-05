//! Numerical primal/JVP/VJP over a point-bound active scalar program.
use super::*;

impl LinearizedRelation<f64> for ScalarLinearization<'_> {
    fn unknown_dimension(&self) -> usize {
        self.unknown_dimension
    }

    fn parameter_dimension(&self) -> usize {
        self.parameter_dimension
    }

    fn residual_dimension(&self) -> usize {
        self.roots.len()
    }

    fn primal(&self, residual: &mut [f64]) -> Result<(), Diagnostic> {
        require_length(residual, self.residual_dimension(), "primal residual")?;
        let values = evaluate_instructions(&self.instructions, &self.inputs)?;
        write_roots(&self.roots, &values, residual)
    }

    fn jvp(
        &self,
        tangent: RelationTangent<'_, f64>,
        residual_tangent: &mut [f64],
    ) -> Result<(), Diagnostic> {
        let (unknown_tangent, parameter_tangent) = match tangent {
            RelationTangent::Unknown(unknown) => (Some(unknown), None),
            RelationTangent::Parameter(parameter) => (None, Some(parameter)),
            RelationTangent::Both { unknown, parameter } => (Some(unknown), Some(parameter)),
        };
        if let Some(unknown) = unknown_tangent {
            require_length(unknown, self.unknown_dimension, "unknown tangent")?;
            require_finite(unknown, "unknown tangent")?;
        }
        if let Some(parameter) = parameter_tangent {
            require_length(parameter, self.parameter_dimension, "parameter tangent")?;
            require_finite(parameter, "parameter tangent")?;
        }
        require_length(
            residual_tangent,
            self.residual_dimension(),
            "residual tangent",
        )?;

        let values = evaluate_instructions(&self.instructions, &self.inputs)?;
        let mut tangents = Vec::with_capacity(self.instructions.len());
        let mut maps = map_evaluation::MapCache::default();
        for (index, instruction) in self.instructions.iter().enumerate() {
            let tangent = match *instruction {
                Instruction::Select { .. }
                | Instruction::Require { .. }
                | Instruction::PureOperator { .. }
                | Instruction::Compare(_, _, _)
                | Instruction::Not(_)
                | Instruction::And(_, _)
                | Instruction::Or(_, _)
                | Instruction::Array { .. }
                | Instruction::Index(_, _)
                | Instruction::TypedConstant(_)
                | Instruction::Quotient(_, _)
                | Instruction::Remainder(_, _)
                | Instruction::ToReal(_)
                | Instruction::ToInteger(_)
                | Instruction::Ordinal(_) => {
                    return Err(ir_builder_error(
                        "discrete operations cannot be differentiated",
                    ));
                }
                Instruction::Constant(_) => 0.0,
                Instruction::ComplexDiv {
                    operands,
                    imaginary,
                } => complex_quotient::jvp(operands, &values, &tangents, index, imaginary)?,
                Instruction::MapInvariant {
                    start,
                    extent,
                    component,
                } => {
                    let gradient = maps
                        .get(start, extent, &values[..index])?
                        .gradient(component)?;
                    let range = map_evaluation::operand_range(start, extent, index)?;
                    gradient
                        .iter()
                        .zip(&tangents[range])
                        .map(|(g, h)| g * h)
                        .sum()
                }
                Instruction::Read(slot) => match self.bindings[slot_index(slot, index)?] {
                    InputBinding::Unknown(coordinate) => {
                        unknown_tangent.map_or(0.0, |values| values[coordinate])
                    }
                    InputBinding::Parameter(coordinate) => {
                        parameter_tangent.map_or(0.0, |values| values[coordinate])
                    }
                    InputBinding::Frozen => 0.0,
                },
                Instruction::Exp(value) => {
                    read(&tangents, value, index)? * read(&values, value, index)?.exp()
                }
                Instruction::Sin(value) => {
                    read(&tangents, value, index)? * read(&values, value, index)?.cos()
                }
                Instruction::Sqrt(value) => {
                    let root = read(&values, value, index)?.sqrt();
                    if root == 0. {
                        return Err(ir_builder_error(
                            "square-root derivative is undefined at zero",
                        ));
                    }
                    read(&tangents, value, index)? / (2. * root)
                }
                Instruction::Neg(value) => -read(&tangents, value, index)?,
                Instruction::Add(left, right) => {
                    read(&tangents, left, index)? + read(&tangents, right, index)?
                }
                Instruction::Sub(left, right) => {
                    read(&tangents, left, index)? - read(&tangents, right, index)?
                }
                Instruction::Mul(left, right) => {
                    read(&tangents, left, index)? * read(&values, right, index)?
                        + read(&values, left, index)? * read(&tangents, right, index)?
                }
                Instruction::Div(left, right) => {
                    let denominator = read(&values, right, index)?;
                    (read(&tangents, left, index)? * denominator
                        - read(&values, left, index)? * read(&tangents, right, index)?)
                        / denominator.powi(2)
                }
                Instruction::PowI(base, exponent) => {
                    powi_derivative(read(&values, base, index)?, exponent)
                        * read(&tangents, base, index)?
                }
            };
            require_finite_value(tangent, "JVP", index)?;
            tangents.push(tangent);
        }
        write_roots(&self.roots, &tangents, residual_tangent)
    }

    fn vjp(
        &self,
        residual_cotangent: &[f64],
        cotangent: RelationCotangent<'_, f64>,
    ) -> Result<(), Diagnostic> {
        let (mut unknown_cotangent, mut parameter_cotangent) = match cotangent {
            RelationCotangent::Unknown(unknown) => (Some(unknown), None),
            RelationCotangent::Parameter(parameter) => (None, Some(parameter)),
            RelationCotangent::Both { unknown, parameter } => (Some(unknown), Some(parameter)),
        };
        require_length(
            residual_cotangent,
            self.residual_dimension(),
            "residual cotangent",
        )?;
        if let Some(unknown) = unknown_cotangent.as_deref_mut() {
            require_length(unknown, self.unknown_dimension, "unknown cotangent")?;
            unknown.fill(0.0);
        }
        if let Some(parameter) = parameter_cotangent.as_deref_mut() {
            require_length(parameter, self.parameter_dimension, "parameter cotangent")?;
            parameter.fill(0.0);
        }
        require_finite(residual_cotangent, "residual cotangent")?;

        let values = evaluate_instructions(&self.instructions, &self.inputs)?;
        let mut adjoints = vec![0.0; self.instructions.len()];
        let mut maps = map_evaluation::MapCache::default();
        for (root, seed) in self.roots.iter().zip(residual_cotangent) {
            accumulate(&mut adjoints, *root, *seed, self.instructions.len())?;
        }

        for (index, instruction) in self.instructions.iter().enumerate().rev() {
            let cotangent = adjoints[index];
            match *instruction {
                Instruction::Select { .. }
                | Instruction::Require { .. }
                | Instruction::PureOperator { .. }
                | Instruction::Compare(_, _, _)
                | Instruction::Not(_)
                | Instruction::And(_, _)
                | Instruction::Or(_, _)
                | Instruction::Array { .. }
                | Instruction::Index(_, _)
                | Instruction::TypedConstant(_)
                | Instruction::Quotient(_, _)
                | Instruction::Remainder(_, _)
                | Instruction::ToReal(_)
                | Instruction::ToInteger(_)
                | Instruction::Ordinal(_) => {
                    return Err(ir_builder_error(
                        "discrete operations cannot be differentiated",
                    ));
                }
                Instruction::Constant(_) => {}
                Instruction::ComplexDiv {
                    operands,
                    imaginary,
                } => {
                    let contributions =
                        complex_quotient::vjp(operands, &values, index, imaginary, cotangent)?;
                    for (operand, contribution) in operands.into_iter().zip(contributions) {
                        accumulate(&mut adjoints, operand, contribution, index)?;
                    }
                }
                Instruction::MapInvariant {
                    start,
                    extent,
                    component,
                } => {
                    let gradient = maps
                        .get(start, extent, &values[..index])?
                        .gradient(component)?;
                    let range = map_evaluation::operand_range(start, extent, index)?;
                    for (coordinate, g) in range.zip(gradient) {
                        accumulate(
                            &mut adjoints,
                            ValueId(coordinate as u32),
                            cotangent * g,
                            index,
                        )?;
                    }
                }
                Instruction::Read(slot) => match self.bindings[slot_index(slot, index)?] {
                    InputBinding::Unknown(coordinate) => {
                        if let Some(values) = unknown_cotangent.as_deref_mut() {
                            accumulate_coordinate(values, coordinate, cotangent, "unknown VJP")?;
                        }
                    }
                    InputBinding::Parameter(coordinate) => {
                        if let Some(values) = parameter_cotangent.as_deref_mut() {
                            accumulate_coordinate(values, coordinate, cotangent, "parameter VJP")?;
                        }
                    }
                    InputBinding::Frozen => {}
                },
                Instruction::Neg(value) => {
                    accumulate(&mut adjoints, value, -cotangent, index)?;
                }
                Instruction::Add(left, right) => {
                    accumulate(&mut adjoints, left, cotangent, index)?;
                    accumulate(&mut adjoints, right, cotangent, index)?;
                }
                Instruction::Sub(left, right) => {
                    accumulate(&mut adjoints, left, cotangent, index)?;
                    accumulate(&mut adjoints, right, -cotangent, index)?;
                }
                Instruction::Mul(left, right) => {
                    accumulate(
                        &mut adjoints,
                        left,
                        cotangent * read(&values, right, index)?,
                        index,
                    )?;
                    accumulate(
                        &mut adjoints,
                        right,
                        cotangent * read(&values, left, index)?,
                        index,
                    )?;
                }
                Instruction::Div(left, right) => {
                    let denominator = read(&values, right, index)?;
                    accumulate(&mut adjoints, left, cotangent / denominator, index)?;
                    accumulate(
                        &mut adjoints,
                        right,
                        -cotangent * read(&values, left, index)? / denominator.powi(2),
                        index,
                    )?;
                }
                Instruction::Exp(value) => {
                    accumulate(
                        &mut adjoints,
                        value,
                        cotangent * read(&values, value, index)?.exp(),
                        index,
                    )?;
                }
                Instruction::Sin(value) => {
                    accumulate(
                        &mut adjoints,
                        value,
                        cotangent * read(&values, value, index)?.cos(),
                        index,
                    )?;
                }
                Instruction::Sqrt(value) => {
                    let root = read(&values, value, index)?.sqrt();
                    if root == 0. {
                        return Err(ir_builder_error(
                            "square-root derivative is undefined at zero",
                        ));
                    }
                    accumulate(&mut adjoints, value, cotangent / (2. * root), index)?;
                }
                Instruction::PowI(base, exponent) => {
                    if exponent != 0 {
                        accumulate(
                            &mut adjoints,
                            base,
                            cotangent * powi_derivative(read(&values, base, index)?, exponent),
                            index,
                        )?;
                    }
                }
            }
        }
        if let Some(unknown) = unknown_cotangent.as_deref() {
            require_finite(unknown, "unknown VJP")?;
        }
        if let Some(parameter) = parameter_cotangent.as_deref() {
            require_finite(parameter, "parameter VJP")?;
        }
        Ok(())
    }
}

impl<'a> ScalarLinearization<'a> {
    pub(super) fn bind(
        instructions: &'a [Instruction],
        roots: &'a [ValueId],
        input_count: usize,
        inputs: &[f64],
        roles: &[DifferentiationRole],
    ) -> Result<Self, Diagnostic> {
        validate_linearization_inputs(instructions, input_count, inputs, roles)?;
        let mut unknown_dimension = 0;
        let mut parameter_dimension = 0;
        let bindings = roles
            .iter()
            .map(|role| match role {
                DifferentiationRole::Unknown => {
                    let coordinate = unknown_dimension;
                    unknown_dimension += 1;
                    InputBinding::Unknown(coordinate)
                }
                DifferentiationRole::Parameter => {
                    let coordinate = parameter_dimension;
                    parameter_dimension += 1;
                    InputBinding::Parameter(coordinate)
                }
                DifferentiationRole::Frozen => InputBinding::Frozen,
            })
            .collect();
        Ok(Self {
            instructions: std::borrow::Cow::Borrowed(instructions),
            roots: std::borrow::Cow::Borrowed(roots),
            inputs: inputs.to_vec(),
            bindings,
            unknown_dimension,
            parameter_dimension,
        })
    }
}
