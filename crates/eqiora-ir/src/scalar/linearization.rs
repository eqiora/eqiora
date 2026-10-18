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
        require_length(
            residual_tangent,
            self.residual_dimension(),
            "residual tangent",
        )?;
        let values = evaluate_instructions(&self.instructions, &self.inputs)?;
        let active = self.derivative_demand(&values)?;
        let tangents = self.forward(
            tangent,
            &values,
            &active,
            &mut map_evaluation::MapCache::default(),
            None,
        )?;
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
        let active = self.derivative_demand(&values)?;
        let mut adjoints = vec![Scaled::ZERO; self.instructions.len()];
        let mut unknown_sums = vec![Scaled::ZERO; self.unknown_dimension];
        let mut parameter_sums = vec![Scaled::ZERO; self.parameter_dimension];
        let mut maps = map_evaluation::MapCache::default();
        for (root, seed) in self.roots.iter().zip(residual_cotangent) {
            accumulate(
                &mut adjoints,
                *root,
                Scaled::new(*seed),
                self.instructions.len(),
            )?;
        }

        for (index, instruction) in self.instructions.iter().enumerate().rev() {
            if !active[index] {
                continue;
            }
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
                Instruction::CoefficientJet { start, len, shape } => {
                    let operands =
                        coefficient_jet::numerical_operands(start, len, shape, &values[..index])?;
                    for operand in operands.iter().filter(|id| active[id.0 as usize]) {
                        let contribution = operands.first(&values, |id| {
                            Ok(if id == operand {
                                cotangent
                            } else {
                                Scaled::ZERO
                            })
                        })?;
                        accumulate(&mut adjoints, operand, contribution, index)?;
                    }
                }
                Instruction::MapInvariant {
                    start,
                    extent,
                    output,
                } => {
                    let range = map_evaluation::operand_range(start, extent, index)?;
                    let gradient = maps
                        .get(start, extent, &values[..index])?
                        .gradient(output, &|coordinate| {
                            active[range.start + coordinate] && !cotangent.is_zero()
                        })?;
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
                        if unknown_cotangent.is_some() {
                            accumulate_coordinate(
                                &mut unknown_sums,
                                coordinate,
                                cotangent,
                                "unknown VJP",
                            )?;
                        }
                    }
                    InputBinding::Parameter(coordinate) => {
                        if parameter_cotangent.is_some() {
                            accumulate_coordinate(
                                &mut parameter_sums,
                                coordinate,
                                cotangent,
                                "parameter VJP",
                            )?;
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
                    let numerator = Scaled::new(read(&values, left, index)?);
                    let denominator = Scaled::new(read(&values, right, index)?);
                    for (operand, da, db) in [
                        (left, Some(cotangent), None),
                        (right, None, Some(cotangent)),
                    ]
                    .into_iter()
                    .filter(|(operand, _, _)| active[operand.0 as usize])
                    {
                        let contribution = differential::quotient_direction(
                            &mut differential::Numerical,
                            numerator,
                            denominator,
                            da,
                            db,
                        )?
                        .expect("one quotient direction is present");
                        accumulate(&mut adjoints, operand, contribution, index)?;
                    }
                }
                Instruction::Math(function, value) => {
                    accumulate(
                        &mut adjoints,
                        value,
                        real_math::action_in(
                            &mut differential::Numerical,
                            function,
                            Scaled::new(read(&values, value, index)?),
                            cotangent,
                        )?,
                        index,
                    )?;
                }
                Instruction::PowI(base, exponent) => {
                    if exponent != 0 {
                        accumulate(
                            &mut adjoints,
                            base,
                            cotangent
                                * differential::power_slope(
                                    &mut differential::Numerical,
                                    Scaled::new(read(&values, base, index)?),
                                    exponent,
                                )?,
                            index,
                        )?;
                    }
                }
            }
        }
        for (destination, sums) in [
            (unknown_cotangent, unknown_sums),
            (parameter_cotangent, parameter_sums),
        ] {
            if let Some(destination) = destination {
                for (output, value) in destination.iter_mut().zip(sums) {
                    *output = value.value();
                }
                require_finite(destination, "VJP output")?;
            }
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
            .collect::<Vec<_>>();
        let active = activity::numerical_activity(instructions, &bindings)?;
        Ok(Self {
            instructions: std::borrow::Cow::Borrowed(instructions),
            roots: std::borrow::Cow::Borrowed(roots),
            inputs: inputs.to_vec(),
            bindings,
            active,
            unknown_dimension,
            parameter_dimension,
        })
    }
}

impl ScalarLinearization<'_> {
    pub(super) fn forward(
        &self,
        tangent: RelationTangent<'_, f64>,
        values: &[f64],
        active: &[bool],
        maps: &mut map_evaluation::MapCache,
        mixed: Option<(&[Scaled], &[Scaled])>,
    ) -> Result<Vec<Scaled>, Diagnostic> {
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

        let mut tangents = Vec::with_capacity(self.instructions.len());
        for (index, instruction) in self.instructions.iter().enumerate() {
            if !active[index] {
                tangents.push(Scaled::ZERO);
                continue;
            }
            let mut tangent = match *instruction {
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
                Instruction::Constant(_) => Scaled::ZERO,
                Instruction::ComplexDiv {
                    operands,
                    imaginary,
                } => complex_quotient::jvp(operands, values, &tangents, index, imaginary)?,
                Instruction::CoefficientJet { start, len, shape } => {
                    coefficient_jet::numerical_operands(start, len, shape, &values[..index])?
                        .first(values, |id| read(&tangents, id, index))?
                }
                Instruction::MapInvariant {
                    start,
                    extent,
                    output,
                } => {
                    let range = map_evaluation::operand_range(start, extent, index)?;
                    let gradient = maps
                        .get(start, extent, &values[..index])?
                        .gradient(output, &|coordinate| {
                            !tangents[range.start + coordinate].is_zero()
                        })?;
                    gradient
                        .iter()
                        .zip(&tangents[range])
                        .map(|(g, h)| *g * *h)
                        .sum()
                }
                Instruction::Read(slot) => match self.bindings[slot_index(slot, index)?] {
                    InputBinding::Unknown(coordinate) => unknown_tangent
                        .map_or(Scaled::ZERO, |values| Scaled::new(values[coordinate])),
                    InputBinding::Parameter(coordinate) => parameter_tangent
                        .map_or(Scaled::ZERO, |values| Scaled::new(values[coordinate])),
                    InputBinding::Frozen => Scaled::ZERO,
                },
                Instruction::Math(function, value) => real_math::action_in(
                    &mut differential::Numerical,
                    function,
                    Scaled::new(read(values, value, index)?),
                    read(&tangents, value, index)?,
                )?,
                Instruction::Neg(_)
                | Instruction::Add(..)
                | Instruction::Sub(..)
                | Instruction::Mul(..)
                | Instruction::Div(..)
                | Instruction::PowI(..) => differential::apply(
                    &mut differential::Numerical,
                    *instruction,
                    |id| read(values, id, index).map(Scaled::new),
                    |id| {
                        if active[id.0 as usize] {
                            read(&tangents, id, index).map(Some)
                        } else {
                            Ok(None)
                        }
                    },
                )?
                .unwrap_or(Scaled::ZERO),
            };
            if let Some((left, right)) = mixed {
                tangent += super::mixed::curvature(*instruction, values, left, right, index, maps)?;
            }
            require_valid_derivative(tangent, "JVP", index)?;
            tangents.push(tangent);
        }
        Ok(tangents)
    }
}
