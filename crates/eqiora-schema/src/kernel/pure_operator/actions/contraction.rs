//! Real adjoints contract closed output coordinates into the selected input axes.
use super::*;

pub(super) fn vjp(
    caller: &mut CalculusBuilder,
    value: &CalculusArgument,
    cotangent: &CalculusArgument,
    formal: u16,
) -> Result<CalculusNodeId, PureOperatorError> {
    let selected = caller
        .formals
        .get(usize::from(formal))
        .ok_or(PureOperatorError::InvalidFormal(formal))?
        .clone();
    let input = selected
        .exact_type()
        .cloned()
        .ok_or(PureOperatorError::FormalTypeMismatch)?;
    let output = value.value_type(caller)?;
    let seed_type = cotangent.value_type(caller)?;
    let expected = ExpressionType::<()>::new(output.clone(), None)
        .vjp(
            &ExpressionType::new(input, None),
            &ExpressionType::new(seed_type.clone(), None),
        )
        .map_err(action_type_error)?
        .value_type;
    let result = PureValueClass::from_value_type(expected.clone())?;
    if !result.is_invariant_scalar()
        && !composition::compatible_component_layout(&result, &caller.result)
    {
        return Err(PureOperatorError::FormalTypeMismatch);
    }
    let value = value.closed(caller)?;
    let seed = cotangent.closed(caller)?;
    let coordinates = coordinates(&output)?;
    let mut staged = CalculusBuilder {
        formals: caller.formals.clone(),
        result: caller.result.clone(),
        nodes: caller.nodes.clone(),
        depths: caller.depths.clone(),
    };
    let mut real = None;
    let mut imag = None;
    for axes in coordinates {
        let source = value.project(&mut staged, &axes)?;
        let seed = seed.project(&mut staged, &axes)?;
        let linear = Linearization::bind(&mut staged, source, &[(formal, selected.clone())])?;
        for coordinate in &linear.coordinates {
            let partial = staged.push(CalculusNode::Differentiated {
                source: linear.root,
                wrt: coordinate.value,
            })?;
            let mut term =
                pairing::component(&mut staged, seed, partial, seed_type.scalar_domain())?;
            // A projected source uses fixed input coordinates. Route each real
            // coordinate's contribution into its matching gradient component.
            for (axis, coordinate) in coordinate.axes.iter().enumerate() {
                let mask = staged.push(CalculusNode::KroneckerDelta(
                    ComponentIndex::Result(axis as u16),
                    *coordinate,
                ))?;
                term = staged.push(CalculusNode::Mul(mask, term))?;
            }
            let sum = if coordinate.part == Some(UnaryMathFunction::Imag) {
                &mut imag
            } else {
                &mut real
            };
            *sum = Some(match *sum {
                Some(prior) => staged.push(CalculusNode::Add(prior, term))?,
                None => term,
            });
        }
    }
    let real = real.ok_or(PureOperatorError::FormalTypeMismatch)?;
    let root = match imag {
        Some(imag) => staged.push(CalculusNode::Complex { real, imag })?,
        None => real,
    };
    let actual = staged.value_type(root)?;
    if actual.scalar_domain() != expected.scalar_domain()
        || actual.dimension() != expected.dimension()
    {
        return Err(PureOperatorError::FormalTypeMismatch);
    }
    *caller = staged;
    Ok(root)
}
