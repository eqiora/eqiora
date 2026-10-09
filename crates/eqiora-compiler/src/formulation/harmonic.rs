//! Compile the request against original typed occurrences without reducing the Model.
use super::*;
use eqiora_schema::kernel::{ClockKind, SignalDirection};
use std::collections::BTreeSet;

#[allow(clippy::too_many_arguments)]
pub(super) fn compile(
    file: &str,
    name: &str,
    relation_names: &[String],
    binding: &eqiora_lang::FormulationBinding,
    range: TextRange,
    source_identity: AuthoredFormSourceIdentity,
    symbols: &ModelSymbols,
    index: &KernelIndex<'_>,
    geometry: Option<&eqiora_geometry::CanonicalGeometryV1>,
) -> Result<CompiledAuthoredFormulation, Diagnostic> {
    let eqiora_lang::FormulationBinding::Harmonic {
        angular_frequency,
        excitations,
        amplitudes,
    } = binding
    else {
        unreachable!("harmonic dispatch");
    };
    let invalid = |message| error(file, range, message);
    let mut relations = Vec::new();
    for name in relation_names {
        let id = resolve_symbol(file, range, name, symbols)?
            .downcast::<kinds::Relation>()
            .ok_or_else(|| invalid("harmonic owner is not an original Relation"))?;
        if relations.contains(&id) {
            return Err(invalid("harmonic Relations must be distinct"));
        }
        relations.push(id);
    }
    let actual_relations = index
        .nodes
        .values()
        .filter_map(|node| match node {
            KernelNode::Relation(relation) if !relation.is_initial() => Some(relation.id().erase()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    if actual_relations != relations.iter().map(|id| id.erase()).collect() {
        return Err(invalid(
            "harmonic selection must account for every original noninitial Relation",
        ));
    }
    let continuous = |id| {
        if let Some(clock) = index.clocked_by.get(&id)
            && !matches!(index.nodes.get(clock), Some(KernelNode::ClockDomain(clock)) if clock.kind()==ClockKind::Continuous)
        {
            return Err(invalid(
                "harmonic dependencies require continuous model time, not a periodic or event clock",
            ));
        }
        Ok(())
    };
    let mut names = BTreeSet::new();
    let mut trials = Vec::new();
    let mut mappings = Vec::new();
    for (amplitude, original) in amplitudes {
        let id = resolve_symbol(file, amplitude.range(), original, symbols)?;
        let Some(KernelNode::Field(field)) = index.nodes.get(&id).copied() else {
            return Err(invalid(
                "harmonic amplitude must map an original real Field",
            ));
        };
        continuous(id)?;
        let suffix = format!(".{}", amplitude.name());
        if !names.insert(amplitude.name())
            || symbols.get(amplitude.name()).is_some()
            || symbols.iter().any(|(name, _)| name.ends_with(&suffix))
            || trials.contains(&field.id())
        {
            return Err(invalid(
                "harmonic amplitude names and original unknown mappings must be distinct",
            ));
        }
        let support = index.defined_on.get(&id).copied();
        if let Some(assertion) = amplitude.domain()
            && Some(resolve_symbol(file, amplitude.range(), assertion, symbols)?) != support
        {
            return Err(invalid(
                "harmonic amplitude support assertion differs from its original Field",
            ));
        }
        let declared =
            crate::value_types::lower_value_type::<RawId>(file, amplitude.value_type(), None)?;
        if field.value_type().scalar_domain() != ScalarDomain::Real
            || declared.scalar_domain() != ScalarDomain::Complex
            || field
                .value_type()
                .clone()
                .with_common_scalar_domain(&declared)
                .as_ref()
                != Some(&declared)
        {
            return Err(invalid(
                "harmonic amplitude must preserve the original real Field's dimension, shape and exact frame as complex values",
            ));
        }
        trials.push(field.id());
        mappings.push((amplitude.name().to_owned(), field.id().ulid().to_string()));
    }
    let original_fields = index
        .nodes
        .values()
        .filter_map(|node| match node {
            KernelNode::Field(field) => Some(field.id().erase()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    if original_fields != trials.iter().map(|id| id.erase()).collect() {
        return Err(invalid(
            "harmonic request leaves an original unknown without an amplitude mapping",
        ));
    }
    let mut context = ExpressionContext {
        file,
        symbols,
        index,
        ambient_dimension: geometry.map_or(0, |g| g.ambient_dimension()),
        topological_dimension: geometry.map_or(0, |g| g.topological_dimension()),
        relation_domain: None,
        tests: BTreeMap::new(),
        integration_domain: None,
        used_tests: BTreeSet::new(),
    };
    let frequency = context.compile_root(angular_frequency)?;
    let inverse_time =
        DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("frequency dimension");
    if frequency.value_type.scalar_domain() != ScalarDomain::Real
        || frequency.value_type.dimension() != inverse_time
    {
        return Err(invalid(
            "angular frequency requires a real scalar with dimension 1/time",
        ));
    }
    let frequency = wire::expression(&frequency);
    closed(&frequency, false)?;
    let mut inputs = BTreeSet::new();
    let mut excitation_values = Vec::new();
    for (original, expression) in excitations {
        let id = resolve_symbol(file, expression.range(), original, symbols)?;
        let Some(KernelNode::Port(port)) = index.nodes.get(&id).copied() else {
            return Err(invalid(
                "harmonic excitation must map an original signal input",
            ));
        };
        let Some((SignalDirection::Input, value_type)) = port.signal_contract() else {
            return Err(invalid(
                "harmonic excitation requires an original signal input, not an output or physical Port",
            ));
        };
        continuous(id)?;
        if !inputs.insert(id) {
            return Err(invalid("repeated harmonic excitation"));
        }
        context.relation_domain = index.defined_on.get(&id).and_then(|id| id.downcast());
        context.integration_domain = context.relation_domain;
        let amplitude = context.compile(expression)?;
        if value_type.scalar_domain() != ScalarDomain::Real
            || amplitude.value_type.scalar_domain() != ScalarDomain::Complex
            || value_type
                .clone()
                .with_common_scalar_domain(&amplitude.value_type)
                .as_ref()
                != Some(&amplitude.value_type)
            || amplitude
                .support
                .is_some_and(|support| Some(support) != context.relation_domain)
        {
            return Err(invalid(
                "harmonic excitation must preserve its original real input type and exact support as a complex amplitude",
            ));
        }
        let amplitude = wire::expression(&amplitude);
        closed(&amplitude, true)?;
        excitation_values.push((port.id().ulid().to_string(), amplitude));
    }
    let original_inputs = index
        .nodes
        .values()
        .filter_map(|node| match node {
            KernelNode::Port(port)
                if matches!(port.signal_contract(), Some((SignalDirection::Input, _))) =>
            {
                Some(port.id().erase())
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    if original_inputs != inputs {
        return Err(invalid(
            "harmonic request leaves an original signal input without an excitation",
        ));
    }
    let projection = AuthoredFormulationProjection::encode_harmonic(
        source_identity.to_string(),
        name.into(),
        HarmonicFormulationRequest::new(
            frequency,
            relations.iter().map(|id| id.ulid().to_string()).collect(),
            excitation_values,
            mappings,
        ),
    )?;
    Ok(CompiledAuthoredFormulation {
        relations,
        domain: None,
        trials,
        projection,
        file: file.into(),
        range,
    })
}

fn closed(expression: &AuthoredFormExpressionV1, coordinates: bool) -> Result<(), Diagnostic> {
    use AuthoredFormExpressionV1 as E;
    let mut pending = vec![expression];
    let mut remaining = 65_536usize;
    while let Some(expression) = pending.pop() {
        remaining = remaining
            .checked_sub(1)
            .ok_or_else(|| wire::rejection("harmonic coefficient exceeds the expression budget"))?;
        match expression {
            E::Number { .. } | E::Rational { .. } | E::Parameter { .. } => {}
            E::Coordinate { .. } if coordinates => {}
            E::Complex { real, imag } => pending.extend([real.as_ref(), imag.as_ref()]),
            E::Neg { value } | E::Conjugate { value } | E::Sin { value } => pending.push(value),
            E::Add { left, right }
            | E::Sub { left, right }
            | E::Mul { left, right }
            | E::Div { left, right } => pending.extend([left.as_ref(), right.as_ref()]),
            E::Pow { base, .. } => pending.push(base),
            _ => {
                return Err(wire::rejection(
                    "harmonic coefficients require closed Parameters and fixed coordinates, not unknowns or time-dependent values",
                ));
            }
        }
    }
    Ok(())
}
