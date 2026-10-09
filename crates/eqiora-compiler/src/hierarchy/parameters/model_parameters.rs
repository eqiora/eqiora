//! Model defaults share the same typed dependency graph as Component defaults.

use super::*;
use eqiora_lang::{Item, ModelDecl, SignatureItem, SourceAstFactory};
use eqiora_schema::kernel::RationalTime;

fn resolve(
    file: &str,
    model: &ModelDecl,
    required_policy: RequiredParameterPolicy,
    resolve_clock: &mut dyn FnMut(&str) -> Option<Option<RationalTime>>,
    frames: BTreeMap<String, SpatialSupport<String>>,
    records: &RecordContext,
) -> Result<SymbolicParameterMap, Vec<Diagnostic>> {
    resolver(file, model, required_policy, frames, records)?.resolve_all(resolve_clock)
}

pub(super) fn resolver<'a>(
    file: &'a str,
    model: &ModelDecl,
    required_policy: RequiredParameterPolicy,
    frames: BTreeMap<String, SpatialSupport<String>>,
    records: &RecordContext,
) -> Result<SymbolicParameterResolver<'a>, Vec<Diagnostic>> {
    let mut declarations = model
        .signature()
        .iter()
        .filter_map(|item| match item {
            SignatureItem::Parameter(value) => Some(value.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    for item in model.items() {
        if let Item::Parameter(value) = item {
            declarations.push(
                SourceAstFactory::component_parameter(
                    VisibilitySyntax::Private,
                    value.name(),
                    value.value_type().clone(),
                    Some(value.value().clone()),
                    value.range(),
                )
                .map_err(|error| vec![hierarchy_error(error.to_string())])?,
            );
        }
    }
    Ok(SymbolicParameterResolver {
        declaration_file: file,
        declarations: records::expand(
            file,
            declarations
                .into_iter()
                .map(|value| (value.name().to_owned(), value))
                .collect(),
            records,
        )?,
        bindings: None,
        bound_values: BTreeMap::new(),
        resolved: BTreeMap::new(),
        required_policy,
        frames,
    })
}

pub(in crate::hierarchy) fn resolve_model_parameters_symbolically(
    file: &str,
    model: &ModelDecl,
    mut resolve_clock: impl FnMut(&str) -> Option<Option<RationalTime>>,
    records: &RecordContext,
) -> Result<SymbolicParameterMap, Vec<Diagnostic>> {
    resolve(
        file,
        model,
        RequiredParameterPolicy::PublicIsFree,
        &mut resolve_clock,
        super::super::supports::model_spatial_supports(file, model)?,
        records,
    )
}

pub(in crate::hierarchy) fn resolve_model_parameters(
    file: &str,
    model: &ModelDecl,
    mut resolve_clock: impl FnMut(&str) -> Option<Option<RationalTime>>,
    bound_frames: BTreeMap<String, SpatialSupport<String>>,
    records: &RecordContext,
) -> Result<BTreeMap<String, ResolvedParameter>, Vec<Diagnostic>> {
    let mut frames = super::super::supports::model_spatial_supports(file, model)?;
    frames.extend(bound_frames);
    resolve(
        file,
        model,
        RequiredParameterPolicy::RejectUnbound,
        &mut resolve_clock,
        frames,
        records,
    )
    .and_then(concrete_parameters)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model(source: &str) -> ModelDecl {
        let document = eqiora_lang::parse("period.eqi", source)
            .into_compilation_document()
            .unwrap();
        document.models()[0].clone()
    }
    #[test]
    fn period_defaults_are_typed_symbolically_then_resolved_in_dependency_order() {
        let model = model(
            "model M(clock tick: periodic, parameter twice: s = 2 * dt, parameter dt: s = period(tick)) { parameter four: s = 2 * twice; }",
        );
        let symbolic = resolve_model_parameters_symbolically(
            "period.eqi",
            &model,
            |name| (name == "tick").then_some(None),
            &RecordContext::default(),
        )
        .unwrap();
        assert_eq!(
            symbolic["twice"].value_type.dimension(),
            crate::dimensions::time_dimension()
        );
        assert!(symbolic["four"].value.is_none());
        let concrete = resolve_model_parameters(
            "period.eqi",
            &model,
            |name| (name == "tick").then_some(Some(RationalTime::new(1, 8).unwrap())),
            BTreeMap::new(),
            &RecordContext::default(),
        )
        .unwrap();
        assert_eq!(
            concrete["dt"].value.real_scalar_value().unwrap().value(),
            0.125
        );
        assert_eq!(
            concrete["twice"].value.real_scalar_value().unwrap().value(),
            0.25
        );
        assert_eq!(
            concrete["four"].value.real_scalar_value().unwrap().value(),
            0.5
        );
    }
    #[test]
    fn period_aliases_remain_static_and_resolve_forward_dependencies() {
        let model = model(
            "model M(clock tick: periodic) { let twice = 2 * dt; let dt: s = period(tick); }",
        );
        let mut values = BTreeMap::new();
        resolve_model_lets("period.eqi", &model, &mut values, |name| {
            (name == "tick").then_some(None)
        })
        .unwrap();
        assert!(values["twice"].value.is_none());
        assert_eq!(
            values["twice"].value_type.dimension(),
            crate::dimensions::time_dimension()
        );
        values.clear();
        resolve_model_lets("period.eqi", &model, &mut values, |name| {
            (name == "tick").then_some(Some(RationalTime::new(1, 8).unwrap()))
        })
        .unwrap();
        assert_eq!(
            values["twice"]
                .value
                .as_ref()
                .unwrap()
                .real_scalar_value()
                .unwrap()
                .value(),
            0.25
        );
        assert!(resolve_model_lets("period.eqi", &model, &mut BTreeMap::new(), |_| None).is_err());
    }
    #[test]
    fn period_requires_nominal_clock_and_time_dimension() {
        for source in [
            "model M(parameter dt: s = period(missing)) {}",
            "model M(parameter dt: s = period(1)) {}",
            "model M(clock tick: periodic, parameter dt: s = period(tick, tick)) {}",
            "model M(clock tick: periodic, parameter dt: m = period(tick)) {}",
        ] {
            let model = model(source);
            assert!(
                resolve_model_parameters_symbolically(
                    "period.eqi",
                    &model,
                    |name| (name == "tick").then_some(None),
                    &RecordContext::default()
                )
                .is_err(),
                "{source}"
            );
        }
        let model = model("model M(clock tick: periodic, parameter dt: s = period(tick)) {}");
        assert!(
            resolve_model_parameters(
                "period.eqi",
                &model,
                |_| Some(None),
                BTreeMap::new(),
                &RecordContext::default()
            )
            .is_err()
        );
    }
}
