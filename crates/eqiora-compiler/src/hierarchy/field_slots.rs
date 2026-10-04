//! Occurrence-bound continuum Field interfaces for reusable Components.
//!
//! A Field slot is a definition-time obligation. Support binding specializes
//! its identity-parametric type, then one exact enclosing Field satisfies the
//! obligation. Neither the slot nor its binding becomes a Kernel entity.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_lang::{
    ComponentDecl, ComponentItem, FieldDecl, InstanceDecl, Item, ModelDecl, SignatureItem,
};
use eqiora_schema::kernel::typing::{ExpressionType, SpatialSupport};

use crate::diagnostics::source_error;

use super::body_check::field_expression_type;
use super::supports::SupportInterface;

/// Closed semantic representation family admitted by Field-slot v1.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FieldRepresentationContract {
    Continuum,
}

/// Complete identity-parametric contract for one semantic Field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct FieldContract<I> {
    value: ExpressionType<I>,
    representation: FieldRepresentationContract,
    pub(super) role: eqiora_lang::FieldRoleSyntax,
    pub(super) activation: eqiora_lang::ActivationSyntax,
}

impl<I> FieldContract<I> {
    pub(super) fn continuum(
        value: ExpressionType<I>,
        role: eqiora_lang::FieldRoleSyntax,
        activation: eqiora_lang::ActivationSyntax,
    ) -> Self {
        Self {
            value,
            representation: FieldRepresentationContract::Continuum,
            role,
            activation,
        }
    }

    pub(super) const fn value(&self) -> &ExpressionType<I> {
        &self.value
    }
}

#[derive(Clone, Debug)]
struct FieldSlotContract {
    support_slot: Option<String>,
    field: FieldContract<String>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct FieldInterface {
    slots: BTreeMap<String, FieldSlotContract>,
}

impl FieldInterface {
    pub(super) fn field(&self, name: &str) -> Option<&FieldContract<String>> {
        self.slots.get(name).map(|slot| &slot.field)
    }

    fn get(&self, name: &str) -> Option<&FieldSlotContract> {
        self.slots.get(name)
    }

    fn iter(&self) -> impl Iterator<Item = (&str, &FieldSlotContract)> {
        self.slots
            .iter()
            .map(|(name, contract)| (name.as_str(), contract))
    }
}

pub(super) fn component_field_interface(
    file: &str,
    component: &ComponentDecl,
    supports: &SupportInterface,
    values: &super::parameters::SymbolicParameterMap,
) -> Result<FieldInterface, Vec<Diagnostic>> {
    signature_field_interface(file, component.signature(), supports, values)
}

pub(super) fn signature_field_interface(
    file: &str,
    signature: &[SignatureItem],
    supports: &SupportInterface,
    values: &super::parameters::SymbolicParameterMap,
) -> Result<FieldInterface, Vec<Diagnostic>> {
    let mut slots = BTreeMap::new();
    let mut diagnostics = Vec::new();
    for item in signature {
        let SignatureItem::Field(declaration) = item else {
            continue;
        };
        if let eqiora_lang::ActivationSyntax::Named(clock) = declaration.activation()
            && !signature.iter().any(|item| matches!(item, SignatureItem::Clock(requirement) if requirement.name() == clock)) {
                diagnostics.push(source_error(codes::LANGUAGE_TYPE_ERROR, file, declaration.range(), "required field clock must name a clock requirement in the signature"));
                continue;
            }
        match field_slot_contract(file, declaration, supports, values) {
            Ok(contract) => {
                if slots
                    .insert(declaration.name().to_owned(), contract)
                    .is_some()
                {
                    diagnostics.push(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        declaration.range(),
                        format!("duplicate Field slot `{}`", declaration.name()),
                    ));
                }
            }
            Err(error) => diagnostics.push(error),
        }
    }
    if diagnostics.is_empty() {
        Ok(FieldInterface { slots })
    } else {
        Err(diagnostics)
    }
}

fn field_slot_contract(
    file: &str,
    declaration: &FieldDecl,
    supports: &SupportInterface,
    values: &super::parameters::SymbolicParameterMap,
) -> Result<FieldSlotContract, Diagnostic> {
    let support = declaration
        .domain()
        .map(|name| {
            supports
                .get(name)
                .map(|s| s.support().clone())
                .ok_or_else(|| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        declaration.range(),
                        format!("unknown support `{name}`"),
                    )
                })
        })
        .transpose()?;
    if support.as_ref().is_some_and(|support| {
        !matches!(
            support,
            SpatialSupport::Volume { .. }
                | SpatialSupport::Boundary { .. }
                | SpatialSupport::Coordinates { .. }
        )
    }) {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            declaration.range(),
            "source Field requirement requires a volume, boundary, or coordinate-factor support",
        ));
    }
    let syntax = super::parameters::specialize_type(file, declaration.value_type(), values)?;
    let value = ExpressionType::new(
        crate::value_types::lower_value_type(file, &syntax, support.as_ref())?,
        support,
    );
    Ok(FieldSlotContract {
        support_slot: declaration.domain().map(str::to_owned),
        field: FieldContract::continuum(
            value,
            declaration.role(),
            declaration.activation().clone(),
        ),
    })
}

/// Collect Fields visible as bare binding targets in one Component body.
pub(super) fn component_field_contracts(
    file: &str,
    component: &ComponentDecl,
    supports: &SupportInterface,
    slots: &FieldInterface,
    values: &super::parameters::SymbolicParameterMap,
    records: &super::parameters::RecordContext,
) -> BTreeMap<String, FieldContract<String>> {
    let mut fields = slots
        .iter()
        .map(|(name, slot)| (name.to_owned(), slot.field.clone()))
        .collect::<BTreeMap<_, _>>();
    for item in component.items() {
        let ComponentItem::Field(declaration) = item else {
            continue;
        };
        let support = declaration
            .domain()
            .and_then(|name| supports.visible_support(name).cloned());
        if let Some(record) = records.record_for_type(declaration.value_type()) {
            fields.extend(record_field_contracts(declaration, record, support));
            continue;
        }
        if let Ok(value) = field_expression_type(file, declaration, support, values) {
            fields.insert(
                declaration.name().to_owned(),
                FieldContract::continuum(
                    value,
                    declaration.role(),
                    declaration.activation().clone(),
                ),
            );
        }
    }
    fields
}

/// Collect exact Fields and record members visible as binding targets in one root Model.
pub(super) fn model_field_contracts(
    file: &str,
    model: &ModelDecl,
    supports: &BTreeMap<String, SpatialSupport<String>>,
    values: &super::parameters::SymbolicParameterMap,
    records: &super::parameters::RecordContext,
) -> BTreeMap<String, FieldContract<String>> {
    let mut fields = BTreeMap::new();
    for item in model.items() {
        let Item::Field(declaration) = item else {
            continue;
        };
        let support = declaration
            .domain()
            .and_then(|name| supports.get(name).cloned());
        if let Some(record) = records.record_for_type(declaration.value_type()) {
            fields.extend(record_field_contracts(declaration, record, support));
            continue;
        }
        if let Ok(value) = field_expression_type(file, declaration, support, values) {
            fields.insert(
                declaration.name().to_owned(),
                FieldContract::continuum(
                    value,
                    declaration.role(),
                    declaration.activation().clone(),
                ),
            );
        }
    }
    fields
}

fn record_field_contracts(
    declaration: &eqiora_lang::FieldDecl,
    record: &crate::record::BoundRecord,
    support: Option<SpatialSupport<String>>,
) -> impl Iterator<Item = (String, FieldContract<String>)> {
    record
        .definition
        .members()
        .iter()
        .map(move |(name, value)| {
            (
                format!("{}.{name}", declaration.name()),
                FieldContract::continuum(
                    ExpressionType::new(value.clone(), support.clone()),
                    declaration.role(),
                    declaration.activation().clone(),
                ),
            )
        })
}

pub(super) fn resolve_instance_clocks(
    file: &str,
    component: &ComponentDecl,
    instance: &InstanceDecl,
    mut resolve: impl FnMut(&str) -> Option<String>,
) -> Result<BTreeMap<String, String>, Vec<Diagnostic>> {
    let required: BTreeSet<_> = component
        .signature()
        .iter()
        .filter_map(|item| match item {
            SignatureItem::Clock(clock) => Some(clock.name()),
            _ => None,
        })
        .collect();
    let mut result = BTreeMap::new();
    let mut errors = Vec::new();
    for binding in super::named_bindings::references(
        file,
        instance,
        |binding| required.contains(binding.name()),
        &mut errors,
    ) {
        if let Some(value) = resolve(binding.target())
            .filter(|_| required.contains(binding.slot()) && !result.contains_key(binding.slot()))
        {
            result.insert(binding.slot().to_owned(), value);
        } else {
            errors.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                binding.range(),
                "clock binding must name one unbound requirement and an exact enclosing clock",
            ));
        }
    }

    for name in required {
        if !result.contains_key(name) {
            errors.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                instance.range(),
                format!("missing required clock binding `{name}`"),
            ));
        }
    }
    if errors.is_empty() {
        Ok(result)
    } else {
        Err(errors)
    }
}

/// Validate one instance and return `child Field slot -> enclosing target`.
pub(super) fn resolve_instance_fields<I: Clone + Eq>(
    binding_file: &str,
    component: &ComponentDecl,
    interface: &FieldInterface,
    instance: &InstanceDecl,
    mut resolve_parent_clock: impl FnMut(&str) -> Option<String>,
    mut resolve_child_support: impl FnMut(&str) -> Option<SpatialSupport<I>>,
    mut resolve_parent: impl FnMut(&str) -> Option<FieldContract<I>>,
) -> Result<BTreeMap<String, String>, Vec<Diagnostic>> {
    let clock_bindings =
        resolve_instance_clocks(binding_file, component, instance, &mut resolve_parent_clock)?;
    let mut diagnostics = Vec::new();
    let mut targets = BTreeMap::new();
    let mut actual = BTreeMap::new();
    let mut seen = BTreeSet::new();

    for binding in super::named_bindings::field_references(
        binding_file,
        instance,
        |binding| interface.get(binding.name()).is_some(),
        &mut diagnostics,
    ) {
        if !seen.insert(binding.slot()) {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                binding_file,
                binding.range(),
                format!(
                    "duplicate binding for Field slot `{}` in instance `{}`",
                    binding.slot(),
                    instance.name()
                ),
            ));
            continue;
        }
        let Some(slot) = interface.get(binding.slot()) else {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                binding_file,
                binding.range(),
                format!(
                    "unknown Field slot `{}` on component `{}`",
                    binding.slot(),
                    component.name()
                ),
            ));
            continue;
        };
        let Some(target) = resolve_parent(binding.target()) else {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                binding_file,
                binding.range(),
                format!(
                    "Field binding target `{}` is not an enclosing Field",
                    binding.target()
                ),
            ));
            continue;
        };
        let support = match &slot.support_slot {
            Some(name) => match resolve_child_support(name) {
                Some(support) => Some(support),
                None => {
                    diagnostics.push(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        binding_file,
                        binding.range(),
                        "required support is unresolved",
                    ));
                    continue;
                }
            },
            None => None,
        };
        let expected = FieldContract {
            value: ExpressionType::new(slot.field.value.value_type.clone(), support),
            representation: slot.field.representation,
            role: slot.field.role,
            activation: match &slot.field.activation {
                eqiora_lang::ActivationSyntax::Continuous => {
                    eqiora_lang::ActivationSyntax::Continuous
                }
                eqiora_lang::ActivationSyntax::Named(clock) => match clock_bindings.get(clock) {
                    Some(target) => eqiora_lang::ActivationSyntax::Named(target.clone()),
                    None => {
                        diagnostics.push(source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            binding_file,
                            binding.range(),
                            "required field clock is not a borrowed clock requirement",
                        ));
                        continue;
                    }
                },
                _ => {
                    diagnostics.push(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        binding_file,
                        binding.range(),
                        "unsupported field activation",
                    ));
                    continue;
                }
            },
        };
        if let Some(message) = field_contract_mismatch(binding.slot(), &expected, &target) {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                binding_file,
                binding.range(),
                message,
            ));
            continue;
        }
        targets.insert(binding.slot().to_owned(), binding.target().to_owned());
        actual.insert(binding.slot().to_owned(), binding.range());
    }

    for (name, _) in interface.iter() {
        if !actual.contains_key(name) {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                binding_file,
                instance.range(),
                format!(
                    "instance `{}` has no binding for required Field slot `{name}`",
                    instance.name()
                ),
            ));
        }
    }

    if diagnostics.is_empty() {
        Ok(targets)
    } else {
        Err(diagnostics)
    }
}

fn field_contract_mismatch<I: Eq>(
    slot: &str,
    expected: &FieldContract<I>,
    actual: &FieldContract<I>,
) -> Option<String> {
    let mismatch = if expected.value.value_type.scalar_domain()
        != actual.value.value_type.scalar_domain()
    {
        "mathematical scalar domain"
    } else if expected.value.dimension() != actual.value.dimension() {
        "physical dimension"
    } else if expected.value.shape() != actual.value.shape() {
        "exact value shape"
    } else if expected.value.value_type.array_rank() != actual.value.value_type.array_rank() {
        "array and spatial axis roles"
    } else if expected.value.frame() != actual.value.frame() {
        "coordinate frame"
    } else if expected.value.support != actual.value.support {
        "exact spatial support"
    } else if expected.role == eqiora_lang::FieldRoleSyntax::State && actual.role != expected.role {
        "declared state role"
    } else if expected.activation != actual.activation {
        "exact activation"
    } else if expected.representation != actual.representation {
        "representation family"
    } else {
        return None;
    };
    Some(format!(
        "Field slot `{slot}` and its target disagree in {mismatch}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_lang::{Document, Item, ModelDecl};

    #[test]
    fn source_slots_preserve_complete_types_across_component_binding() {
        let source = |slot_type, field_type| {
            format!(
                r#"
component Law(variable value: {slot_type} on body, support body: volume(ambient_dimension = 2)) {{


  relation balance on body {{ value - value = 0; }}
}}
model Main() {{
  domain body = box(0, 1, 0, 1);
  variable value: {field_type} on body;
  instance law: Law(body = body, value = value);
}}
"#
            )
        };
        for value_type in [
            "complex<V>",
            "vector<complex<V>, 2>",
            "array<vector<complex<V>, 2>, 3>",
            "tensor<complex<Pa>, 2, 2>",
        ] {
            crate::compile("types.eqi", &source(value_type, value_type)).unwrap();
        }
        for (slot, field, mismatch) in [
            ("complex<V>", "V", "mathematical scalar domain"),
            (
                "array<vector<complex<V>, 2>, 2>",
                "tensor<complex<V>, 2, 2>",
                "array and spatial axis roles",
            ),
        ] {
            let diagnostics = crate::compile("mismatch.eqi", &source(slot, field)).unwrap_err();
            assert!(
                diagnostics
                    .iter()
                    .any(|error| error.message().contains(mismatch))
            );
        }
    }

    #[test]
    fn field_binding_retains_array_and_spatial_axis_roles() {
        use eqiora_core::{DimExponents, ScalarDomain, ValueShape};
        use eqiora_core::{ValueFrame, ValueType};

        let spatial = |extents| {
            ValueType::shaped(
                ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
                ValueShape::new(extents).unwrap(),
                ValueFrame::SpatialCartesian,
            )
            .unwrap()
        };
        let array = FieldContract::continuum(
            ExpressionType::<()>::new(spatial(vec![2]).array(2).unwrap(), None),
            eqiora_lang::FieldRoleSyntax::Variable,
            eqiora_lang::ActivationSyntax::Continuous,
        );
        let tensor = FieldContract::continuum(
            ExpressionType::<()>::new(spatial(vec![2, 2]), None),
            eqiora_lang::FieldRoleSyntax::Variable,
            eqiora_lang::ActivationSyntax::Continuous,
        );
        assert!(field_contract_mismatch("input", &array, &array).is_none());
        assert_eq!(
            field_contract_mismatch("input", &tensor, &array).as_deref(),
            Some("Field slot `input` and its target disagree in array and spatial axis roles")
        );
    }

    fn parse(source: &str) -> Document {
        eqiora_lang::parse("field_slots.eqi", source)
            .into_document()
            .expect("Field-slot fixture parses")
    }

    fn component(document: &Document) -> &ComponentDecl {
        document.components().first().expect("fixture component")
    }

    fn instance(document: &Document) -> &InstanceDecl {
        document
            .models()
            .iter()
            .flat_map(ModelDecl::items)
            .find_map(|item| match item {
                Item::Instance(instance) => Some(instance),
                _ => None,
            })
            .expect("fixture instance")
    }

    #[test]
    fn exact_support_shape_frame_and_dimension_are_required() {
        let document = parse(
            r#"
component Law(variable displacement: vector<m, 2> on body, support body: volume(ambient_dimension = 2)) {


}
model Use() {
  domain body = box(0, 1, 0, 1);
  variable displacement: vector<m, 2> on body;
  instance law: Law(body = body, displacement = displacement);
}
"#,
        );
        let component = component(&document);
        let supports =
            super::super::supports::component_support_interface("field_slots.eqi", component)
                .expect("support interface");
        let interface =
            component_field_interface("field_slots.eqi", component, &supports, &BTreeMap::new())
                .expect("Field interface");
        let exact_support = SpatialSupport::Volume {
            domain: "body-id",
            dimensions: 2,
        };
        let target = FieldContract::continuum(
            ExpressionType::shaped(
                eqiora_core::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0])
                    .expect("bounded dimension"),
                eqiora_core::ValueShape::new([2]).expect("shape"),
                eqiora_core::ValueFrame::SpatialCartesian,
                Some(exact_support.clone()),
            )
            .unwrap(),
            eqiora_lang::FieldRoleSyntax::Variable,
            eqiora_lang::ActivationSyntax::Continuous,
        );
        let resolved = resolve_instance_fields(
            "field_slots.eqi",
            component,
            &interface,
            instance(&document),
            |_| None,
            |_| Some(exact_support.clone()),
            |name| (name == "displacement").then(|| target.clone()),
        )
        .expect("exact Field contract binds");
        assert_eq!(
            resolved.get("displacement").map(String::as_str),
            Some("displacement")
        );
    }
}
