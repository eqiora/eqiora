//! Nominal value dimensions must name the selected declaration and its exact extent.
use super::*;
use eqiora_core::{ScalarDomain, ValueFrame, ValueType};

pub(super) fn validate(nodes: &BTreeMap<RawId, KernelNode>, diagnostics: &mut Vec<Diagnostic>) {
    for (&owner, node) in nodes {
        match node {
            KernelNode::FiniteSpace(space) => {
                if let Some(factors) = space.factors() {
                    for factor in factors {
                        if !matches!(nodes.get(&factor.space().expect("atomic factor").erase()), Some(KernelNode::FiniteSpace(definition)) if definition.basis() == factor)
                        {
                            diagnostics.push(kernel_error(owner, "finite product requires exact selected atomic factor declarations and cardinalities"));
                        }
                    }
                }
            }

            KernelNode::Field(field) => check(owner, field.value_type(), nodes, diagnostics),
            KernelNode::Parameter(parameter) => {
                check_literal(owner, parameter.value(), nodes, diagnostics)
            }
            KernelNode::Port(port) => {
                if let Some((_, value_type)) = port.signal_contract() {
                    check(owner, value_type, nodes, diagnostics);
                }
            }
            KernelNode::Domain(domain) => {
                if let DomainKind::ScalarPhysical {
                    across_type,
                    through_type,
                } = domain.kind()
                {
                    check(owner, across_type, nodes, diagnostics);
                    check(owner, through_type, nodes, diagnostics);
                }
            }
            _ => {}
        }
    }
}

pub(super) fn check(
    owner: RawId,
    value: &ValueType,
    nodes: &BTreeMap<RawId, KernelNode>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if value.scalar_domain() == ScalarDomain::Enum {
        let selected = value
            .enum_definition()
            .and_then(|id| nodes.get(&id.erase()));
        if !matches!(selected,Some(KernelNode::Enum(definition)) if *value==definition.value_type())
        {
            diagnostics.push(kernel_error(
                owner,
                "enum value requires its exact selected Enum declaration and complete scalar type",
            ));
        }
    }
    if value.scalar_domain() == ScalarDomain::Boolean && *value != ValueType::boolean() {
        diagnostics.push(kernel_error(
            owner,
            "Boolean values require the invariant dimensionless scalar Boolean type",
        ));
    }
    if value.scalar_domain() == ScalarDomain::Integer
        && (value.dimension() != DimExponents::DIMENSIONLESS
            || value.frame() != ValueFrame::Invariant)
    {
        diagnostics.push(kernel_error(
            owner,
            "integer values require dimensionless invariant types",
        ));
    }
    for basis in value.finite_bases().flat_map(|basis| basis.atoms()) {
        let matches = matches!(nodes.get(&basis.space().expect("atomic factor").erase()), Some(KernelNode::FiniteSpace(definition)) if definition.basis().space() == basis.space() && definition.basis().extent() == basis.extent());
        if !matches {
            diagnostics.push(kernel_error(owner,
                "nominal value requires each exact selected FiniteSpace declaration and cardinality"));
        }
    }
    if let Some(index) = value.index_set() {
        let matches = matches!(nodes.get(&index.erase()), Some(KernelNode::IndexSet(definition)) if *value == definition.value_type());
        if !matches {
            diagnostics.push(kernel_error(
                owner,
                "index value requires its exact selected IndexSet declaration and bound",
            ));
        }
    }
}

pub(super) fn check_literal(
    owner: RawId,
    value: &eqiora_core::ValueLiteral,
    nodes: &BTreeMap<RawId, KernelNode>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    check(owner, value.value_type(), nodes, diagnostics);
    if value.value_type().scalar_domain() == ScalarDomain::Enum
        && !value
            .enum_tag()
            .zip(value.value_type().enum_member_count())
            .is_some_and(|(tag, count)| tag < count)
    {
        diagnostics.push(kernel_error(
            owner,
            "enum member tag is absent or outside its declaration",
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_schema::kernel::{
        FieldDef, FieldRole, FiniteSpaceDef, IndexSetDef, PortDef, SignalDirection,
    };

    #[test]
    fn enum_owners_and_literals_require_the_full_selected_declaration() {
        use eqiora_schema::kernel::{EnumDef, ParameterDef};
        let definition = EnumDef::new(Id::new(), ["Off".into(), "On".into()]).unwrap();
        let foreign = EnumDef::new(Id::new(), ["Off".into(), "On".into()]).unwrap();
        for value in [
            definition.value_type(),
            ValueType::enumeration(definition.id(), 3).unwrap(),
            foreign.value_type(),
        ] {
            let expected = value == definition.value_type();
            for node in [
                KernelNode::from(FieldDef::new(Id::new(), value.clone(), FieldRole::State)),
                KernelNode::from(PortDef::signal(
                    Id::new(),
                    SignalDirection::Output,
                    value.clone(),
                )),
            ] {
                let nodes = BTreeMap::from([
                    (definition.id().erase(), definition.clone().into()),
                    (node.id(), node),
                ]);
                let mut errors = Vec::new();
                validate(&nodes, &mut errors);
                assert_eq!(errors.is_empty(), expected, "{value:?}");
            }
        }
        for literal in [
            definition.value(1).unwrap(),
            foreign.value(1).unwrap(),
            eqiora_core::ValueLiteral::enum_value(
                ValueType::enumeration(definition.id(), 3).unwrap(),
                2,
            )
            .unwrap(),
        ] {
            let expected = literal.value_type() == &definition.value_type();
            let parameter = ParameterDef::new(Id::new(), literal.clone());
            let nodes = BTreeMap::from([
                (definition.id().erase(), definition.clone().into()),
                (parameter.id().erase(), parameter.clone().into()),
            ]);
            let mut errors = Vec::new();
            validate(&nodes, &mut errors);
            assert_eq!(errors.is_empty(), expected);
            errors.clear();
            check_literal(parameter.id().erase(), &literal, &nodes, &mut errors);
            assert_eq!(errors.is_empty(), expected);
        }
        assert!(definition.value(2).is_err());
        let field = Id::<kinds::Field>::new();
        let nodes = BTreeMap::from([
            (definition.id().erase(), definition.clone().into()),
            (
                field.erase(),
                FieldDef::new(field, definition.value_type(), FieldRole::State).into(),
            ),
        ]);
        assert!(
            symbol_type(
                SymbolRef::Derivative(field, std::num::NonZeroU32::MIN),
                &nodes,
                &[],
                &BTreeMap::new()
            )
            .is_err()
        );
    }

    #[test]
    fn maps_require_both_selected_declarations_and_exact_extents() {
        let input = FiniteSpaceDef::new(Id::new(), ["q0".into(), "q1".into()]).unwrap();
        let output = FiniteSpaceDef::new(Id::new(), ["u0".into(), "u1".into()]).unwrap();
        let field = Id::<kinds::Field>::new();
        for (source, target, expected) in [
            (input.basis(), output.basis(), true),
            (output.basis().dual(), input.basis().dual(), true),
            (
                eqiora_core::FiniteBasis::new(Id::new(), 2).unwrap(),
                output.basis(),
                false,
            ),
            (
                input.basis(),
                eqiora_core::FiniteBasis::new(Id::new(), 2).unwrap(),
                false,
            ),
            (
                eqiora_core::FiniteBasis::new(input.id(), 3).unwrap(),
                output.basis(),
                false,
            ),
            (
                input.basis(),
                eqiora_core::FiniteBasis::new(output.id(), 3).unwrap(),
                false,
            ),
        ] {
            let value = ValueType::linear_map(
                source,
                target,
                ScalarDomain::Complex,
                DimExponents::DIMENSIONLESS,
            )
            .unwrap();
            let nodes = BTreeMap::from([
                (input.id().erase(), input.clone().into()),
                (output.id().erase(), output.clone().into()),
                (
                    field.erase(),
                    FieldDef::new(field, value, FieldRole::Variable).into(),
                ),
            ]);
            let mut errors = vec![];
            validate(&nodes, &mut errors);
            assert_eq!(errors.is_empty(), expected);
        }
    }

    #[test]
    fn nominal_cardinality_and_selected_identity_are_both_required() {
        let space = FiniteSpaceDef::new(Id::new(), ["A".into(), "B".into()]).unwrap();
        let index = IndexSetDef::new(Id::new(), 3).unwrap();
        let field = Id::<kinds::Field>::new();
        let mut nodes = BTreeMap::from([
            (space.id().erase(), space.clone().into()),
            (index.id().erase(), index.clone().into()),
        ]);
        for (value, expected) in [
            (space.counts().unwrap(), true),
            (ValueType::counts(space.id(), 3).unwrap(), false),
            (ValueType::counts(Id::new(), 2).unwrap(), false),
            (ValueType::index(index.id(), 3).unwrap(), true),
            (ValueType::index(index.id(), 2).unwrap(), false),
            (ValueType::index(Id::new(), 3).unwrap(), false),
        ] {
            nodes.insert(
                field.erase(),
                FieldDef::new(field, value, FieldRole::State).into(),
            );
            let mut errors = Vec::new();
            validate(&nodes, &mut errors);
            assert_eq!(errors.is_empty(), expected);
        }
    }
    #[test]
    fn discrete_state_has_no_continuous_derivative() {
        let field = Id::<kinds::Field>::new();
        let value = ValueType::scalar(
            eqiora_core::ScalarDomain::Integer,
            DimExponents::DIMENSIONLESS,
        )
        .expect("valid scalar type");
        let nodes = BTreeMap::from([(
            field.erase(),
            FieldDef::new(field, value, FieldRole::State).into(),
        )]);
        assert!(
            symbol_type(
                SymbolRef::Derivative(field, std::num::NonZeroU32::MIN),
                &nodes,
                &[],
                &BTreeMap::new()
            )
            .is_err()
        );
    }
    #[test]
    fn field_and_port_integer_types_reject_units_and_spatial_frames_without_literals() {
        let space = FiniteSpaceDef::new(Id::new(), ["A".into(), "B".into()]).unwrap();
        let index = IndexSetDef::new(Id::new(), 3).unwrap();
        let field = Id::<kinds::Field>::new();
        let port = Id::<kinds::Port>::new();
        let seconds = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
        let ordinary = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("valid scalar type");
        let mut cases = Vec::new();
        for value in [
            ValueType::boolean(),
            ordinary,
            space.counts().unwrap(),
            space
                .coordinates(
                    eqiora_core::ScalarDomain::Integer,
                    eqiora_core::DimExponents::DIMENSIONLESS,
                )
                .expect("integer coordinates"),
            index.value_type(),
        ] {
            cases.push((value.clone(), true));
            if value.scalar_domain() == ScalarDomain::Boolean {
                assert!(value.with_dimension(seconds).is_err());
            } else {
                cases.push((
                    value
                        .with_dimension(seconds)
                        .expect("constructible integer type; semantic profile rejects dimensions"),
                    false,
                ));
            }
        }
        cases.push((
            ValueType::shaped(
                ScalarDomain::Integer,
                DimExponents::DIMENSIONLESS,
                eqiora_core::ValueShape::new([2]).unwrap(),
                ValueFrame::SpatialCartesian,
            )
            .unwrap(),
            false,
        ));
        for (value, expected) in cases {
            for node in [
                KernelNode::from(FieldDef::new(field, value.clone(), FieldRole::State)),
                KernelNode::from(PortDef::signal(
                    port,
                    SignalDirection::Output,
                    value.clone(),
                )),
            ] {
                let nodes = BTreeMap::from([
                    (space.id().erase(), space.clone().into()),
                    (index.id().erase(), index.clone().into()),
                    (node.id(), node),
                ]);
                let mut errors = Vec::new();
                validate(&nodes, &mut errors);
                assert_eq!(errors.is_empty(), expected, "type: {value:?}");
            }
        }
    }
    #[test]
    fn scalar_physical_port_domain_cannot_hide_dimensioned_integer_types() {
        let integer = ValueType::scalar(
            ScalarDomain::Integer,
            DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap(),
        )
        .expect("constructible integer type; semantic profile rejects dimensions");
        let domain =
            eqiora_schema::kernel::DomainDef::scalar_physical(Id::new(), integer.clone(), integer)
                .unwrap();
        let port = PortDef::scalar_physical(Id::new(), domain.id());
        let nodes = BTreeMap::from([
            (domain.id().erase(), domain.into()),
            (port.id().erase(), port.into()),
        ]);
        let mut errors = Vec::new();
        validate(&nodes, &mut errors);
        assert_eq!(errors.len(), 2);
    }
}
