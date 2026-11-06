use super::super::*;
use eqiora_core::{DimExponents, ScalarDomain, ValueFrame, ValueShape};
use eqiora_schema::kernel::typing::SpatialSupport;

fn context(dimensions: usize, complex: bool) -> Context<'static> {
    let domain = eqiora_core::Id::<eqiora_core::entity::kinds::Domain>::new().erase();
    let support = SpatialSupport::Volume { domain, dimensions };
    let ty = ValueType::shaped(
        if complex {
            ScalarDomain::Complex
        } else {
            ScalarDomain::Real
        },
        DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
        ValueShape::new([dimensions as u32]).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap();
    Context {
        name: "eta",
        field: "u",
        dimensions,
        remaining: 65536,
        test_dimension: eqiora_core::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
        symbols: BTreeMap::from([("u".into(), ty)]),
        supports: BTreeMap::from([("u".into(), support.clone())]),
        domains: BTreeMap::from([("body".into(), support)]),
        integration_domain: None,
    }
}
fn field() -> E {
    E::Field { ulid: "u".into() }
}
fn gradient() -> E {
    E::Gradient {
        value: Box::new(field()),
    }
}
fn symmetric() -> E {
    E::SymmetricPart {
        value: Box::new(gradient()),
    }
}
fn lift() -> E {
    E::IsotropicLift {
        value: Box::new(E::Divergence {
            value: Box::new(field()),
        }),
    }
}
fn gradient_atom(i: usize, j: usize, complex: bool) -> Polynomial {
    Polynomial::symbol(Atom::FieldGradient("u".into(), vec![i, j]), complex)
}

#[test]
fn structure_calculus_matches_independent_components_and_tensor_arithmetic() {
    for dimensions in [2, 3] {
        for complex in [false, true] {
            for i in 0..dimensions {
                for j in 0..dimensions {
                    let mut context = context(dimensions, complex);
                    // sym(A)_ij = (A_ij+A_ji)/2; (div(u) I)_ij = delta_ij sum_k u_k,k.
                    let expected = gradient_atom(i, j, complex)
                        .checked_add(&gradient_atom(j, i, complex))
                        .unwrap()
                        .divide_constant(&Polynomial::constant(ExactRational::integer(2)))
                        .unwrap();
                    assert_eq!(
                        context.tensor(&symmetric(), i, j, 0),
                        Some(expected.clone())
                    );
                    let mut diagonal = Polynomial::constant(ExactRational::integer(0));
                    if i == j {
                        for k in 0..dimensions {
                            diagonal = diagonal.checked_add(&gradient_atom(k, k, complex)).unwrap();
                        }
                    }
                    assert_eq!(context.tensor(&lift(), i, j, 0), Some(diagonal.clone()));
                    let combined = E::Div {
                        left: Box::new(E::Add {
                            left: Box::new(E::Mul {
                                left: Box::new(E::Number { value: 2.0 }),
                                right: Box::new(symmetric()),
                            }),
                            right: Box::new(E::Neg {
                                value: Box::new(lift()),
                            }),
                        }),
                        right: Box::new(E::Number { value: 2.0 }),
                    };
                    let expected = expected
                        .checked_add(
                            &diagonal
                                .checked_neg()
                                .unwrap()
                                .divide_constant(&Polynomial::constant(ExactRational::integer(2)))
                                .unwrap(),
                        )
                        .unwrap();
                    assert_eq!(context.tensor(&combined, i, j, 0), Some(expected.clone()));
                    assert_eq!(
                        context.tensor(
                            &E::Conjugate {
                                value: Box::new(combined)
                            },
                            i,
                            j,
                            0
                        ),
                        expected.conjugate().ok()
                    );
                }
            }
        }
    }
}

#[test]
fn structure_calculus_rejects_invalid_physical_operands_before_zero_cancellation() {
    for dimensions in [2, 3] {
        let mut context = context(dimensions, false);
        let zero = E::Mul {
            left: Box::new(E::Number { value: 0.0 }),
            right: Box::new(gradient()),
        };
        let symmetric_zero = E::SymmetricPart {
            value: Box::new(zero),
        };
        assert!(context.tensor(&symmetric_zero, 0, 0, 0).is_some());
        context.symbols.insert(
            "u".into(),
            ValueType::shaped(
                ScalarDomain::Real,
                DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
                ValueShape::new([dimensions as u32]).unwrap(),
                ValueFrame::Invariant,
            )
            .unwrap(),
        );
        assert!(context.tensor(&symmetric_zero, 0, 0, 0).is_none());
        assert!(context.tensor(&lift(), 0, 1, 0).is_none());
        let mut context = self::context(dimensions, false);
        *context.supports.get_mut("u").unwrap() = SpatialSupport::Volume {
            domain: eqiora_core::Id::<eqiora_core::entity::kinds::Domain>::new().erase(),
            dimensions: dimensions + 1,
        };
        assert!(context.tensor(&symmetric_zero, 0, 0, 0).is_none());
        assert!(context.tensor(&lift(), 0, 1, 0).is_none());
    }
}

#[test]
fn native_source_projection_matches_typed_source_components_without_solving() {
    use eqiora_graph::{GraphStore, InMemoryGraphStore};
    use eqiora_schema::kernel::{ExprNode, KernelNode};
    for dimensions in [2, 3] {
        let bounds = vec!["0,1"; dimensions].join(",");
        let source = format!(
            r#"model M() {{
            domain body=box({bounds});
            variable u:vector<m,{dimensions}> on body in smooth;
            relation law on body {{
                -div(2[Pa]*symmetric_part(grad(u))+3[Pa]*isotropic_lift(div(u)))=0;
            }}
        }}"#
        );
        let compiled = eqiora_compiler::CompiledModel::compile_selected(
            "native-calculus.eqi",
            &source,
            "M",
            &[],
        )
        .unwrap();
        let (transaction, model, _) = compiled.into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let relation = program
            .nodes()
            .find_map(|node| match node {
                KernelNode::Relation(r) => Some(r.id()),
                _ => None,
            })
            .unwrap();
        let typed = program.typed_relation_residual(relation).unwrap();
        let field = program
            .nodes()
            .find_map(|node| match node {
                KernelNode::Field(f) => Some(f.id().ulid().to_string()),
                _ => None,
            })
            .unwrap();
        let mut context = Context {
            name: "eta",
            field: &field,
            dimensions,
            remaining: 65536,
            test_dimension: eqiora_core::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0])
                .unwrap(),
            symbols: symbol_types(&program),
            supports: field_supports(&program),
            domains: domain_supports(&program),
            integration_domain: None,
        };
        let mut count = 0;
        for (index, node) in typed.expression().nodes().iter().enumerate() {
            if !matches!(
                node,
                ExprNode::SymmetricPart(_) | ExprNode::IsotropicLift(_)
            ) {
                continue;
            }
            count += 1;
            let id = typed.expression().node_id(index as u32).unwrap();
            let projected = E::from_expression(typed.expression(), id).unwrap().unwrap();
            for i in 0..dimensions {
                for j in 0..dimensions {
                    assert_eq!(
                        context.tensor(&projected, i, j, 0),
                        context.source(&typed, id, &[i, j], 0)
                    );
                }
            }
        }
        assert_eq!(count, 2);
    }
}

#[test]
fn structure_calculus_rejects_foreign_support_and_exhausted_budgets() {
    let mut context = context(3, false);
    context
        .symbols
        .insert("v".into(), context.symbols["u"].clone());
    context.supports.insert(
        "v".into(),
        SpatialSupport::Volume {
            domain: eqiora_core::Id::<eqiora_core::entity::kinds::Domain>::new().erase(),
            dimensions: 3,
        },
    );
    let canceled_foreign = E::IsotropicLift {
        value: Box::new(E::Mul {
            left: Box::new(E::Number { value: 0.0 }),
            right: Box::new(E::Divergence {
                value: Box::new(E::Field { ulid: "v".into() }),
            }),
        }),
    };
    assert!(context.tensor(&canceled_foreign, 0, 1, 0).is_none());
    context.remaining = 0;
    assert!(context.tensor(&symmetric(), 0, 0, 0).is_none());
    context.remaining = 65536;
    assert!(context.tensor(&symmetric(), 0, 0, 129).is_none());
}

#[test]
fn native_operator_volume_support_is_required_even_for_zero_components() {
    let mut context = context(3, false);
    for scalar in [
        E::Number { value: 0.0 },
        E::Rational {
            value: ExactRational::integer(0),
            dimension: eqiora_core::DimExponents::DIMENSIONLESS.exponents(),
        },
    ] {
        let unsupported = E::IsotropicLift {
            value: Box::new(scalar),
        };
        assert!(context.tensor(&unsupported, 0, 1, 0).is_none());
    }
    context.symbols.insert(
        "uniform_tensor".into(),
        ValueType::shaped(
            ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
            ValueShape::new([3, 3]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap(),
    );
    assert!(
        context
            .tensor(
                &E::SymmetricPart {
                    value: Box::new(E::Parameter {
                        ulid: "uniform_tensor".into()
                    })
                },
                0,
                0,
                0
            )
            .is_none()
    );
    let closed = E::SpatialTensor {
        complex: false,
        dimension: eqiora_core::DimExponents::DIMENSIONLESS.exponents(),
        shape: vec![3, 3],
        values: vec![(0.0, 0.0); 9],
    };
    assert!(
        context
            .tensor(
                &E::SymmetricPart {
                    value: Box::new(closed)
                },
                0,
                0,
                0
            )
            .is_none()
    );
    context.symbols.insert(
        "constant".into(),
        ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap(),
    );
    assert!(
        context
            .tensor(
                &E::IsotropicLift {
                    value: Box::new(E::Parameter {
                        ulid: "constant".into()
                    })
                },
                0,
                1,
                0
            )
            .is_none()
    );
}

#[test]
fn scalar_component_rejects_incompatible_full_values_before_cancellation() {
    let mut context = context(3, false);
    let space = eqiora_core::Id::<eqiora_core::entity::kinds::FiniteSpace>::new()
        .ulid()
        .to_string();
    let other_space = eqiora_core::Id::<eqiora_core::entity::kinds::FiniteSpace>::new()
        .ulid()
        .to_string();
    let map = |n, basis: &str| E::LinearMap {
        source_basis: vec![(basis.into(), n, false)],
        target_basis: vec![(basis.into(), n, false)],
        complex: false,
        dimension: DimExponents::DIMENSIONLESS.exponents(),
        values: vec![(0.0, 0.0); (n * n) as usize],
    };
    let spatial = E::SpatialTensor {
        complex: false,
        dimension: DimExponents::DIMENSIONLESS.exponents(),
        shape: vec![3, 3],
        values: vec![(0.0, 0.0); 9],
    };
    let component = |value| E::Component {
        value: Box::new(value),
        indices: vec![0, 0],
    };
    assert!(context.scalar(&component(map(3, &space)), 0).is_some());
    assert!(
        context
            .scalar(
                &component(E::Mul {
                    left: Box::new(E::Time),
                    right: Box::new(map(3, &space))
                }),
                0
            )
            .is_some()
    );
    let domain = match context.supports["u"].clone() {
        SpatialSupport::Volume { domain, .. } => domain,
        _ => unreachable!(),
    };
    context.domains.insert(
        "face".into(),
        SpatialSupport::Boundary {
            domain: eqiora_core::Id::<eqiora_core::entity::kinds::Domain>::new().erase(),
            parent: domain,
            dimensions: 3,
        },
    );
    context.integration_domain = Some("face".into());
    assert!(
        context
            .scalar(
                &E::Component {
                    value: Box::new(E::Trace {
                        value: Box::new(field()),
                        on_ulid: "face".into()
                    }),
                    indices: vec![0]
                },
                0
            )
            .is_some()
    );
    assert!(
        context
            .scalar(
                &E::Component {
                    value: Box::new(E::TangentialTrace {
                        value: Box::new(field()),
                        on_ulid: "face".into()
                    }),
                    indices: vec![0]
                },
                0
            )
            .is_some()
    );
    context.integration_domain = None;

    for (left, right) in [
        (spatial.clone(), map(2, &space)),
        (spatial, map(3, &space)),
        (map(3, &space), map(3, &other_space)),
    ] {
        let invalid_tensor = E::Add {
            left: Box::new(left),
            right: Box::new(right),
        };
        assert!(context.tensor(&invalid_tensor, 0, 0, 0).is_none());
        let pairing = E::Frobenius {
            left: Box::new(invalid_tensor.clone()),
            right: Box::new(invalid_tensor.clone()),
        };
        assert!(context.scalar(&pairing, 0).is_none());
        let invalid = component(invalid_tensor);
        let canceled = E::Sub {
            left: Box::new(invalid.clone()),
            right: Box::new(invalid),
        };
        assert!(context.scalar(&canceled, 0).is_none());
    }
}

#[test]
fn declared_test_units_survive_native_operand_reconstruction() {
    let mut context = context(3, false);
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let time = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    context.test_dimension = time;
    let test = E::Test {
        field_ulid: "u".into(),
    };
    let direction = E::Direction {
        name: "eta".into(),
        field_ulid: "u".into(),
    };
    for argument in [test, direction] {
        assert_eq!(
            context.retained_type(&argument, 0).unwrap().dimension(),
            time
        );
        let test_gradient = E::SymmetricPart {
            value: Box::new(E::Gradient {
                value: Box::new(argument),
            }),
        };
        assert_eq!(
            context
                .retained_type(&test_gradient, 0)
                .unwrap()
                .dimension(),
            time.div(length).unwrap()
        );
        let adjusted_trial = E::Mul {
            left: Box::new(E::Rational {
                value: ExactRational::integer(1),
                dimension: time.div(length).unwrap().exponents(),
            }),
            right: Box::new(symmetric()),
        };
        assert!(
            context
                .retained_type(
                    &E::Add {
                        left: Box::new(test_gradient.clone()),
                        right: Box::new(adjusted_trial)
                    },
                    0
                )
                .is_some()
        );
        assert!(
            context
                .retained_type(
                    &E::Add {
                        left: Box::new(test_gradient),
                        right: Box::new(symmetric())
                    },
                    0
                )
                .is_none()
        );
    }
}
