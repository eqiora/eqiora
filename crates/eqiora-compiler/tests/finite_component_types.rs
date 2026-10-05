use eqiora_compiler::CompiledModel;
use eqiora_core::{ScalarDomain, ValueType};
use eqiora_graph::Op;
use eqiora_schema::kernel::KernelNode;

fn compile(source: &str) -> CompiledModel {
    CompiledModel::compile_selected("finite-types.eqi", source, "M", &[])
        .unwrap_or_else(|errors| panic!("{errors:?}"))
}
fn field_types(model: &CompiledModel) -> Vec<ValueType> {
    model
        .transaction()
        .ops()
        .iter()
        .filter_map(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Field(field),
            } => Some(field.value_type().clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn finite_source_types_preserve_nominal_duality_dimensions_and_map_direction() {
    let source = "space Spin=orthonormal(up,down); space Control=orthonormal(position,velocity);
model M() { variable ket:coordinates<complex<1>,Spin>; variable bra:coordinates<complex<1>,dual<Spin>>; variable map_value:map<complex<1/s>,Spin,Control>; relation a {ket=ket;} relation b {bra=bra;} relation c {map_value=map_value;} }";
    let original = compile(source);
    let types = field_types(&original);
    assert_eq!(types.len(), 3);
    assert_eq!(
        types
            .iter()
            .filter(|ty| ty.coordinate_basis().is_some())
            .count(),
        2
    );
    assert!(
        types
            .iter()
            .any(|ty| ty.coordinate_basis().is_some_and(|basis| basis.is_dual()))
    );
    let map = types.iter().find(|ty| ty.map_bases().is_some()).unwrap();
    let (input, output) = map.map_bases().unwrap();
    assert_ne!(input.space(), output.space());
    assert_eq!(input.extent(), 2);
    assert_eq!(output.extent(), 2);
    assert_eq!(map.scalar_domain(), ScalarDomain::Complex);
    assert_eq!(
        map.dimension(),
        eqiora_core::DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap()
    );
    let document = eqiora_lang::parse("finite.eqi", source)
        .into_document()
        .unwrap();
    let formatted = eqiora_lang::format(&document);
    assert_eq!(
        original.transaction().ops(),
        compile(&formatted).transaction().ops()
    );
    let reordered = source.replace(
        "space Spin=orthonormal(up,down); space Control=orthonormal(position,velocity);",
        "space Control=orthonormal(position,velocity); space Spin=orthonormal(up,down);",
    );
    assert_eq!(field_types(&original), field_types(&compile(&reordered)));
}

#[test]
fn finite_source_types_reject_equal_extent_substitutions_and_unsupported_domains() {
    let base = "space A=orthonormal(a,b); space B=orthonormal(c,d); model M(){ variable x:coordinates<complex<1>,A>; variable y:coordinates<complex<1>,A>; relation r{x=y;} }";
    compile(base);
    for source in [
        base.replace(
            "variable y:coordinates<complex<1>,A>",
            "variable y:coordinates<complex<1>,B>",
        ),
        base.replace(
            "variable y:coordinates<complex<1>,A>",
            "variable y:coordinates<complex<1>,dual<A>>",
        ),
        base.replace(
            "variable y:coordinates<complex<1>,A>",
            "variable y:array<complex<1>,2>",
        ),
        base.replace("coordinates<complex<1>,A>", "map<integer,A,A>"),
        base.replace("coordinates<complex<1>,A>", "coordinates<bool,A>"),
        base.replace("orthonormal(a,b)", "metric(a,b)"),
    ] {
        assert!(
            CompiledModel::compile_selected("bad.eqi", &source, "M", &[]).is_err(),
            "{source}"
        );
    }
}

#[test]
fn finite_literals_use_the_existing_complex_and_dimensioned_scalar_evaluator() {
    let source = r#"space Spin=orthonormal(up,down); space Control=orthonormal(position,velocity);
model M() {
 parameter ket:coordinates<complex<1>,Spin> = coordinates(Spin,[math.complex(1,2),math.complex(3,-4)]);
 parameter bra:coordinates<complex<1>,dual<Spin>> = coordinates(dual(Spin),[math.complex(1,-2),math.complex(3,4)]);
 parameter h:map<complex<1>,Spin,Spin> = linear_map(Spin,Spin,[[0,math.complex(0,-1)],[math.complex(0,1),0]]);
 parameter a:map<1/s,Control,Control> = linear_map(Control,Control,[[-1[1/s],2[1/s]],[3[1/s],-4[1/s]]]);
 variable observed:coordinates<complex<1>,Spin>; relation r {observed=ket;}
}"#;
    let model = compile(source);
    let values = model
        .transaction()
        .ops()
        .iter()
        .filter_map(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Parameter(parameter),
            } => Some(parameter.value()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(values.len(), 4);
    let h = values
        .iter()
        .find(|value| {
            value.value_type().map_bases().is_some()
                && value.value_type().scalar_domain() == ScalarDomain::Complex
        })
        .unwrap();
    assert_eq!(
        (0..4).map(|i| h.component(i).unwrap()).collect::<Vec<_>>(),
        [(0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (0.0, 0.0)]
    );
    let a = values
        .iter()
        .find(|value| {
            value.value_type().map_bases().is_some()
                && value.value_type().scalar_domain() == ScalarDomain::Real
        })
        .unwrap();
    assert_eq!(
        (0..4)
            .map(|i| a.component(i).unwrap().0)
            .collect::<Vec<_>>(),
        [-1.0, 2.0, 3.0, -4.0]
    );
    for bad in [
        source.replace("linear_map(Spin,Spin,[[0,", "linear_map(Control,Spin,[[0,"),
        source.replace("coordinates(dual(Spin),", "coordinates(Spin),"),
        source.replace("[3[1/s],-4[1/s]]", "[3[s],-4[1/s]]"),
        source.replace(
            "[[0,math.complex(0,-1)],[math.complex(0,1),0]]",
            "[0,math.complex(0,-1),math.complex(0,1),0]",
        ),
        source.replace("math.complex(1,2)", "live_parameter"),
    ] {
        assert!(
            CompiledModel::compile_selected("bad.eqi", &bad, "M", &[]).is_err(),
            "{bad}"
        );
    }
}

#[test]
fn native_finite_map_projection_compiles_with_exact_registered_endpoints() {
    use eqiora_core::{DimExponents, Id, ValueLiteral};
    use eqiora_lang::{DraftDeclaration, DraftParameter, DraftRelation, Module};
    use eqiora_schema::kernel::FiniteSpaceDef;
    let source = FiniteSpaceDef::new(Id::new(), ["q0".into(), "q1".into()]).unwrap();
    let target = FiniteSpaceDef::new(Id::new(), ["u0".into(), "u1".into(), "u2".into()]).unwrap();
    let value_type = ValueType::linear_map(
        source.basis().dual(),
        target.basis().dual(),
        ScalarDomain::Complex,
        DimExponents::DIMENSIONLESS,
    )
    .unwrap();
    let value = ValueLiteral::new(
        value_type,
        [
            (1.0, 2.0),
            (3.0, -4.0),
            (-5.0, 6.0),
            (7.0, 8.0),
            (9.0, -10.0),
            (11.0, 12.0),
        ],
    )
    .unwrap();
    let parameter = DraftParameter::new("matrix", value.clone());
    let relation = DraftRelation::continuous(
        "retained",
        [(parameter.expression(), parameter.expression())],
    );
    assert!(Module::new("M", [parameter.clone().into(), relation.clone().into()]).is_err());
    let draft = Module::new(
        "M",
        [
            DraftDeclaration::FiniteSpace {
                name: "Input".into(),
                definition: source.clone(),
            },
            DraftDeclaration::FiniteSpace {
                name: "Output".into(),
                definition: target.clone(),
            },
            parameter.into(),
            relation.into(),
        ],
    )
    .unwrap();
    let model = eqiora_compiler::lower_module(&draft, None, &[])
        .unwrap_or_else(|errors| panic!("{errors:?}"));
    assert!(model.transaction().ops().iter().any(|op| matches!(op,Op::DefineKernelNode{node:KernelNode::Parameter(parameter)} if parameter.value()==&value)));
    let replay = compile(&eqiora_lang::format(draft.document()));
    assert!(replay.transaction().ops().iter().any(|op| matches!(op,Op::DefineKernelNode{node:KernelNode::Parameter(parameter)} if (0..6).all(|i|parameter.value().component(i)==value.component(i)))));
}

#[test]
fn imported_finite_map_aliases_share_exact_spaces_and_foreign_targets_reject() {
    use eqiora_compiler::{
        CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit,
        analyze_resolved_hierarchy,
    };
    let library = "public space Input=orthonormal(q0,q1); public space Output=orthonormal(u0,u1); space Hidden=orthonormal(h0,h1); public component Sink(parameter matrix:map<complex<1>,Input,Output>) { relation retained {matrix=matrix;} }";
    let source = "import org.example.finite.types as first; import org.example.finite.types as second; model Main(){ instance sink:first.Sink(matrix=linear_map(second.Input,second.Output,[[math.complex(1,2),0],[0,math.complex(3,4)]])); }";
    let compile_modules = |source: &str| {
        let owner =
            CompilationNamespaceId::new(["org.example.finite", "0.1.0", "finite-map-types"])
                .unwrap();
        let input = ResolvedHierarchyInput::new(
            owner.clone(),
            vec![
                ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", source).unwrap(),
                ResolvedSourceUnit::new(owner, "src/types.eqi", library).unwrap(),
            ],
            vec![],
        );
        analyze_resolved_hierarchy(input)?
            .validate_definitions()?
            .compile_root("Main")
    };
    compile_modules(source).unwrap_or_else(|errors| panic!("{errors:?}"));
    for wrong in [
        source.replace(
            "linear_map(second.Input,second.Output",
            "linear_map(second.Input,second.Input",
        ),
        source.replace(
            "linear_map(second.Input,second.Output",
            "linear_map(second.Input,second.Hidden",
        ),
    ] {
        assert!(compile_modules(&wrong).is_err(), "{wrong}");
    }
}

#[test]
fn finite_source_algebra_checks_exact_endpoints_and_retains_operations() {
    let source = r#"space Spin=orthonormal(up,down); space Control=orthonormal(q,v);
model M() {
 parameter h:map<complex<1>,Spin,Spin> = linear_map(Spin,Spin,[[0,math.complex(0,-1)],[math.complex(0,1),0]]);
 variable ket:coordinates<complex<1>,Spin>;
 variable applied:coordinates<complex<1>,Spin>;
 variable dual:coordinates<complex<1>,dual<Spin>>;
 variable transposed:map<complex<1>,dual<Spin>,dual<Spin>>;
 variable adjoint:map<complex<1>,Spin,Spin>;
 variable squared:map<complex<1>,Spin,Spin>;
 variable norm:complex<1>;
 relation a { applied=apply(h,ket); }
 relation b { dual=adjoint(ket); }
 relation c { transposed=transpose(h); }
 relation d { adjoint=adjoint(h); }
 relation e { squared=compose(h,h); }
 relation f { norm=pair(adjoint(ket),ket); }
}"#;
    let model = compile(source);
    let formatted = eqiora_lang::format(
        &eqiora_lang::parse("finite.eqi", source)
            .into_document()
            .unwrap(),
    );
    assert_eq!(
        model.transaction().ops(),
        compile(&formatted).transaction().ops()
    );
    for bad in [
        source.replace(
            "variable ket:coordinates<complex<1>,Spin>",
            "variable ket:coordinates<complex<1>,Control>",
        ),
        source.replace("pair(adjoint(ket),ket)", "pair(ket,ket)"),
        source.replace("apply(h,ket)", "apply(ket,h)"),
        source.replace("compose(h,h)", "compose(h,transpose(h))"),
        source.replace("adjoint(h)", "transpose(h)"),
        source.replace("apply(h,ket)", "apply(h)"),
        source.replace("adjoint(ket)", "adjoint(ket,ket)"),
    ] {
        assert!(
            CompiledModel::compile_selected("bad.eqi", &bad, "M", &[]).is_err(),
            "{bad}"
        );
    }
}

#[test]
fn native_finite_algebra_retains_registered_parameter_references() {
    use eqiora_core::{DimExponents, Id, ValueLiteral};
    use eqiora_lang::{DraftDeclaration, DraftParameter, DraftRelation, Module};
    use eqiora_schema::kernel::FiniteSpaceDef;
    let space = FiniteSpaceDef::new(Id::new(), ["up".into(), "down".into()]).unwrap();
    let basis = space.basis();
    let matrix = DraftParameter::new(
        "h",
        ValueLiteral::new(
            ValueType::linear_map(
                basis,
                basis,
                ScalarDomain::Complex,
                DimExponents::DIMENSIONLESS,
            )
            .unwrap(),
            [(0., 0.), (0., -1.), (0., 1.), (0., 0.)],
        )
        .unwrap(),
    );
    let vector = DraftParameter::new(
        "ket",
        ValueLiteral::new(
            ValueType::coordinates(basis, ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
                .unwrap(),
            [(1., 1.), (2., -1.)],
        )
        .unwrap(),
    );
    let mapped = matrix
        .expression()
        .compose_map(matrix.expression())
        .apply(vector.expression());
    let norm = vector.expression().adjoint().pair(vector.expression());
    let transpose = matrix.expression().transpose().transpose();
    let relation = DraftRelation::continuous(
        "retained",
        [
            (mapped, vector.expression()),
            (norm.clone(), norm),
            (transpose, matrix.expression()),
        ],
    );
    let declarations: Vec<DraftDeclaration> = vec![
        DraftDeclaration::FiniteSpace {
            name: "Spin".into(),
            definition: space,
        },
        matrix.into(),
        vector.into(),
        relation.into(),
    ];
    let draft = Module::new("M", declarations.clone()).unwrap();
    eqiora_compiler::lower_module(&draft, None, &[]).unwrap_or_else(|errors| panic!("{errors:?}"));
    compile(&eqiora_lang::format(draft.document()));
    let mut omitted = declarations;
    omitted.remove(1);
    assert!(Module::new("M", omitted).is_err());
}

#[test]
fn finite_product_aliases_are_structural_and_source_order_independent() {
    let source = r#"
space AB=product(A,B); space Alias=product(A,B); space BA=product(B,A);
space A=orthonormal(a,b); space B=orthonormal(x,y,z);
model M(){
 parameter x:coordinates<complex<1>,A> = coordinates(A,[math.complex(1,1),2]);
 parameter y:coordinates<complex<1>,B> = coordinates(B,[3,math.complex(0,-1),4]);
 variable xy:coordinates<complex<1>,AB>;
 variable alias:coordinates<complex<1>,Alias>;
 variable yx:coordinates<complex<1>,BA>;
 relation r {xy=tensor_product(x,y); alias=xy; yx=permute_factors(xy,[1,0]);}
}"#;
    let original = compile(source);
    // Transaction order follows canonical IDs, not declaration order. Select
    // the exact named fields so a source-identity epoch cannot swap the oracle.
    let ordered_fields = |model: &CompiledModel| {
        ["xy", "alias", "yx"].map(|name| {
            let id = model.symbols().get(name).unwrap();
            model
                .transaction()
                .ops()
                .iter()
                .find_map(|op| match op {
                    Op::DefineKernelNode {
                        node: KernelNode::Field(field),
                    } if field.id().erase() == id => Some(field.value_type().clone()),
                    _ => None,
                })
                .unwrap()
        })
    };
    let fields = ordered_fields(&original);
    assert_eq!(fields[0], fields[1]);
    assert_ne!(fields[0], fields[2]);
    let reordered = source
        .replace("space AB=product(A,B);", "")
        .replace("model M()", "space AB=product(A,B); model M()");
    assert_eq!(fields, ordered_fields(&compile(&reordered)));
    let formatted = eqiora_lang::format(
        &eqiora_lang::parse("finite.eqi", source)
            .into_document()
            .unwrap(),
    );
    assert_eq!(
        original.transaction().ops(),
        compile(&formatted).transaction().ops()
    );
    for invalid in [
        source.replace("[1,0]", "[0,0]"),
        source.replace("[1,0]", "[2,0]"),
        source.replace("permute_factors(xy,[1,0])", "xy"),
        source.replace("product(A,B)", "product(BA,B)"),
        source.replace("product(A,B)", "product(dual(A),B)"),
    ] {
        assert!(
            CompiledModel::compile_selected("bad.eqi", &invalid, "M", &[]).is_err(),
            "{invalid}"
        );
    }
}

#[test]
fn imported_product_aliases_resolve_atomic_factors_before_any_declaration_order() {
    use eqiora_compiler::{
        CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit,
        analyze_resolved_hierarchy,
    };
    let library = "public space Pair=product(A,B); public space A=orthonormal(a,b); public space B=orthonormal(x,y,z);";
    let source = "import org.example.finite.types as lib; space Local=product(lib.A,lib.B); model Main(){variable x:coordinates<1,Local>; variable y:coordinates<1,lib.Pair>; relation r{x=y;}}";
    let owner =
        CompilationNamespaceId::new(["org.example.finite", "0.1.0", "finite-product-types"])
            .unwrap();
    let input = ResolvedHierarchyInput::new(
        owner.clone(),
        vec![
            ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", source).unwrap(),
            ResolvedSourceUnit::new(owner, "src/types.eqi", library).unwrap(),
        ],
        vec![],
    );
    let compiled = analyze_resolved_hierarchy(input)
        .unwrap()
        .validate_definitions()
        .unwrap()
        .compile_root("Main")
        .unwrap();
    let types = field_types(&compiled);
    assert_eq!(types[0], types[1]);
}

#[test]
fn native_product_aliases_project_without_replacing_exact_factor_identities() {
    use eqiora_core::{DimExponents, Id, ValueLiteral};
    use eqiora_lang::{DraftDeclaration, DraftParameter, DraftRelation, Module};
    use eqiora_schema::kernel::FiniteSpaceDef;
    let a = FiniteSpaceDef::new(Id::new(), ["a".into(), "b".into()]).unwrap();
    let b = FiniteSpaceDef::new(Id::new(), ["x".into(), "y".into(), "z".into()]).unwrap();
    let ab = FiniteSpaceDef::product(Id::new(), a.basis(), b.basis()).unwrap();
    let p = DraftParameter::new(
        "state",
        ValueLiteral::new(
            ab.coordinates(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
                .unwrap(),
            [(3., 3.), (1., -1.), (4., 4.), (6., 0.), (0., -2.), (8., 0.)],
        )
        .unwrap(),
    );
    let relation = DraftRelation::continuous(
        "r",
        [(
            p.expression(),
            p.expression()
                .permute_factors([1, 0])
                .permute_factors([1, 0]),
        )],
    );
    let declarations = vec![
        DraftDeclaration::FiniteSpace {
            name: "AB".into(),
            definition: ab,
        },
        DraftDeclaration::FiniteSpace {
            name: "A".into(),
            definition: a,
        },
        DraftDeclaration::FiniteSpace {
            name: "B".into(),
            definition: b,
        },
        p.into(),
        relation.into(),
    ];
    let draft = Module::new("M", declarations.clone()).unwrap();
    eqiora_compiler::lower_module(&draft, None, &[]).unwrap();
    compile(&eqiora_lang::format(draft.document()));
    let omitted = declarations
        .into_iter()
        .enumerate()
        .filter_map(|(i, d)| (i != 1).then_some(d));
    assert!(Module::new("M", omitted).is_err());
}

#[test]
fn finite_three_channels_reject_spatial_vectors_and_implicit_reshape() {
    let source = "space Channels=orthonormal(a,b,c); model M(){parameter rhs:coordinates<1,Channels>=coordinates(Channels,[1,2,3]); variable state:coordinates<1,Channels>; relation r{state=rhs;}}";
    compile(source);
    for ty in ["vector<1,3>", "array<1,3>"] {
        let invalid = source.replace(
            "variable state:coordinates<1,Channels>",
            &format!("variable state:{ty}"),
        );
        assert!(CompiledModel::compile_selected("bad.eqi", &invalid, "M", &[]).is_err());
    }
}

#[test]
fn native_space_invalid_labels_reject_before_source_projection() {
    use eqiora_lang::{DraftDeclaration, Module};
    use eqiora_schema::kernel::FiniteSpaceDef;
    for label in ["not a name", "a.b", "1"] {
        let definition = FiniteSpaceDef::new(eqiora_core::Id::new(), [label.to_owned()]).unwrap();
        assert!(
            Module::new(
                "M",
                [DraftDeclaration::FiniteSpace {
                    name: "A".into(),
                    definition
                }]
            )
            .is_err()
        );
    }
}

#[test]
fn inferred_finite_map_units_span_zero_rows_without_erasing_explicit_units() {
    let source = "space A=orthonormal(a,b,c); model M(){ variable u:coordinates<complex<1>,A>; variable y:coordinates<complex<J>,A>; relation r{y=apply(linear_map(A,A,[[0,2[J],0],[0,0,0],[0,0,math.complex(0[J],3[J])]]),u);}}";
    compile(source);
    // A bare zero is contextual even in an entirely zero row. An explicitly
    // dimensioned zero must not be recast, nor may a nonzero dimensionless value.
    for bad in [
        source.replace("[0,0,0]", "[0,0[s],0]"),
        source.replace("[0,0,0]", "[0,1,0]"),
        source.replace("[0,0,0]", "[0,0]"),
    ] {
        assert!(CompiledModel::compile_selected("bad-finite.eqi", &bad, "M", &[]).is_err());
    }
}
