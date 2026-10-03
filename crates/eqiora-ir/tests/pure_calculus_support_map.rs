use eqiora_core::ValueFrame;
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, Id, ValueShape};
use eqiora_ir::{
    CalculusBuilder, CalculusError, CalculusNode, ComponentIndex, ComponentScalarization,
    ExactRational, OperatorApplicationProof, OperatorExpansionExt, PureOperatorDefinition,
    PureValueClass, StandardPureOperator, SupportMap, SupportMapOrientation, SupportMapPairing,
};
use eqiora_schema::kernel::typing::{ExpressionType, RootContract, SpatialSupport, TypedResidual};
use eqiora_schema::kernel::{ExprDagBuilder, SymbolRef};

fn volume_tensor(domain: &'static str) -> ExpressionType<&'static str> {
    volume_tensor_with_dimension(domain, DimExponents::DIMENSIONLESS)
}

fn volume_tensor_with_dimension(
    domain: &'static str,
    dimension: DimExponents,
) -> ExpressionType<&'static str> {
    ExpressionType::shaped(
        dimension,
        ValueShape::new([2, 2]).expect("shape"),
        ValueFrame::SpatialCartesian,
        Some(SpatialSupport::Volume {
            domain,
            dimensions: 2,
        }),
    )
    .unwrap()
}

fn volume_scalar(domain: &'static str) -> ExpressionType<&'static str> {
    ExpressionType::scalar(
        DimExponents::DIMENSIONLESS,
        Some(SpatialSupport::Volume {
            domain,
            dimensions: 2,
        }),
    )
}

fn equivalent_symmetry_definition(expanded: bool) -> PureOperatorDefinition {
    let tensor = PureValueClass::spatial_tensor(2).expect("tensor class");
    let mut definition = CalculusBuilder::new([tensor], PureValueClass::spatial_tensor(2).unwrap())
        .expect("definition");
    let direct = definition
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: [ComponentIndex::Result(0), ComponentIndex::Result(1)].into(),
        })
        .expect("direct component");
    let transposed = definition
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: [ComponentIndex::Result(1), ComponentIndex::Result(0)].into(),
        })
        .expect("transposed component");
    let sum = definition
        .push(CalculusNode::Add(direct, transposed))
        .expect("sum");
    let root = if expanded {
        sum
    } else {
        let half = definition
            .push(CalculusNode::Rational {
                value: ExactRational::new(1, 2).expect("half"),
                dimension: eqiora_core::DimExponents::DIMENSIONLESS,
            })
            .expect("half node");
        let symmetric = definition
            .push(CalculusNode::Mul(half, sum))
            .expect("symmetric part");
        let two = definition
            .push(CalculusNode::Rational {
                value: ExactRational::integer(2),
                dimension: eqiora_core::DimExponents::DIMENSIONLESS,
            })
            .expect("two node");
        definition
            .push(CalculusNode::Mul(two, symmetric))
            .expect("twice symmetric part")
    };
    definition.finish(root).expect("valid definition")
}

#[test]
fn standard_tensor_operators_expand_and_exact_equivalence_replays() {
    let field = Id::<kinds::Field>::new();
    let mut expression = ExprDagBuilder::new();
    let field_value = expression.symbol(SymbolRef::Field(field)).expect("field");
    let symmetric = expression
        .symmetric_part(field_value)
        .expect("symmetric part");
    let residual = expression.finish([symmetric]).expect("residual");
    let tensor_type = volume_tensor("body");
    let typed = TypedResidual::infer(
        residual,
        Some(SpatialSupport::Volume {
            domain: "body",
            dimensions: 2,
        }),
        RootContract::ComponentwiseResidual,
        |_| Ok::<_, ()>(tensor_type.clone()),
    )
    .expect("typed residual");
    let application =
        OperatorApplicationProof::classify(&typed, symmetric, StandardPureOperator::SymmetricPart)
            .expect("classification")
            .expect("standard application");
    assert_eq!(application.operand(), field_value);
    assert_eq!(
        application.definition_digest(),
        PureOperatorDefinition::symmetric_part()
            .expect("standard definition")
            .digest()
    );

    let scalar_field = Id::<kinds::Field>::new();
    let mut isotropic_expression = ExprDagBuilder::new();
    let scalar = isotropic_expression
        .symbol(SymbolRef::Field(scalar_field))
        .expect("scalar field");
    let isotropic = isotropic_expression
        .isotropic_lift(scalar)
        .expect("isotropic lift");
    let isotropic_residual = isotropic_expression
        .finish([isotropic])
        .expect("isotropic residual");
    let isotropic_typed = TypedResidual::infer(
        isotropic_residual,
        Some(SpatialSupport::Volume {
            domain: "body",
            dimensions: 2,
        }),
        RootContract::ComponentwiseResidual,
        |_| Ok::<_, ()>(volume_scalar("body")),
    )
    .expect("typed isotropic residual");
    let isotropic_application = OperatorApplicationProof::classify(
        &isotropic_typed,
        isotropic,
        StandardPureOperator::IsotropicLift,
    )
    .expect("isotropic classification")
    .expect("standard isotropic application");
    assert_eq!(
        isotropic_application.definition_digest(),
        PureOperatorDefinition::isotropic_lift()
            .expect("isotropic definition")
            .digest()
    );
    let isotropic_definition =
        PureOperatorDefinition::isotropic_lift().expect("isotropic definition");
    let isotropic_arguments = [volume_scalar("body")];
    let isotropic_expansion = isotropic_definition
        .instantiate(&isotropic_arguments)
        .expect("isotropic typing");
    let diagonal = isotropic_expansion
        .component(&[0, 0])
        .expect("diagonal component");
    let off_diagonal = isotropic_expansion
        .component(&[0, 1])
        .expect("off-diagonal component");
    assert_ne!(
        diagonal.normalize().expect("diagonal proof").after_digest(),
        off_diagonal
            .normalize()
            .expect("off-diagonal proof")
            .after_digest()
    );

    let arguments = [tensor_type];
    let compact_definition = equivalent_symmetry_definition(false);
    let expanded_definition = equivalent_symmetry_definition(true);
    assert_ne!(compact_definition.digest(), expanded_definition.digest());
    let compact = compact_definition
        .instantiate(&arguments)
        .expect("compact typing")
        .component(&[0, 1])
        .expect("compact component");
    let expanded = expanded_definition
        .instantiate(&arguments)
        .expect("expanded typing")
        .component(&[0, 1])
        .expect("expanded component");
    let compact_proof = compact.normalize().expect("compact proof");
    let expanded_proof = expanded.normalize().expect("expanded proof");
    compact_proof.verify(&compact).expect("compact replay");
    expanded_proof.verify(&expanded).expect("expanded replay");
    assert!(compact_proof.same_normal_form(&expanded_proof));
    assert!(compact_proof.verify(&expanded).is_err());

    let pressure = DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
    let dimensioned_arguments = [volume_tensor_with_dimension("body", pressure)];
    let dimensioned = compact_definition
        .instantiate(&dimensioned_arguments)
        .expect("dimensioned typing")
        .component(&[0, 1])
        .expect("dimensioned component");
    let dimensioned_proof = dimensioned.normalize().expect("dimensioned proof");
    assert_eq!(
        compact_proof.verify(&dimensioned),
        Err(CalculusError::ProofTypeMismatch)
    );
    assert!(!compact_proof.same_normal_form(&dimensioned_proof));
}

#[test]
fn component_lowering_retains_distinct_parameter_coordinates() {
    let tensor = Id::<kinds::Port>::new();
    let first = Id::<kinds::Parameter>::new();
    let second = Id::<kinds::Parameter>::new();
    let mut expression = ExprDagBuilder::new();
    let tensor_value = expression
        .symbol(SymbolRef::PortFlux(tensor))
        .expect("tensor");
    let symmetric = expression
        .symmetric_part(tensor_value)
        .expect("symmetric part");
    let first_value = expression
        .symbol(SymbolRef::Parameter(first))
        .expect("first Parameter");
    let second_value = expression
        .symbol(SymbolRef::Parameter(second))
        .expect("second Parameter");
    let first_term = expression.mul(first_value, symmetric).expect("first term");
    let second_term = expression
        .mul(second_value, symmetric)
        .expect("second term");
    let root = expression.add(first_term, second_term).expect("root");
    let residual = expression.finish([root]).expect("residual");
    let support = SpatialSupport::Volume {
        domain: "body",
        dimensions: 2,
    };
    let typed = TypedResidual::infer(
        residual,
        Some(support.clone()),
        RootContract::ComponentwiseResidual,
        |symbol| {
            Ok::<_, ()>(match symbol {
                SymbolRef::PortFlux(_) => volume_tensor("body"),
                SymbolRef::Parameter(_) => {
                    ExpressionType::scalar(DimExponents::DIMENSIONLESS, None)
                }
                _ => unreachable!("closed fixture"),
            })
        },
    )
    .expect("typed residual");
    let lowered = ComponentScalarization::lower(&typed).expect("component scalarization");
    let off_diagonal = lowered
        .rows()
        .iter()
        .find(|row| row.component_index() == [0, 1])
        .expect("off-diagonal row");
    let parameters = off_diagonal
        .symbols()
        .iter()
        .filter_map(|coordinate| match coordinate.symbol() {
            SymbolRef::Parameter(parameter) => Some(parameter),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(parameters.contains(&first));
    assert!(parameters.contains(&second));
    assert_ne!(first, second);

    assert_eq!(
        off_diagonal.input_slots().len(),
        off_diagonal.symbols().len()
    );
    for (ordinal, (slot, coordinate)) in off_diagonal
        .input_slots()
        .iter()
        .zip(off_diagonal.symbols())
        .enumerate()
    {
        assert_eq!(usize::try_from(slot.ordinal()), Ok(ordinal));
        assert_eq!(slot.source(), coordinate);
        assert!(
            matches!(
                slot.source().symbol(),
                SymbolRef::PortFlux(id) if id == tensor
            ) || matches!(
                slot.source().symbol(),
                SymbolRef::Parameter(id) if id == first || id == second
            )
        );
    }
}

#[test]
fn trace_support_map_is_semantic_and_fails_closed_on_parent_identity() {
    let field = Id::<kinds::Field>::new();
    let volume = SpatialSupport::Volume {
        domain: "body",
        dimensions: 2,
    };
    let boundary = SpatialSupport::Boundary {
        domain: "wall",
        parent: "body",
        dimensions: 2,
    };
    let mut expression = ExprDagBuilder::new();
    let value = expression.symbol(SymbolRef::Field(field)).expect("field");
    let trace = expression.trace(value).expect("trace");
    let residual = expression.finish([trace]).expect("residual");
    let typed = TypedResidual::infer(
        residual,
        Some(boundary.clone()),
        RootContract::ComponentwiseResidual,
        |_| {
            Ok::<_, ()>(ExpressionType::scalar(
                DimExponents::DIMENSIONLESS,
                Some(volume.clone()),
            ))
        },
    )
    .expect("typed trace");
    let map = SupportMap::classify_trace(&typed, trace)
        .expect("support-map classification")
        .expect("trace map");
    assert_eq!(map.source(), &volume);
    assert_eq!(map.target(), &boundary);
    assert_eq!(map.orientation(), SupportMapOrientation::ParentOutward);
    assert_eq!(map.pairing(), SupportMapPairing::PointwiseValue);

    let foreign = SpatialSupport::Boundary {
        domain: "wall",
        parent: "other",
        dimensions: 2,
    };
    assert!(SupportMap::trace(volume, foreign).is_err());
}
