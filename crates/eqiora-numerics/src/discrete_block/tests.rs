use std::num::NonZeroU16;

use eqiora_assembly::{
    AssemblyMap, AssemblyTarget, DofId, IndexedAssemblyWork, LocalContribution, LocalUnknown,
    REFERENCE_ASSEMBLY_BACKEND, TargetAssemblyMap,
};
use eqiora_core::entity::kinds;
use eqiora_core::{Id, OntologyId};
use eqiora_realization::{RealizationRevision, SemanticRevision, TraceFieldEndpoint};

use super::*;

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const TIME: DimExponents =
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension");

#[test]
fn block_identity_preserves_complete_field_types_without_admitting_nonreal_execution() {
    let mut system = minimal(MinimalIds::new(), false);
    let real = ValueType::scalar(ScalarDomain::Real, LENGTH).expect("numeric scalar type");
    let complex = ValueType::scalar(ScalarDomain::Complex, LENGTH).expect("numeric scalar type");
    let integer = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
        .expect("numeric scalar type");
    let vector = ValueType::shaped(
        ScalarDomain::Real,
        LENGTH,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap();
    let tensor = ValueType::shaped(
        ScalarDomain::Real,
        LENGTH,
        ValueShape::new([2, 2]).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap();
    let channels = real.clone().array(2).unwrap();
    let vector_channels = vector.clone().array(2).unwrap();
    assert_eq!(vector.shape(), channels.shape());
    assert_eq!(tensor.shape(), vector_channels.shape());
    assert_eq!(tensor.frame(), vector_channels.frame());
    let types = [
        real,
        complex,
        integer,
        ValueType::enumeration(Id::new(), 2).unwrap(),
        ValueType::enumeration(Id::new(), 2).unwrap(),
        vector,
        channels,
        tensor,
        vector_channels,
    ];
    let mut identities = std::collections::HashSet::new();
    for value_type in types {
        system.fields[0] = FieldBlock::discrete(
            system.fields[0].domain,
            system.fields[0].field,
            system.fields[0].space.unwrap(),
            value_type.clone(),
            DynQuantity::new(1.0, value_type.dimension()),
            FieldBlockRole::Algebraic,
        )
        .unwrap();
        assert_eq!(system.fields[0].value_type, value_type);
        assert!(identities.insert(system.compute_identity().0));
        if value_type.scalar_domain() != ScalarDomain::Real {
            let error = system.validate().unwrap_err();
            assert_eq!(error.code(), codes::INVALID_REALIZATION);
            assert_eq!(
                error.message(),
                "discrete block execution requires real Field types"
            );
        } else {
            system.validate().unwrap();
        }
    }
}

#[test]
fn complete_field_type_retains_scale_validation() {
    let ids = MinimalIds::new();
    for scale in [
        DynQuantity::new(1.0, TIME),
        DynQuantity::new(0.0, LENGTH),
        DynQuantity::new(-1.0, LENGTH),
        DynQuantity::new(f64::INFINITY, LENGTH),
    ] {
        assert!(
            FieldBlock::discrete(
                ids.domain,
                ids.fields[0],
                Space::continuous_lagrange(NonZeroU16::MIN),
                ValueType::scalar(ScalarDomain::Real, LENGTH).expect("numeric scalar type"),
                scale,
                FieldBlockRole::Algebraic
            )
            .is_err()
        );
    }
}

#[derive(Clone, Copy)]
struct MinimalIds {
    model: OntologyId<Model>,
    domain: Id<kinds::Domain>,
    fields: [Id<kinds::Field>; 2],
    relations: [Id<kinds::Relation>; 2],
    parameters: [Id<kinds::Parameter>; 2],
}

impl MinimalIds {
    fn new() -> Self {
        Self {
            model: OntologyId::new(),
            domain: Id::new(),
            fields: [Id::new(), Id::new()],
            relations: [Id::new(), Id::new()],
            parameters: [Id::new(), Id::new()],
        }
    }
}

fn minimal(ids: MinimalIds, order_reversed: bool) -> DiscreteBlockSystem {
    let mut fields = ids
        .fields
        .into_iter()
        .map(|field| {
            FieldBlock::discrete(
                ids.domain,
                field,
                Space::continuous_lagrange(NonZeroU16::MIN),
                ValueType::scalar(ScalarDomain::Real, LENGTH).expect("numeric scalar type"),
                DynQuantity::new(1.0, LENGTH),
                FieldBlockRole::Algebraic,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut relations = ids
        .fields
        .into_iter()
        .zip(ids.relations)
        .map(|(field, relation)| {
            RelationBlock::new(
                relation,
                BlockSupport::Volume(ids.domain),
                RelationDisposition::Residual {
                    tested: AlgebraicBlock::Field(field),
                },
            )
        })
        .collect::<Vec<_>>();
    let mut residuals = ids
        .fields
        .into_iter()
        .zip(ids.relations)
        .map(|(field, relation)| {
            ResidualBlock::new(
                AlgebraicBlock::Field(field),
                BlockSupport::Volume(ids.domain),
                [ResidualOrigin::Relation(relation)],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut contributions = ids
        .fields
        .into_iter()
        .zip(ids.relations)
        .zip(ids.parameters)
        .enumerate()
        .map(|(packet, ((field, relation), parameter))| {
            ContributionBatch::new(
                [BlockSupport::Volume(ids.domain)],
                [packet],
                [0],
                [ResidualOrigin::Relation(relation)],
                [parameter],
                [AlgebraicBlock::Field(field)],
                [AlgebraicBlock::Field(field)],
                [ContributionTerm::Stiffness],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut closures = ids
        .fields
        .into_iter()
        .zip(ids.relations)
        .map(|(field, relation)| AlgebraicClosure::CompleteOperator {
            field,
            relations: vec![relation],
        })
        .collect::<Vec<_>>();
    if order_reversed {
        fields.reverse();
        relations.reverse();
        residuals.reverse();
        contributions.reverse();
        closures.reverse();
    }
    DiscreteBlockSystem::new(
        DiscreteBlockContext::new(
            ids.model,
            SemanticRevision::new(3),
            BlockRealizationIdentity::Explicit(RealizationRevision::new(4)),
            None,
        ),
        fields,
        vec![],
        relations,
        residuals,
        vec![],
        closures,
        contributions,
        2,
        1,
        0,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap()
}

fn stateful(ids: MinimalIds) -> DiscreteBlockSystem {
    let base = minimal(ids, false);
    let state = Id::<kinds::Field>::new();
    let state_relation = Id::<kinds::Relation>::new();
    let mut fields = base.fields;
    fields.push(
        FieldBlock::discrete(
            ids.domain,
            state,
            Space::continuous_lagrange(NonZeroU16::MIN),
            ValueType::scalar(ScalarDomain::Real, LENGTH).expect("numeric scalar type"),
            DynQuantity::new(1.0, LENGTH),
            FieldBlockRole::EliminatedState,
        )
        .unwrap(),
    );
    let mut relations = base.relations;
    relations.push(RelationBlock::new(
        state_relation,
        BlockSupport::Volume(ids.domain),
        RelationDisposition::StateElimination {
            state,
            rate: ids.fields[0],
        },
    ));
    DiscreteBlockSystem::new(
        base.context,
        fields,
        base.auxiliaries,
        relations,
        base.residuals,
        vec![BlockTransformation::BackwardEulerElimination {
            relation: state_relation,
            state,
            rate: ids.fields[0],
            duration: DynQuantity::new(0.25, TIME),
        }],
        base.closures,
        base.contributions,
        base.packet_count,
        base.target_count,
        base.primary_target,
        base.required_properties,
    )
    .unwrap()
}

fn coupled(ids: MinimalIds) -> DiscreteBlockSystem {
    let base = minimal(ids, false);
    let second_domain = Id::<kinds::Domain>::new();
    let connection = Id::<kinds::Connection>::new();
    let interface_relations = [Id::<kinds::Relation>::new(), Id::new()];
    let boundaries = [Id::<kinds::Domain>::new(), Id::new()];
    let mut fields = base.fields;
    fields
        .iter_mut()
        .find(|field| field.field == ids.fields[1])
        .unwrap()
        .domain = second_domain;
    let mut relations = base.relations;
    relations
        .iter_mut()
        .find(|relation| relation.relation == ids.relations[1])
        .unwrap()
        .support = BlockSupport::Volume(second_domain);
    for side in 0..2 {
        relations.push(RelationBlock::new(
            interface_relations[side],
            BlockSupport::Boundary(boundaries[side]),
            RelationDisposition::BoundaryCondition {
                field: ids.fields[side],
                treatment: BoundaryTreatment::ConformingInterface { connection },
            },
        ));
    }
    let mut residuals = base.residuals;
    residuals
        .iter_mut()
        .find(|residual| residual.tested == AlgebraicBlock::Field(ids.fields[1]))
        .unwrap()
        .support = BlockSupport::Volume(second_domain);
    let mut contributions = base.contributions;
    contributions
        .iter_mut()
        .find(|contribution| {
            contribution
                .row_blocks
                .contains(&AlgebraicBlock::Field(ids.fields[1]))
        })
        .unwrap()
        .supports = vec![BlockSupport::Volume(second_domain)];
    DiscreteBlockSystem::new(
        base.context,
        fields,
        base.auxiliaries,
        relations,
        residuals,
        vec![BlockTransformation::ConformingTraceQuotient {
            quotient: ConformingTraceQuotient::new(
                connection,
                TraceFieldEndpoint::new(ids.domain, ids.fields[0]),
                TraceFieldEndpoint::new(second_domain, ids.fields[1]),
            )
            .unwrap(),
            interface_relations: interface_relations.to_vec(),
        }],
        base.closures,
        contributions,
        base.packet_count,
        base.target_count,
        base.primary_target,
        base.required_properties,
    )
    .unwrap()
}

#[test]
fn construction_order_does_not_change_block_identity() {
    let ids = MinimalIds::new();
    let direct = minimal(ids, false);
    let reversed = minimal(ids, true);
    assert_eq!(direct.identity, reversed.identity);
    assert_ne!(direct.identity.0, [0; 32]);
}

#[test]
fn omitted_packet_and_unknown_relation_fail_closed() {
    let system = minimal(MinimalIds::new(), false);
    let mut omitted = system.clone();
    omitted.packet_count = 3;
    assert_eq!(
        omitted.validate().unwrap_err().code(),
        codes::INVALID_REALIZATION
    );

    let mut foreign = system;
    foreign.contributions[0].origins[0] = ResidualOrigin::Relation(Id::<kinds::Relation>::new());
    assert_eq!(
        foreign.validate().unwrap_err().code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn exact_parameter_support_and_time_elimination_fail_closed() {
    let ids = MinimalIds::new();
    let direct = minimal(ids, false);
    let mut changed_ids = ids;
    changed_ids.parameters[1] = Id::new();
    assert_ne!(direct.identity, minimal(changed_ids, false).identity);

    let mut wrong_support = direct;
    wrong_support.contributions[0].supports =
        vec![BlockSupport::Boundary(Id::<kinds::Domain>::new())];
    assert_eq!(
        wrong_support.validate().unwrap_err().code(),
        codes::INVALID_REALIZATION
    );

    let mut missing_step = stateful(ids);
    missing_step.transformations.clear();
    assert_eq!(
        missing_step.validate().unwrap_err().code(),
        codes::INVALID_REALIZATION
    );

    let mut missing_quotient = coupled(ids);
    missing_quotient.transformations.clear();
    assert_eq!(
        missing_quotient.validate().unwrap_err().code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn checked_work_rejects_packet_target_drift_before_scatter() {
    let mut system = minimal(MinimalIds::new(), false);
    system.target_count = 2;
    let plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(1).unwrap(),
        AssemblyTarget::new(1).unwrap(),
    ])
    .unwrap();
    let expected = plan.target_id(0).unwrap();
    let foreign = plan.target_id(1).unwrap();
    let work = IndexedAssemblyWork::new(2, move |packet| {
        let target = if packet == 0 { foreign } else { expected };
        AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![1.0], vec![0.0])?,
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![Some(DofId::new(0))],
                    vec![LocalUnknown::Free(DofId::new(0))],
                )?,
            )],
        )
    });
    let diagnostic = system
        .checked_backend(&REFERENCE_ASSEMBLY_BACKEND)
        .assemble(&plan, &work)
        .unwrap_err();
    assert_eq!(diagnostic.code(), codes::INVALID_REALIZATION);
}

mod plural;
