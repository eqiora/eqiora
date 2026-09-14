use super::*;

fn rebuild(s: DiscreteBlockSystem) -> Result<DiscreteBlockSystem, Diagnostic> {
    DiscreteBlockSystem::new(
        s.context,
        s.fields,
        s.auxiliaries,
        s.relations,
        s.residuals,
        s.transformations,
        s.closures,
        s.contributions,
        s.packet_count,
        s.target_count,
        s.primary_target,
        s.required_properties,
    )
}

fn chain() -> DiscreteBlockSystem {
    let ids = MinimalIds::new();
    let mut s = coupled(ids);
    let domain = Id::new();
    let field = Id::new();
    let relation = Id::new();
    let connection = Id::new();
    let previous_domain = s
        .fields
        .iter()
        .find(|f| f.field == ids.fields[1])
        .unwrap()
        .domain;
    let mut block = s.fields[0].clone();
    block.field = field;
    block.domain = domain;
    s.fields.push(block);
    s.relations.push(RelationBlock::new(
        relation,
        BlockSupport::Volume(domain),
        RelationDisposition::Residual {
            tested: AlgebraicBlock::Field(field),
        },
    ));
    s.residuals.push(
        ResidualBlock::new(
            AlgebraicBlock::Field(field),
            BlockSupport::Volume(domain),
            [ResidualOrigin::Relation(relation)],
        )
        .unwrap(),
    );
    s.contributions.push(
        ContributionBatch::new(
            [BlockSupport::Volume(domain)],
            [2],
            [0],
            [ResidualOrigin::Relation(relation)],
            [],
            [AlgebraicBlock::Field(field)],
            [AlgebraicBlock::Field(field)],
            [ContributionTerm::Stiffness],
        )
        .unwrap(),
    );
    s.closures.push(AlgebraicClosure::CompleteOperator {
        field,
        relations: vec![relation],
    });
    let interface_relations = [Id::new(), Id::new()];
    for (field, relation) in [ids.fields[1], field].into_iter().zip(interface_relations) {
        s.relations.push(RelationBlock::new(
            relation,
            BlockSupport::Boundary(Id::new()),
            RelationDisposition::BoundaryCondition {
                field,
                treatment: BoundaryTreatment::ConformingInterface { connection },
            },
        ));
    }
    s.transformations
        .push(BlockTransformation::ConformingTraceQuotient {
            quotient: ConformingTraceQuotient::new(
                connection,
                TraceFieldEndpoint::new(previous_domain, ids.fields[1]),
                TraceFieldEndpoint::new(domain, field),
            )
            .unwrap(),
            interface_relations: interface_relations.to_vec(),
        });
    s.packet_count = 3;
    rebuild(s).unwrap()
}

#[test]
fn plural_quotients_preserve_exact_identity_and_checked_assembly() {
    let s = chain();
    let mut reverse = s.clone();
    reverse.fields.reverse();
    reverse.relations.reverse();
    reverse.transformations.reverse();
    reverse.contributions.reverse();
    assert_eq!(s.identity, rebuild(reverse).unwrap().identity);
    // Three springs tied to one coordinate: K=1+2+3=6, f=2K=12.
    // This independent packet fixture checks plural inventory through actual assembly.
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    let target = plan.target_id(0).unwrap();
    let work = IndexedAssemblyWork::new(3, move |packet| {
        let k = (packet + 1) as f64;
        AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![k], vec![2.0 * k])?,
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![Some(DofId::new(0))],
                    vec![LocalUnknown::Free(DofId::new(0))],
                )?,
            )],
        )
    });
    let result = s
        .checked_backend(&REFERENCE_ASSEMBLY_BACKEND)
        .assemble(&plan, &work)
        .unwrap();
    let (systems, _) = result.into_parts();
    assert_eq!(systems[0].rhs(), &[12.0]);
    assert_eq!(systems[0].matrix().values(), &[6.0]);
    assert_eq!(
        systems[0].matrix().multiply(&[2.0]).unwrap(),
        systems[0].rhs()
    );
}

#[test]
fn plural_quotient_ownership_rejects_missing_duplicate_and_stale_bindings() {
    let s = chain();
    let mut missing = s.clone();
    missing.transformations.pop();
    assert!(rebuild(missing).is_err());
    let mut duplicate = s.clone();
    duplicate.transformations.push(s.transformations[0].clone());
    assert!(rebuild(duplicate).is_err());
    let mut one_sided = s.clone();
    let BlockTransformation::ConformingTraceQuotient {
        interface_relations,
        ..
    } = &mut one_sided.transformations[0]
    else {
        unreachable!()
    };
    let omitted = interface_relations.pop().unwrap();
    one_sided.relations.retain(|r| r.relation != omitted);
    assert!(rebuild(one_sided).is_err());
    let mut stale = s.clone();
    let BlockTransformation::ConformingTraceQuotient { quotient, .. } =
        &mut stale.transformations[0]
    else {
        unreachable!()
    };
    let endpoints = quotient.endpoints();
    *quotient = ConformingTraceQuotient::new(
        quotient.connection(),
        TraceFieldEndpoint::new(Id::new(), endpoints[0].field()),
        endpoints[1],
    )
    .unwrap();
    assert!(rebuild(stale).is_err());
    let mut foreign = s.clone();
    let BlockTransformation::ConformingTraceQuotient { quotient, .. } =
        &mut foreign.transformations[0]
    else {
        unreachable!()
    };
    let endpoints = quotient.endpoints();
    *quotient = ConformingTraceQuotient::new(
        quotient.connection(),
        TraceFieldEndpoint::new(endpoints[0].domain(), Id::new()),
        endpoints[1],
    )
    .unwrap();
    assert!(rebuild(foreign).is_err());
}
