use super::*;

impl DiscreteBlockSystem {
    pub(super) fn validate(&self) -> Result<(), Diagnostic> {
        if self
            .fields
            .iter()
            .any(|field| field.value_type.scalar_domain() != ScalarDomain::Real)
        {
            return Err(invalid(
                "discrete block execution requires real Field types",
            ));
        }
        if self.fields.is_empty()
            || self.relations.is_empty()
            || self.residuals.is_empty()
            || self.contributions.is_empty()
            || self.packet_count == 0
            || self.target_count == 0
            || self.primary_target >= self.target_count
        {
            return Err(invalid(
                "a discrete block system requires nonempty semantic, residual, contribution, packet, and target inventories",
            ));
        }
        if self
            .fields
            .windows(2)
            .any(|pair| pair[0].field == pair[1].field)
            || self
                .relations
                .windows(2)
                .any(|pair| pair[0].relation == pair[1].relation)
            || self
                .residuals
                .windows(2)
                .any(|pair| pair[0].tested == pair[1].tested && pair[0].support == pair[1].support)
        {
            return Err(invalid(
                "a discrete block system contains duplicate Field, Relation, or residual identity",
            ));
        }

        let field_ids = self
            .fields
            .iter()
            .map(|field| field.field)
            .collect::<Vec<_>>();
        let algebraic = self
            .fields
            .iter()
            .filter(|field| field.role == FieldBlockRole::Algebraic)
            .map(|field| AlgebraicBlock::Field(field.field))
            .chain(self.auxiliaries.iter().map(|auxiliary| auxiliary.block()))
            .collect::<Vec<_>>();
        if algebraic.is_empty() {
            return Err(invalid(
                "a discrete block system requires at least one algebraic unknown",
            ));
        }
        for relation in &self.relations {
            let valid = match relation.disposition {
                RelationDisposition::CoefficientDefinition { field }
                | RelationDisposition::BoundaryCondition { field, .. } => {
                    field_ids.contains(&field)
                }
                RelationDisposition::Residual { tested } => algebraic.contains(&tested),
                RelationDisposition::StateElimination { state, rate } => {
                    field_ids.contains(&state) && field_ids.contains(&rate)
                }
            };
            if !valid {
                return Err(invalid(
                    "a Relation block refers outside the exact Field/algebraic inventory",
                ));
            }
        }

        let relation_ids = self
            .relations
            .iter()
            .map(|relation| relation.relation)
            .collect::<Vec<_>>();
        let mut residual_relation_ids = self
            .residuals
            .iter()
            .flat_map(|residual| residual.origins.iter())
            .filter_map(|origin| match origin {
                ResidualOrigin::Relation(relation) => Some(*relation),
                ResidualOrigin::AlgebraicConstraint(_) => None,
            })
            .collect::<Vec<_>>();
        residual_relation_ids.sort_by_key(Id::ulid);
        let expected_residual_relations = self
            .relations
            .iter()
            .filter_map(|relation| match relation.disposition {
                RelationDisposition::Residual { .. } => Some(relation.relation),
                _ => None,
            })
            .collect::<Vec<_>>();
        if residual_relation_ids != expected_residual_relations {
            return Err(invalid(
                "residual blocks must cover every and only residual Relation exactly",
            ));
        }
        for residual in &self.residuals {
            if !algebraic.contains(&residual.tested)
                || residual.origins.iter().any(|origin| match origin {
                    ResidualOrigin::Relation(relation) => self
                        .relations
                        .iter()
                        .find(|candidate| candidate.relation == *relation)
                        .is_none_or(|candidate| candidate.support != residual.support),
                    ResidualOrigin::AlgebraicConstraint(constraint) => !self
                        .auxiliaries
                        .iter()
                        .any(|entry| entry.constraint == *constraint),
                })
            {
                return Err(invalid(
                    "a residual block refers outside the exact Relation/algebraic inventory",
                ));
            }
        }

        let mut covered_packets = BTreeSet::new();
        let mut incident_blocks = Vec::new();
        let mut contribution_relations = Vec::new();
        for contribution in &self.contributions {
            if contribution
                .target_indices
                .iter()
                .any(|target| *target >= self.target_count)
            {
                return Err(invalid(
                    "a contribution batch refers outside the assembly target inventory",
                ));
            }
            for packet in &contribution.packet_indices {
                if *packet >= self.packet_count || !covered_packets.insert(*packet) {
                    return Err(invalid(
                        "contribution batches must partition the exact assembly packet inventory",
                    ));
                }
            }
            for block in contribution
                .row_blocks
                .iter()
                .chain(&contribution.column_blocks)
            {
                if !algebraic.contains(block) {
                    return Err(invalid(
                        "a contribution batch refers outside the algebraic block inventory",
                    ));
                }
                if !incident_blocks.contains(block) {
                    incident_blocks.push(*block);
                }
            }
            for origin in &contribution.origins {
                match origin {
                    ResidualOrigin::Relation(relation) => {
                        let Some(relation_block) = self
                            .relations
                            .iter()
                            .find(|candidate| candidate.relation == *relation)
                        else {
                            return Err(invalid(
                                "a contribution batch refers to an unknown Semantic Relation",
                            ));
                        };
                        if !contribution.supports.contains(&relation_block.support) {
                            return Err(invalid(
                                "a contribution batch omits an exact Relation support",
                            ));
                        }
                        if !contribution_relations.contains(relation) {
                            contribution_relations.push(*relation);
                        }
                    }
                    ResidualOrigin::AlgebraicConstraint(constraint) => {
                        if !self
                            .auxiliaries
                            .iter()
                            .any(|entry| entry.constraint == *constraint)
                        {
                            return Err(invalid(
                                "a contribution batch refers to an unknown algebraic constraint",
                            ));
                        }
                    }
                }
            }
        }
        incident_blocks.sort_by(block_order);
        if covered_packets != (0..self.packet_count).collect()
            || incident_blocks != algebraic
            || !expected_residual_relations
                .iter()
                .all(|relation| contribution_relations.contains(relation))
        {
            return Err(invalid(
                "block contributions omit a packet, algebraic block, or residual Relation",
            ));
        }

        self.validate_transformations(&field_ids, &relation_ids)?;
        self.validate_closures(&field_ids, &relation_ids)?;
        Ok(())
    }

    fn validate_transformations(
        &self,
        fields: &[Id<kinds::Field>],
        relations: &[Id<kinds::Relation>],
    ) -> Result<(), Diagnostic> {
        if self
            .transformations
            .windows(2)
            .any(|pair| pair[0] == pair[1])
        {
            return Err(invalid(
                "a discrete block system contains a duplicate transformation",
            ));
        }
        for transformation in &self.transformations {
            match transformation {
                BlockTransformation::EssentialElimination {
                    field,
                    boundary_relations,
                } => {
                    if !fields.contains(field)
                        || boundary_relations.is_empty()
                        || boundary_relations
                            .iter()
                            .any(|relation| !relations.contains(relation))
                    {
                        return Err(invalid(
                            "essential elimination requires an exact Field and nonempty boundary Relation set",
                        ));
                    }
                    if boundary_relations.iter().any(|relation| {
                        !self.relations.iter().any(|candidate| {
                            candidate.relation == *relation
                                && candidate.disposition
                                    == (RelationDisposition::BoundaryCondition {
                                        field: *field,
                                        treatment: BoundaryTreatment::EssentialElimination,
                                    })
                        })
                    }) {
                        return Err(invalid(
                            "essential elimination may own only matching essential boundary Relations",
                        ));
                    }
                }
                BlockTransformation::BackwardEulerElimination {
                    relation,
                    state,
                    rate,
                    duration,
                } => {
                    if !relations.contains(relation)
                        || !fields.contains(state)
                        || !fields.contains(rate)
                        || state == rate
                        || !duration.value().is_finite()
                        || duration.value() <= 0.0
                    {
                        return Err(invalid(
                            "Backward Euler elimination requires exact distinct Fields, Relation, and positive duration",
                        ));
                    }
                    if !self.relations.iter().any(|candidate| {
                        candidate.relation == *relation
                            && candidate.disposition
                                == (RelationDisposition::StateElimination {
                                    state: *state,
                                    rate: *rate,
                                })
                    }) {
                        return Err(invalid(
                            "Backward Euler transformation does not match its exact state Relation",
                        ));
                    }
                }
                BlockTransformation::ConformingTraceQuotient {
                    quotient,
                    interface_relations,
                } => {
                    if quotient
                        .endpoints()
                        .iter()
                        .any(|endpoint| !fields.contains(&endpoint.field()))
                        || interface_relations.is_empty()
                        || interface_relations
                            .iter()
                            .any(|relation| !relations.contains(relation))
                    {
                        return Err(invalid(
                            "a trace quotient requires exact endpoints and interface Relations",
                        ));
                    }
                    if interface_relations.iter().any(|relation| {
                        !self.relations.iter().any(|candidate| {
                            candidate.relation == *relation
                                && matches!(
                                    candidate.disposition,
                                    RelationDisposition::BoundaryCondition {
                                        field,
                                        treatment: BoundaryTreatment::ConformingInterface {
                                            connection,
                                        },
                                    } if connection == quotient.connection()
                                        && quotient
                                            .endpoints()
                                            .iter()
                                            .any(|endpoint| endpoint.field() == field)
                                )
                        })
                    }) {
                        return Err(invalid(
                            "a trace quotient may own only matching interface boundary Relations",
                        ));
                    }
                }
                BlockTransformation::BackwardEulerDerivative {
                    relation,
                    state,
                    duration,
                } => {
                    if !relations.contains(relation)
                        || !fields.contains(state)
                        || !duration.value().is_finite()
                        || duration.value() <= 0.0
                    {
                        return Err(invalid(
                            "Backward Euler derivative realization requires one exact state Field, residual Relation, and positive duration",
                        ));
                    }
                    if !self.relations.iter().any(|candidate| {
                        candidate.relation == *relation
                            && candidate.disposition
                                == (RelationDisposition::Residual {
                                    tested: AlgebraicBlock::Field(*state),
                                })
                    }) {
                        return Err(invalid(
                            "Backward Euler derivative realization does not match its tested residual Relation",
                        ));
                    }
                }
                BlockTransformation::EnergySkewConvection { relation, velocity } => {
                    if !relations.contains(relation) || !fields.contains(velocity) {
                        return Err(invalid(
                            "energy-skew convection requires one exact residual Relation and velocity Field",
                        ));
                    }
                    if !self.relations.iter().any(|candidate| {
                        candidate.relation == *relation
                            && candidate.disposition
                                == (RelationDisposition::Residual {
                                    tested: AlgebraicBlock::Field(*velocity),
                                })
                    }) {
                        return Err(invalid(
                            "energy-skew convection does not match its tested momentum Relation",
                        ));
                    }
                }
            }
        }
        let state_eliminations = self
            .relations
            .iter()
            .filter_map(|relation| match relation.disposition {
                RelationDisposition::StateElimination { state, rate } => {
                    Some((relation.relation, state, rate))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if state_eliminations.iter().any(|(relation, state, rate)| {
            self.transformations
                .iter()
                .filter(|transformation| {
                    matches!(
                        transformation,
                        BlockTransformation::BackwardEulerElimination {
                            relation: candidate,
                            state: candidate_state,
                            rate: candidate_rate,
                            ..
                        } if candidate == relation && candidate_state == state && candidate_rate == rate
                    )
                })
                .count()
                != 1
        }) {
            return Err(invalid(
                "every eliminated-state Relation requires one exact Backward Euler transformation",
            ));
        }
        self.validate_quotient_ownership()?;
        self.validate_boundary_treatments()?;
        Ok(())
    }

    fn validate_quotient_ownership(&self) -> Result<(), Diagnostic> {
        let mut identities = std::collections::BTreeSet::new();
        for transformation in &self.transformations {
            let BlockTransformation::ConformingTraceQuotient {
                quotient,
                interface_relations,
            } = transformation
            else {
                continue;
            };
            let mut fields = quotient
                .endpoints()
                .map(|endpoint| endpoint.field().erase());
            fields.sort();
            if !identities.insert((quotient.connection().erase(), fields))
                || interface_relations
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    != interface_relations.len()
            {
                return Err(invalid("trace quotient ownership must be duplicate-free"));
            }
            for endpoint in quotient.endpoints() {
                if !self.fields.iter().any(|field| {
                    field.field == endpoint.field()
                        && field.domain == endpoint.domain()
                        && field.role == FieldBlockRole::Algebraic
                }) || !interface_relations.iter().any(|relation| {
                    self.relations.iter().any(|candidate| {
                        candidate.relation == *relation
                            && candidate.disposition
                                == RelationDisposition::BoundaryCondition {
                                    field: endpoint.field(),
                                    treatment: BoundaryTreatment::ConformingInterface {
                                        connection: quotient.connection(),
                                    },
                                }
                    })
                }) {
                    return Err(invalid(
                        "each trace endpoint requires its exact algebraic Field/Domain and interface Relation",
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_boundary_treatments(&self) -> Result<(), Diagnostic> {
        for relation in &self.relations {
            let RelationDisposition::BoundaryCondition { field, treatment } = relation.disposition
            else {
                continue;
            };
            match treatment {
                BoundaryTreatment::EssentialElimination => {
                    let owners = self
                        .transformations
                        .iter()
                        .filter(|transformation| {
                            matches!(
                                transformation,
                                BlockTransformation::EssentialElimination {
                                    field: candidate,
                                    boundary_relations,
                                } if *candidate == field
                                    && boundary_relations.contains(&relation.relation)
                            )
                        })
                        .count();
                    if owners != 1 {
                        return Err(invalid(
                            "an essential boundary Relation requires one exact elimination",
                        ));
                    }
                }
                BoundaryTreatment::Natural { inhomogeneous } => {
                    let owners = self
                        .contributions
                        .iter()
                        .filter(|contribution| {
                            contribution.terms.contains(&ContributionTerm::Boundary)
                                && contribution
                                    .origins
                                    .contains(&ResidualOrigin::Relation(relation.relation))
                        })
                        .count();
                    if inhomogeneous && owners != 1 {
                        return Err(invalid(
                            "an inhomogeneous natural boundary requires one exact contribution batch",
                        ));
                    }
                }
                BoundaryTreatment::ConformingInterface { connection } => {
                    let owners = self
                        .transformations
                        .iter()
                        .filter(|transformation| {
                            matches!(
                                transformation,
                                BlockTransformation::ConformingTraceQuotient {
                                    quotient,
                                    interface_relations,
                                } if quotient.connection() == connection
                                    && quotient
                                        .endpoints()
                                        .iter()
                                        .any(|endpoint| endpoint.field() == field)
                                    && interface_relations.contains(&relation.relation)
                            )
                        })
                        .count();
                    if owners != 1 {
                        return Err(invalid(
                            "an interface Relation requires one exact Connection-owned trace quotient",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_closures(
        &self,
        fields: &[Id<kinds::Field>],
        relations: &[Id<kinds::Relation>],
    ) -> Result<(), Diagnostic> {
        if self.closures.is_empty() || self.closures.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid(
                "a discrete block system requires a nonempty duplicate-free closure inventory",
            ));
        }
        for closure in &self.closures {
            let (field, origins) = match closure {
                AlgebraicClosure::EssentialBoundary { field, relations }
                | AlgebraicClosure::BoundaryTraction { field, relations }
                | AlgebraicClosure::CompleteOperator { field, relations } => (*field, relations),
                AlgebraicClosure::ZeroIntegral { field } => {
                    if !fields.contains(field)
                        || !self.auxiliaries.iter().any(|auxiliary| {
                            auxiliary.constraint
                                == AlgebraicConstraint::ZeroIntegral { field: *field }
                        })
                    {
                        return Err(invalid(
                            "a zero-integral closure requires its exact auxiliary block",
                        ));
                    }
                    continue;
                }
            };
            if !fields.contains(&field)
                || origins.is_empty()
                || origins.iter().any(|relation| !relations.contains(relation))
            {
                return Err(invalid(
                    "an algebraic closure refers outside the exact Field/Relation inventory",
                ));
            }
            match closure {
                AlgebraicClosure::EssentialBoundary { relations, .. } => {
                    if !self.transformations.iter().any(|transformation| {
                        matches!(
                            transformation,
                            BlockTransformation::EssentialElimination {
                                field: candidate,
                                boundary_relations,
                            } if *candidate == field && boundary_relations == relations
                        )
                    }) {
                        return Err(invalid(
                            "an essential closure requires the exact elimination transformation",
                        ));
                    }
                }
                AlgebraicClosure::BoundaryTraction { relations, .. } => {
                    if !self.contributions.iter().any(|contribution| {
                        contribution.terms.contains(&ContributionTerm::Boundary)
                            && relations.iter().all(|relation| {
                                contribution
                                    .origins
                                    .contains(&ResidualOrigin::Relation(*relation))
                            })
                    }) {
                        return Err(invalid(
                            "a boundary-determined closure requires the exact boundary contribution",
                        ));
                    }
                }
                AlgebraicClosure::CompleteOperator { relations, .. } => {
                    if !relations.iter().all(|relation| {
                        self.contributions.iter().any(|contribution| {
                            contribution
                                .origins
                                .contains(&ResidualOrigin::Relation(*relation))
                        })
                    }) {
                        return Err(invalid(
                            "a complete-operator closure requires every stated Relation contribution",
                        ));
                    }
                }
                AlgebraicClosure::ZeroIntegral { .. } => unreachable!(),
            }
        }
        Ok(())
    }

    pub(super) fn validate_report(&self, report: &AssemblyReport) -> Result<(), Diagnostic> {
        if report.packet_count() != self.packet_count || report.target_count() != self.target_count
        {
            return Err(invalid(
                "assembly evidence differs from the exact block contribution/target inventory",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_packet(
        &self,
        packet_index: usize,
        packet: &AssemblyPacket,
    ) -> Result<(), Diagnostic> {
        let batch = self
            .contributions
            .iter()
            .find(|batch| batch.packet_indices.binary_search(&packet_index).is_ok())
            .ok_or_else(|| invalid("assembly evaluated a packet absent from block incidence"))?;
        let actual_targets = packet
            .mappings()
            .iter()
            .map(|mapping| mapping.target().index())
            .collect::<Vec<_>>();
        if actual_targets != batch.target_indices {
            return Err(invalid(
                "assembly packet targets differ from its exact block contribution batch",
            ));
        }
        Ok(())
    }
}
