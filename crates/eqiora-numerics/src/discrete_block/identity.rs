use super::*;

impl DiscreteBlockSystem {
    pub(super) fn compute_identity(&self) -> BlockSystemIdentity {
        let mut hash = Sha256::new();
        hash.update(BLOCK_SYSTEM_IDENTITY_DOMAIN);
        hash.update(self.context.model.ulid().to_string().as_bytes());
        hash.update(self.context.semantic_revision.get().to_le_bytes());
        match self.context.realization {
            BlockRealizationIdentity::Default(policy) => {
                hash.update([0]);
                hash.update(policy.get().to_le_bytes());
            }
            BlockRealizationIdentity::Explicit(revision) => {
                hash.update([1]);
                hash.update(revision.get().to_le_bytes());
            }
        }
        match self.context.mesh {
            Some(mesh) => {
                hash.update([1]);
                hash.update(mesh.sha256());
            }
            None => hash.update([0]),
        }
        hash.update((self.packet_count as u64).to_le_bytes());
        hash.update((self.target_count as u64).to_le_bytes());
        hash.update((self.primary_target as u64).to_le_bytes());
        hash.update([property_tag(self.required_properties)]);
        for field in &self.fields {
            hash.update(b"field\0");
            hash_id(&mut hash, field.domain);
            hash_id(&mut hash, field.field);
            match field.space {
                Some(space) => {
                    hash.update([1]);
                    hash_space(&mut hash, space);
                }
                None => hash.update([0]),
            }
            hash_value_type(&mut hash, &field.value_type);
            hash.update([field.role as u8]);
            match field.scale {
                Some(scale) => {
                    hash.update([1]);
                    hash_quantity(&mut hash, scale);
                }
                None => hash.update([0]),
            }
        }
        for auxiliary in &self.auxiliaries {
            hash.update(b"aux\0");
            hash_block(&mut hash, auxiliary.block());
            hash_quantity(&mut hash, auxiliary.scale);
        }
        for relation in &self.relations {
            hash.update(b"relation\0");
            hash_id(&mut hash, relation.relation);
            hash_support(&mut hash, relation.support);
            hash_relation_disposition(&mut hash, relation.disposition);
        }
        for residual in &self.residuals {
            hash.update(b"residual\0");
            hash_block(&mut hash, residual.tested);
            hash_support(&mut hash, residual.support);
            for origin in &residual.origins {
                hash_residual_origin(&mut hash, *origin);
            }
        }
        for transformation in &self.transformations {
            hash_transformation(&mut hash, transformation);
        }
        for closure in &self.closures {
            hash_closure(&mut hash, closure);
        }
        for contribution in &self.contributions {
            hash.update(b"contribution\0");
            for support in &contribution.supports {
                hash_support(&mut hash, *support);
            }
            for packet in &contribution.packet_indices {
                hash.update((*packet as u64).to_le_bytes());
            }
            hash.update([0xfe]);
            for target in &contribution.target_indices {
                hash.update((*target as u64).to_le_bytes());
            }
            for origin in &contribution.origins {
                hash_residual_origin(&mut hash, *origin);
            }
            hash.update([0xfd]);
            for parameter in &contribution.parameters {
                hash_id(&mut hash, *parameter);
            }
            for block in &contribution.row_blocks {
                hash_block(&mut hash, *block);
            }
            hash.update([0xff]);
            for block in &contribution.column_blocks {
                hash_block(&mut hash, *block);
            }
            for term in &contribution.terms {
                hash.update([*term as u8]);
            }
        }
        BlockSystemIdentity(hash.finalize().into())
    }
}

fn hash_id<E: eqiora_core::Entity>(hash: &mut Sha256, id: Id<E>) {
    hash.update(id.ulid().to_string().as_bytes());
    hash.update([0]);
}

fn hash_space(hash: &mut Sha256, space: Space) {
    match space.family() {
        SpaceFamily::ContinuousLagrange { order } => {
            hash.update([0]);
            hash.update(order.get().to_le_bytes());
        }
        SpaceFamily::SimplexP1Bubble => hash.update([1]),
        SpaceFamily::CellConstant => hash.update([2]),
        SpaceFamily::TetrahedralEdge => hash.update([3]),
        SpaceFamily::TetrahedralFace => hash.update([4]),
    }
}

fn hash_shape(hash: &mut Sha256, shape: &ValueShape) {
    hash.update((shape.rank() as u64).to_le_bytes());
    for extent in shape.extents() {
        hash.update(extent.get().to_le_bytes());
    }
}

fn hash_value_type(hash: &mut Sha256, value_type: &ValueType) {
    hash.update([match value_type.scalar_domain() {
        ScalarDomain::Real => 0,
        ScalarDomain::Complex => 1,
        ScalarDomain::Integer => 2,
        ScalarDomain::Boolean => 3,
        ScalarDomain::Enum => 4,
    }]);
    if let Some(definition) = value_type.enum_definition() {
        hash_id(hash, definition);
        hash.update(
            value_type
                .enum_member_count()
                .expect("enum member count")
                .to_le_bytes(),
        );
    }
    hash_dimension(hash, value_type.dimension());
    hash_shape(hash, value_type.shape());
    hash.update([frame_tag(value_type.frame())]);
    hash.update((value_type.array_rank() as u64).to_le_bytes());
}

fn hash_dimension(hash: &mut Sha256, dimension: DimExponents) {
    for (numerator, denominator) in dimension.exponents() {
        hash.update(numerator.to_le_bytes());
        hash.update(denominator.to_le_bytes());
    }
}

fn hash_quantity(hash: &mut Sha256, quantity: DynQuantity) {
    hash.update(quantity.value().to_bits().to_le_bytes());
    hash_dimension(hash, quantity.dim());
}

const fn frame_tag(frame: ValueFrame) -> u8 {
    match frame {
        ValueFrame::Invariant => 0,
        ValueFrame::SpatialCartesian => 1,
    }
}

const fn property_tag(properties: LinearOperatorProperties) -> u8 {
    match properties {
        LinearOperatorProperties::General => 0,
        LinearOperatorProperties::SymmetricPositiveDefinite => 1,
        LinearOperatorProperties::SymmetricIndefinite => 2,
        LinearOperatorProperties::Symmetric => 3,
        LinearOperatorProperties::ComplexSymmetric => 4,
        LinearOperatorProperties::Hermitian => 5,
        LinearOperatorProperties::HermitianPositiveDefinite => 6,
    }
}

fn hash_block(hash: &mut Sha256, block: AlgebraicBlock) {
    hash.update([block_tag(block)]);
    hash_id(hash, block_field(block));
}

fn hash_support(hash: &mut Sha256, support: BlockSupport) {
    match support {
        BlockSupport::Volume(domain) => {
            hash.update([0]);
            hash_id(hash, domain);
        }
        BlockSupport::Boundary(boundary) => {
            hash.update([1]);
            hash_id(hash, boundary);
        }
    }
}

fn hash_relation_disposition(hash: &mut Sha256, disposition: RelationDisposition) {
    match disposition {
        RelationDisposition::CoefficientDefinition { field } => {
            hash.update([0]);
            hash_id(hash, field);
        }
        RelationDisposition::Residual { tested } => {
            hash.update([1]);
            hash_block(hash, tested);
        }
        RelationDisposition::StateElimination { state, rate } => {
            hash.update([2]);
            hash_id(hash, state);
            hash_id(hash, rate);
        }
        RelationDisposition::BoundaryCondition { field, treatment } => {
            hash.update([3]);
            hash_id(hash, field);
            match treatment {
                BoundaryTreatment::EssentialElimination => hash.update([0]),
                BoundaryTreatment::Natural { inhomogeneous } => {
                    hash.update([1, u8::from(inhomogeneous)]);
                }
                BoundaryTreatment::ConformingInterface { connection } => {
                    hash.update([2]);
                    hash_id(hash, connection);
                }
            }
        }
    }
}

fn hash_residual_origin(hash: &mut Sha256, origin: ResidualOrigin) {
    match origin {
        ResidualOrigin::Relation(relation) => {
            hash.update([0]);
            hash_id(hash, relation);
        }
        ResidualOrigin::AlgebraicConstraint(constraint) => {
            hash.update([1]);
            hash_id(hash, constraint.field());
        }
    }
}

fn hash_transformation(hash: &mut Sha256, transformation: &BlockTransformation) {
    hash.update(b"transformation\0");
    match transformation {
        BlockTransformation::EssentialElimination {
            field,
            boundary_relations,
        } => {
            hash.update([0]);
            hash_id(hash, *field);
            for relation in boundary_relations {
                hash_id(hash, *relation);
            }
        }
        BlockTransformation::BackwardEulerElimination {
            relation,
            state,
            rate,
            duration,
        } => {
            hash.update([1]);
            hash_id(hash, *relation);
            hash_id(hash, *state);
            hash_id(hash, *rate);
            hash_quantity(hash, *duration);
        }
        BlockTransformation::ConformingTraceQuotient {
            quotient,
            interface_relations,
        } => {
            hash.update([2]);
            hash_id(hash, quotient.connection());
            for endpoint in quotient.endpoints() {
                hash_id(hash, endpoint.domain());
                hash_id(hash, endpoint.field());
            }
            for relation in interface_relations {
                hash_id(hash, *relation);
            }
        }
        BlockTransformation::BackwardEulerDerivative {
            relation,
            state,
            duration,
        } => {
            hash.update([3]);
            hash_id(hash, *relation);
            hash_id(hash, *state);
            hash_quantity(hash, *duration);
        }
        BlockTransformation::EnergySkewConvection { relation, velocity } => {
            hash.update([4]);
            hash_id(hash, *relation);
            hash_id(hash, *velocity);
        }
    }
}

fn hash_closure(hash: &mut Sha256, closure: &AlgebraicClosure) {
    hash.update(b"closure\0");
    match closure {
        AlgebraicClosure::EssentialBoundary { field, relations } => {
            hash.update([0]);
            hash_id(hash, *field);
            for relation in relations {
                hash_id(hash, *relation);
            }
        }
        AlgebraicClosure::ZeroIntegral { field } => {
            hash.update([1]);
            hash_id(hash, *field);
        }
        AlgebraicClosure::BoundaryTraction { field, relations } => {
            hash.update([2]);
            hash_id(hash, *field);
            for relation in relations {
                hash_id(hash, *relation);
            }
        }
        AlgebraicClosure::CompleteOperator { field, relations } => {
            hash.update([3]);
            hash_id(hash, *field);
            for relation in relations {
                hash_id(hash, *relation);
            }
        }
    }
}
