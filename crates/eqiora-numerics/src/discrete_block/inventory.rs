//! Shared projection of semantic inventories into discrete blocks.

use super::*;
use crate::canonical_boundary::{
    BoundaryRelationBinding, CartesianBoundaryInventory, PhysicalBoundaryDisposition,
    PhysicalBoundaryQuantity,
};
use eqiora_core::RawId;

pub(crate) fn parameter_inventory<'a>(
    fields: impl IntoIterator<Item = &'a [Id<kinds::Parameter>]>,
) -> Vec<Id<kinds::Parameter>> {
    let mut result = fields.into_iter().flatten().copied().collect::<Vec<_>>();
    result.sort_by_key(Id::ulid);
    result.dedup();
    result
}

pub(crate) fn boundary_relation_blocks(
    bindings: &[BoundaryRelationBinding],
    field: Id<kinds::Field>,
    disposition: impl Fn(RawId) -> Option<PhysicalBoundaryDisposition>,
) -> Result<Vec<RelationBlock>, Diagnostic> {
    bindings
        .iter()
        .map(|binding| {
            let disposition = disposition(binding.boundary())
                .ok_or_else(|| invalid("a boundary Relation is absent from its exact inventory"))?;
            Ok(RelationBlock::new(
                relation(binding.relation())?,
                BlockSupport::Boundary(domain(binding.boundary())?),
                RelationDisposition::BoundaryCondition {
                    field,
                    treatment: boundary_treatment_for(disposition)?,
                },
            ))
        })
        .collect()
}

pub(crate) fn relations_with_treatment(
    blocks: &[RelationBlock],
    predicate: impl Fn(BoundaryTreatment) -> bool,
) -> Vec<Id<kinds::Relation>> {
    blocks
        .iter()
        .filter_map(|block| match block.disposition {
            RelationDisposition::BoundaryCondition { treatment, .. } if predicate(treatment) => {
                Some(block.relation)
            }
            _ => None,
        })
        .collect()
}

pub(crate) fn domain(id: RawId) -> Result<Id<kinds::Domain>, Diagnostic> {
    id.downcast::<kinds::Domain>()
        .ok_or_else(|| invalid_identity("Domain", id))
}

pub(crate) fn field(id: RawId) -> Result<Id<kinds::Field>, Diagnostic> {
    id.downcast::<kinds::Field>()
        .ok_or_else(|| invalid_identity("Field", id))
}

pub(crate) fn relation(id: RawId) -> Result<Id<kinds::Relation>, Diagnostic> {
    id.downcast::<kinds::Relation>()
        .ok_or_else(|| invalid_identity("Relation", id))
}

pub(crate) fn invalid_identity(expected: &str, id: RawId) -> Diagnostic {
    invalid(format!(
        "block inventory expected {expected} identity, received {id}"
    ))
}

fn boundary_treatment<const D: usize>(
    inventory: &CartesianBoundaryInventory<D>,
    binding: BoundaryRelationBinding,
) -> Result<BoundaryTreatment, Diagnostic> {
    let disposition = inventory
        .entries()
        .find(|(_, entry)| entry.boundary() == binding.boundary())
        .map(|(_, entry)| entry.disposition())
        .ok_or_else(|| invalid("a boundary Relation is absent from its exact inventory"))?;
    boundary_treatment_for(disposition)
}

fn boundary_treatment_for(
    disposition: PhysicalBoundaryDisposition,
) -> Result<BoundaryTreatment, Diagnostic> {
    match disposition {
        PhysicalBoundaryDisposition::TraceZero => Ok(BoundaryTreatment::EssentialElimination),
        PhysicalBoundaryDisposition::FluxZero => Ok(BoundaryTreatment::Natural {
            inhomogeneous: false,
        }),
        PhysicalBoundaryDisposition::Prescribed(law) => Ok(match law.quantity() {
            PhysicalBoundaryQuantity::Trace => BoundaryTreatment::EssentialElimination,
            PhysicalBoundaryQuantity::Flux => BoundaryTreatment::Natural {
                inhomogeneous: true,
            },
        }),
        PhysicalBoundaryDisposition::PortBinding { connection, .. } => {
            Ok(BoundaryTreatment::ConformingInterface {
                connection: connection.downcast::<kinds::Connection>().ok_or_else(|| {
                    invalid("a Port-boundary treatment requires a Connection identity")
                })?,
            })
        }
    }
}

pub(crate) fn conforming_interface_relations<'a, const D: usize>(
    inventories: impl IntoIterator<
        Item = (
            &'a CartesianBoundaryInventory<D>,
            &'a [BoundaryRelationBinding],
        ),
    >,
    connection: Id<kinds::Connection>,
) -> Result<Vec<Id<kinds::Relation>>, Diagnostic> {
    let mut result = Vec::new();
    for (inventory, bindings) in inventories {
        for binding in bindings {
            if boundary_treatment(inventory, *binding)?
                == (BoundaryTreatment::ConformingInterface { connection })
            {
                let relation = relation(binding.relation())?;
                result.push(relation);
            }
        }
    }
    result.sort_by_key(Id::ulid);
    result.dedup();
    if result.is_empty() {
        return Err(invalid(
            "a conforming trace quotient requires exact interface Relations",
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_boundary::PrescribedBoundaryLaw;

    #[test]
    fn boundary_selection_retains_exact_relation_order_and_treatment() {
        let field = Id::<kinds::Field>::new();
        let boundary = Id::<kinds::Domain>::new().erase();
        let ids = [Id::<kinds::Relation>::new(); 1];
        let relation = ids[0].erase();
        let binding = [BoundaryRelationBinding::new(boundary, relation)];
        for (disposition, expected) in [
            (
                PhysicalBoundaryDisposition::TraceZero,
                BoundaryTreatment::EssentialElimination,
            ),
            (
                PhysicalBoundaryDisposition::FluxZero,
                BoundaryTreatment::Natural {
                    inhomogeneous: false,
                },
            ),
            (
                PhysicalBoundaryDisposition::Prescribed(PrescribedBoundaryLaw::new(
                    PhysicalBoundaryQuantity::Trace,
                    relation,
                )),
                BoundaryTreatment::EssentialElimination,
            ),
            (
                PhysicalBoundaryDisposition::Prescribed(PrescribedBoundaryLaw::new(
                    PhysicalBoundaryQuantity::Flux,
                    relation,
                )),
                BoundaryTreatment::Natural {
                    inhomogeneous: true,
                },
            ),
        ] {
            let blocks = boundary_relation_blocks(&binding, field, |id| {
                (id == boundary).then_some(disposition)
            })
            .unwrap();
            assert_eq!(
                blocks[0].support,
                BlockSupport::Boundary(boundary.downcast().unwrap())
            );
            assert_eq!(
                blocks[0].disposition,
                RelationDisposition::BoundaryCondition {
                    field,
                    treatment: expected,
                }
            );
            assert_eq!(
                relations_with_treatment(&blocks, |treatment| treatment == expected),
                ids
            );
        }
        let second = Id::<kinds::Relation>::new();
        let bindings = [
            BoundaryRelationBinding::new(boundary, second.erase()),
            binding[0],
        ];
        let blocks = boundary_relation_blocks(&bindings, field, |_| {
            Some(PhysicalBoundaryDisposition::TraceZero)
        })
        .unwrap();
        assert_eq!(
            relations_with_treatment(&blocks, |_| true),
            [second, ids[0]]
        );
        assert!(boundary_relation_blocks(&binding, field, |_| None).is_err());
        assert!(
            boundary_relation_blocks(
                &[BoundaryRelationBinding::new(field.erase(), relation)],
                field,
                |_| Some(PhysicalBoundaryDisposition::TraceZero),
            )
            .is_err()
        );
        assert!(
            boundary_relation_blocks(
                &[BoundaryRelationBinding::new(boundary, field.erase())],
                field,
                |_| Some(PhysicalBoundaryDisposition::TraceZero),
            )
            .is_err()
        );
        assert!(
            boundary_relation_blocks(&binding, field, |_| {
                Some(PhysicalBoundaryDisposition::PortBinding {
                    connection: field.erase(),
                    port: field.erase(),
                })
            })
            .is_err()
        );
    }
}
