//! Canonical model declaration tags and payloads.
use super::*;

pub(super) fn encode_model_item(item: &Item, budget: &mut Budget) -> Result<Vec<u8>, Diagnostic> {
    let mut encoder = Encoder::new(budget.limits.max_canonical_bytes);
    match item {
        Item::Domain(declaration) => {
            encoder.u16(1)?;
            encode_domain(&mut encoder, declaration, budget)?;
        }
        Item::Initial(declaration) => {
            encoder.u16(14)?;
            encode_initial(&mut encoder, declaration, budget)?;
        }
        Item::Field(declaration) => {
            encoder.u16(3)?;
            encode_field(&mut encoder, declaration, budget)?;
        }
        Item::Observable(declaration) => {
            encoder.u16(30)?;
            compile_time::encode_observable(&mut encoder, declaration, budget)?;
        }
        Item::Parameter(declaration) => {
            encoder.u16(4)?;
            encode_parameter(&mut encoder, declaration, budget)?;
        }
        Item::IndexSet(declaration) => {
            encoder.u16(15)?;
            encode_let(&mut encoder, declaration, budget)?;
        }
        Item::Coordinate(declaration) => {
            encoder.u16(40)?;
            encode_let(&mut encoder, declaration, budget)?;
        }
        Item::Let(declaration) => {
            encoder.u16(MODEL_LET_ITEM_TAG)?;
            encode_let(&mut encoder, declaration, budget)?;
        }
        Item::Port(declaration) => {
            encoder.u16(5)?;
            encode_port(&mut encoder, declaration, budget)?;
        }
        Item::Event(declaration) => {
            encoder.u16(17)?;
            declarations::encode_event(&mut encoder, declaration, budget)?;
        }
        Item::Clock(declaration) => {
            encoder.u16(6)?;
            declarations::encode_clock(&mut encoder, declaration, budget)?;
        }
        Item::Relation(declaration) => {
            encoder.u16(7)?;
            encode_relation(&mut encoder, declaration, budget)?;
        }
        Item::RelationFamily(declaration) => {
            encoder.u16(16)?;
            encode_relation_family(&mut encoder, declaration, budget)?;
        }
        Item::Connection(declaration) => {
            encoder.u16(MODEL_CONNECTION_ITEM_TAG)?;
            encode_connection(&mut encoder, declaration, budget)?;
        }
        Item::BoundaryConnection(declaration) => {
            encoder.u16(match declaration.syntax() {
                ConnectionSyntax::Conserving => MODEL_BOUNDARY_CONNECTION_ITEM_TAG,
                ConnectionSyntax::SpatialPeriodic => MODEL_SPATIAL_PERIODIC_CONNECTION_ITEM_TAG,
                ConnectionSyntax::Signal => {
                    return Err(source_identity_error(
                        "boundary Connection cannot use signal semantics",
                    ));
                }
            })?;
            encode_boundary_connection(&mut encoder, declaration, budget)?;
        }
        Item::Instance(declaration) => {
            encoder.u16(10)?;
            encode_instance(&mut encoder, declaration, budget)?;
        }
        _ => {
            return Err(source_identity_error(
                "model item is newer than source identity v1",
            ));
        }
    }
    encoder.finish()
}
