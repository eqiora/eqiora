//! Authenticate caller-owned selections before creating a shared immutable snapshot.
use crate::external::ExternalSupportBinding;
use eqiora_core::Diagnostic;
use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet};
use std::collections::BTreeMap;

pub(super) type GeometryGroups<'a> = BTreeMap<
    [u8; 32],
    (
        &'a CanonicalGeometryV1,
        Vec<(
            &'a str,
            &'a NamedEntitySet,
            Option<(&'a str, &'a NamedEntitySet)>,
        )>,
    ),
>;
pub(super) type GeometryPartition<'a> = (
    &'a CanonicalGeometryV1,
    &'a NamedEntitySet,
    &'a [&'a NamedEntitySet],
);

pub(super) fn authenticate(
    file: &str,
    geometry_groups: GeometryGroups<'_>,
    geometry_partitions: Vec<GeometryPartition<'_>>,
    supports: &mut [ExternalSupportBinding],
) -> Result<(), Vec<Diagnostic>> {
    for (geometry, group_bindings) in geometry_groups.values() {
        crate::external_compile::validate_geometry_bindings(file, geometry, group_bindings)?;
    }
    for (geometry, parent, members) in geometry_partitions {
        geometry
            .validate_complete_exterior_partition(
                parent,
                members,
                eqiora_geometry::CanonicalGeometryLimits::default(),
            )
            .map_err(|error| vec![error])?;
    }
    for (_, (geometry, _)) in geometry_groups {
        // Validate original catalog references before any clone/name lookup. Rebinding
        // a foreign same-bytes selection first would erase its ownership evidence.
        let snapshot = std::sync::Arc::new(geometry.clone());
        for support in &mut *supports {
            match support {
                ExternalSupportBinding::Region {
                    geometry: digest,
                    context,
                    ..
                }
                | ExternalSupportBinding::Boundary {
                    geometry: digest,
                    context,
                    ..
                }
                | ExternalSupportBinding::CompleteExterior {
                    geometry: digest,
                    context,
                    ..
                } if digest.bytes() == geometry.digest_bytes() => {
                    *context = Some(std::sync::Arc::clone(&snapshot));
                }
                _ => {}
            }
        }
    }
    Ok(())
}
