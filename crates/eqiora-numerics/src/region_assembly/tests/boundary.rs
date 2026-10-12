use super::*;
use eqiora_meshing::{EntityIncidence, MeshEntity, OrientationCode, ReferenceCell};

fn add_flux(fixture: &mut Fixture, ordinal: usize, value: f64) {
    let form = &fixture.forms[0].0;
    let cell = &fixture.cells[0];
    let facet = AffineGeometryMap::new(
        ReferenceCell::point(),
        1,
        vec![0.5 * ordinal as f64],
        vec![],
    )
    .unwrap();
    let prepared = form
        .prepare_natural_facet(
            form.fields()[0].field,
            &cell.geometry,
            (
                &facet,
                EntityIncidence {
                    entity: MeshEntity::new(1, 0),
                    local_ordinal: ordinal,
                    orientation: OrientationCode::new(0),
                },
                &[ordinal],
            ),
            &QuadratureRule::point(),
            |_, _| Ok(vec![value]),
        )
        .unwrap();
    fixture.boundary_packets.push((0, prepared));
}

#[test]
fn boundary_packets_follow_cells_and_preserve_constrained_flux_rows() {
    let mut fixture = Fixture::new(1, false);
    let base = fixture.prepare().unwrap();
    let base_result = REFERENCE_ASSEMBLY_BACKEND
        .assemble(&fixture.plan, &base)
        .unwrap();
    add_flux(&mut fixture, 0, 2.);
    add_flux(&mut fixture, 1, 3.);
    let prepared = fixture.prepare().unwrap();
    assert_eq!(prepared.packet_count(), base.packet_count() + 2);
    assert!(prepared.evaluate(prepared.packet_count()).is_err());
    let result = REFERENCE_ASSEMBLY_BACKEND
        .assemble(&fixture.plan, &prepared)
        .unwrap();
    let (base_systems, _) = base_result.into_parts();
    let (systems, _) = result.into_parts();
    assert_eq!(dense(&systems[0]), dense(&base_systems[0]));
    assert_eq!(dense(&systems[1]), dense(&base_systems[1]));
    close(systems[0].rhs()[0], base_systems[0].rhs()[0] + 3.0);
    close(systems[1].rhs()[0], base_systems[1].rhs()[0] + 2.0);
    close(systems[1].rhs()[1], base_systems[1].rhs()[1] + 3.0);
    close(systems[1].rhs()[2], base_systems[1].rhs()[2]);
}

#[test]
fn preparation_rejects_foreign_boundary_parent_and_geometry() {
    let mut fixture = Fixture::new(1, false);
    add_flux(&mut fixture, 0, 2.);
    fixture.boundary_packets[0].0 = fixture.cells.len();
    assert!(
        fixture
            .prepare()
            .unwrap_err()
            .message()
            .contains("foreign parent cell")
    );
    fixture.boundary_packets[0].0 = 1;
    assert!(
        fixture
            .prepare()
            .unwrap_err()
            .message()
            .contains("exact parent geometry or Field binding")
    );
}
