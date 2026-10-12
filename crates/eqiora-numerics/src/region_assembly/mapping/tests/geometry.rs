//! Independent coupled scalar oracle for Field/geometry Jacobian composition.
use super::*;
use crate::form_compiler::region::RegionTimeBinding;
use eqiora_meshing::{
    AffineGeometryLinearization, MeshQualityGate, SimplicialMesh, simplex_duffy_gauss_legendre,
};

fn coupled_geometry<const D: usize>() {
    let bounds = vec!["0,1"; D].join(",");
    let source = format!(
        "model Coupled() {{
        domain body = box({bounds});
        state a: 1 on body;
        state b: 1 on body;
        relation first on body {{ derivative(a) - div(1[m^2/s]*grad(a)) + (a-2*b)/1[s] = 0; }}
        relation second on body {{ derivative(b) - div(1[m^2/s]*grad(b)) + (3*b-a)/1[s] = 0; }}
    }}"
    );
    let (transaction, model, symbols) = compile("geometry.eqi", &source)
        .unwrap()
        .remove(0)
        .into_parts();
    let a = symbols.get("a").unwrap();
    let b = symbols.get("b").unwrap();
    let domain = symbols.get("body").unwrap();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let compiled = CompiledRegionForm::<f64>::derive(&program, domain, D).unwrap();
    let fields = compiled
        .fields()
        .map(|(field, ty)| RegionFieldBinding {
            field,
            space: Space::continuous_lagrange(NonZeroU16::MIN),
            scale: DynQuantity::new(if field == a { 2. } else { 3. }, ty.dimension()),
        })
        .collect::<Vec<_>>();
    let volume =
        DimExponents::from_integers([0, i32::try_from(D).unwrap(), 0, 0, 0, 0, 0]).unwrap();
    let rows = compiled
        .rows()
        .map(|(row, _, ty)| {
            (
                row,
                DynQuantity::new(1., ty.dimension().mul(volume).unwrap().pow(-1, 1).unwrap()),
            )
        })
        .collect();
    let reference = ReferenceCell::simplex(D).unwrap();
    let form = compiled
        .bind(
            reference,
            &fields,
            &rows,
            Some(&RegionTimeBinding {
                step: DynQuantity::new(
                    1.,
                    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap(),
                ),
                states: vec![],
            }),
        )
        .unwrap();
    let mesh = SimplicialMesh::new(
        D,
        (0..=D)
            .map(|node| {
                (0..D)
                    .map(|axis| if node == axis + 1 { 1. } else { 0. })
                    .collect()
            })
            .collect(),
        vec![(0..=D).collect()],
        MeshQualityGate::new(0.01).unwrap(),
    )
    .unwrap();
    let mapping = RegionDofMap::new(
        &mesh,
        &BTreeMap::from([(domain, form.fields().to_vec())]),
        reference,
        &[domain],
        &[],
        &BTreeMap::new(),
    )
    .unwrap();
    let recover = |old: bool| {
        let mut values = vec![0.; mapping.free_count()];
        for key in mapping.keys() {
            let physical = match (key.field == a, old) {
                (true, true) => 1.,
                (true, false) => 2.,
                (false, true) => 4.,
                (false, false) => 3.,
            };
            values[mapping.free_dof(key).unwrap().index()] =
                physical / mapping.field_scale(key.field).unwrap();
        }
        mapping.recover(&values, &[a, b]).unwrap()
    };
    let previous = mapping
        .physical_cell_history(0, &form, Some(&recover(true)))
        .unwrap();
    let point = mapping
        .physical_cell_point(0, &form, &recover(false))
        .unwrap();
    let direction = |fields: bool, geometry: bool| RegionDirection {
        physical: mapping
            .keys()
            .map(|key| {
                (
                    key,
                    if !fields {
                        0.
                    } else if key.field == a {
                        0.5
                    } else {
                        -0.25
                    },
                )
            })
            .collect(),
        coordinate: mesh
            .vertices()
            .iter()
            .map(|p| std::array::from_fn(|axis| if geometry { p[axis] } else { 0. }))
            .collect::<Vec<[f64; D]>>(),
    };
    let directions = [
        direction(true, false),
        direction(false, true),
        direction(true, true),
    ];
    let map = mesh.geometry_map(MeshEntity::new(D, 0)).unwrap();
    let stationary = AffineGeometryLinearization::stationary(map.clone()).unwrap();
    let quadrature = simplex_duffy_gauss_legendre(D, 3).unwrap();
    for moving_history in [false, true] {
        let prepare = |direction: Option<&AffineGeometryLinearization>| {
            form.prepare_cell_actions(
                &map,
                &map,
                direction
                    .map(|current| (current, if moving_history { current } else { &stationary })),
                &quadrature,
            )
        };
        let matrix = mapping
            .linearize_geometry_cell(0, &form, &mesh, &previous, &point, &directions, prepare)
            .unwrap();
        // Uniform values have no diffusion contribution. For dt=1 the current
        // residual densities are 2*a-2*b and 4*b-a, with previous a,b subtracted.
        // Dilation differentiates the current measure by D; previous mass only
        // contributes when the selected history map moves with it. On the unit
        // simplex, each nodal test integrates to 1/(D+1)!.
        let test_integral = 1. / (1..=D + 1).product::<usize>() as f64;
        for field in form.fields() {
            let (state, current, old) = if field.field == a {
                (1.5, -2., 1.)
            } else {
                (-1.5, 10., 4.)
            };
            let shape = D as f64 * (current - if moving_history { old } else { 0. });
            for row in field.range.clone() {
                for (column, expected) in [state, shape, state + shape].iter().enumerate() {
                    let actual = matrix[row * directions.len() + column];
                    assert!(
                        (actual - expected * test_integral).abs() < 1e-11,
                        "D={D}, row={row}, column={column}: {actual} != {}",
                        expected * test_integral
                    );
                }
            }
        }
        let mut incomplete = directions.to_vec();
        incomplete[0].coordinate.pop();
        let error = mapping
            .linearize_geometry_cell(0, &form, &mesh, &previous, &point, &incomplete, prepare)
            .unwrap_err();
        assert!(error.message().contains("vertex inventory"), "{error:?}");
        let mut incomplete = directions.to_vec();
        incomplete[0].physical.pop_first();
        let error = mapping
            .linearize_geometry_cell(0, &form, &mesh, &previous, &point, &incomplete, prepare)
            .unwrap_err();
        assert!(
            error.message().contains("exact Field coefficient"),
            "{error:?}"
        );
    }
}

#[test]
fn composed_scalar_and_geometry_columns_have_independent_simplex_oracles() {
    coupled_geometry::<2>();
    coupled_geometry::<3>();
}
