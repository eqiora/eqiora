//! Independent two-cell flux derivatives include both coefficient and trace products.
use super::*;
use crate::region_assembly::ConservativeFacets;
use crate::region_assembly::RegionPoint;
use crate::region_assembly::mapping::{RegionAlgebraicCoordinates, RegionDofMap, RegionSolveInput};
use eqiora_meshing::CartesianMesh;
use eqiora_realization::Space;
use eqiora_schema::kernel::BoundarySide;
use eqiora_solver::LinearOperatorProperties;

#[test]
fn conservative_input_partial_preserves_primal_flux_and_boundary_products() {
    let source = "model Products() {
        domain body=box(0,1);
        domain left=boundary(body,axis=0,side=lower);
        domain right=boundary(body,axis=0,side=upper);
        parameter p:1=2;
        variable u:1 on body in smooth;
        relation balance on body {-div(p*p*grad(u))=0;}
        relation left_value on left {trace(u)=p*p*p;}
        relation right_value on right {trace(u)=p*p*p;}
    }";
    let (transaction, model, symbols) = compile("conservative-partial.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let domain = symbols.get("body").unwrap();
    let form =
        CompiledSpatialForm::<f64>::derive_at_time(&program, domain, 1, &Default::default(), None)
            .unwrap();
    let coordinate = crate::parameter_point::coordinates(
        &program,
        &[symbols.get("p").unwrap().downcast().unwrap()],
    )
    .unwrap()
    .remove(0);
    let partial = form
        .input_partial(&coordinate, InputPartialScope::Full)
        .unwrap();
    let mesh = CartesianMesh::from_axes(vec![vec![0., 0.5, 1.]]).unwrap();
    let reference = ReferenceCell::hypercube(1).unwrap();
    let input = |form: &CompiledSpatialForm<f64>| {
        let bound = form.bind_space(reference, Space::cell_constant()).unwrap();
        let field = bound.fields()[0].field;
        let boundaries = [
            (BoundarySide::Lower, "left"),
            (BoundarySide::Upper, "right"),
        ]
        .into_iter()
        .map(|(side, name)| {
            (
                (field, 0, side),
                form.boundary_laws()[&field][&symbols.get(name).unwrap()].clone(),
            )
        })
        .collect();
        RegionSolveInput {
            operator_properties: LinearOperatorProperties::General,
            forms: vec![(
                bound,
                QuadratureRule::tensor_product_gauss_legendre(1, 1).unwrap(),
            )],
            conservative: Some(Box::new(ConservativeFacets {
                mesh: mesh.clone(),
                boundaries,
                primal: None,
            })),
            natural: Vec::new(),
            previous: None,
            prescribed_states: BTreeMap::new(),
            geometry_action: None,
        }
    };
    let primal = input(&form);
    let mut differentiated = input(&partial);
    differentiated.retain_primal_integration(&primal).unwrap();
    let mapping = RegionDofMap::<f64>::new(
        &mesh,
        &BTreeMap::from([(domain, primal.forms[0].0.fields().to_vec())]),
        reference,
        &[domain; 2],
        &[],
        &BTreeMap::new(),
    )
    .unwrap();
    // At p=2, a=4, a'=4, g=8, g'=12. The interior distance is 1/2;
    // each exterior distance is 1/4. At u=(1,3), R=(-128,-64), while
    // R_p includes T'*(u-g)-T*g' at each boundary: (-320,-256).
    for (input, expected) in [(&primal, [-128., -64.]), (&differentiated, [-320., -256.])] {
        let prepared = mapping.prepare_numeric_assembly(&mesh, input).unwrap();
        let actual = prepared
            .work
            .residual(
                &prepared.plan,
                prepared.plan.target_id(0).unwrap(),
                RegionPoint::Bound(&[1., 3.]),
            )
            .unwrap();
        assert_eq!(actual.len(), 2);
        for (&actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
        }
    }
}
