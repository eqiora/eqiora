use super::*;
use eqiora_core::{Id, ScalarDomain, ValueLiteral, ValueType, entity::kinds};
use eqiora_schema::kernel::{ExprNode, SymbolRef};

pub(super) struct Motion {
    program: KernelProgram,
    relation: Id<kinds::Relation>,
}

impl Motion {
    pub(super) fn new(map: &str, reverse_source: bool) -> Self {
        let source_coordinates = if reverse_source { "eta,xi" } else { "xi,eta" };
        let source = format!(
            "model Motion() {{
            domain reference=box(0,1,0,1);
            domain physical=box(-10,10,-10,10);
            coordinate xi:m on reference from reference[0];
            coordinate eta:m on reference from reference[1];
            coordinate x:m on physical from physical[0];
            coordinate y:m on physical from physical[1];
            variable volume:1 on reference;
            relation map_volume on reference {{
                volume=volume_jacobian(from=({source_coordinates}),at=({map}));
            }}
        }}"
        );
        let (transaction, model, _) = compile("motion.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let relation = program
            .nodes()
            .find_map(|node| match node {
                KernelNode::Relation(relation) => Some(relation.id()),
                _ => None,
            })
            .unwrap();
        Self { program, relation }
    }

    pub(super) fn state(
        &self,
        reference: &SimplicialMesh,
        time: f64,
    ) -> Result<FixedTopologyGeometryState<2>, Diagnostic> {
        let typed = self
            .program
            .typed_relation_residual(self.relation)
            .map_err(|errors| errors.into_iter().next().unwrap())?;
        let expression = typed.expression();
        let (id, at) = expression
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(index, node)| match node {
                ExprNode::CoordinateMapFactor { at, .. } => {
                    Some((expression.node_id(index as u32).unwrap(), at))
                }
                _ => None,
            })
            .unwrap();
        let rows =
            eqiora_ir::ScalarOperatorIr::bind_affine_coordinate_map(&typed, id, &mut |symbol| {
                if symbol == SymbolRef::Time {
                    Some(
                        ValueLiteral::from_real(
                            ValueType::scalar(
                                ScalarDomain::Real,
                                DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap(),
                            )
                            .unwrap(),
                            time,
                        )
                        .unwrap(),
                    )
                } else if let SymbolRef::Parameter(id) = symbol {
                    self.program.typed_value(id.erase()).cloned()
                } else {
                    None
                }
            })?;
        let coordinates = reference
            .vertices()
            .iter()
            .map(|point| {
                let mut mapped = vec![0.; 2];
                for (row, (target, _)) in rows.iter().zip(at) {
                    let Some(ExprNode::Symbol(SymbolRef::Coordinate { axis, .. })) =
                        expression.node(*target)
                    else {
                        panic!("typed target coordinate");
                    };
                    mapped[*axis] = row.offsets()[0]
                        + row
                            .selected_symbols()
                            .iter()
                            .zip(row.coefficients())
                            .map(|(coordinate, coefficient)| {
                                let SymbolRef::Coordinate { axis, .. } = coordinate.symbol() else {
                                    panic!("typed source coordinate");
                                };
                                coefficient * point[axis]
                            })
                            .sum::<f64>();
                }
                mapped
            })
            .collect();
        FixedTopologyGeometryState::new(reference, coordinates)
    }
}

#[test]
fn retained_map_binding_preserves_axis_order_and_rejects_nonlinearity() {
    let mesh = reference();
    for reverse in [false, true] {
        for map in [
            "x=(1+0.5[1/s]*time())*xi+0.25[m/s]*time(),y=(1+0.25[1/s]*time())*eta",
            "y=(1+0.25[1/s]*time())*eta,x=(1+0.5[1/s]*time())*xi+0.25[m/s]*time()",
        ] {
            let motion = Motion::new(map, reverse);
            for time in [0., 1., 2.] {
                let state = motion.state(&mesh, time).unwrap();
                for (actual, point) in state.coordinates().iter().zip(mesh.vertices()) {
                    assert_eq!(
                        *actual,
                        vec![
                            (1. + 0.5 * time) * point[0] + 0.25 * time,
                            (1. + 0.25 * time) * point[1]
                        ]
                    );
                }
            }
        }
    }
    let nonlinear = Motion::new("x=xi*xi/1[m],y=eta", false);
    assert!(nonlinear.state(&mesh, 0.).is_err());
    let singular = Motion::new("x=(1-1[1/s]*time())*xi,y=eta", false);
    assert!(singular.state(&mesh, 1.).is_err());
}
