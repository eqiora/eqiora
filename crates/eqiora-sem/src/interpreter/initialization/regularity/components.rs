//! Initial Jacobian rows retain the original root, channel, and real/imaginary part.
use super::*;
use eqiora_ir::{ComponentScalarization, ScalarSymbolCoordinate};

pub(super) fn jacobian(
    program: &KernelProgram,
    owner: RawId,
    expression: &ExprDag,
    roots: &[ExprId],
    variables: &[Variable],
    context: &EvalContext<'_>,
) -> Result<Vec<Vec<Vec<f64>>>, Diagnostic> {
    let typed = point_residual(program, owner, expression, roots, variables, context)?;
    // Preserve the scalar IR's exact discrete control and lazy branch semantics.
    // Shaped/complex mathematical values use its existing component adapter.
    if typed
        .node_types()
        .iter()
        .all(|node| node.value_type.scalar_domain() != eqiora_core::ScalarDomain::Complex)
        && roots.iter().all(|root| {
            typed
                .node_type(*root)
                .expect("typed root")
                .shape()
                .is_scalar()
        })
        && variables.iter().all(|variable| {
            variable.value_type(program).is_ok_and(|ty| {
                ty.shape().is_scalar() && ty.scalar_domain() == eqiora_core::ScalarDomain::Real
            })
        })
    {
        let operator = ScalarOperatorIr::lower_typed_scalar(&typed)?;
        return Ok(differentiate(&operator, variables, context, roots.len())?
            .into_iter()
            .map(|row| vec![row])
            .collect());
    }
    let operator = ComponentScalarization::lower(&typed)?;
    let coordinates = variables
        .iter()
        .map(|variable| {
            ScalarSymbolCoordinate::for_value(variable.symbol(), &variable.value_type(program)?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let width = coordinates.iter().map(Vec::len).sum();
    let column = |source: &ScalarSymbolCoordinate| {
        let variable = coordinate(source.symbol(), variables, context)?;
        let component = coordinates[variable].iter().position(|candidate| {
            candidate.component_index() == source.component_index()
                && candidate.is_imaginary() == source.is_imaginary()
        })?;
        Some(coordinates[..variable].iter().map(Vec::len).sum::<usize>() + component)
    };
    let linearization = operator.linearize(|source| {
        let value = resolve(source, context)?;
        Some((
            value,
            if column(source).is_some() {
                DifferentiationRole::Unknown
            } else {
                DifferentiationRole::Frozen
            },
        ))
    })?;
    let columns = linearization
        .unknown_coordinates()
        .iter()
        .map(column)
        .collect::<Vec<_>>();
    let mut rows = vec![vec![0.; width]; operator.rows().len()];
    for index in 0..width {
        let direction = columns
            .iter()
            .map(|&column| f64::from(column == Some(index)))
            .collect::<Vec<_>>();
        let mut output = vec![0.; rows.len()];
        linearization.jvp(RelationTangent::Unknown(&direction), &mut output)?;
        for (row, value) in rows.iter_mut().zip(output) {
            row[index] = value;
        }
    }
    let mut grouped = vec![Vec::new(); roots.len()];
    for (source, row) in operator.rows().iter().zip(rows) {
        grouped[source.root_index()].push(row);
    }
    if owner.kind() == eqiora_core::EntityKind::Relation {
        for (sides, pair) in roots
            .as_chunks::<2>()
            .0
            .iter()
            .zip(grouped.as_chunks_mut::<2>().0.iter_mut())
        {
            let complex = sides.iter().any(|side| {
                typed
                    .node_type(*side)
                    .expect("typed equation side")
                    .value_type
                    .scalar_domain()
                    == eqiora_core::ScalarDomain::Complex
            });
            if complex {
                for (&side, rows) in sides.iter().zip(pair) {
                    if typed
                        .node_type(side)
                        .expect("typed equation side")
                        .value_type
                        .scalar_domain()
                        == eqiora_core::ScalarDomain::Real
                    {
                        *rows = std::mem::take(rows)
                            .into_iter()
                            .flat_map(|row| [row, vec![0.; width]])
                            .collect();
                    }
                }
            }
        }
    }
    Ok(grouped)
}

pub(super) fn resolve(source: &ScalarSymbolCoordinate, context: &EvalContext<'_>) -> Option<f64> {
    let value = evaluate::resolve_symbol(source.symbol(), context)?;
    value_component(source, &value)
}

pub(super) fn value_component(
    source: &ScalarSymbolCoordinate,
    value: &eqiora_core::ValueLiteral,
) -> Option<f64> {
    let extents = value.value_type().shape().extents();
    if extents.len() != source.component_index().len() {
        return None;
    }
    let flat = extents.iter().zip(source.component_index()).try_fold(
        0usize,
        |flat, (extent, &index)| {
            if index >= extent.get() {
                return None;
            }
            flat.checked_mul(extent.get() as usize)?
                .checked_add(index as usize)
        },
    )?;
    let (real, imaginary) = value.component(flat)?;
    Some(if source.is_imaginary() {
        imaginary
    } else {
        real
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{
        DimExponents, Id, OntologyId, ScalarDomain, ValueLiteral, ValueType, entity::kinds,
    };
    use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
    use eqiora_schema::{
        Model, ModelView,
        kernel::{FieldDef, FieldRole, RelationDef, UnaryMathFunction},
    };

    #[test]
    fn mixed_channels_keep_root_pairing_and_conjugate_sign() {
        let z = Id::<kinds::Field>::new();
        let w = Id::<kinds::Field>::new();
        let relation = Id::<kinds::Relation>::new();
        let model = OntologyId::<Model>::new();
        let complex = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
            .unwrap()
            .array(6)
            .unwrap();
        let real = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
        let mut builder = ExprDagBuilder::new();
        let zs = builder.symbol(SymbolRef::Field(z)).unwrap();
        let conjugates = (0..6)
            .map(|index| {
                let component = builder.index(zs, index).unwrap();
                builder
                    .unary_math(UnaryMathFunction::Conj, component)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let conjugate = builder.array(conjugates).unwrap();
        let ws = builder.symbol(SymbolRef::Field(w)).unwrap();
        let square = builder.mul(ws, ws).unwrap();
        let zero = builder
            .constant(
                ValueLiteral::new(
                    ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS).unwrap(),
                    [(0., 0.)],
                )
                .unwrap(),
            )
            .unwrap();
        let expression = builder
            .finish([zs, conjugate, ws, square, ws, zero])
            .unwrap();
        let mut transaction = Transaction::new("mixed initial differential");
        for (field, ty) in [(z, complex.clone()), (w, real)] {
            transaction.push(Op::DefineKernelNode {
                node: FieldDef::new(field, ty, FieldRole::State).into(),
            });
        }
        transaction.push(Op::DefineKernelNode {
            node: RelationDef::initial(relation, expression.clone())
                .unwrap()
                .into(),
        });
        for field in [z, w] {
            transaction.push(Op::Connect {
                from: relation.erase(),
                to: field.erase(),
                edge: EdgeKind::DependsOn,
            });
        }
        transaction.push(Op::DefineOntologyView {
            view: ModelView::new(model, [z.erase(), w.erase(), relation.erase()], [])
                .unwrap()
                .into(),
        });
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let typed_fields = BTreeMap::from([(
            z.erase(),
            ValueLiteral::new(complex, [(2., 3.); 6]).unwrap(),
        )]);
        let fields = BTreeMap::from([(w.erase(), 5.)]);
        let context = EvalContext {
            program: &program,
            time: 0.,
            typed_fields: &typed_fields,
            typed_ports: &BTreeMap::new(),
            typed_next: &BTreeMap::new(),
            fields: &fields,
            field_candidates: &BTreeMap::new(),
            derivatives: &BTreeMap::new(),
            next_fields: &BTreeMap::new(),
            ports: &BTreeMap::new(),
            port_candidates: &BTreeMap::new(),
            signal_sources: &BTreeMap::new(),
            physical: &BTreeMap::new(),
            physical_candidates: &BTreeMap::new(),
        };
        // Unknown order deliberately differs from first appearance in the expression.
        let rows = jacobian(
            &program,
            relation.erase(),
            &expression,
            expression.roots(),
            &[Variable::Field(w.erase()), Variable::Field(z.erase())],
            &context,
        )
        .unwrap();
        assert_eq!(
            rows.iter().map(Vec::len).collect::<Vec<_>>(),
            [12, 12, 1, 1, 2, 2]
        );
        // d(x+iy)/d(x,y)=I and d(conj(x+iy))/d(x,y)=diag(1,-1).
        for channel in 0..12 {
            for column in 0..13 {
                let identity = f64::from(column == channel + 1);
                assert_eq!(rows[0][channel][column], identity);
                assert_eq!(
                    rows[1][channel][column],
                    identity * if channel % 2 == 0 { 1. } else { -1. }
                );
            }
        }
        let mut identity = vec![0.; 13];
        identity[0] = 1.;
        assert_eq!(rows[2][0], identity);
        assert_eq!(rows[4][0], identity);
        assert_eq!(rows[4][1], vec![0.; 13]);
        assert_eq!(rows[5], vec![vec![0.; 13]; 2]);
        identity[0] = 10.; // d(w²)/dw=2w at w=5, independent of the complex channels.
        assert_eq!(rows[3][0], identity);
    }
}
