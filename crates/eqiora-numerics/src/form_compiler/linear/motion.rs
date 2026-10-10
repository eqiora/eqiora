//! Exact storage-map selection and the first affine moving scalar chart.
use super::*;
use eqiora_core::{DimExponents, ValueLiteral};
use eqiora_meshing::MeshGeometry;
use eqiora_schema::kernel::{CoordinateMapFactor, ExprDag, ExprId, FieldRole};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StorageMotion {
    relation: RawId,
    factor: ExprId,
    pub(crate) domain: RawId,
    pub(crate) target: RawId,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct UniformChart {
    pub(crate) scale: f64,
    pub(crate) offset: [f64; 2],
}

impl StorageMotion {
    pub(super) fn bind_transport<S: Coefficient>(
        &self,
        context: &Context<'_, S>,
        field: RawId,
        row: &mut super::lowering::Terms<S>,
        current: &UniformChart,
        previous: Option<(f64, f64)>,
    ) -> Result<(), Diagnostic> {
        if row.transport.is_empty() {
            return Ok(());
        }
        let Some(KernelNode::Relation(relation)) = context.program.node(self.relation) else {
            unreachable!()
        };
        let RelationMeaning::Conservation(law) = relation.meaning() else {
            unreachable!()
        };
        if context.owner != self.relation {
            return Err(invalid(
                "moving transport must belong to its exact storage Law",
            ));
        }
        let typed = typed_relation(context.program, self.relation)?;
        let velocities = typed.verify_uniform_ale_transport(
            self.factor, law.storage().expect("mapped storage").0, law.flux(), field.downcast().expect("Field"),
        ).map_err(|error| invalid(&format!("moving transport requires exact capacity, map cofactor and explicit material-minus-mesh velocity correspondence: {error}")))?;
        let Some((previous_time_s, step_s)) = previous else {
            return Ok(());
        };
        if !step_s.is_finite() || step_s <= 0.0 {
            return Err(invalid(
                "moving transport requires a positive finite geometry step",
            ));
        }
        let previous = self.bind(context.program, previous_time_s)?;
        let Some(ExprNode::CoordinateMapFactor { source, .. }) = context.dag.node(self.factor)
        else {
            unreachable!()
        };
        let constant = |value| Data::constant(2, <S as From<f64>>::from(value));
        let capacity = row
            .storage
            .get(&field)
            .ok_or_else(|| invalid("moving transport lost its exact storage Field"))?
            .clone()
            .multiply(constant(1.0 / (current.scale * current.scale)));
        // The sealed GeometryAction uses a linear path between accepted maps.
        // In 2D its cofactor is affine, so its path average is the midpoint
        // cofactor. Only mesh flux uses this geometric time integral; material
        // velocity keeps the Backward Euler endpoint evaluation.
        let average_scale = 0.5 * previous.scale + 0.5 * current.scale;
        for &coordinate in source {
            let Some(ExprNode::Symbol(SymbolRef::Coordinate { axis, .. })) =
                context.dag.node(coordinate)
            else {
                unreachable!()
            };
            let secant = context
                .data(coordinate, 0)?
                .multiply(constant((current.scale - previous.scale) / step_s))
                .add(constant(
                    (current.offset[*axis] - previous.offset[*axis]) / step_s,
                ));
            let mesh_flux = secant.multiply(constant(average_scale));
            let material_flux = context
                .data(velocities[*axis], 0)?
                .multiply(constant(-current.scale));
            // Flux pairs with minus grad(test) in reference coordinates.
            row.transport.insert(
                (field, *axis),
                capacity.clone().multiply(mesh_flux.add(material_flux)),
            );
        }
        Ok(())
    }

    pub(crate) fn select(
        program: &KernelProgram,
        domain: RawId,
    ) -> Result<Option<Self>, Diagnostic> {
        let mut selected = None;
        for relation in crate::canonical::relations_on(program, domain) {
            let Some(KernelNode::Relation(definition)) = program.node(relation) else {
                continue;
            };
            let RelationMeaning::Conservation(law) = definition.meaning() else {
                continue;
            };
            let Some((stored, _)) = law.storage() else {
                continue;
            };
            let dag = definition.expression();
            let mut pending = vec![stored];
            let mut leaves = Vec::new();
            let mut maps = Vec::new();
            let mut work = 0usize;
            while let Some(id) = pending.pop() {
                work += 1;
                if work > 1_000_000 {
                    return Err(invalid("mapped storage exceeds traversal budget"));
                }
                match dag.node(id) {
                    Some(ExprNode::Mul(a, b)) => pending.extend([*a, *b]),
                    Some(ExprNode::CoordinateMapFactor {
                        factor: CoordinateMapFactor::VolumeScale,
                        ..
                    }) => maps.push(id),
                    _ => leaves.push(id),
                }
            }
            if maps.is_empty() {
                continue;
            }
            let [factor] = maps.as_slice() else {
                return Err(invalid("moving storage requires one exact volume map"));
            };
            let Some(ExprNode::CoordinateMapFactor { source, at, .. }) = dag.node(*factor) else {
                unreachable!()
            };
            if coordinate_domain(dag, source.iter().copied()) != Some(domain) {
                return Err(invalid(
                    "moving storage map must use its complete planar source coordinates",
                ));
            }
            let target = coordinate_domain(dag, at.iter().map(|(id, _)| *id)).ok_or_else(|| {
                invalid("moving storage map requires complete planar target coordinates")
            })?;
            // The existing history operator integrates the same normalized
            // capacity on old and new geometries. Prove that capacity cannot
            // change with Time or position, rather than compare two samples.
            let mut seen = BTreeSet::new();
            while let Some(id) = leaves.pop() {
                if !seen.insert(id) {
                    continue;
                }
                let node = dag.node(id).expect("retained storage node");
                match node {
                    ExprNode::Constant(_) | ExprNode::Symbol(SymbolRef::Parameter(_)) => {}
                    ExprNode::Symbol(SymbolRef::Field(field)) if matches!(program.node(field.erase()), Some(KernelNode::Field(field)) if field.role() == FieldRole::State) =>
                        {}
                    ExprNode::Neg(_)
                    | ExprNode::Add(..)
                    | ExprNode::Sub(..)
                    | ExprNode::Mul(..)
                    | ExprNode::Div(..)
                    | ExprNode::PowI(..)
                    | ExprNode::UnaryMath(..) => {
                        super::super::scalar::push_operands(node, &mut leaves)
                    }
                    _ => {
                        return Err(invalid(
                            "moving storage requires Time-independent, spatially constant normalized capacity",
                        ));
                    }
                }
            }
            if selected
                .replace(Self {
                    relation,
                    factor: *factor,
                    domain,
                    target,
                })
                .is_some()
            {
                return Err(invalid(
                    "moving scalar Region requires a unique storage motion",
                ));
            }
        }
        Ok(selected)
    }

    pub(crate) fn bind(
        &self,
        program: &KernelProgram,
        time_s: f64,
    ) -> Result<UniformChart, Diagnostic> {
        let typed = typed_relation(program, self.relation)?;
        let Some(ExprNode::CoordinateMapFactor { at, .. }) = typed.expression().node(self.factor)
        else {
            return Err(invalid("moving storage lost its exact map factor"));
        };
        let time = ValueLiteral::from_real(
            ValueType::scalar(
                ScalarDomain::Real,
                DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap(),
            )
            .unwrap(),
            time_s,
        )
        .map_err(|_| invalid("moving chart requires finite physical Time"))?;
        let rows = eqiora_ir::ScalarOperatorIr::bind_affine_coordinate_map(
            &typed,
            self.factor,
            &mut |symbol| match symbol {
                SymbolRef::Time => Some(time.clone()),
                SymbolRef::Parameter(id) => program.typed_value(id.erase()).cloned(),
                _ => None,
            },
        )?;
        let mut matrix = [[0.; 2]; 2];
        let mut offset = [0.; 2];
        for (row, (target, _)) in rows.iter().zip(at) {
            let Some(ExprNode::Symbol(SymbolRef::Coordinate { axis, .. })) =
                typed.expression().node(*target)
            else {
                unreachable!("typed target")
            };
            offset[*axis] = row.offsets()[0];
            for (coordinate, value) in row.selected_symbols().iter().zip(row.coefficients()) {
                let SymbolRef::Coordinate { axis: source, .. } = coordinate.symbol() else {
                    unreachable!("typed source")
                };
                matrix[*axis][source] = *value;
            }
        }
        let scale = matrix[0][0];
        if matrix[0][1] != 0.
            || matrix[1][0] != 0.
            || matrix[1][1] != scale
            || !scale.is_finite()
            || scale <= 0.
        {
            return Err(invalid(
                "moving scalar diffusion currently requires positive uniform affine scaling and translation",
            ));
        }
        Ok(UniformChart { scale, offset })
    }
}

fn coordinate_domain(dag: &ExprDag, ids: impl Iterator<Item = ExprId>) -> Option<RawId> {
    let mut domain = None;
    let mut axes = BTreeSet::new();
    for id in ids {
        let ExprNode::Symbol(SymbolRef::Coordinate {
            support,
            factor,
            axis,
        }) = dag.node(id)?
        else {
            return None;
        };
        if support != factor
            || domain.is_some_and(|prior| prior != support.erase())
            || !axes.insert(*axis)
        {
            return None;
        }
        domain = Some(support.erase());
    }
    (axes == BTreeSet::from([0, 1])).then_some(domain).flatten()
}

impl UniformChart {
    pub(crate) fn geometry_state(
        &self,
        reference: &eqiora_meshing::SimplicialMesh,
    ) -> Result<eqiora_meshing::FixedTopologyGeometryState<2>, Diagnostic> {
        if reference.geometric_dimension() != 2 {
            return Err(invalid("moving scalar chart requires a planar Mesh"));
        }
        let coordinates = reference
            .vertices()
            .iter()
            .map(|point| {
                (0..2)
                    .map(|axis| self.scale * point[axis] + self.offset[axis])
                    .collect()
            })
            .collect();
        eqiora_meshing::FixedTopologyGeometryState::new(reference, coordinates)
    }
}
