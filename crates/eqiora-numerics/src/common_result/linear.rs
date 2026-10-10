use super::*;

impl CommonResult {
    /// Accept the complete scalar Field inventory from one validated solve.
    pub(crate) fn accept_linear(
        plan: CommonLinearPlan,
        elapsed_seconds: f64,
        output: CommonLinearRunOutput<f64>,
    ) -> Result<Self, Diagnostic> {
        require_elapsed(elapsed_seconds)?;
        let elasticity = plan
            .elasticity_observation(output.reactions.as_ref())?
            .map(|value| ElasticityResultObservation {
                constrained_reaction: value.constrained_reaction(),
                integrated_body_force: value.integrated_body_force(),
                exact_bounds: value.exact_bounds(),
            });
        if output.fields.len() != plan.fields().len()
            || output.fields.iter().zip(plan.fields()).any(
                |((actual, value_type, _, actual_space), (expected, expected_type))| {
                    *actual != expected
                        || value_type != expected_type
                        || !plan
                            .field_space(expected)
                            .is_ok_and(|space| *actual_space == space)
                },
            )
        {
            return Err(invalid(
                "scalar output differs from the complete typed Plan Field inventory or coefficient Space",
            ));
        }
        plan.check_nullspace_evidence(&output.fields[0].2, output.nullspace.as_ref())?;
        let fields = output
            .fields
            .into_iter()
            .map(|(field, value_type, values, _)| {
                let (space, layouts) = field_layout(&plan, field)?;
                let mut remaining = values.as_slice();
                let mut blocks = Vec::new();
                let scalar_width: usize =
                    if value_type.scalar_domain() == eqiora_core::ScalarDomain::Complex {
                        2
                    } else {
                        1
                    };
                for (association, shape) in layouts {
                    let count = shape.iter().try_fold(scalar_width, |count, extent| {
                        count
                            .checked_mul(*extent)
                            .ok_or_else(|| invalid("linear Result block cardinality overflows"))
                    })?;
                    let (values, rest) = remaining.split_at_checked(count).ok_or_else(|| {
                        invalid("linear Result omits selected coefficient values")
                    })?;
                    blocks.push(CommonResultFieldBlock::new(
                        association,
                        values.to_vec(),
                        shape,
                    )?);
                    remaining = rest;
                }
                if !remaining.is_empty() {
                    return Err(invalid("linear Result contains excess coefficient values"));
                }
                CommonResultField::new(
                    field.ulid().to_string(),
                    value_type.scalar_domain(),
                    value_type.dimension(),
                    fields::value_shape(&value_type),
                    space,
                    blocks,
                )
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        Self::finish_static(
            ResolvedCommonPlan::Linear(Box::new(plan)),
            CommonResultFamily::Linear,
            elapsed_seconds,
            fields,
            CommonSolveEvidence::from_report(&output.solve_report),
            CommonAssemblyEvidence::from_report(&output.assembly_report),
            StaticObservation::Linear {
                nullspace: output.nullspace,
                elasticity,
            },
        )
    }
}

type BlockLayout = (CommonFieldAssociation, Vec<usize>);

/// Exact coefficient block inventory shared by publication and artifact replay.
pub(super) fn field_layout(
    plan: &CommonLinearPlan,
    field: eqiora_core::Id<eqiora_core::entity::kinds::Field>,
) -> Result<(Space, Vec<BlockLayout>), Diagnostic> {
    use eqiora_realization::SpaceFamily;
    let space = plan.field_space(field)?;
    let association = match space.family() {
        SpaceFamily::ContinuousLagrange { .. } => CommonFieldAssociation::Vertex,
        SpaceFamily::CellConstant => CommonFieldAssociation::Cell,
        SpaceFamily::TetrahedralEdge => CommonFieldAssociation::Edge,
        SpaceFamily::TetrahedralFace => CommonFieldAssociation::Face,
        SpaceFamily::SimplexP1Bubble => {
            let (_, ty) = plan
                .fields()
                .find(|(id, _)| *id == field)
                .ok_or_else(|| invalid("linear Result Field is outside the Plan"))?;
            let entities = plan.field_coefficient_entities(field)?;
            let vertices = entities
                .iter()
                .filter(|entity| entity.dimension() == 0)
                .count();
            let cells = entities.len() - vertices;
            if vertices == 0 || cells == 0 {
                return Err(invalid(
                    "bubble Result requires vertex and cell coefficients",
                ));
            }
            let mut vertex_shape = vec![vertices];
            vertex_shape.extend(fields::value_shape(ty));
            let mut cell_shape = vec![cells];
            cell_shape.extend(fields::value_shape(ty));
            return Ok((
                space,
                vec![
                    (CommonFieldAssociation::Vertex, vertex_shape),
                    (CommonFieldAssociation::CellBubble, cell_shape),
                ],
            ));
        }
    };
    Ok((
        space,
        vec![(association, plan.field_support(field.erase())?.0)],
    ))
}
