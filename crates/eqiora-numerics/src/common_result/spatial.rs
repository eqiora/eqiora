use super::*;

impl CommonResult {
    /// Accept the complete scalar Field inventory from one validated solve.
    #[cfg(test)]
    pub(crate) fn accept_linear(
        plan: CommonSpatialPlan,
        initial_state: Option<crate::CommonState>,
        elapsed_seconds: f64,
        output: CommonLinearRunOutput<f64>,
    ) -> Result<Self, Diagnostic> {
        Self::accept_spatial(
            plan,
            initial_state,
            elapsed_seconds,
            output.fields,
            crate::numerical_admission::CommonSpatialExecution {
                solve: CommonSolveHistory::Linear(Box::new(CommonSolveEvidence::from_report(
                    &output.solve_report,
                ))),
                assembly: CommonAssemblyEvidence::from_report(&output.assembly_report),
            },
            output.nullspace,
        )
    }

    pub(crate) fn accept_spatial(
        plan: CommonSpatialPlan,
        initial_state: Option<crate::CommonState>,
        elapsed_seconds: f64,
        fields: Vec<(
            eqiora_core::Id<eqiora_core::entity::kinds::Field>,
            eqiora_core::ValueType,
            Vec<f64>,
            Space,
        )>,
        execution: crate::numerical_admission::CommonSpatialExecution,
        nullspace: Option<crate::nullspace::NullspaceEvidence>,
    ) -> Result<Self, Diagnostic> {
        Self::accept_spatial_with(
            plan,
            initial_state,
            elapsed_seconds,
            fields,
            execution,
            nullspace,
            None,
        )
    }

    pub(crate) fn accept_live_linear(
        mut live: crate::numerical_admission::LiveStaticExecution<'_>,
    ) -> Result<Self, Diagnostic> {
        let _acceptance =
            eqiora_execution::telemetry_span!(backend("scalar_live_acceptance", "scalar_static"))
                .entered();
        let output = live.take_output()?;
        let execution = crate::numerical_admission::CommonSpatialExecution {
            solve: CommonSolveHistory::Linear(Box::new(CommonSolveEvidence::from_report(
                &output.solve_report,
            ))),
            assembly: CommonAssemblyEvidence::from_report(&output.assembly_report),
        };
        Self::accept_spatial_with(
            live.plan().clone(),
            live.initial().cloned(),
            0.0,
            output.fields,
            execution,
            output.nullspace,
            Some(&live),
        )
    }

    fn accept_spatial_with(
        plan: CommonSpatialPlan,
        initial_state: Option<crate::CommonState>,
        elapsed_seconds: f64,
        fields: Vec<(
            eqiora_core::Id<eqiora_core::entity::kinds::Field>,
            eqiora_core::ValueType,
            Vec<f64>,
            Space,
        )>,
        execution: crate::numerical_admission::CommonSpatialExecution,
        nullspace: Option<crate::nullspace::NullspaceEvidence>,
        live: Option<&crate::numerical_admission::LiveStaticExecution<'_>>,
    ) -> Result<Self, Diagnostic> {
        let _packing =
            eqiora_execution::telemetry_span!(backend("scalar_result_packing", "scalar_static"))
                .entered();
        require_elapsed(elapsed_seconds)?;
        if let Some(state) = &initial_state {
            plan.static_initial_fields(state)?;
        }
        if fields.len() != plan.fields().len()
            || fields.iter().zip(plan.fields()).any(
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
        let result_fields = fields
            .iter()
            .map(|(field, value_type, values, _)| {
                let (space, layouts) = field_layout(&plan, *field)?;
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
                CommonResultField::new(field.ulid().to_string(), value_type.clone(), space, blocks)
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        // Shape and finite-value validation precede equation reacceptance.
        plan.check_nullspace_evidence(&fields[0].2, nullspace.as_ref())?;
        let values = fields
            .iter()
            .map(|(id, _, values, _)| (id.erase(), values.clone()))
            .collect();
        drop(_packing);
        let residual = match live {
            Some(live) => live.validate(&values, &execution, nullspace.as_ref())?,
            None => plan.reaccept_static_execution(
                &values,
                nullspace.as_ref(),
                initial_state.as_ref(),
                &execution,
            )?,
        };
        let _publication = eqiora_execution::telemetry_span!(backend(
            "scalar_result_publication",
            "scalar_static"
        ))
        .entered();
        Self::finish_static(
            ResolvedCommonPlan::Spatial(Box::new(plan)),
            CommonResultFamily::Spatial,
            elapsed_seconds,
            result_fields,
            execution.solve,
            execution.assembly,
            StaticAcceptedEvidence::Spatial {
                nullspace,
                initial_state,
                residual,
            },
        )
    }
}

use super::fields::BlockLayout;

/// Exact coefficient block inventory shared by publication and artifact replay.
pub(super) fn field_layout(
    plan: &CommonSpatialPlan,
    field: eqiora_core::Id<eqiora_core::entity::kinds::Field>,
) -> Result<(Space, Vec<BlockLayout>), Diagnostic> {
    use eqiora_realization::SpaceFamily;
    let space = plan.field_space(field)?;
    if space.family() == SpaceFamily::SimplexP1Bubble
        || matches!(space.family(), SpaceFamily::ContinuousLagrange { order } if order.get() == 2)
    {
        let (_, ty) = plan
            .fields()
            .find(|(id, _)| *id == field)
            .ok_or_else(|| invalid("Result Field is outside the Plan"))?;
        let owner = plan.authenticated_mesh();
        let dimension = owner
            .geometry()
            .ok_or_else(|| invalid("nodal Result requires physical mesh geometry"))?
            .ambient_dimension();
        let entities = plan.field_coefficient_entities(field)?;
        return Ok((
            space,
            fields::entity_block_layout(space, dimension, &entities, &fields::value_shape(ty))?,
        ));
    }
    let association = match space.family() {
        SpaceFamily::ContinuousLagrange { order } if order.get() == 1 => {
            CommonFieldAssociation::Vertex
        }
        SpaceFamily::CellConstant => CommonFieldAssociation::Cell,
        SpaceFamily::TetrahedralEdge => CommonFieldAssociation::Edge,
        SpaceFamily::TetrahedralFace => CommonFieldAssociation::Face,
        _ => return Err(invalid("unsupported Result coefficient Space")),
    };
    Ok((
        space,
        vec![(association, plan.field_support(field.erase())?.0)],
    ))
}
