//! Exact numerical policy admission for the bounded ALE consumer.
use super::*;

pub(super) fn require_supported_dimension<const D: usize>() -> Result<(), Diagnostic> {
    if matches!(D, 2 | 3) {
        Ok(())
    } else {
        Err(invalid_realization(
            "fixed-topology ALE FSI finalization supports dimensions two and three",
        ))
    }
}

pub(super) fn required_quadrature_policy<const D: usize>() -> Result<QuadraturePolicy, Diagnostic> {
    match D {
        2 => Ok(QuadraturePolicy::TriangleDuffyGaussLegendre {
            points_per_axis: NonZeroUsize::new(LEGACY_TRIANGLE_DUFFY_POINTS_PER_AXIS)
                .expect("five is non-zero"),
        }),
        3 => Ok(QuadraturePolicy::SimplexDuffyGaussLegendre {
            spatial_dimension: NonZeroUsize::new(3).expect("three is non-zero"),
            points_per_axis: NonZeroUsize::new(TETRAHEDRON_DUFFY_POINTS_PER_AXIS)
                .expect("seven is non-zero"),
        }),
        _ => Err(invalid_realization(
            "fixed-topology ALE FSI has no quadrature policy for this dimension",
        )),
    }
}

fn required_quadrature_rule<const D: usize>() -> Result<QuadratureRule, Diagnostic> {
    match D {
        2 => triangle_duffy_gauss_legendre(LEGACY_TRIANGLE_DUFFY_POINTS_PER_AXIS),
        3 => simplex_duffy_gauss_legendre(3, TETRAHEDRON_DUFFY_POINTS_PER_AXIS),
        _ => Err(invalid_realization(
            "fixed-topology ALE FSI has no simplex quadrature rule for this dimension",
        )),
    }
}

pub(super) fn weak_functional_power<const D: usize>(
    pressure: eqiora_core::DynQuantity,
    velocity: eqiora_core::DynQuantity,
    length: eqiora_core::DynQuantity,
) -> Result<eqiora_core::DynQuantity, Diagnostic> {
    require_supported_dimension::<D>()?;
    let mut power = pressure.try_mul(velocity)?;
    for _ in 1..D {
        power = power.try_mul(length)?;
    }
    Ok(power)
}

pub(super) fn require_exact_plan<const D: usize>(
    model: &AleFsiCartesianModel<D>,
    resolved: &ResolvedFixedTopologyAleCoupledRealization,
    mesh_artifact: MeshArtifactReference,
    mesh: &SimplicialMesh,
) -> Result<
    (
        AleFsiFieldIdentities<D>,
        FixedReferenceFsiScale<D>,
        QuadratureRule,
    ),
    Diagnostic,
> {
    if resolved.model() != model.model()
        || resolved.semantic_revision().get() != model.semantic_revision()
    {
        return Err(invalid_realization(
            "resolved fixed-topology ALE Realization does not reference the exact lowered Semantic Model revision",
        ));
    }
    require_supported_dimension::<D>()?;
    let expected_requirements = fixed_topology_ale_fsi_requirements(model);
    if resolved.requirements() != &expected_requirements {
        return Err(invalid_realization(
            "resolved fixed-topology ALE requirements differ from the exact canonical Domain, Field, Relation, Connection, state, or execution roles",
        ));
    }

    let ale = resolved.plan();
    let plan = ale.coupled();
    let spatial = plan.spatial();
    if plan.target()
        != (Target::HostCpu {
            threads: NonZeroUsize::MIN,
        })
        || plan.schedule() != ExecutionSchedule::Offline
        || plan.operator_properties() != LinearOperatorProperties::General
    {
        return Err(invalid_realization(
            "fixed-topology ALE FSI v1 requires one offline serial-host general nonlinear operator",
        ));
    }
    let discretization = spatial.discretization();
    if discretization.method() != DiscretizationMethod::ContinuousGalerkin
        || discretization.mesh()
            != (MeshPolicy::ImportedSimplicial {
                artifact: mesh_artifact,
            })
        || discretization.quadrature() != required_quadrature_policy::<D>()?
    {
        return Err(invalid_realization(
            "fixed-topology ALE FSI requires the exact authenticated mesh digest, continuous Galerkin method, and dimension-appropriate Duffy simplex quadrature",
        ));
    }
    let fields = field_identities(model);
    require_spaces(model, resolved)?;
    require_dimension(
        spatial.coordinate_length_scale().quantity().dim(),
        LENGTH,
        "ALE FSI coordinate scale",
    )?;
    let length = spatial.coordinate_length_scale().quantity();
    let velocity = scale_for(plan, AlgebraicBlock::Field(fields.fluid_velocity))?;
    let pressure = scale_for(plan, AlgebraicBlock::Field(fields.fluid_pressure))?;
    let solid_velocity_scale = scale_for(plan, AlgebraicBlock::Field(fields.solid_velocity))?;
    require_dimension(velocity.dim(), VELOCITY, "ALE FSI velocity scale")?;
    require_dimension(pressure.dim(), PRESSURE, "ALE FSI pressure scale")?;
    if solid_velocity_scale != velocity
        || plan
            .time_step()
            .eliminated_states()
            .iter()
            .find(|state| state.pair().state() == fields.solid_displacement)
            .ok_or_else(|| invalid_realization("ALE eliminated state has no exact Field binding"))?
            .state_scale()
            .quantity()
            != length
        || plan.scaling().weak_functional_scale().quantity()
            != weak_functional_power::<D>(pressure, velocity, length)?
    {
        return Err(invalid_realization(
            "fixed-topology ALE FSI requires one shared velocity scale, displacement scale L, and derived weak-functional scale P U L^(D - 1)",
        ));
    }
    let quality = ale.mesh_motion().quality_gate().minimum_mean_ratio();
    if mesh.quality_gate().minimum_mean_ratio() != quality {
        return Err(invalid_realization(
            "authenticated ALE reference mesh must carry the exact resolved geometry-quality gate",
        ));
    }
    Ok((
        fields,
        FixedReferenceFsiScale::<D>::new(length.value(), velocity.value(), pressure.value())?,
        required_quadrature_rule::<D>()?,
    ))
}

fn require_spaces<const D: usize>(
    model: &AleFsiCartesianModel<D>,
    resolved: &ResolvedFixedTopologyAleCoupledRealization,
) -> Result<(), Diagnostic> {
    let p1 = Space::continuous_lagrange(NonZeroU16::MIN);
    let expected = [
        (
            fluid_domain(model),
            vec![
                (fluid_velocity(model), Space::simplex_p1_bubble()),
                (fluid_pressure(model), p1),
            ],
        ),
        (solid_domain(model), vec![(solid_velocity(model), p1)]),
    ];
    let expected_domain_count = expected.len();
    let spatial = resolved.plan().coupled().spatial();
    for (domain, mut bindings) in expected {
        bindings.sort_by_key(|(field, _)| field.ulid());
        let Some(actual) = spatial
            .domains()
            .iter()
            .find(|selection| selection.domain() == domain)
        else {
            return Err(invalid_realization(
                "fixed-topology ALE FSI spatial plan omits an exact canonical Domain",
            ));
        };
        let actual_bindings = actual
            .field_spaces()
            .iter()
            .map(|binding| (binding.field(), binding.space()))
            .collect::<Vec<_>>();
        if actual_bindings != bindings || !actual.constraints().is_empty() {
            return Err(invalid_realization(
                "fixed-topology ALE FSI requires exact MINI/P1/P1 spaces without an independent pressure-gauge block",
            ));
        }
    }
    if spatial.domains().len() != expected_domain_count {
        return Err(invalid_realization(
            "fixed-topology ALE FSI spatial plan contains a foreign Domain",
        ));
    }
    Ok(())
}

fn scale_for(
    plan: &eqiora_realization::CoupledFieldwiseRealizationPlan,
    block: AlgebraicBlock,
) -> Result<eqiora_core::DynQuantity, Diagnostic> {
    plan.scaling()
        .block_scales()
        .iter()
        .find(|entry| entry.block() == block)
        .map(|entry| entry.scale().quantity())
        .ok_or_else(|| invalid_realization("fixed-topology ALE FSI omits an exact block scale"))
}

fn require_dimension(
    actual: DimExponents,
    expected: DimExponents,
    label: &str,
) -> Result<(), Diagnostic> {
    if actual != expected {
        return Err(invalid_realization(format!(
            "{label} has incompatible physical dimension {actual:?}"
        )));
    }
    Ok(())
}

// Lineage and concrete execution placement are checked by the remesh caller.
// The common graph owns every numerical choice, including all Region geometry
// and Field transformations. Only imported mesh identities may differ.
pub(super) fn same_spatial_policy_except_imported_mesh(
    source: &PortableRealizationGraph,
    target: &PortableRealizationGraph,
) -> bool {
    source.domains().len() == target.domains().len()
        && source
            .domains()
            .iter()
            .zip(target.domains())
            .all(|(source, target)| {
                let source_discretization = source.discretization();
                let target_discretization = target.discretization();
                source.domain() == target.domain()
                    && source.coordinates() == target.coordinates()
                    && source.configuration() == target.configuration()
                    && source_discretization.method() == target_discretization.method()
                    && source_discretization.quadrature() == target_discretization.quadrature()
                    && matches!(
                        (source_discretization.mesh(), target_discretization.mesh()),
                        (
                            MeshPolicy::ImportedSimplicial { .. },
                            MeshPolicy::ImportedSimplicial { .. }
                        )
                    )
            })
        && source.fields() == target.fields()
        && source.geometry_actions() == target.geometry_actions()
        && source.transformations() == target.transformations()
        && source.systems() == target.systems()
        && source.linear_solves() == target.linear_solves()
        && source.nonlinear_solves() == target.nonlinear_solves()
        && source.placements() == target.placements()
        && source.root() == target.root()
}
