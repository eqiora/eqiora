use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResourceDigests {
    pub(crate) geometry: String,
    pub(crate) mesh: String,
    pub(crate) correspondence: String,
    pub(crate) production: String,
}

struct DigestedResources<'a> {
    geometry: &'a CanonicalGeometryV1,
    mesh: eqiora_artifact::ArtifactDigest,
    correspondence: eqiora_artifact::ArtifactDigest,
    production: eqiora_artifact::ArtifactDigest,
}

pub(crate) fn resource_digests(
    resources: &NativeMeshResources,
) -> Result<ResourceDigests, Diagnostic> {
    let digests = digest_resources(resources)?;
    Ok(ResourceDigests {
        geometry: hex_bytes(&digests.geometry.digest_bytes()),
        mesh: digests.mesh.to_string(),
        correspondence: digests.correspondence.to_string(),
        production: digests.production.to_string(),
    })
}

fn digest_resources(resources: &NativeMeshResources) -> Result<DigestedResources<'_>, Diagnostic> {
    let (geometry, mesh, correspondence, production) = match resources {
        NativeMeshResources::Cartesian {
            geometry,
            mesh,
            correspondence,
            production,
        } => (
            geometry,
            mesh.digest()?,
            correspondence.digest()?,
            production.digest()?,
        ),
        NativeMeshResources::AffineTriangleSimplicial {
            geometry,
            mesh,
            correspondence,
            production,
        }
        | NativeMeshResources::AdjacentPartitionSimplicial {
            geometry,
            mesh,
            correspondence,
            production,
        }
        | NativeMeshResources::GmshSimplicial {
            geometry,
            mesh,
            correspondence,
            production,
            ..
        } => (
            geometry,
            mesh.digest()?,
            correspondence.digest()?,
            production.digest()?,
        ),
    };
    Ok(DigestedResources {
        geometry,
        mesh,
        correspondence,
        production,
    })
}

pub(crate) fn resource_artifact_digests(
    resources: &NativeMeshResources,
) -> Result<
    (
        eqiora_artifact::ArtifactDigest,
        eqiora_artifact::ArtifactDigest,
        eqiora_artifact::ArtifactDigest,
        eqiora_artifact::ArtifactDigest,
    ),
    Diagnostic,
> {
    let digests = digest_resources(resources)?;
    Ok((
        eqiora_artifact::ArtifactDigest::from_sha256(digests.geometry.digest_bytes()),
        digests.mesh,
        digests.correspondence,
        digests.production,
    ))
}

pub(crate) fn recognize_capability(
    program: &KernelProgram,
    scalar: &Result<ExecutableScalarEquations, Diagnostic>,
    transient: &Result<TransientIncompressibleNavierStokesCartesianModel2d, Diagnostic>,
    transient_geometry: &Result<(), Diagnostic>,
    fsi: &Result<FixedReferenceFsiCartesianModel2d, Diagnostic>,
) -> Result<NativeCapability, Diagnostic> {
    let elasticity = recognize_isotropic_elasticity_geometry_mathematics(program);
    let stokes = recognize_steady_incompressible_stokes_geometry_mathematics(program);
    let recognized = [
        scalar.is_ok(),
        elasticity.is_ok(),
        stokes.is_ok(),
        transient.is_ok() || transient_geometry.is_ok(),
        fsi.is_ok(),
    ];
    if recognized.into_iter().filter(|matched| *matched).count() > 1 {
        return Err(invalid(
            "Model mathematical meaning is ambiguous across native capabilities",
        ));
    }
    if scalar.is_ok() {
        return Ok(NativeCapability::ScalarElliptic);
    }
    if elasticity.is_ok() {
        return Ok(NativeCapability::IsotropicElasticity);
    }
    if stokes.is_ok() {
        return Ok(NativeCapability::SteadyIncompressibleStokes);
    }
    if transient.is_ok() || transient_geometry.is_ok() {
        return Ok(NativeCapability::TransientIncompressibleFlow);
    }
    if fsi.is_ok() {
        return Ok(NativeCapability::FixedReferenceFsi);
    }
    let scalar = scalar.as_ref().unwrap_err();
    let elasticity = elasticity.unwrap_err();
    let stokes = stokes.unwrap_err();
    let transient_message = match (transient.as_ref(), transient_geometry.as_ref()) {
        (Err(cartesian), Err(geometry)) => format!(
            "Cartesian [{}: {}]; Geometry [{}: {}]",
            cartesian.code(),
            cartesian.message(),
            geometry.code(),
            geometry.message()
        ),
        _ => unreachable!("recognized transient handled above"),
    };
    let fsi = fsi.as_ref().unwrap_err();
    Err(invalid(format!(
        "Model mathematical meaning matches no native capability: scalar [{}: {}]; elasticity [{}: {}]; Stokes [{}: {}]; transient flow [{transient_message}]; FSI [{}: {}]",
        scalar.code(),
        scalar.message(),
        elasticity.code(),
        elasticity.message(),
        stokes.code(),
        stokes.message(),
        fsi.code(),
        fsi.message(),
    )))
}

pub(crate) fn recognize_exact_model(
    capability: NativeCapability,
    program: &KernelProgram,
    resources: &NativeMeshResources,
    scalar: Result<ExecutableScalarEquations, Diagnostic>,
    transient: Result<TransientIncompressibleNavierStokesCartesianModel2d, Diagnostic>,
    fsi: Result<FixedReferenceFsiCartesianModel2d, Diagnostic>,
) -> Result<RecognizedNativeModel, Diagnostic> {
    match (capability, resources) {
        (NativeCapability::ScalarElliptic, NativeMeshResources::Cartesian { .. }) => {
            scalar.map(Box::new).map(RecognizedNativeModel::Scalar)
        }
        (
            NativeCapability::IsotropicElasticity,
            NativeMeshResources::Cartesian {
                geometry,
                mesh,
                correspondence,
                ..
            },
        ) => lower_isotropic_elasticity_geometry_2d(program, geometry, mesh, correspondence)
            .map(Box::new)
            .map(RecognizedNativeModel::Elasticity),
        (
            NativeCapability::SteadyIncompressibleStokes,
            NativeMeshResources::GmshSimplicial {
                geometry,
                mesh,
                correspondence,
                ..
            },
        ) => SteadyStokesGeometryBinding2d::new_authenticated(
            program,
            geometry,
            mesh,
            correspondence,
        )
        .map(Box::new)
        .map(RecognizedNativeModel::Stokes),
        (
            NativeCapability::TransientIncompressibleFlow,
            NativeMeshResources::GmshSimplicial {
                geometry,
                mesh,
                correspondence,
                ..
            },
        ) => TransientNavierStokesGeometryBinding2d::new_authenticated(
            program,
            geometry,
            mesh,
            correspondence,
        )
        .map(Box::new)
        .map(RecognizedNativeModel::TransientGeometry),
        (NativeCapability::TransientIncompressibleFlow, _) => {
            let transient = transient?;
            let exact_bounds = resources
                .geometry()
                .planar_rectangle_bounds()
                .ok_or_else(|| {
                    invalid("transient flow requires an exact planar rectangle Geometry")
                })?;
            if !exact_bounds
                .iter()
                .zip(transient.bounds())
                .all(|(caller, model)| {
                    caller[0].to_bits() == model[0].to_bits()
                        && caller[1].to_bits() == model[1].to_bits()
                })
            {
                return Err(invalid(
                    "caller Mesh Geometry bounds differ from Model-owned transient Domain",
                ));
            }
            Ok(RecognizedNativeModel::Transient(Box::new(transient)))
        }
        (
            NativeCapability::FixedReferenceFsi,
            NativeMeshResources::AdjacentPartitionSimplicial { .. },
        ) => fsi.map(Box::new).map(RecognizedNativeModel::Fsi),
        _ => Err(invalid(
            "recognized Model capability and authenticated common Mesh kind are cross-wired",
        )),
    }
}

pub(crate) fn lower_scalar_candidate(
    program: &KernelProgram,
    resources: &NativeMeshResources,
) -> Result<ExecutableScalarEquations, Diagnostic> {
    let NativeMeshResources::Cartesian {
        geometry,
        mesh,
        correspondence,
        ..
    } = resources
    else {
        return Err(invalid(
            "scalar elliptic lowering requires authenticated Cartesian resources",
        ));
    };
    let source_domains = program
        .nodes()
        .filter(|node| {
            matches!(node,
        eqiora_schema::kernel::KernelNode::Domain(domain)
            if matches!(domain.kind(), eqiora_schema::kernel::DomainKind::CartesianBox { .. }))
        })
        .count();
    if source_domains > 0 {
        return ExecutableScalarEquations::source_regions(program, mesh.mesh());
    }
    let (domain, bounds, boundaries) =
        geometry_cartesian_support(program, geometry, mesh, correspondence)?;
    ExecutableScalarEquations::new(program, domain, bounds, boundaries)
}

pub(crate) fn require_policy_compatibility(
    capability: NativeCapability,
    spatial: NativeSpatialPolicy,
    linear: &NativeLinearPolicy,
) -> Result<(), Diagnostic> {
    let properties = match (capability, spatial) {
        (NativeCapability::ScalarElliptic, NativeSpatialPolicy::ScalarQ1)
        | (
            NativeCapability::TransientIncompressibleFlow,
            NativeSpatialPolicy::TransientMiniP1(_),
        )
        | (
            NativeCapability::TransientIncompressibleFlow,
            NativeSpatialPolicy::TransientCellCentered(_),
        ) => LinearOperatorProperties::General,
        (NativeCapability::ScalarElliptic, NativeSpatialPolicy::ScalarTpfa)
        | (NativeCapability::IsotropicElasticity, NativeSpatialPolicy::ElasticityQ1) => {
            LinearOperatorProperties::SymmetricPositiveDefinite
        }
        (NativeCapability::SteadyIncompressibleStokes, NativeSpatialPolicy::StokesMiniP1(_)) => {
            LinearOperatorProperties::SymmetricIndefinite
        }
        _ => {
            return Err(invalid(
                "Model capability and spatial policy are cross-wired",
            ));
        }
    };
    if !linear.planning_audit_is_coherent()
        || linear.execution != SERIAL_EXECUTION_PROVIDER
        || linear.workers != NonZeroUsize::MIN
    {
        return Err(invalid(
            "linear solver, preconditioner, reduction, or placement is unsupported",
        ));
    }
    linear
        .capabilities
        .require_problem(linear.solver, ScalarType::F64, properties)
}
