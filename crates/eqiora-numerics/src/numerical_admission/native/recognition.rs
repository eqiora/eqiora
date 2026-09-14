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

pub(crate) fn recognize_exact_model(
    program: &KernelProgram,
    resources: &NativeMeshResources,
    scalar: Result<ExecutableScalarEquations, Diagnostic>,
    transient: Result<TransientIncompressibleNavierStokesCartesianModel2d, Diagnostic>,
    transient_geometry: Result<(), Diagnostic>,
    fsi: Result<FixedReferenceFsiCartesianModel2d, Diagnostic>,
) -> Result<RecognizedNativeModel, Diagnostic> {
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
        if !matches!(resources, NativeMeshResources::Cartesian { .. }) {
            return Err(invalid(
                "scalar conservation realization requires authenticated Cartesian resources",
            ));
        }
        return scalar.map(Box::new).map(RecognizedNativeModel::Scalar);
    }
    if elasticity.is_ok() {
        let NativeMeshResources::Cartesian {
            geometry,
            mesh,
            correspondence,
            ..
        } = resources
        else {
            return Err(invalid(
                "isotropic small-strain realization requires authenticated Cartesian resources",
            ));
        };
        return lower_isotropic_elasticity_geometry_2d(program, geometry, mesh, correspondence)
            .map(Box::new)
            .map(RecognizedNativeModel::Elasticity);
    }
    if stokes.is_ok() {
        let NativeMeshResources::GmshSimplicial {
            geometry,
            mesh,
            correspondence,
            ..
        } = resources
        else {
            return Err(invalid(
                "steady incompressible mixed form requires authenticated Gmsh simplicial resources",
            ));
        };
        return SteadyStokesGeometryBinding2d::new_authenticated(
            program,
            geometry,
            mesh,
            correspondence,
        )
        .map(Box::new)
        .map(RecognizedNativeModel::Stokes);
    }
    if transient.is_ok() || transient_geometry.is_ok() {
        if let NativeMeshResources::GmshSimplicial {
            geometry,
            mesh,
            correspondence,
            ..
        } = resources
        {
            return TransientNavierStokesGeometryBinding2d::new_authenticated(
                program,
                geometry,
                mesh,
                correspondence,
            )
            .map(Box::new)
            .map(RecognizedNativeModel::TransientGeometry);
        }
        let transient = transient?;
        let exact_bounds = resources
            .geometry()
            .planar_rectangle_bounds()
            .ok_or_else(|| {
                invalid("transient storage realization requires an exact planar rectangle Geometry")
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
        return Ok(RecognizedNativeModel::Transient(Box::new(transient)));
    }
    if fsi.is_ok() {
        if !matches!(
            resources,
            NativeMeshResources::AdjacentPartitionSimplicial { .. }
        ) {
            return Err(invalid(
                "coupled interface realization requires authenticated adjacent-partition simplicial resources",
            ));
        }
        return fsi.map(Box::new).map(RecognizedNativeModel::Fsi);
    }
    let scalar = scalar.unwrap_err();
    let elasticity = elasticity.unwrap_err();
    let stokes = stokes.unwrap_err();
    let transient_message = match (&transient, &transient_geometry) {
        (Err(cartesian), Err(geometry)) => format!(
            "Cartesian [{}: {}]; Geometry [{}: {}]",
            cartesian.code(),
            cartesian.message(),
            geometry.code(),
            geometry.message()
        ),
        _ => unreachable!("recognized transient handled above"),
    };
    let fsi = fsi.unwrap_err();
    Err(invalid(format!(
        "Model has no admitted exact mathematical realization: scalar conservation form [{}: {}]; isotropic small-strain form [{}: {}]; steady incompressible mixed form [{}: {}]; transient storage form [{transient_message}]; coupled interface form [{}: {}]",
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
    spatial: NativeSpatialPolicy,
    linear: &NativeLinearPolicy,
) -> Result<(), Diagnostic> {
    let properties = match spatial {
        NativeSpatialPolicy::ScalarQ1
        | NativeSpatialPolicy::TransientMiniP1(_)
        | NativeSpatialPolicy::TransientCellCentered(_) => LinearOperatorProperties::General,
        NativeSpatialPolicy::ScalarTpfa | NativeSpatialPolicy::ElasticityQ1 => {
            LinearOperatorProperties::SymmetricPositiveDefinite
        }
        NativeSpatialPolicy::StokesMiniP1(_) => LinearOperatorProperties::SymmetricIndefinite,
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
