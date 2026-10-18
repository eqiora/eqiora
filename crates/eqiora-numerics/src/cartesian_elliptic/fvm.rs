//! Cell-centered assembly, explicit reference and field reconstruction.
use super::*;
mod boundary;
mod facets;
mod flux;
pub(crate) use flux::FvmFlux;

/// Method-private state retained between finalized TPFA assembly and field
/// reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalizedCartesianFvmAssembly<S = f64> {
    mesh: CartesianMesh,
    cell_centers: Vec<Vec<f64>>,
    linear_system: LinearSystem<S>,
    operator_properties: LinearOperatorProperties,
    scalar_reconstruction: Option<(CartesianMesh, Vec<Option<S>>)>,
    facets: Vec<CartesianFacetPacket<S>>,
    cell_dofs: Vec<Vec<DofId>>,
    assembly_report: AssemblyReport,
}

impl<S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + Sync>
    FinalizedCartesianFvmAssembly<S>
{
    pub(crate) fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    pub(crate) fn canonical_system(&self) -> Result<Arc<CanonicalCsrSystemView<S>>, Diagnostic> {
        Ok(Arc::new(CanonicalCsrSystemView::new(
            &self.linear_system,
            self.linear_system.rhs(),
            self.operator_properties,
        )?))
    }
}

impl FinalizedCartesianFvmAssembly {
    pub(crate) fn into_canonical(
        mut self,
        mean_reference: Option<f64>,
    ) -> Result<(Arc<CanonicalCsrSystemView>, FinalizedCartesianFvmState), Diagnostic> {
        if self
            .cell_dofs
            .iter()
            .enumerate()
            .any(|(cell, dofs)| dofs.as_slice() != [DofId::new(cell)])
        {
            return Err(invalid(
                "scalar TPFA reconstruction requires its canonical cell ordering",
            ));
        }
        let (reconstruction_mesh, reconstruction_boundary_values) =
            self.scalar_reconstruction.take().ok_or_else(|| {
                invalid("scalar TPFA reconstruction requires exactly one Field component")
            })?;
        let dimension = self.mesh.topological_dimension();
        let cell_count = self.cell_centers.len();
        let pure_neumann = self.facets.iter().all(|facet| {
            !matches!(
                facet.kind,
                CartesianFacetKind::Boundary { natural: false, .. }
            )
        });
        let reference = match mean_reference {
            Some(mean) if pure_neumann && dimension == 1 && mean.is_finite() => {
                let weights = self
                    .mesh
                    .axis_coordinates(0)
                    .expect("one-dimensional mesh")
                    .windows(2)
                    .map(|pair| pair[1] - pair[0])
                    .collect::<Vec<_>>();
                let measure: f64 = weights.iter().sum();
                Some(crate::nullspace::NullspaceConstraint::new(
                    vec![1.; cell_count],
                    weights,
                    mean * measure,
                )?)
            }
            Some(_) => {
                return Err(invalid(
                    "TPFA spatial reference requires finite mean and exactly one pure-Neumann dimension",
                ));
            }
            None if pure_neumann => {
                return Err(invalid(
                    "Cartesian TPFA system requires an essential boundary or explicit spatial reference",
                ));
            }
            None => None,
        };

        if reference.is_some() {
            self.operator_properties = LinearOperatorProperties::Symmetric;
        }
        let canonical_system = self.canonical_system()?;
        let Self {
            mesh,
            cell_centers,
            linear_system: _,
            operator_properties: _,
            scalar_reconstruction: _,
            facets,
            assembly_report,
            cell_dofs: _,
        } = self;
        Ok((
            canonical_system,
            FinalizedCartesianFvmState {
                mesh,
                cell_centers,
                reconstruction_mesh,
                reconstruction_boundary_values,
                facets,
                assembly_report,
                reference,
            },
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalizedCartesianFvmState {
    mesh: CartesianMesh,
    cell_centers: Vec<Vec<f64>>,
    reconstruction_mesh: CartesianMesh,
    reconstruction_boundary_values: Vec<Option<f64>>,
    facets: Vec<CartesianFacetPacket>,
    assembly_report: AssemblyReport,
    reference: Option<crate::nullspace::NullspaceConstraint>,
}

impl FinalizedCartesianFvmState {
    pub(crate) const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    pub(crate) fn point(
        &self,
        cell_values: Vec<f64>,
        canonical_system: Arc<CanonicalCsrSystemView>,
    ) -> Result<CartesianFvmPoint, Diagnostic> {
        if cell_values.len() != canonical_system.rows() {
            return Err(invalid(
                "Cartesian FVM solution shape differs from its finalized system",
            ));
        }
        Ok(CartesianFvmPoint {
            mesh: self.mesh.clone(),
            cell_centers: self.cell_centers.clone(),
            cell_values,
            canonical_system,
        })
    }

    pub(crate) fn assess(
        &self,
        system: &CanonicalCsrSystemView,
        values: &[f64],
        multiplier: f64,
        plan: eqiora_solver::SolverPlan,
    ) -> Result<crate::nullspace::NullspaceEvidence, Diagnostic> {
        let reference = self
            .reference
            .as_ref()
            .ok_or_else(|| invalid("TPFA has no declared reference"))?;
        crate::nullspace::assess_canonical_with_nullspace(
            system, reference, values, multiplier, plan,
        )
    }

    pub(crate) fn solve(
        self,
        request: LinearSolveRequest<'_>,
        system: Arc<CanonicalCsrSystemView>,
    ) -> Result<ScalarEllipticCartesianFvmSolution, Diagnostic> {
        if let Some(reference) = &self.reference {
            let solved =
                crate::nullspace::solve_canonical_with_nullspace(request, &system, reference)?;
            self.finish_parts(solved.values, solved.report, Some(solved.evidence), system)
        } else {
            let solved = request.solve(&system.linear_problem()?)?;
            self.finish(solved, system)
        }
    }

    pub(crate) fn finish(
        self,
        solved: LinearSolution,
        system: Arc<CanonicalCsrSystemView>,
    ) -> Result<ScalarEllipticCartesianFvmSolution, Diagnostic> {
        if self.reference.is_some() {
            return Err(invalid(
                "constrained TPFA requires its checked nullspace solve",
            ));
        }
        let (values, report) = solved.into_parts();
        self.finish_parts(values, report, None, system)
    }

    fn finish_parts(
        self,
        cell_values: Vec<f64>,
        solve_report: SolveReport,
        nullspace: Option<crate::nullspace::NullspaceEvidence>,
        canonical_system: Arc<CanonicalCsrSystemView>,
    ) -> Result<ScalarEllipticCartesianFvmSolution, Diagnostic> {
        let point = self.point(cell_values, canonical_system)?;
        let cell_values = &point.cell_values;
        let canonical_system = &point.canonical_system;
        let (boundary_flux_sum, boundary_load_sum) =
            self.facets
                .iter()
                .try_fold((0.0, 0.0), |(flux, load), facet| match facet.kind {
                    CartesianFacetKind::Interior { .. } => Ok((flux, load)),
                    CartesianFacetKind::Boundary {
                        cell,
                        coefficient,
                        load: contribution,
                        ..
                    } => {
                        let cell_value = cell_values.get(cell).copied().ok_or_else(|| {
                            invalid("Cartesian FVM boundary facet exceeds its finalized field")
                        })?;
                        Ok::<_, Diagnostic>((
                            flux + contribution - coefficient * cell_value,
                            load + contribution,
                        ))
                    }
                })?;
        if !boundary_flux_sum.is_finite() {
            return Err(invalid("Cartesian FVM boundary flux sum is non-finite"));
        }
        let integrated_source =
            canonical_system.right_hand_side().iter().sum::<f64>() - boundary_load_sum;
        if !integrated_source.is_finite() {
            return Err(invalid("Cartesian FVM source integral is non-finite"));
        }
        let reconstruction = reconstruct_cell_field_from_boundary_values(
            self.reconstruction_mesh,
            &self.mesh,
            cell_values,
            self.reconstruction_boundary_values,
            &self.facets,
        )?;

        Ok(ScalarEllipticCartesianFvmSolution {
            point,
            reconstruction,
            boundary_flux_sum,
            integrated_source,
            assembly_report: self.assembly_report,
            solve_report,
            nullspace,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_elliptic_cartesian_fvm<V, R, S>(
    mesh: &CartesianMesh,
    components: usize,
    flux: Option<FvmFlux<'_, V>>,
    reaction: &R,
    source: &S,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<V>,
    operator_properties: LinearOperatorProperties,
) -> Result<FinalizedCartesianFvmAssembly<V>, Diagnostic>
where
    V: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + From<f64> + Send + Sync,
    R: Fn(usize, usize, &[f64]) -> Result<V, Diagnostic> + Sync + ?Sized,
    S: Fn(usize, &[f64]) -> Result<V, Diagnostic> + Sync + ?Sized,
{
    let cells = mesh.entity_count(mesh.topological_dimension()).unwrap_or(0);
    cells
        .checked_mul(components)
        .ok_or_else(|| invalid("Cartesian FVM DOF count overflow"))?;
    let cell_dofs = (0..cells)
        .map(|cell| {
            (0..components)
                .map(|component| DofId::new(cell * components + component))
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    let (prepared, result) = with_elliptic_cartesian_fvm(
        mesh,
        components,
        &cell_dofs,
        flux,
        reaction,
        source,
        cell_quadrature,
        facet_quadrature,
        |plan, work| assembly.assemble(plan, work),
        operator_properties,
    )?;
    Ok(prepared.finish(result))
}

/// Geometry and facet data shared by complete systems and partial residuals.
pub(crate) struct PreparedCartesianFvmAssembly<S> {
    mesh: CartesianMesh,
    cell_centers: Vec<Vec<f64>>,
    scalar_reconstruction: Option<(CartesianMesh, Vec<Option<S>>)>,
    facets: Vec<CartesianFacetPacket<S>>,
    cell_dofs: Vec<Vec<DofId>>,
    operator_properties: LinearOperatorProperties,
}

impl<S: eqiora_core::Scalar + num_complex::ComplexFloat + Sync> PreparedCartesianFvmAssembly<S> {
    pub(crate) fn finish(
        self,
        result: eqiora_assembly::AssemblyResult<S>,
    ) -> FinalizedCartesianFvmAssembly<S> {
        let (systems, assembly_report) = result.into_parts();
        let mut systems = systems.into_iter();
        let linear_system = systems
            .next()
            .expect("one-target FVM assembly returns its system");
        debug_assert!(systems.next().is_none());
        FinalizedCartesianFvmAssembly {
            mesh: self.mesh,
            cell_centers: self.cell_centers,
            scalar_reconstruction: self.scalar_reconstruction,
            facets: self.facets,
            cell_dofs: self.cell_dofs,
            operator_properties: self.operator_properties,
            linear_system,
            assembly_report,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn with_elliptic_cartesian_fvm<V, R, S, T>(
    mesh: &CartesianMesh,
    components: usize,
    cell_dofs: &[Vec<DofId>],
    flux: Option<FvmFlux<'_, V>>,
    reaction: &R,
    source: &S,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    consume: impl FnOnce(&AssemblyPlan, &dyn eqiora_assembly::AssemblyWork<V>) -> Result<T, Diagnostic>,
    operator_properties: LinearOperatorProperties,
) -> Result<(PreparedCartesianFvmAssembly<V>, T), Diagnostic>
where
    V: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + From<f64> + Send + Sync,
    R: Fn(usize, usize, &[f64]) -> Result<V, Diagnostic> + Sync + ?Sized,
    S: Fn(usize, &[f64]) -> Result<V, Diagnostic> + Sync + ?Sized,
{
    validate_problem(mesh, cell_quadrature)?;
    let dimension = mesh.topological_dimension();
    if flux.is_some() {
        require_facet_rule(dimension, facet_quadrature)?;
    }
    let cell_count = mesh.entity_count(dimension).expect("mesh owns cells");
    if components == 0 {
        return Err(invalid("Cartesian FVM requires nonempty Field components"));
    }
    let dof_count = cell_count
        .checked_mul(components)
        .ok_or_else(|| invalid("Cartesian FVM component DOF count overflow"))?;
    if cell_dofs.len() != cell_count
        || cell_dofs.iter().any(|row| row.len() != components)
        || cell_dofs
            .iter()
            .flatten()
            .map(|dof| dof.index())
            .collect::<std::collections::BTreeSet<_>>()
            != (0..dof_count).collect()
    {
        return Err(invalid(
            "Cartesian FVM requires a complete exact cell/component DOF permutation",
        ));
    }
    let cell_centers = (0..cell_count)
        .map(|cell_index| {
            mesh.geometry_map(MeshEntity::new(dimension, cell_index))
                .expect("mesh cell has affine geometry")
                .origin()
                .to_vec()
        })
        .collect::<Vec<_>>();

    let facets = flux
        .map(|flux| {
            facets::prepare(
                mesh,
                &cell_centers,
                components,
                flux,
                facet_quadrature,
                operator_properties,
            )
        })
        .transpose()?
        .unwrap_or_default();
    let scalar_reconstruction = flux
        .filter(|flux| components == 1 && flux.direction.is_none())
        .map(|flux| {
            prepare_cell_field_reconstruction(mesh, &|axis, side, point| {
                (flux.value.boundary)(0, axis, side, point)
            })
        })
        .transpose()?;

    let assembly_plan = AssemblyPlan::new(vec![AssemblyTarget::new(dof_count)?])?;
    let target = assembly_plan
        .target_id(0)
        .expect("one-target FVM assembly plan owns its system");
    let packet_count = cell_count
        .checked_add(facets.len())
        .ok_or_else(|| invalid("Cartesian FVM packet count overflows usize"))?;
    let work = IndexedAssemblyWork::new(packet_count, |packet_index| {
        let (local, map) = if packet_index < cell_count {
            let volume_operator = CartesianVolumeCell {
                components,
                reaction,
                source,
            };
            let cell = MeshEntity::new(dimension, packet_index);
            let geometry = mesh
                .geometry_map(cell)
                .expect("mesh cell has affine geometry");
            let local = volume_operator.evaluate(&geometry, cell_quadrature)?;
            let dofs = (0..components)
                .map(|component| cell_dofs[packet_index][component])
                .collect::<Vec<_>>();
            let map = AssemblyMap::new(
                dofs.iter().copied().map(Some).collect(),
                dofs.into_iter().map(LocalUnknown::Free).collect(),
            )?;
            (local, map)
        } else {
            let facet_index = packet_index - cell_count;
            let facet = &facets[facet_index];
            match facet.kind {
                CartesianFacetKind::Interior { left, right } => {
                    let rows = [cell_dofs[left][facet.test], cell_dofs[right][facet.test]];
                    let left = cell_dofs[left][facet.trial];
                    let right = cell_dofs[right][facet.trial];
                    let local = CartesianInteriorFlux {
                        prescribed_flux: facet.prescribed_flux,
                    }
                    .evaluate(&facet.transmissibility, facet_quadrature)?;
                    let map = AssemblyMap::new(
                        rows.into_iter().map(Some).collect(),
                        vec![LocalUnknown::Free(left), LocalUnknown::Free(right)],
                    )?;
                    (local, map)
                }
                CartesianFacetKind::Boundary {
                    cell,
                    coefficient,
                    load,
                    ..
                } => {
                    let row = cell_dofs[cell][facet.test];
                    let column = cell_dofs[cell][facet.trial];
                    let local = LocalContribution::new(1, 1, vec![coefficient], vec![load])?;
                    let map = AssemblyMap::new(vec![Some(row)], vec![LocalUnknown::Free(column)])?;
                    (local, map)
                }
            }
        };
        AssemblyPacket::new(local, vec![TargetAssemblyMap::new(target, map)])
    });
    let result = consume(&assembly_plan, &work)?;
    Ok((
        PreparedCartesianFvmAssembly {
            mesh: mesh.clone(),
            cell_centers,
            scalar_reconstruction,
            facets,
            cell_dofs: cell_dofs.to_vec(),
            operator_properties,
        },
        result,
    ))
}

#[cfg(test)]
mod tests;
