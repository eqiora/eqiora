//! Cell-centered assembly, explicit reference and field reconstruction.
use super::*;

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
        let (reconstruction_mesh, reconstruction_boundary_values) =
            self.scalar_reconstruction.take().ok_or_else(|| {
                invalid("scalar TPFA reconstruction requires exactly one Field component")
            })?;
        let dimension = self.mesh.topological_dimension();
        let cell_count = self.cell_centers.len();
        let pure_neumann = self
            .facets
            .iter()
            .all(|facet| !matches!(facet.kind, CartesianFacetKind::Essential { .. }));
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
                    CartesianFacetKind::Essential { cell, value, .. } => {
                        let cell_value = cell_values.get(cell).copied().ok_or_else(|| {
                            invalid("Cartesian FVM boundary facet exceeds its finalized field")
                        })?;
                        Ok::<_, Diagnostic>((
                            flux + facet.transmissibility * (value - cell_value),
                            load + facet.transmissibility * value,
                        ))
                    }
                    CartesianFacetKind::Natural { flux_integral, .. } => {
                        Ok((flux + flux_integral, load + flux_integral))
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
pub(crate) fn finalize_elliptic_cartesian_fvm<V, K, R, S, B>(
    mesh: &CartesianMesh,
    components: usize,
    coefficient: &K,
    reaction: &R,
    source: &S,
    boundary: &B,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<V>,
    operator_properties: LinearOperatorProperties,
) -> Result<FinalizedCartesianFvmAssembly<V>, Diagnostic>
where
    V: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + From<f64> + Send + Sync,
    K: Fn(&[f64]) -> Result<V, Diagnostic> + Sync + ?Sized,
    R: Fn(&[f64]) -> Result<V, Diagnostic> + Sync + ?Sized,
    S: Fn(usize, &[f64]) -> Result<V, Diagnostic> + Sync + ?Sized,
    B: Fn(usize, usize, BoundarySide, &[f64]) -> Result<CartesianBoundaryValue<V>, Diagnostic>
        + ?Sized,
{
    validate_problem(mesh, cell_quadrature)?;
    let dimension = mesh.topological_dimension();
    require_facet_rule(dimension, facet_quadrature)?;
    let cell_count = mesh.entity_count(dimension).expect("mesh owns cells");
    if components == 0 {
        return Err(invalid("Cartesian FVM requires nonempty Field components"));
    }
    let dof_count = cell_count
        .checked_mul(components)
        .ok_or_else(|| invalid("Cartesian FVM component DOF count overflow"))?;
    let cell_centers = (0..cell_count)
        .map(|cell_index| {
            mesh.geometry_map(MeshEntity::new(dimension, cell_index))
                .expect("mesh cell has affine geometry")
                .origin()
                .to_vec()
        })
        .collect::<Vec<_>>();

    let facet_dimension = dimension - 1;
    let facet_count = mesh
        .entity_count(facet_dimension)
        .expect("mesh owns its facet stratum");
    let component_facets = facet_count
        .checked_mul(components)
        .ok_or_else(|| invalid("Cartesian FVM component facet count overflow"))?;
    let facets = (0..component_facets)
        .map(|index| {
            let facet_index = index / components;
            let component = index % components;
            let facet = MeshEntity::new(facet_dimension, facet_index);
            let facet_geometry = mesh
                .geometry_map(facet)
                .expect("mesh facet has affine geometry");
            let free_axes = mesh
                .entity_free_axes(facet)
                .expect("mesh facet exposes its tangent axes");
            let normal_axis = (0..dimension)
                .find(|axis| free_axes.binary_search(axis).is_err())
                .ok_or_else(|| invalid("Cartesian facet has no normal axis"))?;
            let cells = mesh
                .incidence(facet, dimension)
                .ok_or_else(|| invalid("Cartesian facet adjacency is unavailable"))?;
            let (kind, distance) = match cells.as_slice() {
                [left, right] => {
                    let left_center = &cell_centers[left.entity.index()];
                    let right_center = &cell_centers[right.entity.index()];
                    let distance = (right_center[normal_axis] - left_center[normal_axis]).abs();
                    require_positive_distance(distance)?;
                    (
                        CartesianFacetKind::Interior {
                            left: left.entity.index(),
                            right: right.entity.index(),
                        },
                        distance,
                    )
                }
                [cell] => {
                    let center = &cell_centers[cell.entity.index()];
                    let boundary_coordinates = facet_geometry.origin();
                    let distance = (boundary_coordinates[normal_axis] - center[normal_axis]).abs();
                    let (axis, side) = cartesian_boundary_facet_side(mesh, facet)?
                        .expect("one-cell Cartesian facet is on the boundary");
                    require_positive_distance(distance)?;
                    let kind = match boundary(component, axis, side, boundary_coordinates)? {
                        CartesianBoundaryValue::Essential(value) if value.is_finite() => {
                            CartesianFacetKind::Essential {
                                axis,
                                side,
                                cell: cell.entity.index(),
                                value,
                            }
                        }
                        CartesianBoundaryValue::Natural(_) => {
                            let flux_integral = integrate_boundary_flux(
                                &facet_geometry,
                                facet_quadrature,
                                &|coordinates| match boundary(component, axis, side, coordinates)? {
                                    CartesianBoundaryValue::Natural(value) => Ok(value),
                                    CartesianBoundaryValue::Essential(_) => Err(invalid(
                                        "Cartesian boundary changes kind within a facet",
                                    )),
                                },
                            )?;
                            CartesianFacetKind::Natural {
                                axis,
                                side,
                                cell: cell.entity.index(),
                                flux_integral,
                            }
                        }
                        CartesianBoundaryValue::Essential(_) => {
                            return Err(invalid("Cartesian boundary returned a non-finite value"));
                        }
                    };
                    (kind, distance)
                }
                _ => {
                    return Err(invalid(
                        "Cartesian facet requires exactly one or two adjacent cells",
                    ));
                }
            };
            let mut face_centroid = vec![0.0; dimension];
            facet_geometry.map_point(&vec![0.0; facet_dimension], &mut face_centroid)?;
            let coefficient_value = coefficient(&face_centroid)?;
            let transmissibility = facet_transmissibility(
                &facet_geometry,
                distance,
                coefficient_value,
                facet_quadrature,
                operator_properties,
            )?;
            Ok(CartesianFacetPacket {
                transmissibility,
                kind,
            })
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    let scalar_reconstruction = (components == 1)
        .then(|| {
            prepare_cell_field_reconstruction(mesh, &|axis, side, point| {
                boundary(0, axis, side, point)
            })
        })
        .transpose()?;

    let assembly_plan = AssemblyPlan::new(vec![AssemblyTarget::new(dof_count)?])?;
    let target = assembly_plan
        .target_id(0)
        .expect("one-target FVM assembly plan owns its system");
    let packet_count = dof_count
        .checked_add(component_facets)
        .ok_or_else(|| invalid("Cartesian FVM packet count overflows usize"))?;
    let work = IndexedAssemblyWork::new(packet_count, |packet_index| {
        let (local, map) = if packet_index < dof_count {
            let component = packet_index % components;
            let source = |point: &[f64]| source(component, point);
            let volume_operator = CartesianVolumeCell {
                reaction,
                source: &source,
            };
            let cell = MeshEntity::new(dimension, packet_index / components);
            let geometry = mesh
                .geometry_map(cell)
                .expect("mesh cell has affine geometry");
            let local = volume_operator.evaluate(&geometry, cell_quadrature)?;
            let dof = DofId::new(packet_index);
            let map = AssemblyMap::new(vec![Some(dof)], vec![LocalUnknown::Free(dof)])?;
            (local, map)
        } else {
            let facet_index = packet_index - dof_count;
            let component = facet_index % components;
            let facet = &facets[facet_index];
            match facet.kind {
                CartesianFacetKind::Interior { left, right } => {
                    let left = DofId::new(left * components + component);
                    let right = DofId::new(right * components + component);
                    let local = CartesianInteriorFlux
                        .evaluate(&facet.transmissibility, facet_quadrature)?;
                    let map = AssemblyMap::new(
                        vec![Some(left), Some(right)],
                        vec![LocalUnknown::Free(left), LocalUnknown::Free(right)],
                    )?;
                    (local, map)
                }
                CartesianFacetKind::Essential { cell, value, .. } => {
                    let cell = DofId::new(cell * components + component);
                    let local = CartesianBoundaryFlux
                        .evaluate(&facet.transmissibility, facet_quadrature)?;
                    let map = AssemblyMap::new(
                        vec![Some(cell)],
                        vec![LocalUnknown::Free(cell), LocalUnknown::Fixed(value)],
                    )?;
                    (local, map)
                }
                CartesianFacetKind::Natural {
                    cell,
                    flux_integral,
                    ..
                } => {
                    let cell = DofId::new(cell * components + component);
                    let local = LocalContribution::new(
                        1,
                        1,
                        vec![<V as From<f64>>::from(0.0)],
                        vec![flux_integral],
                    )?;
                    let map = AssemblyMap::new(vec![Some(cell)], vec![LocalUnknown::Free(cell)])?;
                    (local, map)
                }
            }
        };
        AssemblyPacket::new(local, vec![TargetAssemblyMap::new(target, map)])
    });
    let (systems, assembly_report) = assembly.assemble(&assembly_plan, &work)?.into_parts();
    let mut systems = systems.into_iter();
    let system = systems
        .next()
        .expect("one-target FVM assembly returns its system");
    debug_assert!(systems.next().is_none());

    Ok(FinalizedCartesianFvmAssembly {
        mesh: mesh.clone(),
        cell_centers,
        linear_system: system,
        operator_properties,
        scalar_reconstruction,
        facets,
        assembly_report,
    })
}

#[cfg(test)]
mod tests;
