use super::*;

impl<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send>
    ExecutableLinearEquations<S>
{
    pub(super) fn execute_simplicial(
        &self,
        workers: NonZeroUsize,
        request: LinearSolveRequest<'_, S>,
        envelope: &SimplicialMeshEnvelopeV1,
        space: Space,
        complete: impl FnOnce(
            &crate::region_assembly::InterfaceReactions<S>,
            &[S],
        ) -> Result<
            crate::region_assembly::RecoveredInterfaceReactions<S>,
            Diagnostic,
        >,
    ) -> Result<CommonLinearRunOutput<S>, Diagnostic> {
        let (mapping, forms, natural) = self.simplicial_assembly(envelope, space)?;
        let mesh = envelope.mesh();
        let output = mapping.solve(
            mesh,
            crate::region_assembly::mapping::RegionSolveInput {
                geometry_action: None,
                forms,
                natural,
                previous: None,
            },
            workers,
            request,
            complete,
        )?;
        Ok(CommonLinearRunOutput {
            fields: output
                .fields
                .into_iter()
                .map(|(field, recovered)| {
                    (
                        field.downcast().expect("Field"),
                        recovered.value_type,
                        recovered.coefficients.into_values().collect(),
                        recovered.space,
                    )
                })
                .collect(),
            solve_report: output.solve_report,
            assembly_report: output.assembly_report,
            nullspace: None,
        })
    }
    pub(in crate::numerical_admission) fn simplicial_assembly(
        &self,
        envelope: &SimplicialMeshEnvelopeV1,
        space: Space,
    ) -> Result<SimplicialAssembly<S>, Diagnostic> {
        self.simplicial_assembly_at(envelope, space, None)
    }

    pub(in crate::numerical_admission) fn simplicial_assembly_at(
        &self,
        envelope: &SimplicialMeshEnvelopeV1,
        space: Space,
        state: Option<&eqiora_meshing::FixedTopologyGeometryState<2>>,
    ) -> Result<SimplicialAssembly<S>, Diagnostic> {
        let current = state
            .map(|state| state.reconstruct_mesh(envelope.mesh()))
            .transpose()?;
        let mesh = current.as_ref().unwrap_or_else(|| envelope.mesh());
        let identity = envelope.digest()?;
        let dimension = mesh.topological_dimension();
        let nodal = space == Space::continuous_lagrange(std::num::NonZeroU16::MIN);
        if dimension != if nodal { 2 } else { 3 } || !self.interfaces.is_empty() {
            return Err(invalid(
                "simplicial execution requires planar nodal or spatial moment cells without trace quotients",
            ));
        }
        let mut membership = Vec::new();
        for region in &self.regions {
            let LinearRegionSupport::Simplicial(support) = &region.support else {
                return Err(invalid(
                    "moment execution requires authenticated polyhedral support",
                ));
            };
            if support.mesh != identity {
                return Err(invalid(
                    "moment execution Mesh differs from authenticated Model support",
                ));
            }
            membership.extend(
                support
                    .cells
                    .iter()
                    .map(|cell| (*cell, region.form.domain())),
            );
        }
        let (domains, traces) = bind_region_topology(mesh, membership, &[])?;
        let reference = eqiora_meshing::ReferenceCell::simplex(dimension)?;
        let quadrature = eqiora_meshing::simplex_duffy_gauss_legendre(dimension, 3)?;
        // Binding checks the differential/Space pairing and every boundary law.
        // Only authenticated homogeneous natural laws permit an empty facet load.
        let forms = self
            .regions
            .iter()
            .map(|region| {
                Ok((
                    region.form.bind_space(reference, space)?,
                    quadrature.clone(),
                ))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let layouts = self
            .regions
            .iter()
            .zip(&forms)
            .map(|(region, (form, _))| (region.form.domain(), form.fields().to_vec()))
            .collect();
        let mut prescribed = BTreeMap::new();
        let mut natural = Vec::new();
        if nodal {
            let facet_rule = eqiora_meshing::simplex_duffy_gauss_legendre(1, 3)?;
            for (region, (form, _)) in self.regions.iter().zip(&forms) {
                let LinearRegionSupport::Simplicial(support) = &region.support else {
                    unreachable!("validated simplicial support")
                };
                for (field, _) in region.form.fields() {
                    for (boundary, facets) in &support.facets {
                        let law = &region.form.boundary_laws()[field][boundary];
                        for &facet in facets {
                            let vertices = mesh.entity_vertices(facet).expect("facet vertices");
                            if law.quantity == PhysicalBoundaryQuantity::Trace {
                                for vertex in vertices {
                                    let value =
                                        law.evaluate(&mesh.vertices()[vertex.index()], &[])?[0];
                                    let key = FieldDof {
                                        field: *field,
                                        entity: vertex,
                                        slot: 0,
                                        component: 0,
                                    };
                                    let value = crate::cartesian_elliptic::support::require_compatible_boundary_value(
                                        prescribed.get(&key).copied(), value,
                                    )?.expect("finite candidate");
                                    prescribed.insert(key, value);
                                }
                            } else {
                                let parents =
                                    mesh.incidence(facet, dimension).expect("facet parents");
                                let [parent] = parents.as_slice() else {
                                    return Err(invalid(
                                        "simplicial boundary requires exactly one owned parent",
                                    ));
                                };
                                if domains[parent.entity.index()] != region.form.domain() {
                                    return Err(invalid(
                                        "simplicial boundary parent differs from its Region",
                                    ));
                                }
                                let geometry =
                                    mesh.geometry_map(parent.entity).expect("cell geometry");
                                let facet_geometry =
                                    mesh.geometry_map(facet).expect("facet geometry");
                                let cell_vertices =
                                    mesh.entity_vertices(parent.entity).expect("cell vertices");
                                let positions = vertices
                                    .iter()
                                    .map(|vertex| {
                                        cell_vertices
                                            .iter()
                                            .position(|candidate| candidate == vertex)
                                            .expect("incidence closure")
                                    })
                                    .collect::<Vec<_>>();
                                natural.push((
                                    parent.entity.index(),
                                    form.evaluate_natural_facet(
                                        *field,
                                        &geometry,
                                        (&facet_geometry, *parent, &positions),
                                        &facet_rule,
                                        |point, _| law.evaluate(point, &[]),
                                    )?,
                                ));
                            }
                        }
                    }
                }
            }
        }
        let mapping =
            RegionDofMap::<S>::new(mesh, &layouts, reference, &domains, &traces, &prescribed)?;
        Ok((mapping, forms, natural))
    }
}

type SimplicialAssembly<S> = (
    RegionDofMap<S>,
    Vec<(
        crate::form_compiler::region::BoundRegionForm<S>,
        QuadratureRule,
    )>,
    Vec<(usize, eqiora_assembly::LocalContribution<S>)>,
);
