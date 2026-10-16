use super::*;

impl<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send>
    ExecutableSpatialEquations<S>
{
    pub(in crate::numerical_admission) fn simplicial_assembly(
        &self,
        envelope: &SimplicialMeshEnvelopeV1,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
    ) -> Result<SimplicialAssembly<S>, Diagnostic> {
        self.simplicial_assembly_at(envelope, selected, scaling, None)
    }

    pub(in crate::numerical_admission) fn simplicial_assembly_at(
        &self,
        envelope: &SimplicialMeshEnvelopeV1,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
        state: Option<&eqiora_meshing::FixedTopologyGeometryState<2>>,
    ) -> Result<SimplicialAssembly<S>, Diagnostic> {
        self.simplicial_assembly_with_boundary(envelope, selected, scaling, state, |_, law, value| {
            if law.trace_field.is_some_and(|field| field != law.tested) {
                return Err(invalid("eliminated-state boundary data requires an explicit history-dependent rate constraint"));
            }
            Ok(Some(value))
        })
    }

    pub(in crate::numerical_admission) fn simplicial_assembly_with_boundary(
        &self,
        envelope: &SimplicialMeshEnvelopeV1,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
        state: Option<&eqiora_meshing::FixedTopologyGeometryState<2>>,
        mut boundary_value: impl FnMut(
            FieldDof,
            &crate::form_compiler::region::RegionBoundaryLaw<S>,
            S,
        ) -> Result<Option<S>, Diagnostic>,
    ) -> Result<SimplicialAssembly<S>, Diagnostic> {
        let current = state
            .map(|state| state.reconstruct_mesh(envelope.mesh()))
            .transpose()?;
        let mesh = current.as_ref().unwrap_or_else(|| envelope.mesh());
        let identity = envelope.digest()?;
        let dimension = mesh.topological_dimension();
        let nodal = selected
            .iter()
            .flat_map(|domain| domain.field_spaces())
            .all(|field| {
                field.space() == Space::continuous_lagrange(std::num::NonZeroU16::MIN)
                    || field.space() == Space::simplex_p1_bubble()
            });
        if !matches!(dimension, 2 | 3) || (!nodal && dimension != 3) {
            return Err(invalid(
                "simplicial execution requires planar or spatial nodal cells, or spatial moment cells",
            ));
        }
        let mut membership = Vec::new();
        for region in &self.regions {
            let SpatialRegionSupport::Simplicial(support) = &region.support else {
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
        let (domains, traces) = bind_region_topology(mesh, membership, &self.quotients()?)?;
        for (trace, interface) in traces.iter().zip(&self.interfaces) {
            let actual = trace
                .facets
                .iter()
                .map(|witness| witness.facet)
                .collect::<BTreeSet<_>>();
            for side in interface.sides() {
                let region = self
                    .regions
                    .iter()
                    .find(|region| region.form.domain() == side.domain())
                    .ok_or_else(|| invalid("trace has no exact Region"))?;
                let SpatialRegionSupport::Simplicial(support) = &region.support else {
                    unreachable!("validated simplicial support")
                };
                let selected = support
                    .facets
                    .get(&side.boundary())
                    .ok_or_else(|| invalid("trace has no exact Boundary selection"))?;
                if actual != selected.iter().copied().collect() {
                    return Err(invalid(
                        "trace quotient differs from its exact selected Boundary facets",
                    ));
                }
            }
        }
        let reference = eqiora_meshing::ReferenceCell::simplex(dimension)?;
        let bound = self.bind_spaces(reference, selected, scaling)?;
        let points = super::super::quadrature_points(reference, selected, &bound)?;
        let quadrature = eqiora_meshing::simplex_duffy_gauss_legendre(dimension, points)?;
        // Binding checks the differential/Space pairing and every boundary law.
        // Only authenticated homogeneous natural laws permit an empty facet load.
        let forms = bound
            .into_iter()
            .map(|form| (form, quadrature.clone()))
            .collect::<Vec<_>>();
        let layouts = self
            .regions
            .iter()
            .zip(&forms)
            .map(|(region, (form, _))| (region.form.domain(), form.fields().to_vec()))
            .collect();
        let mut prescribed = BTreeMap::new();
        let mut natural = Vec::new();
        if nodal {
            for (region, (form, _)) in self.regions.iter().zip(&forms) {
                let SpatialRegionSupport::Simplicial(support) = &region.support else {
                    unreachable!("validated simplicial support")
                };
                for (field, _) in region.form.fields() {
                    for (boundary, facets) in &support.facets {
                        let Some(law) = region.form.boundary_laws()[field].get(boundary) else {
                            continue;
                        };
                        for &facet in facets {
                            let vertices = mesh.entity_vertices(facet).expect("facet vertices");
                            if law.quantity == PhysicalBoundaryQuantity::Trace {
                                for vertex in vertices {
                                    for (component, value) in law
                                        .evaluate(&mesh.vertices()[vertex.index()], &[])?
                                        .into_iter()
                                        .enumerate()
                                    {
                                        let key = FieldDof {
                                            field: *field,
                                            entity: vertex,
                                            slot: 0,
                                            component,
                                        };
                                        let Some(value) = boundary_value(key, law, value)? else {
                                            continue;
                                        };
                                        let value = crate::cartesian_elliptic::support::require_compatible_boundary_value(
                                            prescribed.get(&key).copied(), value,
                                        )?.expect("finite candidate");
                                        prescribed.insert(key, value);
                                    }
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
                                natural.push((
                                    parent.entity.index(),
                                    crate::region_assembly::mapping::prepare_mesh_facet(
                                        mesh,
                                        form,
                                        *field,
                                        facet,
                                        *parent,
                                        |point, normal| law.evaluate(point, normal),
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
        natural.extend(crate::region_assembly::mapping::prepare_interface_facets(
            mesh,
            forms.iter().map(|(form, _)| form),
            &traces,
        )?);
        Ok((mapping, forms, natural))
    }
}

type SimplicialAssembly<S> = (
    RegionDofMap<S>,
    Vec<(
        crate::form_compiler::region::BoundRegionForm<S>,
        QuadratureRule,
    )>,
    Vec<(usize, crate::form_compiler::region::PreparedRegionFacet<S>)>,
);
