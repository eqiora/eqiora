use super::*;
use crate::spatial_expression::Coefficient;
use eqiora_core::RawId;

use super::super::native::polyhedral::SimplicialRegionSupport;

#[derive(Debug, Clone, PartialEq)]
enum SpatialRegionSupport {
    Cartesian(ScalarRegionSupport),
    Simplicial(SimplicialRegionSupport),
}

/// Checked spatial equations and their exact physical support.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::numerical_admission) struct SpatialRegion<S: Coefficient> {
    pub(in crate::numerical_admission) form: crate::form_compiler::spatial::CompiledSpatialForm<S>,
    support: SpatialRegionSupport,
}

impl<S: Coefficient> SpatialRegion<S> {
    pub(in crate::numerical_admission) fn new(
        program: &KernelProgram,
        domain: eqiora_core::RawId,
        bounds: Vec<[f64; 2]>,
        boundaries: BTreeMap<(usize, BoundarySide), eqiora_core::RawId>,
    ) -> Result<Self, Diagnostic> {
        let form = crate::form_compiler::spatial::CompiledSpatialForm::<S>::derive(
            program,
            domain,
            bounds.len(),
            &std::collections::BTreeSet::new(),
        )?;
        let expected = boundaries.values().copied().collect::<BTreeSet<_>>();
        if form
            .boundary_laws()
            .iter()
            .filter(|(field, _)| form.requires_boundary(**field))
            .any(|(_, laws)| laws.keys().copied().collect::<BTreeSet<_>>() != expected)
        {
            return Err(invalid(
                "compiled boundary laws differ from authenticated Geometry support",
            ));
        }
        Ok(Self {
            form,
            support: SpatialRegionSupport::Cartesian(ScalarRegionSupport::new(
                domain, bounds, boundaries,
            )),
        })
    }

    pub(in crate::numerical_admission) fn cartesian(
        &self,
    ) -> Result<&ScalarRegionSupport, Diagnostic> {
        match &self.support {
            SpatialRegionSupport::Cartesian(support) => Ok(support),
            SpatialRegionSupport::Simplicial(_) => {
                Err(invalid("this operation requires Cartesian Region support"))
            }
        }
    }

    pub(in crate::numerical_admission) fn simplicial(
        &self,
    ) -> Result<&SimplicialRegionSupport, Diagnostic> {
        match &self.support {
            SpatialRegionSupport::Simplicial(support) => Ok(support),
            SpatialRegionSupport::Cartesian(_) => {
                Err(invalid("this operation requires simplicial Region support"))
            }
        }
    }

    pub(in crate::numerical_admission) fn domain_id(
        &self,
    ) -> eqiora_core::Id<eqiora_core::entity::kinds::Domain> {
        self.form
            .domain()
            .downcast()
            .expect("compiled Domain identity")
    }

    /// Independently admit the bounded conservation subset for TPFA or differentiation.
    pub(in crate::numerical_admission) fn conservation_descriptor(
        &self,
        program: &KernelProgram,
    ) -> Result<ScalarConservationDescriptor, Diagnostic> {
        let descriptor =
            recognize_scalar_conservation_on_supports(program, vec![self.cartesian()?.clone()])?;
        let regions = descriptor.regions().collect::<Vec<_>>();
        let [region] = regions.as_slice() else {
            return Err(invalid("steady scalar conservation requires one region"));
        };
        if region.storage().is_some() || descriptor.interfaces().len() != 0 {
            return Err(invalid(
                "steady scalar conservation does not admit storage or interfaces",
            ));
        }
        if region
            .exterior()
            .any(|boundary| matches!(boundary.law(), ScalarExteriorLaw::Robin { .. }))
        {
            return Err(invalid(
                "steady scalar conservation does not admit Robin boundaries",
            ));
        }
        Ok(descriptor)
    }

    /// Existing authored single-equation Formulation metadata, never an execution fallback.
    pub(in crate::numerical_admission) fn primal_form(
        &self,
        program: &KernelProgram,
    ) -> Result<Option<crate::form_compiler::DerivedScalarGalerkinForm>, Diagnostic> {
        if self.form.fields().len() != 1
            || self
                .form
                .fields()
                .iter()
                .any(|(_, ty)| !ty.shape().is_scalar())
        {
            return Ok(None);
        }
        Ok(crate::form_compiler::derive_candidate_with_dimension(
            program,
            self.form.domain(),
            self.form.dimension(),
        )
        .ok()
        .flatten())
    }
}

/// One ordered mathematical inventory; Region count never selects an executor.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExecutableSpatialEquations<S: Coefficient> {
    pub(in crate::numerical_admission) regions: Vec<SpatialRegion<S>>,
    pub(in crate::numerical_admission) interfaces:
        Vec<crate::scalar_conservation::ScalarMaterialInterface>,
}

impl<S: Coefficient> ExecutableSpatialEquations<S> {
    pub(in crate::numerical_admission) fn new(
        program: &KernelProgram,
        domain: RawId,
        bounds: Vec<[f64; 2]>,
        boundaries: BTreeMap<(usize, BoundarySide), RawId>,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            regions: vec![SpatialRegion::new(program, domain, bounds, boundaries)?],
            interfaces: vec![],
        })
    }
    pub(in crate::numerical_admission) fn simplicial(
        program: &KernelProgram,
        resources: &NativeMeshResources,
    ) -> Result<Self, Diagnostic> {
        let supports = super::super::native::polyhedral::bind_model_support(program, resources)?;
        let dimension = resources.geometry()?.ambient_dimension();
        let candidates =
            crate::scalar_conservation::compiled_interfaces::CompiledInterfaceCandidates::discover(
                program,
                &supports
                    .iter()
                    .map(|(&domain, support)| (domain, support.boundaries.clone()))
                    .collect(),
            )?;
        let interface_boundaries = candidates.boundaries()?;
        let mut regions = Vec::new();
        for (&domain, support) in &supports {
            let motion =
                crate::form_compiler::spatial::motion::StorageMotion::select(program, domain)?;
            let form = crate::form_compiler::spatial::CompiledSpatialForm::<S>::derive_at_time(
                program,
                domain,
                dimension,
                &interface_boundaries,
                motion.map(|_| 0.0),
            )?;
            if form.is_transient() && S::DOMAIN != eqiora_core::ScalarDomain::Real {
                return Err(invalid("simplicial storage requires real Fields"));
            }
            for (_, ty) in form.fields() {
                let components = crate::form_compiler::region::components(ty, dimension)?;
                if (components != 1 && components != dimension) || ty.scalar_domain() != S::DOMAIN {
                    return Err(invalid(
                        "simplicial spatial equations require scalars or spatial vectors matching the support dimension and coefficient domain",
                    ));
                }
            }
            if form
                .boundary_laws()
                .iter()
                .filter(|(field, _)| form.requires_boundary(**field))
                .any(|(field, laws)| {
                    let mut covered = laws.keys().copied().collect::<BTreeSet<_>>();
                    covered.extend(
                        interface_boundaries
                            .iter()
                            .filter(|endpoint| endpoint.field == *field)
                            .map(|endpoint| endpoint.boundary),
                    );
                    covered != support.boundaries
                })
            {
                return Err(invalid(
                    "compiled boundary laws differ from authenticated polyhedral supports",
                ));
            }
            regions.push(SpatialRegion {
                form,
                support: SpatialRegionSupport::Simplicial(support.clone()),
            });
        }
        let interfaces = candidates.finish(
            program,
            |domain, boundary, relation, field, normal| {
                let region = regions
                    .iter()
                    .find(|region| region.form.domain() == domain)
                    .ok_or_else(|| invalid("interface has no exact compiled Region"))?;
                region
                    .form
                    .require_interface_flux(program, boundary, relation, field, normal)
            },
            |_, sides| {
                super::super::native::polyhedral::validate_interface_support(
                    &supports, resources, sides,
                )
            },
        )?;
        Ok(Self {
            regions,
            interfaces,
        })
    }

    pub(in crate::numerical_admission) fn single(&self) -> Result<&SpatialRegion<S>, Diagnostic> {
        let [region] = self.regions.as_slice() else {
            return Err(invalid(
                "this numerical operation requires one exact Region",
            ));
        };
        Ok(region)
    }
    pub(in crate::numerical_admission) fn fields(&self) -> Vec<(RawId, eqiora_core::ValueType)> {
        let mut fields = self
            .regions
            .iter()
            .flat_map(|region| region.form.fields().iter().cloned())
            .collect::<Vec<_>>();
        fields.sort_by_key(|(field, _)| *field);
        fields
    }
    pub(in crate::numerical_admission) fn represented_fields(
        &self,
    ) -> Vec<(RawId, eqiora_core::ValueType)> {
        let mut fields = self
            .regions
            .iter()
            .flat_map(|region| region.form.represented_fields())
            .collect::<Vec<_>>();
        fields.sort_by_key(|(field, _)| *field);
        fields
    }
    /// Semantic Field blocks supplied to the sole solver authority before selection.
    pub(in crate::numerical_admission) fn algebraic_structure(
        &self,
        constraint: Option<eqiora_solver::AlgebraicConstraint>,
    ) -> Result<eqiora_solver::AlgebraicStructure, Diagnostic> {
        eqiora_solver::AlgebraicStructure::new(
            self.fields().into_iter().map(|(field, _)| {
                field
                    .downcast()
                    .expect("compiled scalar unknown is a Field")
            }),
            constraint,
        )
    }
    pub(in crate::numerical_admission) fn primal_form(
        &self,
        program: &KernelProgram,
    ) -> Result<Option<crate::form_compiler::DerivedScalarGalerkinForm>, Diagnostic> {
        if self.regions.len() != 1 {
            return Ok(None);
        }
        self.single()?.primal_form(program)
    }
    pub(in crate::numerical_admission) fn conservation_descriptor(
        &self,
        program: &KernelProgram,
    ) -> Result<ScalarConservationDescriptor, Diagnostic> {
        self.single()?.conservation_descriptor(program)
    }
    pub(in crate::numerical_admission) fn source_regions(
        program: &KernelProgram,
        mesh: &eqiora_meshing::CartesianMesh,
    ) -> Result<Self, Diagnostic> {
        let supports = crate::scalar_conservation::cartesian_region_supports(program)?;
        if supports.is_empty() {
            return Err(invalid(
                "scalar equations require at least one Cartesian volume Domain",
            ));
        }
        let candidates =
            crate::scalar_conservation::compiled_interfaces::CompiledInterfaceCandidates::discover(
                program,
                &supports
                    .iter()
                    .map(|support| {
                        (
                            support.domain,
                            support.boundaries.values().copied().collect(),
                        )
                    })
                    .collect(),
            )?;
        let interface_boundaries = candidates.boundaries()?;
        let mut regions = Vec::new();
        for support in &supports {
            if support.bounds.len() != mesh.topological_dimension() {
                return Err(invalid(
                    "steady Region execution requires matching spatial dimension",
                ));
            }
            let form = crate::form_compiler::spatial::CompiledSpatialForm::<S>::derive(
                program,
                support.domain,
                support.bounds.len(),
                &interface_boundaries,
            )?;
            // Storage is bound after the caller selects an explicit temporal Plan.
            if !form.is_transient() {
                form.volume()?;
            }
            regions.push(SpatialRegion {
                form,
                support: SpatialRegionSupport::Cartesian(support.clone()),
            });
        }
        let interfaces = candidates.finish(
            program,
            |domain, boundary, relation, field, normal| {
                let region = regions
                    .iter()
                    .find(|region| region.form.domain() == domain)
                    .ok_or_else(|| invalid("interface has no exact compiled Region"))?;
                region
                    .form
                    .require_interface_flux(program, boundary, relation, field, normal)
            },
            |source, sides| {
                crate::scalar_conservation::interface::validate_cartesian_pair(
                    program, &supports, source, sides,
                )
            },
        )?;
        regions.sort_by_key(|region| region.form.domain());
        let result = Self {
            regions,
            interfaces,
        };
        result.cell_domains(mesh)?;
        Ok(result)
    }
    /// Exact whole-cell inclusion; gaps, overlaps and cut cells are rejected.
    pub(in crate::numerical_admission) fn cell_domains(
        &self,
        mesh: &eqiora_meshing::CartesianMesh,
    ) -> Result<Vec<RawId>, Diagnostic> {
        let dimension = mesh.topological_dimension();
        for region in &self.regions {
            let region = region.cartesian()?;
            if region.bounds.len() != dimension
                || region.bounds.iter().enumerate().any(|(axis, bounds)| {
                    let coordinates = mesh.axis_coordinates(axis).expect("Cartesian axis");
                    !coordinates.contains(&bounds[0]) || !coordinates.contains(&bounds[1])
                })
            {
                return Err(invalid(
                    "each exact Region bound must coincide with an admitted mesh coordinate",
                ));
            }
        }
        let mut used = BTreeSet::new();
        let mut domains = Vec::new();
        for cell in 0..mesh
            .entity_count(dimension)
            .ok_or_else(|| invalid("missing mesh cells"))?
        {
            let entity = eqiora_meshing::MeshEntity::new(dimension, cell);
            let vertices = mesh
                .incidence(entity, 0)
                .ok_or_else(|| invalid("cell has no exact vertex closure"))?;
            let owners = self
                .regions
                .iter()
                .filter(|region| {
                    let region = region
                        .cartesian()
                        .expect("Cartesian supports checked above");
                    region.bounds.len() == dimension
                        && vertices.iter().all(|vertex| {
                            let point = mesh
                                .vertex_coordinates(vertex.entity)
                                .expect("Cartesian vertex");
                            region.bounds.iter().enumerate().all(|(axis, bounds)| {
                                point[axis] >= bounds[0] && point[axis] <= bounds[1]
                            })
                        })
                })
                .collect::<Vec<_>>();
            let [owner] = owners.as_slice() else {
                return Err(invalid(
                    "each mesh cell requires exactly one complete Region owner; gaps, overlaps and cut cells are unsupported",
                ));
            };
            used.insert(owner.form.domain());
            domains.push(owner.form.domain());
        }
        if used.len() != self.regions.len() {
            return Err(invalid(
                "every admitted Region requires nonempty mesh cell coverage",
            ));
        }
        Ok(domains)
    }
}

impl<S: Coefficient> ExecutableSpatialEquations<S> {
    pub(in crate::numerical_admission) fn bind_spaces(
        &self,
        reference: eqiora_meshing::ReferenceCell,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
    ) -> Result<Vec<crate::form_compiler::region::BoundRegionForm<S>>, Diagnostic> {
        let domains = selected
            .iter()
            .map(|binding| binding.domain().erase())
            .collect::<BTreeSet<_>>();
        if domains.len() != selected.len()
            || domains
                != self
                    .regions
                    .iter()
                    .map(|region| region.form.domain())
                    .collect()
        {
            return Err(invalid(
                "selected discretizations must cover exactly the executable Regions",
            ));
        }
        self.regions
            .iter()
            .map(|region| {
                let binding = selected
                    .iter()
                    .find(|binding| binding.domain() == region.domain_id())
                    .expect("checked exact Domain inventory");
                region
                    .form
                    .bind_field_spaces(reference, binding.field_spaces(), scaling)
            })
            .collect()
    }

    pub(in crate::numerical_admission) fn discretizations(
        &self,
        space: Space,
        constraint: Option<eqiora_solver::AlgebraicConstraint>,
    ) -> Result<Vec<eqiora_realization::DomainFieldDiscretization>, Diagnostic> {
        self.regions
            .iter()
            .map(|region| {
                eqiora_realization::DomainFieldDiscretization::new(
                    region.domain_id(),
                    region.form.represented_fields().iter().map(|(field, _)| {
                        eqiora_realization::FieldSpaceBinding::new(
                            field.downcast().expect("Field"),
                            space,
                        )
                    }),
                    constraint,
                )
            })
            .collect()
    }
    pub(in crate::numerical_admission) fn quotients(
        &self,
    ) -> Result<Vec<eqiora_realization::ConformingTraceQuotient>, Diagnostic> {
        self.interfaces
            .iter()
            .map(|interface| {
                let endpoints = interface
                    .sides()
                    .iter()
                    .map(|side| {
                        let region = self
                            .regions
                            .iter()
                            .find(|region| region.form.domain() == side.domain())
                            .ok_or_else(|| invalid("Connection has no exact Region"))?;
                        if !region
                            .form
                            .fields()
                            .iter()
                            .any(|(field, _)| *field == side.field())
                        {
                            return Err(invalid(
                                "Connection endpoint Field is outside its exact Region",
                            ));
                        }
                        Ok(eqiora_realization::TraceFieldEndpoint::new(
                            region.domain_id(),
                            side.field().downcast().expect("Field"),
                        ))
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()?;
                eqiora_realization::ConformingTraceQuotient::new(
                    interface.source(),
                    endpoints[0],
                    endpoints[1],
                )
            })
            .collect()
    }
}
mod discretizations;
mod execute;

#[cfg(test)]
mod complex_tests;

mod quadrature;
pub(super) use quadrature::quadrature_points;
