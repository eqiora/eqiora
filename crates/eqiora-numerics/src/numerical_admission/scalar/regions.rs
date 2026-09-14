use super::*;
use eqiora_core::RawId;

/// Checked scalar equations and their exact Cartesian support.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::numerical_admission) struct ScalarRegion {
    pub(in crate::numerical_admission) form: crate::form_compiler::linear::CompiledLinearBlockForm,
    pub(in crate::numerical_admission) bounds: Vec<[f64; 2]>,
    pub(in crate::numerical_admission) boundaries:
        BTreeMap<(usize, BoundarySide), eqiora_core::RawId>,
}

impl ScalarRegion {
    pub(in crate::numerical_admission) fn new(
        program: &KernelProgram,
        domain: eqiora_core::RawId,
        bounds: Vec<[f64; 2]>,
        boundaries: BTreeMap<(usize, BoundarySide), eqiora_core::RawId>,
    ) -> Result<Self, Diagnostic> {
        let form = crate::form_compiler::linear::CompiledLinearBlockForm::derive(
            program,
            domain,
            bounds.len(),
            &std::collections::BTreeSet::new(),
        )?;
        let expected = boundaries.values().copied().collect::<BTreeSet<_>>();
        if form
            .boundary_laws()
            .values()
            .any(|laws| laws.keys().copied().collect::<BTreeSet<_>>() != expected)
        {
            return Err(invalid(
                "compiled boundary laws differ from authenticated Geometry support",
            ));
        }
        Ok(Self {
            form,
            bounds,
            boundaries,
        })
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
        let descriptor = recognize_scalar_conservation_on_supports(
            program,
            vec![ScalarRegionSupport::new(
                self.form.domain(),
                self.bounds.clone(),
                self.boundaries.clone(),
            )],
        )?;
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
        if self.form.fields().len() != 1 {
            return Ok(None);
        }
        Ok(crate::form_compiler::derive_candidate_with_dimension(
            program,
            self.form.domain(),
            self.bounds.len(),
        )
        .ok()
        .flatten())
    }
}

/// One ordered mathematical inventory; Region count never selects an executor.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExecutableScalarEquations {
    pub(in crate::numerical_admission) regions: Vec<ScalarRegion>,
    pub(in crate::numerical_admission) interfaces:
        Vec<crate::scalar_conservation::ScalarMaterialInterface>,
}

impl ExecutableScalarEquations {
    pub(in crate::numerical_admission) fn new(
        program: &KernelProgram,
        domain: RawId,
        bounds: Vec<[f64; 2]>,
        boundaries: BTreeMap<(usize, BoundarySide), RawId>,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            regions: vec![ScalarRegion::new(program, domain, bounds, boundaries)?],
            interfaces: vec![],
        })
    }
    pub(in crate::numerical_admission) fn single(&self) -> Result<&ScalarRegion, Diagnostic> {
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
    /// Semantic Field blocks supplied to the sole solver authority before selection.
    pub(in crate::numerical_admission) fn algebraic_structure(
        &self,
    ) -> Result<eqiora_solver::AlgebraicStructure, Diagnostic> {
        eqiora_solver::AlgebraicStructure::new(
            self.fields().into_iter().map(|(field, _)| {
                field
                    .downcast()
                    .expect("compiled scalar unknown is a Field")
            }),
            [],
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
        let descriptor = crate::scalar_conservation::recognize_scalar_conservation(program)?;
        let interfaces = descriptor.interfaces().cloned().collect::<Vec<_>>();
        let interface_boundaries = interfaces
            .iter()
            .flat_map(|interface| interface.sides().iter().map(|side| side.boundary()))
            .collect::<BTreeSet<_>>();
        let mut regions = Vec::new();
        for region in descriptor.regions() {
            if region.storage().is_some() || region.dimensions() != mesh.topological_dimension() {
                return Err(invalid(
                    "steady Region execution requires fixed matching spatial dimension without storage",
                ));
            }
            let mut boundaries = region
                .exterior()
                .map(|boundary| ((boundary.axis(), boundary.side()), boundary.boundary()))
                .collect::<BTreeMap<_, _>>();
            for side in interfaces.iter().flat_map(|interface| interface.sides()) {
                if side.domain() == region.domain() {
                    boundaries.insert((side.axis(), side.side()), side.boundary());
                }
            }
            let form = crate::form_compiler::linear::CompiledLinearBlockForm::derive(
                program,
                region.domain(),
                region.dimensions(),
                &interface_boundaries,
            )?;
            regions.push(ScalarRegion {
                form,
                bounds: region.bounds().to_vec(),
                boundaries,
            });
        }
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

impl ExecutableScalarEquations {
    pub(in crate::numerical_admission) fn discretizations(
        &self,
        space: Space,
    ) -> Result<Vec<eqiora_realization::DomainFieldDiscretization>, Diagnostic> {
        self.regions
            .iter()
            .map(|region| {
                eqiora_realization::DomainFieldDiscretization::new(
                    region.domain_id(),
                    region.form.fields().iter().map(|(field, _)| {
                        eqiora_realization::FieldSpaceBinding::new(
                            field.downcast().expect("Field"),
                            space,
                        )
                    }),
                    [],
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
                        let [(field, _)] = region.form.fields() else {
                            return Err(invalid(
                                "scalar Connection requires an exact single Field endpoint",
                            ));
                        };
                        Ok(eqiora_realization::TraceFieldEndpoint::new(
                            region.domain_id(),
                            field.downcast().expect("Field"),
                        ))
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()?;
                eqiora_realization::ConformingTraceQuotient::new(
                    interface.connection().downcast().expect("Connection"),
                    endpoints[0],
                    endpoints[1],
                )
            })
            .collect()
    }
}
mod execute;
