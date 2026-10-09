use super::*;

impl<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send>
    ExecutableLinearEquations<S>
{
    pub(super) fn execute_moments(
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
        let (mapping, forms) = self.moment_assembly(envelope, space)?;
        let mesh = envelope.mesh();
        let output = mapping.solve(mesh, forms, vec![], workers, request, complete)?;
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
    pub(in crate::numerical_admission) fn moment_assembly(
        &self,
        envelope: &SimplicialMeshEnvelopeV1,
        space: Space,
    ) -> Result<MomentAssembly<S>, Diagnostic> {
        let mesh = envelope.mesh();
        let identity = envelope.digest()?;
        if mesh.topological_dimension() != 3 || !self.interfaces.is_empty() {
            return Err(invalid(
                "moment execution requires three-dimensional cells without trace quotients",
            ));
        }
        let mut membership = Vec::new();
        for region in &self.regions {
            let LinearRegionSupport::Polyhedral(support) = &region.support else {
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
        let reference = eqiora_meshing::ReferenceCell::simplex(3)?;
        let quadrature = eqiora_meshing::simplex_duffy_gauss_legendre(3, 3)?;
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
        let mapping = RegionDofMap::<S>::new(
            mesh,
            &layouts,
            reference,
            &domains,
            &traces,
            &BTreeMap::new(),
        )?;
        Ok((mapping, forms))
    }
}

type MomentAssembly<S> = (
    RegionDofMap<S>,
    Vec<(
        crate::form_compiler::region::BoundRegionForm<S>,
        QuadratureRule,
    )>,
);
