use super::*;
use eqiora_realization::{DomainFieldDiscretization, FieldSpaceBinding};

impl<S: Coefficient> ExecutableSpatialEquations<S> {
    pub(in crate::numerical_admission) fn finite_element_operator_properties(
        &self,
    ) -> LinearOperatorProperties {
        // Congruent trace and essential elimination preserve symmetry. Keep
        // multiplier Fields outside trace quotients so their zero block survives;
        // assembled multiplier closure is checked before the solver is called.
        let multipliers = self
            .regions
            .iter()
            .flat_map(|region| {
                region
                    .form
                    .constraint_pairs()
                    .map(|(_, multiplier)| multiplier)
            })
            .collect::<BTreeSet<_>>();
        if self
            .regions
            .iter()
            .any(|region| region.form.has_zero_multiplier_block())
            && self
                .regions
                .iter()
                .all(|region| region.form.has_symmetric_operator())
            && self.regions.iter().all(|region| {
                region.form.constraint_pairs().next().is_none()
                    || region.form.has_zero_multiplier_block()
            })
            && self
                .interfaces
                .iter()
                .flat_map(|interface| interface.sides())
                .all(|side| {
                    !multipliers.contains(&side.field())
                        && self
                            .regions
                            .iter()
                            .find(|region| region.form.domain() == side.domain())
                            .is_some_and(|region| {
                                region.form.interface_test_orientation(
                                    side.field(),
                                    &eqiora_realization::SystemScaling::Dimensional,
                                ) == Some(1.0)
                            })
                })
        {
            LinearOperatorProperties::SymmetricIndefinite
        } else {
            LinearOperatorProperties::General
        }
    }

    pub(in crate::numerical_admission) fn finite_element_formulation(&self) -> FormulationKind {
        if self
            .regions
            .iter()
            .any(|region| region.form.constraint_pairs().next().is_some())
        {
            FormulationKind::MixedGalerkin
        } else {
            FormulationKind::PrimalGalerkin
        }
    }

    pub(in crate::numerical_admission) fn mixed_formulation(
        &self,
        requested: FormulationSelectionMode,
    ) -> Option<CommonFormulationDescription> {
        if self.finite_element_formulation() != FormulationKind::MixedGalerkin {
            return None;
        }
        Some(CommonFormulationDescription {
            requested,
            kind: FormulationKind::MixedGalerkin,
            boundary_treatment:
                crate::form_compiler::vocabulary::BoundaryTreatment::ExplicitTraceFluxLaws.id(),
            rule_ids: self
                .regions
                .iter()
                .flat_map(|region| region.form.formulation_rules())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            selection_reason_codes: Box::new([match requested {
                FormulationSelectionMode::Automatic => {
                    "eqiora.formulation.auto.equation-derived-mixed-galerkin/v1"
                }
                FormulationSelectionMode::Exact => {
                    "eqiora.formulation.exact.equation-derived-mixed-galerkin/v1"
                }
                FormulationSelectionMode::Authored => {
                    unreachable!("authored mixed overrides require their own source correspondence")
                }
            }]),
            requested_source_identity: None,
            source_relation: None,
            state_coordinates: Box::new([]),
        })
    }

    pub(in crate::numerical_admission) fn resolve_discretizations(
        &self,
        request: &CommonSpatialRequest,
        model: &ModelEnvelope,
    ) -> Result<Vec<DomainFieldDiscretization>, Diagnostic> {
        let policies = match request {
            CommonSpatialRequest::Uniform(policy) => self
                .regions
                .iter()
                .map(|region| (region.form.domain(), *policy))
                .collect::<BTreeMap<_, _>>(),
            CommonSpatialRequest::Scoped(bindings) => {
                let digest = model.digest()?;
                let mut policies = BTreeMap::new();
                for binding in bindings {
                    if binding.model() != &digest
                        || policies
                            .insert(binding.domain().erase(), binding.policy())
                            .is_some()
                    {
                        return Err(invalid(
                            "scoped spatial policies require unique Domains of this exact Model",
                        ));
                    }
                }
                if policies.keys().copied().collect::<BTreeSet<_>>()
                    != self
                        .regions
                        .iter()
                        .map(|region| region.form.domain())
                        .collect()
                {
                    return Err(invalid(
                        "scoped spatial policies must cover exactly the executable Regions",
                    ));
                }
                policies
            }
        };
        self.regions
            .iter()
            .map(|region| {
                let policy = policies[&region.form.domain()];
                let pairs = region.form.constraint_pairs().collect::<Vec<_>>();
                if (policy == CommonSpatialPolicy::MiniP1) != !pairs.is_empty() {
                    return Err(invalid(
                        "divergence-constrained Regions require the paired MINI/P1 policy",
                    ));
                }
                let base = match policy {
                    CommonSpatialPolicy::Q1
                    | CommonSpatialPolicy::P1
                    | CommonSpatialPolicy::MiniP1 => {
                        Space::continuous_lagrange(std::num::NonZeroU16::MIN)
                    }
                    CommonSpatialPolicy::TetrahedralEdge => Space::tetrahedral_edge(),
                    CommonSpatialPolicy::TetrahedralFace => Space::tetrahedral_face(),
                    CommonSpatialPolicy::CellCentered | CommonSpatialPolicy::CellCenteredTpfa => {
                        return Err(invalid(
                            "finite-element Region cannot use a cell-centered policy",
                        ));
                    }
                };
                let mut fields = region
                    .form
                    .represented_fields()
                    .iter()
                    .map(|(field, _)| (*field, base))
                    .collect::<BTreeMap<_, _>>();
                for (constrained, _) in pairs {
                    *fields.get_mut(&constrained).ok_or_else(|| {
                        invalid("constraint trial is absent from represented Fields")
                    })? = Space::simplex_p1_bubble();
                }
                for (pair, _) in region.form.kinematics() {
                    let rate_space = fields[&pair.rate().erase()];
                    *fields.get_mut(&pair.state().erase()).ok_or_else(|| {
                        invalid("kinematic state is absent from represented Fields")
                    })? = rate_space;
                }
                DomainFieldDiscretization::new(
                    region.domain_id(),
                    fields.into_iter().map(|(field, space)| {
                        FieldSpaceBinding::new(field.downcast().expect("Field"), space)
                    }),
                    None,
                )
            })
            .collect()
    }
}

impl<S: Coefficient> ExecutableSpatialEquations<S> {
    pub(in crate::numerical_admission) fn validate_discretizations(
        &self,
        selected: &[DomainFieldDiscretization],
        resources: &NativeMeshResources,
        temporal: Option<CommonBackwardEuler>,
        nonlinear: Option<NonlinearSolvePlan>,
        scaling: &eqiora_realization::SystemScaling,
    ) -> Result<(), Diagnostic> {
        let moment = selected
            .iter()
            .flat_map(|domain| domain.field_spaces())
            .any(|field| {
                matches!(
                    field.space().family(),
                    SpaceFamily::TetrahedralEdge | SpaceFamily::TetrahedralFace
                )
            });
        if moment && (temporal.is_some() || nonlinear.is_some() || !self.interfaces.is_empty()) {
            return Err(invalid(
                "moment admission requires static linear equations without full-value trace quotients",
            ));
        }
        let dimension = resources.geometry()?.ambient_dimension();
        let reference = match resources {
            NativeMeshResources::Cartesian { .. } => {
                eqiora_meshing::ReferenceCell::hypercube(dimension)?
            }
            NativeMeshResources::GmshSimplicial { .. }
            | NativeMeshResources::AffineTriangleSimplicial { .. } => {
                eqiora_meshing::ReferenceCell::simplex(dimension)?
            }
            _ => {
                return Err(invalid(
                    "spatial Field spaces require an authenticated Cartesian or simplicial Mesh",
                ));
            }
        };
        if moment
            && (dimension != 3
                || self
                    .regions
                    .iter()
                    .any(|region| !matches!(region.support, SpatialRegionSupport::Simplicial(_))))
        {
            return Err(invalid(
                "moment admission requires authenticated polyhedral support",
            ));
        }
        let mut bound = self.clone();
        if let Some(temporal) = temporal {
            for region in &mut bound.regions {
                region.form = region.form.bind_backward_euler(temporal.step())?;
            }
        }
        // Quotient rows sum the two outward flux equations. Their actual row
        // orientations must agree, even when the selected solver is general.
        for interface in &bound.interfaces {
            let signs = interface.sides().each_ref().map(|side| {
                bound
                    .regions
                    .iter()
                    .find(|region| region.form.domain() == side.domain())
                    .and_then(|region| {
                        region
                            .form
                            .interface_test_orientation(side.field(), scaling)
                    })
            });
            if signs[0].is_none() || signs[0] != signs[1] {
                return Err(invalid(
                    "interface quotient requires matching applied test orientations",
                ));
            }
        }
        bound.bind_spaces(reference, selected, scaling)?;
        Ok(())
    }
}
