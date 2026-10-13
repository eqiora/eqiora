//! Lossless projections from accepted realization families into the portable graph.

use std::num::NonZeroUsize;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};

use super::*;
use crate::{
    AlgebraicBlock, ResolvedCoupledFieldwiseRealization, ResolvedFieldwiseRealization,
    ResolvedFixedTopologyAleCoupledRealization, ResolvedRealization,
    ResolvedTransientCellCenteredIncompressibleFlowRealization,
    ResolvedTransientCellCenteredTransportRealization, ResolvedTransientFieldwiseRealization,
    Target, invalid_realization,
};

impl ResolvedRealization {
    /// Normalize an accepted compatibility plan for one exact semantic Field.
    ///
    /// The claim is deliberately supplied by an equation-aware lowerer because
    /// the old plan contains neither Semantic identities nor operator facts.
    /// This projection validates its structure and seals its operator property
    /// against the exact candidate set retained by compatibility resolution.
    /// The execution finalizer still owns comparison with the accepted
    /// equation identity and coefficients.
    ///
    /// # Errors
    /// Returns `EQ0807` if the resolved value cannot form one connected,
    /// portable linear-solve DAG.
    pub fn portable_graph(
        &self,
        domain: Id<kinds::Domain>,
        field: Id<kinds::Field>,
        operator_properties: LinearOperatorProperties,
    ) -> Result<PortableRealizationGraph, Diagnostic> {
        self.require_admitted_operator_properties(operator_properties)?;
        let plan = self.plan();
        let requirements = self.requirements();
        PortableRealizationGraph::spatial_regions(
            RealizationLineage::new(self.model(), self.semantic_revision(), self.source()),
            [crate::DomainFieldDiscretization::new(
                domain,
                [crate::FieldSpaceBinding::new(field, plan.space())],
                [],
            )?],
            [],
            None,
            plan.discretization(),
            operator_properties,
            requirements.scalar_type(),
            requirements.vector_layout(),
            plan.solver(),
            None,
            plan.target(),
            plan.schedule(),
            SystemScaling::Dimensional,
        )
    }
}

impl ResolvedFieldwiseRealization {
    /// Normalize an accepted single-Domain field-wise plan into the portable DAG.
    ///
    /// # Errors
    /// Returns `EQ0807` if the resolved compatibility value cannot be
    /// represented losslessly by the connected Phase-A graph.
    pub fn portable_graph(&self) -> Result<PortableRealizationGraph, Diagnostic> {
        let plan = self.plan();
        let spatial = plan.spatial();
        let fields = spatial
            .field_spaces()
            .iter()
            .map(|binding| FieldRepresentationNode {
                domain: DomainDiscretizationId::new(0),
                field: binding.field(),
                space: binding.space(),
            })
            .collect::<Vec<_>>();
        let blocks = blocks_from_scaling(&fields, plan.scaling())?;
        let execution = self.requirements().execution();
        let graph = PortableRealizationGraph {
            lineage: RealizationLineage::explicit(
                self.model(),
                self.semantic_revision(),
                self.realization_revision(),
            ),
            domains: vec![DomainDiscretizationNode {
                domain: spatial.domain(),
                coordinates: CoordinateTreatment::Scaled(spatial.coordinate_length_scale()),
                configuration: DomainConfiguration::FixedGeometry,
                discretization: spatial.discretization(),
            }],
            fields,
            geometry_actions: Vec::new(),
            transformations: Vec::new(),
            systems: vec![AlgebraicSystemNode {
                blocks,
                transformations: Vec::new(),
                scaling: SystemScaling::SymmetricCongruence(plan.scaling().clone()),
                operator_properties: plan.operator_properties(),
                scalar_type: execution.scalar_type(),
                partition: execution.vector_layout(),
            }],
            linear_solves: vec![LinearSolveNode {
                system: AlgebraicSystemId::new(0),
                plan: plan.solver(),
                placement: PlacementRequirementId::new(0),
                schedule: plan.schedule(),
            }],
            nonlinear_solves: Vec::new(),
            placements: vec![portable_placement(plan.target())],
            root: SolveRoot::Linear(LinearSolveId::new(0)),
        };
        graph.validate()?;
        Ok(graph)
    }
}

impl ResolvedCoupledFieldwiseRealization {
    /// Normalize every admitted Domain, Field, Connection and kinematic Relation.
    ///
    /// # Errors
    /// Returns `EQ0807` if the exact inventory cannot be represented losslessly.
    pub fn portable_graph(&self) -> Result<PortableRealizationGraph, Diagnostic> {
        let plan = self.plan();
        let spatial = plan.spatial();
        let domains = spatial
            .domains()
            .iter()
            .map(|selection| DomainDiscretizationNode {
                domain: selection.domain(),
                coordinates: CoordinateTreatment::Scaled(spatial.coordinate_length_scale()),
                configuration: DomainConfiguration::FixedGeometry,
                discretization: spatial.discretization(),
            })
            .collect::<Vec<_>>();
        let mut fields = spatial
            .domains()
            .iter()
            .enumerate()
            .flat_map(|(domain_index, selection)| {
                selection
                    .field_spaces()
                    .iter()
                    .map(move |binding| FieldRepresentationNode {
                        domain: DomainDiscretizationId::new(domain_index),
                        field: binding.field(),
                        space: binding.space(),
                    })
            })
            .collect::<Vec<_>>();
        for eliminated in plan.time_step().eliminated_states() {
            let pair = eliminated.pair();
            let rate_domain = fields
                .iter()
                .find(|field| field.field == pair.rate())
                .map(|field| field.domain)
                .ok_or_else(|| {
                    invalid_realization(
                        "coupled portable graph cannot locate the eliminated state's rate Domain",
                    )
                })?;
            fields.push(FieldRepresentationNode {
                domain: rate_domain,
                field: pair.state(),
                space: eliminated.state_space(),
            });
        }
        fields.sort_by_key(|field| field.field.ulid());
        let mut transformations = plan
            .time_step()
            .eliminated_states()
            .iter()
            .map(|eliminated| {
                let pair = eliminated.pair();
                Ok(TransformationNode::BackwardEulerElimination {
                    relation: pair.relation(),
                    state: field_reference(&fields, pair.state())?,
                    rate: field_reference(&fields, pair.rate())?,
                    duration: plan.time_step().duration(),
                    state_scale: eliminated.state_scale(),
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        transformations.extend(spatial.trace_quotients().iter().map(|quotient| {
            TransformationNode::ConformingTraceQuotient {
                source: quotient.source(),
                endpoints: quotient.endpoints().map(|endpoint| {
                    field_reference(&fields, endpoint.field())
                        .expect("resolved trace endpoint is present in the exact Field inventory")
                }),
            }
        }));
        let transformation_references = (0..transformations.len())
            .map(TransformationId::new)
            .collect();
        let blocks = blocks_from_scaling(&fields, plan.scaling())?;
        let execution = self.requirements().execution();
        let graph = PortableRealizationGraph {
            lineage: RealizationLineage::explicit(
                self.model(),
                self.semantic_revision(),
                self.realization_revision(),
            ),
            domains,
            fields,
            geometry_actions: Vec::new(),
            transformations,
            systems: vec![AlgebraicSystemNode {
                blocks,
                transformations: transformation_references,
                scaling: SystemScaling::SymmetricCongruence(plan.scaling().clone()),
                operator_properties: plan.operator_properties(),
                scalar_type: execution.scalar_type(),
                partition: execution.vector_layout(),
            }],
            linear_solves: vec![LinearSolveNode {
                system: AlgebraicSystemId::new(0),
                plan: plan.solver(),
                placement: PlacementRequirementId::new(0),
                schedule: plan.schedule(),
            }],
            nonlinear_solves: Vec::new(),
            placements: vec![portable_placement(plan.target())],
            root: SolveRoot::Linear(LinearSolveId::new(0)),
        };
        graph.validate()?;
        Ok(graph)
    }
}

impl ResolvedTransientFieldwiseRealization {
    /// Normalize this accepted transient plan into the common portable DAG.
    ///
    /// The existing resolver remains the sole compatibility validator. This
    /// projection cannot add a step count, runtime backend, buffer, device
    /// ordinal, or another source of numerical policy.
    ///
    /// # Errors
    /// Returns `EQ0807` if the resolved compatibility value cannot be
    /// represented losslessly by the connected Phase-A graph.
    pub fn portable_graph(&self) -> Result<PortableRealizationGraph, Diagnostic> {
        let plan = self.plan();
        let fieldwise = plan.fieldwise();
        let spatial = fieldwise.spatial();
        let domain_id = DomainDiscretizationId::new(0);
        let domains = vec![DomainDiscretizationNode {
            domain: spatial.domain(),
            coordinates: CoordinateTreatment::Scaled(spatial.coordinate_length_scale()),
            configuration: DomainConfiguration::FixedGeometry,
            discretization: spatial.discretization(),
        }];
        let fields = spatial
            .field_spaces()
            .iter()
            .map(|binding| FieldRepresentationNode {
                domain: domain_id,
                field: binding.field(),
                space: binding.space(),
            })
            .collect::<Vec<_>>();
        let field_id = |field| field_reference(&fields, field);
        let time_step = plan.time_step();
        let convection = plan.convection();
        let transformations = vec![
            TransformationNode::BackwardEulerDerivative {
                relation: time_step.relation(),
                state: field_id(time_step.state())?,
                duration: time_step.duration(),
            },
            TransformationNode::EnergySkewConvection {
                relation: convection.relation(),
                velocity: field_id(convection.velocity())?,
            },
        ];
        let blocks = fieldwise
            .scaling()
            .block_scales()
            .iter()
            .map(|entry| match entry.block() {
                AlgebraicBlock::Field(field) => field_id(field).map(SystemBlock::Field),
                AlgebraicBlock::ConstraintMultiplier { field } => Ok(
                    SystemBlock::ConstraintMultiplier(AlgebraicConstraint::ZeroIntegral { field }),
                ),
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let execution = self.requirements().fieldwise().execution();
        let placement = portable_placement(fieldwise.target());
        let graph = PortableRealizationGraph {
            lineage: RealizationLineage::explicit(
                self.model(),
                self.semantic_revision(),
                self.realization_revision(),
            ),
            domains,
            fields,
            geometry_actions: Vec::new(),
            transformations,
            systems: vec![AlgebraicSystemNode {
                blocks,
                transformations: vec![TransformationId::new(0), TransformationId::new(1)],
                scaling: SystemScaling::SymmetricCongruence(fieldwise.scaling().clone()),
                operator_properties: fieldwise.operator_properties(),
                scalar_type: execution.scalar_type(),
                partition: execution.vector_layout(),
            }],
            linear_solves: vec![LinearSolveNode {
                system: AlgebraicSystemId::new(0),
                plan: fieldwise.solver(),
                placement: PlacementRequirementId::new(0),
                schedule: fieldwise.schedule(),
            }],
            nonlinear_solves: vec![NonlinearSolveNode {
                residual_system: AlgebraicSystemId::new(0),
                linearization: LinearSolveId::new(0),
                plan: plan.nonlinear(),
            }],
            placements: vec![placement],
            root: SolveRoot::Nonlinear(NonlinearSolveId::new(0)),
        };
        graph.validate()?;
        Ok(graph)
    }
}

impl ResolvedTransientCellCenteredTransportRealization {
    /// Normalize this accepted linear transport plan into the common portable DAG.
    ///
    /// The graph records exactly one backward difference, one selected
    /// convection treatment, and orthogonal two-point diffusive flux over the
    /// same Relation/state pair. It cannot add run length, boundary meaning,
    /// or nonlinear policy.
    ///
    /// # Errors
    /// Returns `EQ0807` if the resolved compatibility value cannot be
    /// represented losslessly by the connected Phase-A graph.
    pub fn portable_graph(&self) -> Result<PortableRealizationGraph, Diagnostic> {
        let plan = self.plan();
        let fieldwise = plan.fieldwise();
        let spatial = fieldwise.spatial();
        let fields = spatial
            .field_spaces()
            .iter()
            .map(|binding| FieldRepresentationNode {
                domain: DomainDiscretizationId::new(0),
                field: binding.field(),
                space: binding.space(),
            })
            .collect::<Vec<_>>();
        let state = field_reference(&fields, plan.time_step().state())?;
        let transformations = vec![
            TransformationNode::BackwardEulerDerivative {
                relation: plan.time_step().relation(),
                state,
                duration: plan.time_step().duration(),
            },
            TransformationNode::CellCenteredConvection {
                relation: plan.convection().relation(),
                state,
                scheme: plan.convection().scheme(),
            },
            TransformationNode::OrthogonalTwoPointDiffusion {
                relation: plan.diffusion().relation(),
                state,
            },
        ];
        let blocks = blocks_from_scaling(&fields, fieldwise.scaling())?;
        let execution = self.requirements().fieldwise().execution();
        let graph = PortableRealizationGraph {
            lineage: RealizationLineage::explicit(
                self.model(),
                self.semantic_revision(),
                self.realization_revision(),
            ),
            domains: vec![DomainDiscretizationNode {
                domain: spatial.domain(),
                coordinates: CoordinateTreatment::Scaled(spatial.coordinate_length_scale()),
                configuration: DomainConfiguration::FixedGeometry,
                discretization: spatial.discretization(),
            }],
            fields,
            geometry_actions: Vec::new(),
            transformations,
            systems: vec![AlgebraicSystemNode {
                blocks,
                transformations: vec![
                    TransformationId::new(0),
                    TransformationId::new(1),
                    TransformationId::new(2),
                ],
                scaling: SystemScaling::SymmetricCongruence(fieldwise.scaling().clone()),
                operator_properties: fieldwise.operator_properties(),
                scalar_type: execution.scalar_type(),
                partition: execution.vector_layout(),
            }],
            linear_solves: vec![LinearSolveNode {
                system: AlgebraicSystemId::new(0),
                plan: fieldwise.solver(),
                placement: PlacementRequirementId::new(0),
                schedule: fieldwise.schedule(),
            }],
            nonlinear_solves: Vec::new(),
            placements: vec![portable_placement(fieldwise.target())],
            root: SolveRoot::Linear(LinearSolveId::new(0)),
        };
        graph.validate()?;
        Ok(graph)
    }
}

impl ResolvedTransientCellCenteredIncompressibleFlowRealization {
    /// Normalize the accepted collocated flow plan into the portable DAG.
    ///
    /// One nonlinear root owns backward Euler, centered momentum convection,
    /// Newtonian traction, and the shared momentum-weighted face-flux coupling.
    /// Run length and physical boundary meaning remain outside the graph.
    ///
    /// # Errors
    /// Returns `EQ0807` when the accepted compatibility value cannot be
    /// represented losslessly by this connected graph.
    pub fn portable_graph(&self) -> Result<PortableRealizationGraph, Diagnostic> {
        let plan = self.plan();
        let fieldwise = plan.fieldwise();
        let spatial = fieldwise.spatial();
        let fields = spatial
            .field_spaces()
            .iter()
            .map(|binding| FieldRepresentationNode {
                domain: DomainDiscretizationId::new(0),
                field: binding.field(),
                space: binding.space(),
            })
            .collect::<Vec<_>>();
        let velocity = field_reference(&fields, plan.coupling().velocity())?;
        let pressure = field_reference(&fields, plan.coupling().pressure())?;
        let transformations = vec![
            TransformationNode::BackwardEulerDerivative {
                relation: plan.time_step().relation(),
                state: velocity,
                duration: plan.time_step().duration(),
            },
            TransformationNode::ImplicitCenteredMomentumConvection {
                relation: plan.convection().relation(),
                velocity,
            },
            TransformationNode::CartesianCentralNewtonianTraction {
                relation: plan.traction().relation(),
                velocity,
                pressure,
            },
            TransformationNode::MomentumWeightedLinearExactCoupling {
                momentum_relation: plan.coupling().momentum_relation(),
                incompressibility_relation: plan.coupling().incompressibility_relation(),
                velocity,
                pressure,
                positive_diagonal: plan.coupling().positive_diagonal(),
                transient_history: plan.coupling().transient_history(),
            },
        ];
        let blocks = blocks_from_scaling(&fields, fieldwise.scaling())?;
        let execution = self.requirements().fieldwise().execution();
        let graph = PortableRealizationGraph {
            lineage: RealizationLineage::explicit(
                self.model(),
                self.semantic_revision(),
                self.realization_revision(),
            ),
            domains: vec![DomainDiscretizationNode {
                domain: spatial.domain(),
                coordinates: CoordinateTreatment::Scaled(spatial.coordinate_length_scale()),
                configuration: DomainConfiguration::FixedGeometry,
                discretization: spatial.discretization(),
            }],
            fields,
            geometry_actions: Vec::new(),
            transformations,
            systems: vec![AlgebraicSystemNode {
                blocks,
                transformations: vec![
                    TransformationId::new(0),
                    TransformationId::new(1),
                    TransformationId::new(2),
                    TransformationId::new(3),
                ],
                scaling: SystemScaling::SymmetricCongruence(fieldwise.scaling().clone()),
                operator_properties: fieldwise.operator_properties(),
                scalar_type: execution.scalar_type(),
                partition: execution.vector_layout(),
            }],
            linear_solves: vec![LinearSolveNode {
                system: AlgebraicSystemId::new(0),
                plan: fieldwise.solver(),
                placement: PlacementRequirementId::new(0),
                schedule: fieldwise.schedule(),
            }],
            nonlinear_solves: vec![NonlinearSolveNode {
                residual_system: AlgebraicSystemId::new(0),
                linearization: LinearSolveId::new(0),
                plan: plan.nonlinear(),
            }],
            placements: vec![portable_placement(fieldwise.target())],
            root: SolveRoot::Nonlinear(NonlinearSolveId::new(0)),
        };
        graph.validate()?;
        Ok(graph)
    }
}

impl ResolvedFixedTopologyAleCoupledRealization {
    /// Normalize the accepted fixed-topology ALE plan into one portable DAG.
    ///
    /// The graph contains one sealed geometry action.  The fluid Domain,
    /// endpoint ALE pullback, mesh velocity, and GCL correction all refer to
    /// that action; the solid Domain remains explicitly in the reference
    /// configuration.
    ///
    /// # Errors
    /// Returns `EQ0807` if any exact Domain, Field, transformation, geometry,
    /// or solve reference cannot form the closed nonlinear graph.
    pub fn portable_graph(&self) -> Result<PortableRealizationGraph, Diagnostic> {
        let plan = self.plan();
        let coupled = plan.coupled();
        let spatial = coupled.spatial();
        let regions = spatial
            .domains()
            .iter()
            .map(|domain| {
                let mut fields = domain.field_spaces().to_vec();
                for state in coupled.time_step().eliminated_states() {
                    if fields
                        .iter()
                        .any(|field| field.field() == state.pair().rate())
                    {
                        fields.push(crate::FieldSpaceBinding::new(
                            state.pair().state(),
                            state.state_space(),
                        ));
                    }
                }
                crate::DomainFieldDiscretization::new(
                    domain.domain(),
                    fields,
                    domain.constraints().iter().copied(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let execution = self.requirements().coupled().execution();
        let mut graph = PortableRealizationGraph::spatial_regions(
            RealizationLineage::explicit(
                self.model(),
                self.semantic_revision(),
                self.realization_revision(),
            ),
            regions,
            spatial.trace_quotients().iter().copied(),
            Some(coupled.time_step()),
            spatial.discretization(),
            coupled.operator_properties(),
            execution.scalar_type(),
            execution.vector_layout(),
            coupled.solver(),
            Some(plan.nonlinear()),
            coupled.target(),
            coupled.schedule(),
            SystemScaling::SymmetricCongruence(coupled.scaling().clone()),
        )?
        .with_harmonic_geometry(
            plan.mesh_motion(),
            plan.mesh_motion().harmonic_domains().iter().copied(),
            [plan.fluid_time_step()],
        )?;
        for domain in &mut graph.domains {
            domain.coordinates = CoordinateTreatment::Scaled(spatial.coordinate_length_scale());
        }
        graph.validate()?;
        Ok(graph)
    }
}

fn blocks_from_scaling(
    fields: &[FieldRepresentationNode],
    scaling: &SymmetricCongruenceScaling,
) -> Result<Vec<SystemBlock>, Diagnostic> {
    scaling
        .block_scales()
        .iter()
        .map(|entry| match entry.block() {
            AlgebraicBlock::Field(field) => field_reference(fields, field).map(SystemBlock::Field),
            AlgebraicBlock::ConstraintMultiplier { field } => {
                if fields.iter().any(|node| node.field == field) {
                    Ok(SystemBlock::ConstraintMultiplier(
                        AlgebraicConstraint::ZeroIntegral { field },
                    ))
                } else {
                    Err(invalid_realization(
                        "portable constraint multiplier refers to an unrepresented Field",
                    ))
                }
            }
        })
        .collect()
}

fn field_reference(
    fields: &[FieldRepresentationNode],
    field: Id<kinds::Field>,
) -> Result<FieldRepresentationId, Diagnostic> {
    fields
        .binary_search_by_key(&field.ulid(), |node| node.field.ulid())
        .map(FieldRepresentationId::new)
        .map_err(|_| {
            invalid_realization(
                "portable transformation references an unrepresented Semantic Field",
            )
        })
}

pub(super) fn portable_placement(target: Target) -> PlacementRequirementNode {
    match target {
        Target::HostCpu { threads } => PlacementRequirementNode::HostWorkers {
            workers_per_partition: threads,
        },
        Target::CudaGpu { .. } => PlacementRequirementNode::CudaDevices {
            devices_per_partition: NonZeroUsize::MIN,
        },
    }
}
