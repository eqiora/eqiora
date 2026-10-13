//! Read Region bindings from the selected system of the common numerical graph.
use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::ReferenceCell;
use eqiora_realization::{
    AlgebraicSystemNode, BackwardEulerStateBinding, BackwardEulerStatePair, BackwardEulerStep,
    ConformingTraceQuotient, PortableRealizationGraph, SolveRoot, Space, SystemBlock,
    SystemScaling, TraceFieldEndpoint, TransformationNode,
};

use super::{BoundRegionForm, CompiledRegionForm, RegionTimeBinding, invalid};
use crate::spatial_expression::Coefficient;

pub(crate) struct RegionGraph<'a> {
    graph: &'a PortableRealizationGraph,
    system: &'a AlgebraicSystemNode,
    pub(crate) fields: BTreeMap<RawId, (RawId, Space)>,
    pub(crate) quotients: Vec<ConformingTraceQuotient>,
    time: Option<RegionTimeBinding>,
}

impl<'a> RegionGraph<'a> {
    pub(crate) fn new(graph: &'a PortableRealizationGraph) -> Result<Self, Diagnostic> {
        let system = match graph.root() {
            SolveRoot::Linear(id) => graph.linear_solve(id).expect("validated root").system(),
            SolveRoot::Nonlinear(id) => graph
                .nonlinear_solve(id)
                .expect("validated root")
                .residual_system(),
        };
        let system = graph.system(system).expect("validated root system");
        let mut fields = BTreeMap::new();
        for block in system.blocks() {
            let SystemBlock::Field(id) = *block else {
                return Err(invalid(
                    "Region binding requires Field-owned algebraic blocks",
                ));
            };
            let field = graph.field(id).expect("validated algebraic Field");
            fields.insert(
                field.field().erase(),
                (
                    graph
                        .domain(field.domain())
                        .expect("validated Domain")
                        .domain()
                        .erase(),
                    field.space(),
                ),
            );
        }
        let mut duration = None;
        let mut states = Vec::new();
        let mut quotients = Vec::new();
        for id in system.transformations() {
            let node = graph.transformation(*id).expect("validated transformation");
            let step = match *node {
                TransformationNode::BackwardEulerDerivative { duration, .. }
                | TransformationNode::BackwardEulerElimination { duration, .. } => Some(duration),
                _ => None,
            };
            if let Some(step) = step {
                if duration.is_some_and(|selected| selected != step) {
                    return Err(invalid(
                        "synchronized Region bindings require one exact duration",
                    ));
                }
                duration = Some(step);
            }
            match *node {
                TransformationNode::BackwardEulerElimination {
                    relation,
                    state,
                    rate,
                    state_scale,
                    ..
                } => {
                    let state = graph.field(state).expect("validated state");
                    let rate = graph.field(rate).expect("validated rate");
                    states.push(BackwardEulerStateBinding::new(
                        BackwardEulerStatePair::new(relation, state.field(), rate.field())?,
                        state.space(),
                        state_scale,
                    ));
                }
                TransformationNode::ConformingTraceQuotient { source, endpoints } => {
                    let endpoints = endpoints.map(|id| {
                        let field = graph.field(id).expect("validated quotient Field");
                        TraceFieldEndpoint::new(
                            graph
                                .domain(field.domain())
                                .expect("validated Domain")
                                .domain(),
                            field.field(),
                        )
                    });
                    quotients.push(ConformingTraceQuotient::new(
                        source,
                        endpoints[0],
                        endpoints[1],
                    )?);
                }
                _ => {}
            }
        }
        states.sort_by_key(|state| state.pair().state().erase());
        Ok(Self {
            graph,
            system,
            fields,
            quotients,
            time: duration.map(|step| RegionTimeBinding { step, states }),
        })
    }

    pub(crate) fn kinematic_step(&self) -> Result<BackwardEulerStep, Diagnostic> {
        let time = self
            .time
            .as_ref()
            .ok_or_else(|| invalid("Region graph has no time step"))?;
        BackwardEulerStep::new(time.step, time.states.iter().copied())
    }

    pub(crate) fn domains(&self) -> impl Iterator<Item = RawId> + '_ {
        self.graph
            .domains()
            .iter()
            .map(|domain| domain.domain().erase())
    }

    pub(crate) fn bind<S: Coefficient>(
        &self,
        form: &CompiledRegionForm<S>,
        reference: ReferenceCell,
        orientations: &BTreeMap<RawId, f64>,
    ) -> Result<BoundRegionForm<S>, Diagnostic> {
        let space = |field| {
            self.fields
                .get(&field)
                .filter(|(domain, _)| *domain == form.domain)
                .map(|(_, space)| *space)
                .ok_or_else(|| invalid("Region Field has no exact graph Domain/Space binding"))
        };
        let (fields, rows) = match self.system.scaling() {
            SystemScaling::Dimensional => form.si_bindings(space)?,
            SystemScaling::SymmetricCongruence(scaling) => {
                form.congruence_bindings(space, scaling, orientations)?
            }
        };
        let time = self.time.as_ref().map(|time| RegionTimeBinding {
            step: time.step,
            states: time
                .states
                .iter()
                .copied()
                .filter(|state| {
                    fields
                        .iter()
                        .any(|field| field.field == state.pair().rate().erase())
                })
                .collect(),
        });
        form.bind(reference, &fields, &rows, time.as_ref())
    }
}
