//! Bind numerical geometry choices to the existing exact Region/Field map.
use super::*;
use eqiora_realization::{
    DomainConfiguration, GeometryActionNode, P1HarmonicMeshMotionPolicy, PortableRealizationGraph,
    TransformationNode,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn current_pullbacks<const D: usize>(
    graph: &PortableRealizationGraph,
    layout: &FsiLayout<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    plan: &AleFsiStepPlan<D>,
) -> Result<BTreeMap<RawId, Vec<RawId>>, Diagnostic> {
    let domains = layout
        .partition()
        .domains()
        .map(|id| id.erase())
        .collect::<BTreeSet<_>>();
    if graph
        .domains()
        .iter()
        .map(|node| node.domain().erase())
        .collect::<BTreeSet<_>>()
        != domains
    {
        return Err(invalid(
            "ALE graph Domains differ from the exact Region inventory",
        ));
    }
    let mut expected = BTreeMap::new();
    for &domain in &domains {
        for field in layout.region_form(domain)?.fields() {
            expected.insert(field.field, (domain, field.space));
        }
    }
    for state in layout.state_bindings() {
        let (domain, _) = layout
            .mapping()
            .field_layout(state.pair().rate().erase())
            .ok_or_else(|| invalid("ALE graph driver rate has no mapped Field"))?;
        expected.insert(state.pair().state().erase(), (domain, state.state_space()));
    }
    let represented = graph
        .fields()
        .iter()
        .map(|field| {
            (
                field.field().erase(),
                (
                    graph
                        .domain(field.domain())
                        .expect("validated graph Domain")
                        .domain()
                        .erase(),
                    field.space(),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if represented != expected {
        return Err(invalid(
            "ALE graph Fields or Spaces differ from the exact Region inventory",
        ));
    }
    let [
        GeometryActionNode::P1HarmonicExtension {
            harmonic_domains,
            drivers,
            interfaces,
            duration,
            quality_gate,
            solver,
        },
    ] = graph.geometry_actions()
    else {
        return Err(invalid(
            "ALE execution requires the exact supplied geometry action inventory",
        ));
    };
    let policy = P1HarmonicMeshMotionPolicy::new(
        harmonic_domains.iter().map(|id| {
            graph
                .domain(*id)
                .expect("validated harmonic Domain")
                .domain()
        }),
        drivers.iter().map(|id| {
            let field = graph.field(*id).expect("validated driver Field");
            (
                graph
                    .domain(field.domain())
                    .expect("validated driver Domain")
                    .domain(),
                field.field(),
            )
        }),
        interfaces.iter().copied(),
        *quality_gate,
        *solver,
    )?;
    if &policy != motion.policy() || duration.value() != plan.time_step() {
        return Err(invalid(
            "ALE graph geometry policy or duration differs from the sealed action",
        ));
    }
    let mut current = BTreeMap::new();
    for domain in graph.domains() {
        if let DomainConfiguration::CurrentAleGeometry { action } = domain.configuration() {
            if action.index() != 0 {
                return Err(invalid("ALE Domain selects an unsupplied geometry action"));
            }
            current.insert(domain.domain().erase(), Vec::new());
        }
    }
    for node in graph.transformations() {
        if let TransformationNode::GclCompatibleAlePullback {
            relation,
            velocity,
            geometry,
        } = *node
        {
            let field = graph.field(velocity).expect("validated GCL Field");
            let domain = graph
                .domain(field.domain())
                .expect("validated GCL Domain")
                .domain()
                .erase();
            if geometry.index() != 0
                || layout
                    .region_form(domain)?
                    .equation_for(field.field().erase())
                    != Some(relation.erase())
            {
                return Err(invalid(
                    "ALE pullback differs from its exact Region Relation and Field",
                ));
            }
            current
                .get_mut(&domain)
                .ok_or_else(|| invalid("ALE pullback has no current Domain configuration"))?
                .push(field.field().erase());
        }
    }
    Ok(current)
}
