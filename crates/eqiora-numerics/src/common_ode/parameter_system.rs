//! Real Parameter coordinates shared by continuous flow and canonical event groups.
use super::*;
use eqiora_graph::EdgeKind;
use eqiora_runtime::{CanonicalEventProgram, CanonicalRootSet};
use eqiora_time::{
    EventGuardLinearization, EventResetLinearization, MassParameterDependence,
    ParametricTimeSystem, TimeSystem, TransversalEventLinearization,
};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone)]
pub(crate) struct GlobalParameterSystem<'a> {
    flow: &'a FirstOrderProgram,
    parameter_ids: Vec<Id<kinds::Parameter>>,
    parameter_values: Vec<f64>,
    flow_columns: Vec<usize>,
    initial_jacobian: Vec<f64>,
}

impl<'a> GlobalParameterSystem<'a> {
    pub(crate) fn new(
        plan: &'a CommonOdePlan,
        roots: &CanonicalRootSet,
    ) -> Result<Self, Diagnostic> {
        let flow = &plan.program;
        let mut coordinates = HashMap::<Id<kinds::Parameter>, usize>::new();
        let mut parameter_ids = Vec::new();
        let mut parameter_values: Vec<f64> = Vec::new();
        for (ids, values) in std::iter::once((flow.parameter_fields(), flow.parameters())).chain(
            roots
                .events()
                .iter()
                .map(|event| (event.parameter_fields(), event.parameters())),
        ) {
            if ids.len() != values.len() {
                return Err(invalid(
                    "canonical Parameter IDs and values have different shapes",
                ));
            }
            for (&id, &value) in ids.iter().zip(values) {
                if let Some(&column) = coordinates.get(&id) {
                    if parameter_values[column].to_bits() != value.to_bits() {
                        return Err(invalid(
                            "one exact global Parameter has inconsistent captured values",
                        ));
                    }
                } else {
                    coordinates.insert(id, parameter_ids.len());
                    parameter_ids.push(id);
                    parameter_values.push(value);
                }
            }
        }
        if roots.events().iter().any(|event| event.flow() != flow) {
            return Err(invalid(
                "event Parameter layout belongs to a different canonical flow",
            ));
        }
        let kernel = plan.model.to_program().map_err(|errors| {
            errors
                .into_iter()
                .next()
                .unwrap_or_else(|| invalid("initial Parameter Model replay failed"))
        })?;
        reject_unproven_initial_parameters(&kernel, &parameter_ids, flow.parameter_fields())?;
        let mut initial_tangent = vec![0.0; flow.dimension()];
        // The existing initialization owner proves a unique zero tangent in flow coordinates.
        flow.initial_parameter_jvp(
            0.0,
            &vec![0.0; flow.parameter_dimension()],
            &mut initial_tangent,
        )?;
        let flow_columns = flow
            .parameter_fields()
            .iter()
            .map(|id| coordinates[id])
            .collect();
        let initial_jacobian = vec![0.0; flow.dimension() * parameter_ids.len()];
        Ok(Self {
            flow,
            parameter_ids,
            parameter_values,
            flow_columns,
            initial_jacobian,
        })
    }

    pub(crate) fn parameter_ids(&self) -> &[Id<kinds::Parameter>] {
        &self.parameter_ids
    }

    /// Bind the accepted fixed-time post-event Jacobian, row-major (state, global Parameter).
    pub(crate) fn with_initial_jacobian(
        &self,
        initial_jacobian: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        if self.dimension().checked_mul(self.parameter_dimension()) != Some(initial_jacobian.len())
            || initial_jacobian.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "restart initial Parameter Jacobian has invalid shape or values",
            ));
        }
        let mut result = self.clone();
        result.initial_jacobian = initial_jacobian;
        Ok(result)
    }

    /// Reindex only direct Parameter columns; canonical composition owns saltation.
    pub(crate) fn lift_event(
        &self,
        event: &CanonicalEventProgram,
        linearization: &TransversalEventLinearization,
    ) -> Result<TransversalEventLinearization, Diagnostic> {
        if event.flow() != self.flow
            || linearization.state_dimension() != self.dimension()
            || linearization.parameter_dimension() != event.parameter_fields().len()
        {
            return Err(invalid(
                "event derivatives differ from the selected canonical Parameter layout",
            ));
        }
        let columns = event
            .parameter_fields()
            .iter()
            .map(|id| {
                self.parameter_ids
                    .iter()
                    .position(|candidate| candidate == id)
                    .ok_or_else(|| {
                        invalid("event derivative references a foreign global Parameter")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut guard = vec![0.0; self.parameter_dimension()];
        let mut reset = vec![0.0; self.dimension() * self.parameter_dimension()];
        for (local, &global) in columns.iter().enumerate() {
            guard[global] = linearization.guard().parameter_gradient()[local];
            for state in 0..self.dimension() {
                reset[state * self.parameter_dimension() + global] =
                    linearization.reset().parameter_jacobian()[state * columns.len() + local];
            }
        }
        TransversalEventLinearization::new(
            linearization.flow().clone(),
            EventGuardLinearization::new(
                linearization.guard().state_gradient().to_vec(),
                guard,
                linearization.guard().time_derivative(),
            )?,
            EventResetLinearization::new(
                self.dimension(),
                self.parameter_dimension(),
                linearization.reset().state_jacobian().to_vec(),
                reset,
                linearization.reset().time_derivative().to_vec(),
            )?,
        )
    }
}

impl TimeSystem for GlobalParameterSystem<'_> {
    fn dimension(&self) -> usize {
        self.flow.dimension()
    }
    fn rhs(&self, time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        self.flow.rhs(time, state, output)
    }
    fn rhs_jvp(
        &self,
        time: f64,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.flow.rhs_jvp(time, state, direction, output)
    }
    fn mass_action(
        &self,
        time: f64,
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.flow.mass_action(time, direction, output)
    }
}
impl ParametricTimeSystem for GlobalParameterSystem<'_> {
    fn parameter_dimension(&self) -> usize {
        self.parameter_ids.len()
    }
    fn parameters(&self) -> &[f64] {
        &self.parameter_values
    }
    fn mass_parameter_dependence(&self) -> MassParameterDependence {
        self.flow.mass_parameter_dependence()
    }
    fn rhs_parameter_jvp(
        &self,
        time: f64,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.require_direction(direction)?;
        let local = self
            .flow_columns
            .iter()
            .map(|&column| direction[column])
            .collect::<Vec<_>>();
        self.flow.rhs_parameter_jvp(time, state, &local, output)
    }
    fn initial_parameter_jvp(
        &self,
        time: f64,
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.require_direction(direction)?;
        if !time.is_finite() || output.len() != self.dimension() {
            return Err(invalid(
                "initial Parameter action has invalid time or state shape",
            ));
        }
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.initial_jacobian
                [row * self.parameter_dimension()..(row + 1) * self.parameter_dimension()]
                .iter()
                .zip(direction)
                .map(|(entry, direction)| entry * direction)
                .sum();
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "initial Parameter action produced non-finite values",
            ));
        }
        Ok(())
    }
}
impl GlobalParameterSystem<'_> {
    fn require_direction(&self, direction: &[f64]) -> Result<(), Diagnostic> {
        if direction.len() != self.parameter_dimension()
            || direction.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "global Parameter direction has invalid shape or values",
            ));
        }
        Ok(())
    }
}

fn reject_unproven_initial_parameters(
    kernel: &KernelProgram,
    global: &[Id<kinds::Parameter>],
    flow: &[Id<kinds::Parameter>],
) -> Result<(), Diagnostic> {
    let unsupported = global
        .iter()
        .filter(|id| !flow.contains(id))
        .map(|id| id.erase())
        .collect::<BTreeSet<_>>();
    let mut pending = kernel
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Relation(relation) if relation.is_initial() => Some(relation.id().erase()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut visited = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !visited.insert(id) {
            continue;
        }
        if unsupported.contains(&id) {
            return Err(invalid(
                "Initial constraints depend on a guard/reset-only Parameter whose initial tangent is not derived",
            ));
        }
        pending.extend(
            kernel
                .edges()
                .iter()
                .filter(|edge| {
                    edge.from() == id
                        && matches!(
                            edge.kind(),
                            EdgeKind::DependsOn | EdgeKind::StructurallyDependsOn
                        )
                })
                .map(|edge| edge.to()),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::DynQuantity;
    use eqiora_graph::{GraphStore, InMemoryGraphStore};

    const RAMP: &str = r#"
model Ramp() {
  state x: 1;
  parameter speed: 1/s = 2;
  parameter first: 1 = 1;
  parameter second: 1 = 3;
  initial { x = 0; }
  relation flow { derivative(x) = speed; }
  event first_hit = crossing(x-first, direction=rising);
  event second_hit = crossing(x-second, direction=rising);
  relation first_reset at first_hit { next(x)=0; }
  relation second_reset at second_hit { next(x)=0; }
}
"#;

    fn plan(source: &str) -> CommonOdePlan {
        let compiled = eqiora_compiler::compile("global-parameters.eqi", source)
            .unwrap()
            .pop()
            .unwrap();
        let (transaction, model, _) = compiled.into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let kernel = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let absolute = kernel
            .nodes()
            .filter_map(|node| match node {
                KernelNode::Field(field) => {
                    Some(CommonTsitourasTolerance::new(field.id(), 1e-10).unwrap())
                }
                _ => None,
            })
            .collect();
        let guards = kernel
            .nodes()
            .filter_map(|node| match node {
                KernelNode::Activation(activation)
                    if matches!(activation.kind(), ActivationKind::Event { .. }) =>
                {
                    Some(
                        CommonGuardTolerance::new(
                            activation.id(),
                            DynQuantity::new(1e-10, DimExponents::DIMENSIONLESS),
                        )
                        .unwrap(),
                    )
                }
                _ => None,
            })
            .collect();
        let temporal = CommonTsitouras45::new(1e-3, 1e-9, absolute)
            .unwrap()
            .with_event_policy(CommonEventPolicy::new(4, guards).unwrap());
        CommonOdePlan::resolve(
            &ModelEnvelope::from_program(&kernel).unwrap(),
            &kernel,
            temporal,
            TimeBackendIdentity::new("test.parameters", "1"),
        )
        .unwrap()
    }

    #[test]
    fn real_global_coordinates_project_flow_and_lift_canonical_event_derivatives() {
        let plan = plan(RAMP);
        let roots = plan.root_set().unwrap().unwrap();
        let system = GlobalParameterSystem::new(&plan, &roots).unwrap();
        assert_eq!(system.parameter_dimension(), 3);
        assert_eq!(
            system.parameter_ids()[0],
            plan.program.parameter_fields()[0]
        );
        let speed = system
            .parameters()
            .iter()
            .position(|&value| value == 2.0)
            .unwrap();
        let first = system
            .parameters()
            .iter()
            .position(|&value| value == 1.0)
            .unwrap();
        let second = system
            .parameters()
            .iter()
            .position(|&value| value == 3.0)
            .unwrap();
        let mut direction = vec![0.0; 3];
        let mut output = vec![0.0; 1];
        direction[speed] = 1.0;
        system
            .rhs_parameter_jvp(0.5, &[1.0], &direction, &mut output)
            .unwrap();
        assert_eq!(output, [1.0]);
        direction.fill(0.0);
        direction[first] = 1.0;
        system
            .rhs_parameter_jvp(0.5, &[1.0], &direction, &mut output)
            .unwrap();
        assert_eq!(output, [0.0]);
        system
            .initial_parameter_jvp(0.0, &direction, &mut output)
            .unwrap();
        assert_eq!(output, [0.0]);

        let event = roots
            .events()
            .iter()
            .find(|event| event.parameters().contains(&1.0))
            .unwrap();
        let point = event.linearize_at(0.5, &[1.0], 1e-10).unwrap();
        let lifted = system.lift_event(event, point.derivatives()).unwrap();
        assert_eq!(lifted.parameter_dimension(), 3);
        assert_eq!(
            lifted.saltation_matrix(),
            point.derivatives().saltation_matrix()
        );
        let mut pre = vec![0.0; 3];
        pre[speed] = 0.5;
        let propagated = lifted.propagate_forward(&pre).unwrap();
        // x=vt, tau=p/v, and x_after=vt-p: independently derived fixed-time derivatives.
        assert_eq!(propagated.event_time()[speed], -0.25);
        assert_eq!(propagated.event_time()[first], 0.5);
        assert_eq!(propagated.event_time()[second], 0.0);
        assert_eq!(propagated.post_state()[speed], 0.5);
        assert_eq!(propagated.post_state()[first], -1.0);
        assert_eq!(propagated.post_state()[second], 0.0);
        let restarted = system
            .with_initial_jacobian(propagated.post_state().to_vec())
            .unwrap();
        restarted
            .initial_parameter_jvp(0.5, &direction, &mut output)
            .unwrap();
        assert_eq!(output, [-1.0]);
    }

    #[test]
    fn initial_parameter_dependence_and_foreign_event_layout_fail_closed() {
        for initial in ["x = first", "x = speed * 1[s]"] {
            let plan = plan(&RAMP.replace("x = 0", initial));
            let roots = plan.root_set().unwrap().unwrap();
            assert!(GlobalParameterSystem::new(&plan, &roots).is_err());
        }
        let first_plan = plan(RAMP);
        let foreign_plan = plan(&RAMP.replace("second: 1 = 3", "second: 1 = 4"));
        assert!(
            GlobalParameterSystem::new(&first_plan, &foreign_plan.root_set().unwrap().unwrap())
                .is_err()
        );
    }
}
