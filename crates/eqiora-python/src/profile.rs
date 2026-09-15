//! Process-local collection and Python presentation for Eqiora tracing spans.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::{Layer, Registry};

pub(crate) const TELEMETRY_TARGET: &str = "eqiora::execution";

#[derive(Debug, Clone)]
pub(crate) struct ProfilePhaseData {
    pub(crate) path: Vec<String>,
    pub(crate) fields: BTreeMap<String, String>,
    pub(crate) calls: usize,
    pub(crate) inclusive: Duration,
    pub(crate) self_time: Duration,
}

#[derive(Debug, Clone)]
pub(crate) struct ProfileEventData {
    pub(crate) path: Vec<String>,
    pub(crate) fields: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ProfileData {
    pub(crate) phases: Vec<ProfilePhaseData>,
    pub(crate) events: Vec<ProfileEventData>,
}

#[derive(Default)]
struct FieldVisitor(BTreeMap<String, String>);

impl Visit for FieldVisitor {
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.0.insert(field.name().to_owned(), value.to_string());
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.0.insert(field.name().to_owned(), value.to_string());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().to_owned(), value.to_string());
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.0.insert(field.name().to_owned(), value.to_string());
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().to_owned(), value.to_owned());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().to_owned(), format!("{value:?}"));
    }
}

struct OpenSpan {
    identity: PhaseIdentity,
    path: Vec<String>,
    parent: Option<u64>,
    entered: Option<Instant>,
    inclusive: Duration,
    child_inclusive: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PhaseIdentity {
    path: Vec<String>,
    fields: BTreeMap<String, String>,
}

#[derive(Default)]
struct PhaseAggregate {
    calls: usize,
    inclusive: Duration,
    self_time: Duration,
}

#[derive(Default)]
struct CollectorState {
    open: HashMap<u64, OpenSpan>,
    phases: BTreeMap<PhaseIdentity, PhaseAggregate>,
    phase_order: Vec<PhaseIdentity>,
    events: Vec<ProfileEventData>,
}

#[derive(Clone, Default)]
pub(crate) struct ProfileCollector {
    state: Arc<Mutex<CollectorState>>,
}

impl ProfileCollector {
    pub(crate) fn capture<T>(&self, operation: impl FnOnce() -> T) -> T {
        let subscriber = Registry::default().with(self.clone());
        tracing::subscriber::with_default(subscriber, operation)
    }

    pub(crate) fn finish(&self) -> ProfileData {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        ProfileData {
            phases: state
                .phase_order
                .iter()
                .filter_map(|identity| {
                    let aggregate = state.phases.get(identity)?;
                    Some(ProfilePhaseData {
                        path: identity.path.clone(),
                        fields: identity.fields.clone(),
                        calls: aggregate.calls,
                        inclusive: aggregate.inclusive,
                        self_time: aggregate.self_time,
                    })
                })
                .collect(),
            events: state.events.clone(),
        }
    }
}

impl CollectorState {
    fn close_span(&mut self, id: u64) {
        let Some(span) = self.open.remove(&id) else {
            return;
        };
        if let Some(parent) = span.parent.and_then(|parent| self.open.get_mut(&parent)) {
            parent.child_inclusive += span.inclusive;
        }
        let self_time = span.inclusive.saturating_sub(span.child_inclusive);
        let aggregate = self.phases.entry(span.identity).or_default();
        aggregate.calls += 1;
        aggregate.inclusive += span.inclusive;
        aggregate.self_time += self_time;
    }
}

const OCCURRENCE_FIELDS: [&str; 6] = [
    "dt_s",
    "iteration",
    "residual_norm",
    "solve",
    "step",
    "time_s",
];

fn identity_fields(fields: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    fields
        .iter()
        .filter(|(name, _)| !OCCURRENCE_FIELDS.contains(&name.as_str()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

fn has_occurrence_fields(fields: &BTreeMap<String, String>) -> bool {
    fields
        .keys()
        .any(|name| OCCURRENCE_FIELDS.contains(&name.as_str()))
}

impl<S> Layer<S> for ProfileCollector
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if attrs.metadata().target() != TELEMETRY_TARGET {
            return;
        }
        let mut visitor = FieldVisitor::default();
        attrs.record(&mut visitor);
        let phase = visitor
            .0
            .remove("phase")
            .unwrap_or_else(|| attrs.metadata().name().to_owned());
        let parent = attrs
            .parent()
            .cloned()
            .or_else(|| ctx.current_span().id().cloned());
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let parent = parent.map(|parent| parent.into_u64());
        let mut path = parent
            .and_then(|parent| state.open.get(&parent).map(|span| span.path.clone()))
            .unwrap_or_default();
        path.push(phase.clone());
        let identity = PhaseIdentity {
            path: path.clone(),
            fields: identity_fields(&visitor.0),
        };
        let first_identity_occurrence = !state.phase_order.contains(&identity);
        if first_identity_occurrence {
            state.phase_order.push(identity.clone());
        }
        if first_identity_occurrence || has_occurrence_fields(&visitor.0) {
            visitor.0.insert("phase".to_owned(), phase);
            visitor.0.insert("event".to_owned(), "phase".to_owned());
            state.events.push(ProfileEventData {
                path: path.clone(),
                fields: visitor.0,
            });
        }
        state.open.insert(
            id.into_u64(),
            OpenSpan {
                identity,
                path,
                parent,
                entered: None,
                inclusive: Duration::ZERO,
                child_inclusive: Duration::ZERO,
            },
        );
    }

    fn on_enter(&self, id: &Id, _ctx: Context<'_, S>) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(span) = state.open.get_mut(&id.into_u64()) {
            span.entered = Some(Instant::now());
        }
    }

    fn on_exit(&self, id: &Id, _ctx: Context<'_, S>) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(span) = state.open.get_mut(&id.into_u64())
            && let Some(entered) = span.entered.take()
        {
            span.inclusive += entered.elapsed();
        }
    }

    fn on_close(&self, id: Id, _ctx: Context<'_, S>) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.close_span(id.into_u64());
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        if event.metadata().target() != TELEMETRY_TARGET {
            return;
        }
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let path = ctx
            .current_span()
            .id()
            .and_then(|id| state.open.get(&id.into_u64()).map(|span| span.path.clone()))
            .unwrap_or_default();
        state.events.push(ProfileEventData {
            path,
            fields: visitor.0,
        });
    }
}

#[pyclass(
    name = "ProfilePhase",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Aggregate timing for one hierarchical execution phase and semantic identity.
///
/// Inclusive time contains nested children; self time excludes them. Occurrence
/// fields such as step and iteration remain events and do not split this
/// aggregate.
pub(crate) struct PyProfilePhase(ProfilePhaseData);

#[pymethods]
impl PyProfilePhase {
    #[getter]
    fn path(&self) -> Vec<String> {
        self.0.path.clone()
    }

    #[getter]
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let fields = PyDict::new(py);
        for (name, value) in &self.0.fields {
            fields.set_item(name, value)?;
        }
        Ok(fields)
    }

    #[getter]
    fn name(&self) -> &str {
        self.0.path.last().map_or("", String::as_str)
    }

    #[getter]
    const fn calls(&self) -> usize {
        self.0.calls
    }

    #[getter]
    fn inclusive_seconds(&self) -> f64 {
        self.0.inclusive.as_secs_f64()
    }

    #[getter]
    fn self_seconds(&self) -> f64 {
        self.0.self_time.as_secs_f64()
    }

    #[getter]
    fn mean_seconds(&self) -> f64 {
        self.inclusive_seconds() / self.0.calls as f64
    }

    fn __repr__(&self) -> String {
        format!(
            "ProfilePhase(name={:?}, fields={:?}, calls={}, inclusive_seconds={:.6}, self_seconds={:.6}, mean_seconds={:.6})",
            self.name(),
            self.0.fields,
            self.calls(),
            self.inclusive_seconds(),
            self.self_seconds(),
            self.mean_seconds(),
        )
    }
}

#[pyclass(
    name = "ProfileEvent",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Structured metadata for one occurrence-bearing phase, the first occurrence
/// of an aggregate-only phase identity, or a solver observation.
pub(crate) struct PyProfileEvent(ProfileEventData);

#[pymethods]
impl PyProfileEvent {
    #[getter]
    fn path(&self) -> Vec<String> {
        self.0.path.clone()
    }

    #[getter]
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let fields = PyDict::new(py);
        for (name, value) in &self.0.fields {
            fields.set_item(name, value)?;
        }
        Ok(fields)
    }

    fn __repr__(&self) -> String {
        format!(
            "ProfileEvent(path={:?}, fields={:?})",
            self.0.path, self.0.fields
        )
    }
}

/// Process-local phase timings and structured solver events for one Run.
#[pyclass(name = "Profile", module = "eqiora._eqiora", frozen)]
pub(crate) struct PyProfile {
    data: ProfileData,
}

impl PyProfile {
    pub(crate) fn new(data: ProfileData) -> Self {
        Self { data }
    }
}

#[pymethods]
impl PyProfile {
    #[getter]
    fn phases(&self, py: Python<'_>) -> PyResult<Vec<Py<PyProfilePhase>>> {
        self.data
            .phases
            .iter()
            .cloned()
            .map(|phase| Py::new(py, PyProfilePhase(phase)))
            .collect()
    }

    #[getter]
    fn events(&self, py: Python<'_>) -> PyResult<Vec<Py<PyProfileEvent>>> {
        self.data
            .events
            .iter()
            .cloned()
            .map(|event| Py::new(py, PyProfileEvent(event)))
            .collect()
    }

    #[getter]
    fn total_seconds(&self) -> f64 {
        self.data
            .phases
            .iter()
            .find(|phase| phase.path.as_slice() == ["run"])
            .map_or(0.0, |phase| phase.inclusive.as_secs_f64())
    }

    fn summary(&self) -> String {
        format_summary(&self.data)
    }

    fn __repr__(&self) -> String {
        format!(
            "Profile(phases={}, events={}, total_seconds={:.6})",
            self.data.phases.len(),
            self.data.events.len(),
            self.total_seconds(),
        )
    }
}

fn format_summary(data: &ProfileData) -> String {
    let mut output = String::from(
        "Eqiora run profile\nphase and identity                              total inclusive       self       calls       mean per call\n",
    );
    for phase in &data.phases {
        let indent = "  ".repeat(phase.path.len().saturating_sub(1));
        let name = phase.path.last().map_or("", String::as_str);
        let identity = phase
            .fields
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>()
            .join(", ");
        let label = if identity.is_empty() {
            format!("{indent}{name}")
        } else {
            format!("{indent}{name} [{identity}]")
        };
        let mean = phase.inclusive.as_secs_f64() / phase.calls as f64;
        output.push_str(&format!(
            "{label:<47} {:>10.6} s {:>10.6} s {:>8} calls {:>10.6} s/call\n",
            phase.inclusive.as_secs_f64(),
            phase.self_time.as_secs_f64(),
            phase.calls,
            mean,
        ));
    }
    output
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyProfilePhase>()?;
    module.add_class::<PyProfileEvent>()?;
    module.add_class::<PyProfile>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(path: &[&str], fields: &[(&str, &str)]) -> PhaseIdentity {
        PhaseIdentity {
            path: path.iter().map(|part| (*part).to_owned()).collect(),
            fields: fields
                .iter()
                .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
                .collect(),
        }
    }

    fn open(
        identity: PhaseIdentity,
        parent: Option<u64>,
        inclusive: Duration,
        child_inclusive: Duration,
    ) -> OpenSpan {
        OpenSpan {
            path: identity.path.clone(),
            identity,
            parent,
            entered: None,
            inclusive,
            child_inclusive,
        }
    }

    #[test]
    fn nested_and_repeated_phases_aggregate_inclusive_self_calls_and_mean() {
        let run = identity(&["run"], &[("family", "transient_flow")]);
        let step = identity(&["run", "solve", "time_step"], &[]);
        let mut state = CollectorState {
            phase_order: vec![run.clone(), step.clone()],
            ..CollectorState::default()
        };
        state.open.insert(
            1,
            open(run.clone(), None, Duration::from_secs(20), Duration::ZERO),
        );
        state.open.insert(
            2,
            open(
                step.clone(),
                Some(1),
                Duration::from_secs(6),
                Duration::ZERO,
            ),
        );
        state.close_span(2);
        state.open.insert(
            3,
            open(step, Some(1), Duration::from_secs(4), Duration::ZERO),
        );
        state.close_span(3);
        state.close_span(1);

        let collector = ProfileCollector {
            state: Arc::new(Mutex::new(state)),
        };
        let data = collector.finish();
        assert_eq!(data.phases.len(), 2);
        assert_eq!(data.phases[0].inclusive, Duration::from_secs(20));
        assert_eq!(data.phases[0].self_time, Duration::from_secs(10));
        assert_eq!(data.phases[0].calls, 1);
        assert_eq!(data.phases[1].inclusive, Duration::from_secs(10));
        assert_eq!(data.phases[1].self_time, Duration::from_secs(10));
        assert_eq!(data.phases[1].calls, 2);
        assert_eq!(
            data.phases[1].inclusive.as_secs_f64() / data.phases[1].calls as f64,
            5.0
        );
    }

    #[test]
    fn semantic_fields_separate_same_path_and_observations_do_not() {
        let mut first = BTreeMap::from([
            ("phase".to_owned(), "assembly".to_owned()),
            ("role".to_owned(), "initial_linearization".to_owned()),
            ("iteration".to_owned(), "0".to_owned()),
        ]);
        let first_identity = identity_fields(&first);
        first.insert("iteration".to_owned(), "1".to_owned());
        assert_eq!(first_identity, identity_fields(&first));
        first.insert("role".to_owned(), "line_search_trial".to_owned());
        assert_ne!(first_identity, identity_fields(&first));
    }

    #[test]
    fn collector_retains_hierarchy_and_separates_roles_while_aggregating_calls() {
        let collector = ProfileCollector::default();
        collector.capture(|| {
            let _run = tracing::span!(
                target: TELEMETRY_TARGET,
                tracing::Level::INFO,
                "eqiora_phase",
                phase = "run",
                family = "transient_flow"
            )
            .entered();
            let _solve = tracing::span!(
                target: TELEMETRY_TARGET,
                tracing::Level::INFO,
                "eqiora_phase",
                phase = "solve",
                solve = 1
            )
            .entered();
            for step in 1..=2 {
                let _step = tracing::span!(
                    target: TELEMETRY_TARGET,
                    tracing::Level::INFO,
                    "eqiora_phase",
                    phase = "time_step",
                    step
                )
                .entered();
            }
            for role in ["initial_linearization", "line_search_trial"] {
                let _assembly = tracing::span!(
                    target: TELEMETRY_TARGET,
                    tracing::Level::INFO,
                    "eqiora_phase",
                    phase = "assembly",
                    role
                )
                .entered();
            }
        });
        let phases = collector.finish().phases;
        let steps = phases
            .iter()
            .find(|phase| phase.path == ["run", "solve", "time_step"])
            .expect("time-step aggregate");
        assert_eq!(steps.calls, 2);
        assert_eq!(steps.fields, BTreeMap::new());
        let assembly_roles = phases
            .iter()
            .filter(|phase| phase.path == ["run", "solve", "assembly"])
            .map(|phase| phase.fields["role"].as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            assembly_roles,
            ["initial_linearization", "line_search_trial"]
        );
        let events = collector.finish().events;
        let time_step_events = events
            .iter()
            .filter(|event| {
                event
                    .fields
                    .get("phase")
                    .is_some_and(|phase| phase == "time_step")
            })
            .count();
        assert_eq!(time_step_events, 2);
    }

    #[test]
    fn aggregate_only_phase_events_retain_one_representative() {
        let collector = ProfileCollector::default();
        collector.capture(|| {
            for _ in 0..10_000 {
                let _backend = tracing::span!(
                    target: TELEMETRY_TARGET,
                    tracing::Level::INFO,
                    "eqiora_phase",
                    phase = "assembly_local_evaluation",
                    backend = "fixed-domain-mini"
                )
                .entered();
            }
        });
        let data = collector.finish();
        assert_eq!(data.phases.len(), 1);
        assert_eq!(data.phases[0].calls, 10_000);
        assert_eq!(data.events.len(), 1);
    }

    #[test]
    fn summary_names_units_aggregate_mean_and_call_count() {
        let data = ProfileData {
            phases: vec![ProfilePhaseData {
                path: vec!["run".to_owned(), "nonlinear_iteration".to_owned()],
                fields: BTreeMap::from([("nonlinear_solver".to_owned(), "newton".to_owned())]),
                calls: 20,
                inclusive: Duration::from_micros(18_651_461),
                self_time: Duration::from_micros(10_000_000),
            }],
            events: Vec::new(),
        };
        let summary = format_summary(&data);
        assert!(summary.contains("total inclusive"));
        assert!(summary.contains("self"));
        assert!(summary.contains("mean per call"));
        assert!(summary.contains("18.651461 s"));
        assert!(summary.contains("10.000000 s"));
        assert!(summary.contains("20 calls"));
        assert!(summary.contains("0.932573 s/call"));
        assert!(!summary.contains('×'));
    }
}
