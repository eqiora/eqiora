//! Pure process coefficients lowered through the shared typed component IR.
use super::invalid;
use crate::common_time::{CommonTimeState, TimeStateSpace};
use eqiora_core::{Diagnostic, Id, ValueLiteral, ValueType, entity::kinds};
use eqiora_graph::EdgeKind;
use eqiora_ir::{ComponentScalarRow, ComponentScalarization, ScalarSymbolCoordinate};
use eqiora_schema::kernel::{
    ActivationKind, ExprId, FieldRole, KernelNode, NoiseProcess, RelationDef, RelationMeaning,
    StochasticCalculus, SymbolRef,
    typing::{RootContract, TypedResidual},
};
use eqiora_sem::{Interpreter, KernelProgram, ReferenceConfig};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProcessProgram {
    kernel: KernelProgram,
    coordinates: Vec<ScalarSymbolCoordinate>,
    space: Arc<TimeStateSpace>,
    pub(super) noises: Vec<Noise>,
    pub(super) laws: Vec<Law>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Noise {
    pub(super) id: Id<kinds::Relation>,
    pub(super) process: NoiseProcess,
    pub(super) parameter: Coefficient,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Law {
    pub(super) id: Id<kinds::Relation>,
    pub(super) state: Range<usize>,
    pub(super) calculus: Option<StochasticCalculus>,
    pub(super) drift: Option<Coefficient>,
    pub(super) diffusions: Vec<(Id<kinds::Relation>, Coefficient)>,
    pub(super) jumps: Vec<(Id<kinds::Relation>, Coefficient)>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Coefficient {
    pub(super) value_type: ValueType,
    rows: Vec<(ComponentScalarRow, Vec<Input>)>,
    state_dimension: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum Input {
    State(usize),
    Constant(f64),
    Time,
}

impl ProcessProgram {
    pub(crate) fn new(kernel: &KernelProgram) -> Result<Self, Diagnostic> {
        let mut coordinates = Vec::new();
        let mut time_coordinates = Vec::new();
        let mut fields = HashMap::new();
        for node in kernel.nodes() {
            match node {
                KernelNode::Field(field) => {
                    if field.role() != FieldRole::State
                        || kernel.edges().iter().any(|edge| {
                            edge.from() == field.id().erase() && edge.kind() == EdgeKind::DefinedOn
                        })
                    {
                        return Err(invalid(
                            "serial process execution requires finite owned States",
                        ));
                    }
                    let start = coordinates.len();
                    let field_coordinates = ScalarSymbolCoordinate::for_value(
                        SymbolRef::Field(field.id()),
                        field.value_type(),
                    )?;
                    let parts = if field.value_type().scalar_domain()
                        == eqiora_core::ScalarDomain::Complex
                    {
                        2
                    } else {
                        1
                    };
                    time_coordinates.extend(field_coordinates.iter().enumerate().map(
                        |(index, coordinate)| {
                            eqiora_core::TimeStateCoordinate::new(
                                field.id(),
                                0,
                                index / parts,
                                coordinate.is_imaginary(),
                            )
                        },
                    ));
                    coordinates.extend(field_coordinates);
                    fields.insert(field.id(), start..coordinates.len());
                }
                KernelNode::Activation(activation)
                    if !matches!(activation.kind(), ActivationKind::Continuous) =>
                {
                    return Err(invalid(
                        "serial process execution does not admit hybrid activations",
                    ));
                }
                KernelNode::Port(_) | KernelNode::Connection(_) => {
                    return Err(invalid(
                        "serial process execution does not admit unresolved ports or connections",
                    ));
                }
                _ => {}
            }
        }
        let bindings = coordinates
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, coordinate)| (coordinate, index))
            .collect::<HashMap<_, _>>();
        let mut noises = Vec::new();
        let mut laws = Vec::new();
        for node in kernel.nodes() {
            let KernelNode::Relation(relation) = node else {
                continue;
            };
            if relation.is_initial() {
                continue;
            }
            let coefficient = |root| Coefficient::new(kernel, relation, root, &bindings);
            match relation.meaning() {
                RelationMeaning::Noise {
                    process, parameter, ..
                } => {
                    // KernelProgram already validates independence between
                    // distinct groups; a single group needs no such declaration.
                    noises.push(Noise {
                        id: relation.id(),
                        process: *process,
                        parameter: coefficient(*parameter)?,
                    });
                }
                RelationMeaning::Stochastic {
                    target,
                    calculus,
                    drift,
                    diffusions,
                    jumps,
                } => {
                    let state = fields.remove(target).ok_or_else(|| {
                        invalid("process law must uniquely own an admitted State")
                    })?;
                    let terms = |terms: &[(Id<kinds::Relation>, ExprId)]| {
                        terms
                            .iter()
                            .map(|(group, root)| Ok((*group, coefficient(*root)?)))
                            .collect::<Result<Vec<_>, Diagnostic>>()
                    };
                    laws.push(Law {
                        id: relation.id(),
                        state,
                        calculus: *calculus,
                        drift: drift.map(coefficient).transpose()?,
                        diffusions: terms(diffusions)?,
                        jumps: terms(jumps)?,
                    });
                }
                _ => {
                    return Err(invalid(
                        "serial process execution requires explicit stochastic evolution for every State",
                    ));
                }
            }
        }
        if laws.is_empty() || !fields.is_empty() {
            return Err(invalid(
                "serial process execution requires exactly one law per State",
            ));
        }
        Ok(Self {
            kernel: kernel.clone(),
            space: Arc::new(TimeStateSpace::new(
                &eqiora_artifact::ModelEnvelope::from_program(kernel)?,
                time_coordinates,
                b"stochastic-law-coordinates/v1",
            )?),
            coordinates,
            noises,
            laws,
        })
    }

    pub(crate) fn state_space(&self) -> &Arc<TimeStateSpace> {
        &self.space
    }

    pub(crate) fn initialize(&self, time: f64) -> Result<CommonTimeState, Diagnostic> {
        let initial = Interpreter::new()
            .initialize(&self.kernel, time, ReferenceConfig::new(0., 1.)?)
            .map_err(|errors| {
                errors
                    .into_iter()
                    .next()
                    .expect("initialization diagnostic")
            })?;
        let mut values = HashMap::new();
        for (field, value) in initial.fields() {
            let field = Id::<kinds::Field>::from_ulid(field.ulid());
            values.extend(value_coordinates(SymbolRef::Field(field), value)?);
        }
        let values = self
            .coordinates
            .iter()
            .map(|coordinate| {
                values
                    .get(coordinate)
                    .copied()
                    .ok_or_else(|| invalid("initial solve omitted a process State coordinate"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.accept_state(time, values, "initial")
    }

    pub(super) fn accept_state(
        &self,
        time: f64,
        values: Vec<f64>,
        source_kind: &'static str,
    ) -> Result<CommonTimeState, Diagnostic> {
        CommonTimeState::new(self.space.clone(), time, values, source_kind)
    }

    pub(crate) fn state_from_bytes(&self, bytes: &[u8]) -> Result<CommonTimeState, Diagnostic> {
        CommonTimeState::from_bytes(bytes, self.space.clone())
    }

    pub(super) fn validate_state(&self, state: &CommonTimeState) -> Result<(), Diagnostic> {
        if state.state_space_identity() != self.space.identity() {
            return Err(invalid(
                "process State belongs to a different exact state space",
            ));
        }
        Ok(())
    }
}

impl Coefficient {
    fn new(
        kernel: &KernelProgram,
        relation: &RelationDef,
        root: ExprId,
        states: &HashMap<ScalarSymbolCoordinate, usize>,
    ) -> Result<Self, Diagnostic> {
        // These are coefficient values, never pairs of equation sides or residuals.
        let typed = TypedResidual::infer(
            relation.expression().clone(),
            None,
            |domain| kernel.spatial_support(domain).cloned(),
            RootContract::ValueRoots,
            |symbol| kernel.symbol_type(symbol).ok_or(()),
        )
        .map_err(|_| invalid("process coefficient lost its exact Model type"))?;
        let value_type = typed
            .node_type(root)
            .ok_or_else(|| invalid("missing process coefficient root"))?
            .value_type
            .clone();
        let lowered = ComponentScalarization::lower_selected(&typed, &[root])?;
        let mut parameters = HashMap::new();
        for node in kernel.nodes() {
            if let KernelNode::Parameter(parameter) = node
                && parameter.value().components().is_some()
            {
                parameters.extend(value_coordinates(
                    SymbolRef::Parameter(parameter.id()),
                    parameter.value(),
                )?);
            }
        }
        let rows = lowered
            .rows()
            .iter()
            .map(|row| {
                let inputs = row
                    .symbols()
                    .iter()
                    .map(|coordinate| {
                        if let Some(&index) = states.get(coordinate) {
                            return Ok(Input::State(index));
                        }
                        if let Some(&value) = parameters.get(coordinate) {
                            return Ok(Input::Constant(value));
                        }
                        if coordinate.symbol() == SymbolRef::Time {
                            return Ok(Input::Time);
                        }
                        Err(invalid(
                            "process coefficient has an unbound numerical coordinate",
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((row.clone(), inputs))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        Ok(Self {
            value_type,
            rows,
            state_dimension: states.len(),
        })
    }

    pub(super) fn constant(&self) -> bool {
        self.rows.iter().all(|(_, inputs)| {
            inputs
                .iter()
                .all(|input| matches!(input, Input::Constant(_)))
        })
    }

    pub(super) fn evaluate_constant(&self) -> Result<Vec<f64>, Diagnostic> {
        self.evaluate_inputs(|input| match input {
            Input::Constant(value) => Ok(*value),
            _ => Err(invalid(
                "a constant coefficient cannot depend on State or time",
            )),
        })
    }

    pub(super) fn evaluate(&self, state: &[f64], time: f64) -> Result<Vec<f64>, Diagnostic> {
        if state.len() != self.state_dimension
            || state.iter().any(|x| !x.is_finite())
            || !time.is_finite()
        {
            return Err(invalid(
                "process coefficients require a complete finite State and time",
            ));
        }
        self.evaluate_inputs(|input| {
            Ok(match input {
                Input::State(index) => state[*index],
                Input::Constant(value) => *value,
                Input::Time => time,
            })
        })
    }

    fn evaluate_inputs(
        &self,
        mut resolve: impl FnMut(&Input) -> Result<f64, Diagnostic>,
    ) -> Result<Vec<f64>, Diagnostic> {
        self.rows
            .iter()
            .map(|(row, inputs)| {
                let inputs = inputs
                    .iter()
                    .map(&mut resolve)
                    .collect::<Result<Vec<_>, _>>()?;
                row.evaluate(&inputs)
            })
            .collect()
    }
}

fn value_coordinates(
    symbol: SymbolRef,
    value: &ValueLiteral,
) -> Result<Vec<(ScalarSymbolCoordinate, f64)>, Diagnostic> {
    let coordinates = ScalarSymbolCoordinate::for_value(symbol, value.value_type())?;
    let complex = value.value_type().scalar_domain() == eqiora_core::ScalarDomain::Complex;
    let values = value
        .components()
        .ok_or_else(|| invalid("process coefficient requires numeric values"))?
        .flat_map(|(real, imaginary)| {
            [Some(real), complex.then_some(imaginary)]
                .into_iter()
                .flatten()
        });
    Ok(coordinates.into_iter().zip(values).collect())
}
