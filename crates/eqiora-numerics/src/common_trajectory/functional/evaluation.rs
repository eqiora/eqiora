//! Demand-driven Observable composition at one accepted ODE state.
use super::*;
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationTangent, ScalarOperatorIr};
use eqiora_schema::kernel::ExprNode;
use eqiora_time::{ParametricTimeSystem, TimeSystem};

pub(super) struct Directions {
    pub(super) state: Vec<f64>,
    pub(super) parameters: HashMap<Id<kinds::Parameter>, f64>,
}

#[derive(Clone)]
pub(super) struct Evaluation {
    pub(super) value: ValueLiteral,
    pub(super) tangent: Option<f64>,
    // Structural dependence, never inferred from a numerically zero product.
    active: bool,
}

pub(super) struct Context<'a, 'b> {
    owner: &'a TimeObservable<'b>,
    time: f64,
    state: &'a [f64],
    directions: Option<Directions>,
    rates: Option<Vec<f64>>,
    rate_direction: Option<Vec<f64>>,
    accepted: HashMap<Id<kinds::Observable>, Evaluation>,
    remaining: usize,
}

impl<'a, 'b> Context<'a, 'b> {
    pub(super) fn new(
        owner: &'a TimeObservable<'b>,
        time: f64,
        state: &'a [f64],
        directions: Option<Directions>,
    ) -> Self {
        Self {
            owner,
            time,
            state,
            directions,
            rates: None,
            rate_direction: None,
            accepted: HashMap::new(),
            remaining: 1_000_000,
        }
    }

    pub(super) fn evaluate(
        &mut self,
        id: Id<kinds::Observable>,
        depth: usize,
    ) -> Result<Evaluation, Diagnostic> {
        if let Some(value) = self.accepted.get(&id) {
            return Ok(value.clone());
        }
        if depth > 256 {
            return Err(invalid(
                "Observable composition exceeds its dependency depth bound",
            ));
        }
        let program = &self.owner.program;
        let typed = program.typed_observable(id).map_err(|errors| {
            errors
                .into_iter()
                .next()
                .expect("failed typing has a diagnostic")
        })?;
        self.remaining = self
            .remaining
            .checked_sub(typed.expression().nodes().len())
            .ok_or_else(|| invalid("Observable composition exceeds its expression work bound"))?;
        let input_types: HashMap<_, _> = typed
            .expression()
            .nodes()
            .iter()
            .zip(typed.node_types())
            .filter_map(|(node, ty)| match node {
                ExprNode::Symbol(symbol) => Some((*symbol, ty.value_type.clone())),
                _ => None,
            })
            .collect();
        let mut inputs = HashMap::new();
        let value = program.evaluate_observable_with_points(id, None, &mut |input, _point| {
            let eqiora_sem::EvaluationInput::Value(symbol) = input else {
                return Err(invalid(
                    "ODE Observable has no spatial point reconstruction",
                ));
            };
            let evaluation = if let SymbolRef::Observable(dependency) = symbol {
                self.evaluate(dependency, depth + 1)?
            } else {
                self.input(symbol, &input_types[&symbol])?
            };
            let value = evaluation.value.clone();
            inputs.insert(symbol, evaluation);
            Ok(value)
        })?;
        let mut evaluation = Evaluation {
            value,
            tangent: None,
            active: false,
        };
        if self.directions.is_some() {
            let operator = ScalarOperatorIr::lower(typed.expression())?;
            let mut values = Vec::new();
            let mut roles = Vec::new();
            let mut deltas = Vec::new();
            for symbol in operator.symbols() {
                if let Some(input) = inputs.get(symbol) {
                    values.push(input.value.clone());
                    if input.active {
                        evaluation.active = true;
                        roles.push(DifferentiationRole::Unknown);
                        deltas.push(input.tangent.ok_or_else(|| {
                            invalid("functional sensitivity requires real scalar inputs")
                        })?);
                        continue;
                    }
                } else {
                    // The canonical evaluator did not demand this slot. Preserve its
                    // type for IR admission, without evaluating an inactive dependency.
                    values.push(
                        ValueLiteral::from_real(input_types[symbol].clone(), 0.)
                            .map_err(|error| invalid(error.to_string()))?,
                    );
                }
                roles.push(DifferentiationRole::Frozen);
            }
            let linearized = operator.linearize_typed(&values, &roles)?;
            let mut product = [0.];
            linearized.jvp(RelationTangent::Unknown(&deltas), &mut product)?;
            evaluation.tangent = Some(product[0]);
        }
        self.accepted.insert(id, evaluation.clone());
        Ok(evaluation)
    }

    fn input(&mut self, symbol: SymbolRef, ty: &ValueType) -> Result<Evaluation, Diagnostic> {
        let mut active = false;
        let mut tangent = self.directions.as_ref().map(|_| 0.);
        let value = match symbol {
            SymbolRef::Parameter(id) => {
                active = true;
                tangent = self
                    .directions
                    .as_ref()
                    .map(|directions| directions.parameters.get(&id).copied().unwrap_or(0.));
                self.owner
                    .program
                    .typed_value(id.erase())
                    .cloned()
                    .ok_or_else(|| invalid("Observable Parameter value is unavailable"))?
            }
            SymbolRef::Field(_) | SymbolRef::Derivative(..) => {
                active = true;
                let complex = ty.scalar_domain() == eqiora_core::ScalarDomain::Complex;
                let mut components = Vec::new();
                for component in 0..ty
                    .shape()
                    .component_count()
                    .ok_or_else(|| invalid("Observable component count overflow"))?
                {
                    let real = self.component(symbol, component, false)?;
                    let imaginary = if complex {
                        self.component(symbol, component, true)?
                    } else {
                        0.
                    };
                    components.push((real, imaginary));
                }
                if let Some(directions) = &self.directions {
                    let (index, rate) = self.owner.coordinate(symbol)?;
                    tangent = Some(if rate {
                        self.rate_direction.as_ref().expect("demanded rate product")[index]
                    } else {
                        directions.state[index]
                    });
                }
                ValueLiteral::new(ty.clone(), components)
                    .map_err(|error| invalid(error.to_string()))?
            }
            SymbolRef::Time => ValueLiteral::from_real(ty.clone(), self.time)
                .map_err(|error| invalid(error.to_string()))?,
            _ => {
                return Err(invalid(
                    "Observable input is outside the admitted ODE profile",
                ));
            }
        };
        Ok(Evaluation {
            value,
            tangent,
            active,
        })
    }

    fn component(
        &mut self,
        symbol: SymbolRef,
        component: usize,
        imaginary: bool,
    ) -> Result<f64, Diagnostic> {
        let (index, rate) = self
            .owner
            .component_coordinate(symbol, component, imaginary)?;
        if !rate {
            return Ok(self.state[index]);
        }
        if self.rates.is_none() {
            let plan =
                self.owner.plan.as_ode().ok_or_else(|| {
                    invalid("stochastic trajectory has no classical time derivative")
                })?;
            let problem = eqiora_time::TimeProblem::new(
                plan.system(),
                plan.equation_class(),
                eqiora_time::InitialConditionPolicy::Provided,
                self.state.to_vec(),
            )?;
            self.rates = Some(problem.rate(self.time, self.state)?);
            if let Some(directions) = &self.directions {
                let system = plan.system();
                let mut delta = vec![0.; self.state.len()];
                system.rhs_jvp(self.time, self.state, &directions.state, &mut delta)?;
                let parameters: Vec<_> = system
                    .parameter_fields()
                    .iter()
                    .map(|id| directions.parameters.get(id).copied().unwrap_or(0.))
                    .collect();
                let mut direct = vec![0.; self.state.len()];
                system.rhs_parameter_jvp(self.time, self.state, &parameters, &mut direct)?;
                for (delta, direct) in delta.iter_mut().zip(direct) {
                    *delta += direct;
                }
                self.rate_direction = Some(delta);
            }
        }
        Ok(self.rates.as_ref().expect("demanded rate")[index])
    }
}
