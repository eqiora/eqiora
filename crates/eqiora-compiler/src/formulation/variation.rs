//! Functional derivation binds local jets to exact source and boundary identities.
mod local;

use super::*;
use eqiora_schema::kernel::pure_operator::{CalculusNode, CalculusNodeId, PureOperatorDefinition};
use eqiora_schema::kernel::typing::{ExpressionType, RootContract, SpatialSupport, TypedResidual};
use eqiora_schema::kernel::{ObservableMeasure, ObservableReduction, SymbolRef};

struct Request<'a> {
    wrt: &'a Expr,
    direction: &'a Expr,
    holding: &'a Expr,
}

fn requests(
    mut arguments: &eqiora_lang::CallArguments,
) -> Result<(&Expr, Vec<Request<'_>>), &'static str> {
    let mut requests = Vec::new();
    loop {
        if requests.len() == 2 {
            return Err("only first and second functional variations are admitted");
        }
        let (positional, named) = arguments.parts();
        let [functional] = positional else {
            return Err("variation requires one exact Observable");
        };
        let options = named
            .iter()
            .map(|option| (option.name(), option.value()))
            .collect::<BTreeMap<_, _>>();
        if named.len() != 3
            || options.len() != 3
            || !["wrt", "direction", "holding"]
                .iter()
                .all(|key| options.contains_key(key))
        {
            return Err("variation requires exactly wrt, direction and holding");
        }
        requests.push(Request {
            wrt: options["wrt"],
            direction: options["direction"],
            holding: options["holding"],
        });
        if let ExprKind::Call {
            callee,
            arguments: inner,
        } = functional.kind()
            && callee.as_str() == "variation"
        {
            arguments = inner;
        } else {
            requests.reverse();
            return Ok((functional, requests));
        }
    }
}

impl ExpressionContext<'_> {
    pub(super) fn compile_variation(
        &mut self,
        expression: &Expr,
        arguments: &eqiora_lang::CallArguments,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        let fail = |message| error(self.file, expression.range(), message);
        let (functional, requests) = requests(arguments).map_err(fail)?;
        let name = |value: &Expr| match value.kind() {
            ExprKind::Name(name) => Ok(name.clone()),
            _ => Err(fail("variation binding must be one exact local name")),
        };
        let functional_name = name(functional)?;
        let wrt_name = name(requests[0].wrt)?;
        let wrt = resolve_symbol(self.file, expression.range(), &wrt_name, self.symbols)?
            .downcast::<kinds::Field>()
            .ok_or_else(|| fail("variation wrt must be a Field"))?;
        let Some(KernelNode::Field(field)) = self.index.nodes.get(&wrt.erase()).copied() else {
            return Err(fail("variation Field is unavailable"));
        };
        let mut directions = Vec::new();
        for request in &requests {
            if resolve_symbol(
                self.file,
                request.wrt.range(),
                &name(request.wrt)?,
                self.symbols,
            )? != wrt.erase()
            {
                return Err(fail("ordered variations must select the same exact Field"));
            }
            let direction = name(request.direction)?;
            let Some((trial, dimension)) = self.tests.get(direction.as_str()) else {
                return Err(fail(
                    "variation direction must be an explicitly declared test",
                ));
            };
            if resolve_symbol(self.file, expression.range(), trial, self.symbols)? != wrt.erase()
                || *dimension != field.dimension()
            {
                return Err(fail(
                    "variation direction must have the selected Field identity and dimension",
                ));
            }
            if directions.contains(&direction) {
                return Err(fail(
                    "second variation requires an independent named direction",
                ));
            }
            directions.push(direction);
        }
        let functional_id = resolve_symbol(
            self.file,
            expression.range(),
            &functional_name,
            self.symbols,
        )?;
        let Some(KernelNode::Observable(functional)) =
            self.index.nodes.get(&functional_id).copied()
        else {
            return Err(fail("variation requires an authored Observable"));
        };
        let ObservableReduction::SpatialIntegral {
            domain,
            measure: ObservableMeasure::Volume,
        } = functional.reduction()
        else {
            return Err(fail(
                "initial functional variation requires a fixed volume integral",
            ));
        };
        if Some(domain) != self.relation_domain
            || self.index.defined_on.get(&wrt.erase()) != Some(&domain.erase())
        {
            return Err(fail(
                "functional, varied Field and Formulation must share the exact Domain",
            ));
        }
        let mut required = std::collections::BTreeSet::new();
        for node in functional.expression().nodes() {
            let id = match node {
                eqiora_schema::kernel::ExprNode::Symbol(SymbolRef::Field(id)) => Some(id.erase()),
                eqiora_schema::kernel::ExprNode::Symbol(SymbolRef::Parameter(id)) => {
                    Some(id.erase())
                }
                _ => None,
            };
            if let Some(id) = id.filter(|id| *id != wrt.erase()) {
                required.insert(id);
            }
        }
        for request in &requests {
            let ExprKind::Tuple(held) = request.holding.kind() else {
                return Err(fail(
                    "holding must be an explicit tuple of independent bindings",
                ));
            };
            let mut holding = std::collections::BTreeSet::new();
            for value in held {
                let id = resolve_symbol(self.file, value.range(), &name(value)?, self.symbols)?;
                if id == wrt.erase() || !holding.insert(id) {
                    return Err(fail("holding has a repeated or varied binding"));
                }
            }
            if holding != required {
                return Err(fail(
                    "holding must name exactly the other independent Field and Parameter bindings",
                ));
            }
        }
        let support = SpatialSupport::Volume {
            domain: domain.erase(),
            dimensions: self.ambient_dimension,
        };
        let typed_density = TypedResidual::infer(
            functional.expression().clone(),
            Some(support.clone()),
            RootContract::Observable,
            |symbol| match symbol {
                SymbolRef::Field(id) => match self.index.nodes.get(&id.erase()).copied() {
                    Some(KernelNode::Field(value))
                        if self.index.defined_on.get(&id.erase()) == Some(&domain.erase()) =>
                    {
                        Ok(ExpressionType::new(
                            value.value_type().clone(),
                            Some(support.clone()),
                        ))
                    }
                    _ => Err(()),
                },
                SymbolRef::Parameter(id) => match self.index.nodes.get(&id.erase()).copied() {
                    Some(KernelNode::Parameter(value)) => {
                        Ok(ExpressionType::new(value.value_type().clone(), None))
                    }
                    _ => Err(()),
                },
                _ => Err(()),
            },
        )
        .map_err(|_| fail("functional density has incompatible types or supports"))?;
        functional.validate_type(
            typed_density
                .node_type(typed_density.expression().roots()[0])
                .expect("typed root"),
            Some(&support),
        )?;
        let derived = local::derive(&typed_density, wrt, requests.len() as u8)?;
        let inputs = derived
            .inputs
            .iter()
            .map(|(input, _)| {
                self.variation_input(input, &derived.inputs, &directions, wrt, domain)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let integrand = render(
            &derived.definition,
            derived.definition.root(),
            &inputs,
            &mut 65536,
            0,
        )?;
        let value = typed(
            AuthoredFormExpressionKind::Integrate {
                domain,
                integrand: Box::new(integrand),
            },
            functional.value_type().dimension(),
            ValueShape::scalar(),
            None,
        );
        let result = typed(
            AuthoredFormExpressionKind::Variation {
                functional: functional.id(),
                wrt,
                directions: directions.clone(),
                holding: required.into_iter().collect(),
                value: Box::new(value),
            },
            functional.value_type().dimension(),
            ValueShape::scalar(),
            None,
        );
        self.used_tests.extend(directions);
        Ok(result)
    }

    fn variation_input(
        &self,
        input: &local::Input,
        inputs: &[(local::Input, DimExponents)],
        directions: &[String],
        wrt: Id<kinds::Field>,
        domain: Id<kinds::Domain>,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        let invalid = || wire::rejection("functional input binding is inconsistent");
        let (source, direction) = match input {
            local::Input::Direction { input, order } => (
                &inputs.get(*input).ok_or_else(invalid)?.0,
                Some(
                    directions
                        .get(usize::from(*order) - 1)
                        .ok_or_else(invalid)?,
                ),
            ),
            _ => (input, None),
        };
        let (mut value, indices) = match source {
            local::Input::Value(SymbolRef::Field(id), indices)
            | local::Input::Gradient(id, indices) => {
                let Some(KernelNode::Field(field)) = self.index.nodes.get(&id.erase()).copied()
                else {
                    return Err(invalid());
                };
                if direction.is_some() && *id != wrt {
                    return Err(invalid());
                }
                let kind = match direction {
                    Some(name) => AuthoredFormExpressionKind::Direction {
                        name: name.clone(),
                        trial: *id,
                    },
                    None => AuthoredFormExpressionKind::Field(*id),
                };
                let mut value = typed(kind, field.dimension(), field.shape().clone(), Some(domain));
                if matches!(source, local::Input::Gradient(..)) {
                    let mut axes = field
                        .shape()
                        .extents()
                        .iter()
                        .map(|extent| extent.get())
                        .collect::<Vec<_>>();
                    axes.push(u32::try_from(self.ambient_dimension).map_err(|_| invalid())?);
                    value = typed(
                        AuthoredFormExpressionKind::Gradient(Box::new(value)),
                        field
                            .dimension()
                            .div(length_dimension())
                            .ok_or_else(invalid)?,
                        ValueShape::new(axes).map_err(|_| invalid())?,
                        Some(domain),
                    );
                }
                (value, indices.as_slice())
            }
            local::Input::Value(SymbolRef::Parameter(id), indices) => {
                let Some(KernelNode::Parameter(parameter)) =
                    self.index.nodes.get(&id.erase()).copied()
                else {
                    return Err(invalid());
                };
                (
                    typed(
                        AuthoredFormExpressionKind::Parameter(*id),
                        parameter.value_type().dimension(),
                        parameter.value_type().shape().clone(),
                        None,
                    ),
                    indices.as_slice(),
                )
            }
            local::Input::Coordinate(axis) => {
                return Ok(typed(
                    AuthoredFormExpressionKind::Coordinate(*axis),
                    length_dimension(),
                    ValueShape::scalar(),
                    Some(domain),
                ));
            }
            _ => return Err(invalid()),
        };
        if !indices.is_empty() {
            let dimension = value.dimension;
            let support = value.support;
            value = typed(
                AuthoredFormExpressionKind::Component {
                    value: Box::new(value),
                    indices: indices.to_vec(),
                },
                dimension,
                ValueShape::scalar(),
                support,
            );
        }
        Ok(value)
    }
}

fn render(
    definition: &PureOperatorDefinition,
    id: CalculusNodeId,
    inputs: &[AuthoredFormExpression],
    remaining: &mut usize,
    depth: usize,
) -> Result<AuthoredFormExpression, Diagnostic> {
    if depth > 96 || *remaining == 0 {
        return Err(wire::rejection(
            "derived functional projection exceeds its tree bound",
        ));
    }
    *remaining -= 1;
    let invalid = || wire::rejection("derived local calculus has an incompatible type");
    let mut child = |id| render(definition, id, inputs, remaining, depth + 1);
    Ok(match &definition.nodes()[id.index() as usize] {
        CalculusNode::FormalComponent { formal, .. } => inputs[usize::from(*formal)].clone(),
        CalculusNode::Rational { value, dimension } => typed(
            AuthoredFormExpressionKind::Rational(*value),
            *dimension,
            ValueShape::scalar(),
            None,
        ),
        CalculusNode::Differentiated { value, .. } | CalculusNode::BoundInput(value) => {
            child(*value)?
        }
        CalculusNode::Neg(value) => {
            let value = child(*value)?;
            typed(
                AuthoredFormExpressionKind::Neg(Box::new(value.clone())),
                value.dimension,
                value.shape,
                value.support,
            )
        }
        CalculusNode::Add(left, right) | CalculusNode::Mul(left, right) => {
            let left = child(*left)?;
            let right = child(*right)?;
            let multiply = matches!(
                definition.nodes()[id.index() as usize],
                CalculusNode::Mul(..)
            );
            let dimension = if multiply {
                left.dimension.mul(right.dimension).ok_or_else(invalid)?
            } else {
                if left.dimension != right.dimension {
                    return Err(invalid());
                }
                left.dimension
            };
            if left.support.is_some() && right.support.is_some() && left.support != right.support {
                return Err(invalid());
            }
            let support = left.support.or(right.support);
            typed(
                binary(
                    if multiply {
                        BinaryOp::Mul
                    } else {
                        BinaryOp::Add
                    },
                    left,
                    right,
                ),
                dimension,
                ValueShape::scalar(),
                support,
            )
        }
        _ => return Err(invalid()),
    })
}
