//! Functional derivation binds local jets to exact source and boundary identities.
mod composite;
mod local;
mod replay;

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
        let derived = composite::derive(functional.id(), wrt, &directions, &mut |id| {
            self.typed_functional(id)
        })?;
        if self.relation_domain.map(Id::erase) != Some(derived.volume)
            || self.index.defined_on.get(&wrt.erase()) != Some(&derived.volume)
        {
            return Err(fail(
                "functional, varied Field and Formulation must share the exact parent volume",
            ));
        }
        let required = derived.holding;
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
        let value = derived.value;
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
    fn typed_functional(
        &self,
        id: Id<kinds::Observable>,
    ) -> Result<(eqiora_schema::kernel::ObservableDef, TypedResidual<RawId>), Diagnostic> {
        let fail = || wire::rejection("functional density has incompatible types or supports");
        let Some(KernelNode::Observable(functional)) = self.index.nodes.get(&id.erase()).copied()
        else {
            return Err(wire::rejection("variation requires an authored Observable"));
        };
        let support = match functional.reduction() {
            ObservableReduction::Value => None,
            ObservableReduction::SpatialIntegral {
                input,
                domain,
                measure,
            } if input == domain => Some(match measure {
                ObservableMeasure::Volume => SpatialSupport::Volume {
                    domain: domain.erase(),
                    dimensions: self.ambient_dimension,
                },
                ObservableMeasure::Boundary => SpatialSupport::Boundary {
                    domain: domain.erase(),
                    parent: *self
                        .index
                        .boundary_of
                        .get(&domain.erase())
                        .ok_or_else(fail)?,
                    dimensions: self.ambient_dimension,
                },
            }),
            ObservableReduction::SpatialIntegral { .. } => {
                return Err(wire::rejection(
                    "local variation requires a full fixed-domain integral",
                ));
            }
        };
        let volume = support
            .as_ref()
            .map(|support| support.parent().copied().unwrap_or(*support.domain()));
        let typed = TypedResidual::infer(
            functional.expression().clone(),
            support.clone(),
            RootContract::Observable,
            |symbol| match symbol {
                SymbolRef::Coordinate {
                    support: declared,
                    factor,
                    axis,
                } => {
                    let coordinate_support = match support.as_ref() {
                        Some(current) if *current.domain() == declared.erase() => current.clone(),
                        Some(SpatialSupport::Boundary {
                            parent, dimensions, ..
                        }) if *parent == declared.erase() => SpatialSupport::Volume {
                            domain: *parent,
                            dimensions: *dimensions,
                        },
                        _ => return Err(()),
                    };
                    eqiora_schema::kernel::typing::coordinate(
                        &factor.erase(),
                        axis,
                        Some(&coordinate_support),
                    )
                    .map_err(|_| ())
                }
                SymbolRef::Field(id) => match self.index.nodes.get(&id.erase()).copied() {
                    Some(KernelNode::Field(value))
                        if volume.is_some()
                            && self.index.defined_on.get(&id.erase()).copied() == volume =>
                    {
                        Ok(ExpressionType::new(
                            value.value_type().clone(),
                            Some(SpatialSupport::Volume {
                                domain: volume.expect("checked spatial volume"),
                                dimensions: self.ambient_dimension,
                            }),
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
                SymbolRef::Observable(id) => match self.index.nodes.get(&id.erase()).copied() {
                    Some(KernelNode::Observable(value)) => {
                        Ok(ExpressionType::new(value.value_type().clone(), None))
                    }
                    _ => Err(()),
                },
                _ => Err(()),
            },
        )
        .map_err(|_| fail())?;
        Ok((functional.clone(), typed))
    }
}

fn symbol_type(
    density: &TypedResidual<RawId>,
    symbol: SymbolRef,
) -> Result<&eqiora_core::ValueType, Diagnostic> {
    let dag = density.expression();
    let index = dag.nodes().iter().position(|node| matches!(node, eqiora_schema::kernel::ExprNode::Symbol(value) if *value == symbol))
        .ok_or_else(|| wire::rejection("functional input is absent from its typed density"))?;
    let ty = density
        .node_type(dag.node_id(index as u32).expect("existing node"))
        .ok_or_else(|| wire::rejection("functional input has no type"))?;
    Ok(&ty.value_type)
}

fn functional_support(
    density: &TypedResidual<RawId>,
    domain: Id<kinds::Domain>,
) -> Result<&SpatialSupport<RawId>, Diagnostic> {
    let root = density
        .node_type(density.expression().roots()[0])
        .ok_or_else(|| wire::rejection("functional density has no typed root"))?;
    match &root.support {
        Some(
            support @ (SpatialSupport::Volume { domain: actual, .. }
            | SpatialSupport::Boundary { domain: actual, .. }),
        ) if *actual == domain.erase() => Ok(support),
        _ => Err(wire::rejection(
            "functional density requires its exact fixed integration support",
        )),
    }
}

fn derive_value(
    density: &TypedResidual<RawId>,
    wrt: Id<kinds::Field>,
    directions: &[String],
    domain: Id<kinds::Domain>,
    dimension: DimExponents,
    remaining: &mut usize,
) -> Result<AuthoredFormExpression, Diagnostic> {
    functional_support(density, domain)?;
    let derived = local::derive(
        density,
        wrt,
        u8::try_from(directions.len())
            .map_err(|_| wire::rejection("too many variation directions"))?,
    )?;
    let inputs = derived
        .inputs
        .iter()
        .map(|(input, _)| variation_input(density, input, &derived.inputs, directions, wrt, domain))
        .collect::<Result<Vec<_>, _>>()?;
    let integrand = render(
        &derived.definition,
        derived.definition.root(),
        &inputs,
        remaining,
        0,
    )?;
    Ok(typed(
        AuthoredFormExpressionKind::Integrate {
            domain,
            integrand: Box::new(integrand),
        },
        dimension,
        ValueShape::scalar(),
        None,
    ))
}

fn variation_input(
    density: &TypedResidual<RawId>,
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
            let field = symbol_type(density, SymbolRef::Field(*id))?;
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
            let support = functional_support(density, domain)?;
            let parent = support
                .parent()
                .copied()
                .unwrap_or(domain.erase())
                .downcast::<kinds::Domain>()
                .ok_or_else(invalid)?;
            let mut value = typed(kind, field.dimension(), field.shape().clone(), Some(parent));
            if support.parent().is_some() {
                if matches!(source, local::Input::Gradient(..)) {
                    return Err(wire::rejection(
                        "surface energy currently requires Field traces, not gradient traces",
                    ));
                }
                value = typed(
                    AuthoredFormExpressionKind::Trace(Box::new(value)),
                    field.dimension(),
                    field.shape().clone(),
                    Some(domain),
                );
            }
            if matches!(source, local::Input::Gradient(..)) {
                let mut axes = field
                    .shape()
                    .extents()
                    .iter()
                    .map(|extent| extent.get())
                    .collect::<Vec<_>>();
                axes.push(
                    u32::try_from(
                        functional_support(density, domain)?
                            .ambient_dimensions()
                            .ok_or_else(invalid)?,
                    )
                    .map_err(|_| invalid())?,
                );
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
            let parameter = symbol_type(density, SymbolRef::Parameter(*id))?;
            (
                typed(
                    AuthoredFormExpressionKind::Parameter(*id),
                    parameter.dimension(),
                    parameter.shape().clone(),
                    None,
                ),
                indices.as_slice(),
            )
        }
        local::Input::Value(
            symbol @ SymbolRef::Coordinate {
                support,
                factor,
                axis,
            },
            indices,
        ) if indices.is_empty() => {
            return Ok(typed(
                AuthoredFormExpressionKind::Coordinate {
                    support: *support,
                    factor: *factor,
                    axis: *axis,
                },
                symbol_type(density, *symbol)?.dimension(),
                ValueShape::scalar(),
                Some(*support),
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
