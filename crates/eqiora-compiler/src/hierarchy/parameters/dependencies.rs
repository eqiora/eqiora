use super::*;

pub(super) struct ExpressionDefinition<'a> {
    pub(super) expression: Option<&'a Expr>,
    pub(super) dependencies: BTreeMap<String, TextRange>,
    pub(super) valid: bool,
}

pub(super) struct ExpressionCycle {
    pub(super) members: Vec<String>,
    pub(super) path: Vec<String>,
    pub(super) range: TextRange,
}

pub(super) fn collect_expression_dependencies(
    file: &str,
    expression: &Expr,
    contains: impl Fn(&str) -> bool,
    context: ExpressionContext<'_>,
) -> (BTreeMap<String, TextRange>, Vec<Diagnostic>) {
    let mut dependencies = BTreeMap::new();
    let mut diagnostics = Vec::new();
    let mut pending = vec![expression];
    while let Some(expression) = pending.pop() {
        if expression.resolved_enum().is_some() {
            continue;
        }
        if expression.resolved_nominal().is_some() {
            if let Err(error) = crate::nominal::literal(file, expression) {
                diagnostics.push(error);
            }
            continue;
        }
        match expression.kind() {
            ExprKind::Reduction { .. } => diagnostics.push(source_error(codes::LANGUAGE_TYPE_ERROR,file,expression.range(),"finite reductions are admitted in Relations and runtime let aliases, not Parameter or extent initializers")),
            ExprKind::Boolean(_) | ExprKind::Number(_) | ExprKind::Quantity { .. } => {}
            ExprKind::Name(name) => {
                if contains(name) {
                    dependencies
                        .entry(name.to_owned())
                        .and_modify(|range| {
                            if range_key(expression.range()) < range_key(*range) {
                                *range = expression.range();
                            }
                        })
                        .or_insert(expression.range());
                } else {
                    diagnostics.push(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        expression.range(),
                        context.unknown_name_message(name),
                    ));
                }
            }
            ExprKind::Path(path)
                if (crate::math::constant(path).is_some() || path.as_str() == "math.i") => {}
            ExprKind::Path(path) if contains(path.as_str()) => {
                dependencies.entry(path.as_str().to_owned()).or_insert(path.range());
            }
            ExprKind::Path(path) => diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                path.range(),
                context.qualified_name_message(path),
            )),
            // Clock identity is resolved separately during typed evaluation, never
            // as an edge in the Parameter default dependency graph.
            ExprKind::Call { callee, .. } if callee.as_str() == "period" => {}
            ExprKind::Call { callee, arguments } if callee.as_str() == "tensor_value" => {
                if let Some((_, components)) = super::tensor_values::arguments(arguments) {
                    pending.push(components);
                } else {
                    diagnostics.push(source_error(codes::LANGUAGE_TYPE_ERROR, file, expression.range(),
                        "tensor_value requires exactly named frame and components arguments"));
                }
            }
            ExprKind::Call { callee, arguments }
                if callee.as_str() == "math.complex"
                    || crate::math::unary_function(callee.as_str()).is_some()
                    || crate::math::piecewise::arity(callee.as_str()).is_some()
                    || crate::lower::IntegerBuiltin::named(callee.as_str()).is_some() =>
            {
                if let Some(arguments) = arguments.positional() {
                    pending.extend(arguments);
                } else {
                    diagnostics.push(source_error(codes::LANGUAGE_TYPE_ERROR, file, expression.range(),
                        "static builtins require positional arguments"));
                }
            }
            ExprKind::Call { callee, .. } => diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                expression.range(),
                context.call_message(callee.as_str()),
            )),
            ExprKind::Case { value, arms } => {
                pending.push(value.as_ref());
                pending.extend(arms.iter().map(eqiora_lang::CaseArm::value));
            }
            ExprKind::Select { condition, then_value, else_value } => {
                pending.extend([condition.as_ref(), then_value.as_ref(), else_value.as_ref()]);
            }
            ExprKind::Array(elements) => pending.extend(elements),
            ExprKind::Index { value, index } => pending.extend([value.as_ref(), index.as_ref()]),
            ExprKind::Slice { value, lower, upper } => pending.extend([value.as_ref(), lower.as_ref(), upper.as_ref()]),
            ExprKind::Unary {
                op: UnaryOp::Neg | UnaryOp::Not,
                value,
            } => pending.push(value),
            ExprKind::Binary { left, right, .. } => {
                pending.push(right);
                pending.push(left);
            }
            _ => diagnostics.push(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                file,
                expression.range(),
                context.unsupported_message(),
            )),
        }
    }
    stable_sort(&mut diagnostics);
    (dependencies, diagnostics)
}

pub(super) fn expression_cycles(
    defaults: &BTreeMap<String, ExpressionDefinition<'_>>,
) -> Vec<ExpressionCycle> {
    let adjacency = defaults
        .iter()
        .map(|(name, parameter)| {
            let dependencies = parameter
                .dependencies
                .keys()
                .filter(|dependency| defaults.contains_key(*dependency))
                .cloned()
                .collect::<Vec<_>>();
            (name.clone(), dependencies)
        })
        .collect::<BTreeMap<_, _>>();
    let reverse = reverse_adjacency(&adjacency);
    let mut visited = BTreeSet::new();
    let mut finished = Vec::with_capacity(adjacency.len());
    for root in adjacency.keys() {
        if !visited.insert(root.clone()) {
            continue;
        }
        let mut stack = vec![(root.clone(), 0_usize)];
        while !stack.is_empty() {
            let child = {
                let (node, next) = stack.last_mut().expect("nonempty DFS stack");
                let neighbors = &adjacency[node];
                if *next < neighbors.len() {
                    let child = neighbors[*next].clone();
                    *next += 1;
                    Some(child)
                } else {
                    None
                }
            };
            if let Some(child) = child {
                if visited.insert(child.clone()) {
                    stack.push((child, 0));
                }
            } else {
                let (node, _) = stack.pop().expect("nonempty DFS stack");
                finished.push(node);
            }
        }
    }

    let mut assigned = BTreeSet::new();
    let mut components = Vec::new();
    for root in finished.into_iter().rev() {
        if !assigned.insert(root.clone()) {
            continue;
        }
        let mut component = Vec::new();
        let mut pending = vec![root];
        while let Some(node) = pending.pop() {
            component.push(node.clone());
            for parent in reverse[&node].iter().rev() {
                if assigned.insert(parent.clone()) {
                    pending.push(parent.clone());
                }
            }
        }
        component.sort();
        let is_cycle = component.len() > 1
            || adjacency[&component[0]]
                .iter()
                .any(|dependency| dependency == &component[0]);
        if is_cycle {
            components.push(canonical_expression_cycle(component, &adjacency, defaults));
        }
    }
    components.sort_by(|left, right| left.members.cmp(&right.members));
    components
}

fn reverse_adjacency(adjacency: &BTreeMap<String, Vec<String>>) -> BTreeMap<String, Vec<String>> {
    let mut reverse = adjacency
        .keys()
        .map(|name| (name.clone(), Vec::new()))
        .collect::<BTreeMap<_, _>>();
    for (name, dependencies) in adjacency {
        for dependency in dependencies {
            reverse
                .get_mut(dependency)
                .expect("dependency is an active default")
                .push(name.clone());
        }
    }
    for parents in reverse.values_mut() {
        parents.sort();
    }
    reverse
}

fn canonical_expression_cycle(
    members: Vec<String>,
    adjacency: &BTreeMap<String, Vec<String>>,
    defaults: &BTreeMap<String, ExpressionDefinition<'_>>,
) -> ExpressionCycle {
    let member_set = members.iter().cloned().collect::<BTreeSet<_>>();
    let start = members.first().expect("cyclic component is nonempty");
    let next = adjacency[start]
        .iter()
        .find(|dependency| member_set.contains(*dependency))
        .expect("a strongly connected component has an internal edge");
    let range = defaults[start].dependencies[next];
    let path = if next == start {
        vec![start.clone(), start.clone()]
    } else {
        let mut parents = BTreeMap::<String, String>::new();
        let mut visited = BTreeSet::from([next.clone()]);
        let mut pending = VecDeque::from([next.clone()]);
        while let Some(node) = pending.pop_front() {
            if &node == start {
                break;
            }
            for child in &adjacency[&node] {
                if member_set.contains(child) && visited.insert(child.clone()) {
                    parents.insert(child.clone(), node.clone());
                    pending.push_back(child.clone());
                }
            }
        }
        let mut tail = vec![start.clone()];
        let mut cursor = start;
        while cursor != next {
            cursor = &parents[cursor];
            tail.push(cursor.clone());
        }
        tail.reverse();
        let mut path = vec![start.clone()];
        path.extend(tail);
        path
    };
    ExpressionCycle {
        members,
        path,
        range,
    }
}

pub(super) fn expression_evaluation_order(
    defaults: &BTreeMap<String, ExpressionDefinition<'_>>,
    cyclic: &BTreeSet<String>,
) -> Vec<String> {
    let mut indegree = defaults
        .keys()
        .filter(|name| !cyclic.contains(*name))
        .map(|name| (name.clone(), 0_usize))
        .collect::<BTreeMap<_, _>>();
    let mut dependents = indegree
        .keys()
        .map(|name| (name.clone(), Vec::new()))
        .collect::<BTreeMap<_, _>>();
    for (name, parameter) in defaults {
        if cyclic.contains(name) {
            continue;
        }
        for dependency in parameter.dependencies.keys() {
            if defaults.contains_key(dependency) && !cyclic.contains(dependency) {
                *indegree.get_mut(name).expect("acyclic default is indexed") += 1;
                dependents
                    .get_mut(dependency)
                    .expect("acyclic dependency is indexed")
                    .push(name.clone());
            }
        }
    }
    for values in dependents.values_mut() {
        values.sort();
    }
    let mut ready = indegree
        .iter()
        .filter_map(|(name, &count)| (count == 0).then_some(name.clone()))
        .collect::<BTreeSet<_>>();
    let mut order = Vec::with_capacity(indegree.len());
    while let Some(name) = ready.pop_first() {
        order.push(name.clone());
        for dependent in &dependents[&name] {
            let count = indegree
                .get_mut(dependent)
                .expect("acyclic dependent is indexed");
            *count -= 1;
            if *count == 0 {
                ready.insert(dependent.clone());
            }
        }
    }
    order
}

fn range_key(range: TextRange) -> (u32, u32) {
    (range.start(), range.end())
}
