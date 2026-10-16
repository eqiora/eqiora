use super::*;

#[derive(Debug, Default)]
pub(super) struct FlatPhysicalConnectionPlan {
    pub(super) emissions: BTreeMap<usize, Box<[RawId]>>,
    pub(super) consumed: BTreeSet<usize>,
}

#[derive(Debug)]
struct FlatPhysicalFragment {
    item_index: usize,
    topology: ConnectionFragment<RawId>,
}

pub(super) fn prepare_flat_physical_connections(
    file: &str,
    model: &LoweringModel,
    bindings: &BTreeMap<String, Binding>,
) -> Result<FlatPhysicalConnectionPlan, Vec<Diagnostic>> {
    let limits = ConnectionSetLimits::default();
    let mut fragments = Vec::new();
    for (item_index, item) in model.items.iter().enumerate() {
        let LoweringItem::Connection {
            syntax,
            ports,
            range,
        } = item
        else {
            continue;
        };
        if *syntax != ConnectionSyntax::Conserving {
            continue;
        }

        let mut resolved = Vec::new();
        let mut scalar_physical = true;
        for name in ports {
            let Some(Binding::Port(_, contract)) = bindings.get(name) else {
                scalar_physical = false;
                break;
            };
            let contract = resolve_port_contract(file, *range, contract, bindings)
                .map_err(|diagnostic| vec![diagnostic])?;
            scalar_physical &= matches!(contract, ResolvedPortContract::ScalarPhysical { .. });
            resolved.push(contract);
        }
        if !scalar_physical || resolved.is_empty() {
            continue;
        }

        let mut isolated_membership = BTreeSet::new();
        let (_, ports) = lower_connection(
            file,
            *range,
            *syntax,
            ports,
            Id::new(),
            bindings,
            &mut isolated_membership,
        )
        .map_err(|diagnostic| vec![diagnostic])?;
        let topology = ConnectionFragment::try_new(ports, limits).map_err(|error| {
            vec![source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                *range,
                format!("invalid scalar physical Connection fragment: {error}"),
            )]
        })?;
        fragments.push(FlatPhysicalFragment {
            item_index,
            topology,
        });
    }

    let topologies = fragments
        .iter()
        .map(|fragment| fragment.topology.clone())
        .collect::<Vec<_>>();
    let normalized = normalize_connection_sets(&topologies, limits).map_err(|error| {
        vec![source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            model.range,
            format!("cannot normalize scalar physical Connection fragments: {error}"),
        )]
    })?;
    let mut representatives = vec![usize::MAX; normalized.sets().len()];
    for (fragment, &set_index) in fragments.iter().zip(normalized.fragment_sets()) {
        representatives[set_index] = representatives[set_index].min(fragment.item_index);
    }

    let mut plan = FlatPhysicalConnectionPlan::default();
    for fragment in &fragments {
        plan.consumed.insert(fragment.item_index);
    }
    for (set_index, set) in normalized.sets().iter().enumerate() {
        let representative = representatives[set_index];
        debug_assert_ne!(representative, usize::MAX);
        plan.emissions
            .insert(representative, set.members().to_vec().into_boxed_slice());
    }
    Ok(plan)
}

pub(super) fn lower_connection(
    file: &str,
    range: TextRange,
    syntax: ConnectionSyntax,
    names: &[String],
    id: Id<kinds::Connection>,
    bindings: &BTreeMap<String, Binding>,
    connected_ports: &mut BTreeSet<RawId>,
) -> Result<(ConnectionDef, Vec<RawId>), Diagnostic> {
    let mut ports = Vec::new();
    let mut definitions = Vec::new();
    for name in names {
        match bindings.get(name) {
            Some(Binding::Port(id, contract)) => {
                ports.push(id.erase());
                definitions.push(resolve_port_contract(file, range, contract, bindings)?);
            }
            Some(_) => {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    range,
                    format!("Connection name `{name}` is not a Port"),
                ));
            }
            None => {
                return Err(unresolved(file, range, name, "Connection Port"));
            }
        }
    }
    let unique_ports = ports.iter().copied().collect::<BTreeSet<_>>();
    if unique_ports.len() != ports.len() {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            "Connection repeats the same Port",
        ));
    }
    let members = if syntax == ConnectionSyntax::Signal {
        &ports[ports.len().min(1)..]
    } else {
        &ports[..]
    };
    if let Some(port) = members.iter().find(|port| connected_ports.contains(port)) {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            format!("Port `{port}` already belongs to another Connection"),
        ));
    }
    if syntax == ConnectionSyntax::SpatialPeriodic {
        if definitions.len() != 2
            || definitions.iter().any(|definition| {
                !matches!(definition, ResolvedPortContract::BoundaryPhysical { .. })
            })
        {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                "spatial-periodic Connection requires exactly two field-physical Ports",
            ));
        }
        connected_ports.extend(&ports);
        return Ok((
            ConnectionDef::new(id, ConnectionSemantics::SpatialPeriodic),
            ports,
        ));
    }
    let (kind, semantics) = match syntax {
        ConnectionSyntax::Signal => {
            let Some(Binding::Port(driver, _)) = names.first().and_then(|name| bindings.get(name))
            else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    range,
                    "signal Connection requires an explicit source Port",
                ));
            };
            (
                ScalarConnectionKind::Signal,
                ConnectionSemantics::Signal { driver: *driver },
            )
        }
        ConnectionSyntax::Conserving => (
            ScalarConnectionKind::Conserving,
            ConnectionSemantics::Conserving,
        ),
        ConnectionSyntax::SpatialPeriodic => unreachable!("handled above"),
    };
    let contracts = definitions
        .iter()
        .enumerate()
        .map(|(index, definition)| {
            let mut contract = resolved_scalar_port_contract(definition);
            if kind == ScalarConnectionKind::Signal
                && let ScalarPortContract::Signal { direction, .. } = &mut contract
            {
                *direction = if index == 0 {
                    SignalDirection::Output
                } else {
                    SignalDirection::Input
                };
            }
            contract
        })
        .collect::<Vec<_>>();
    ScalarPortContract::validate_connection(kind, &contracts).map_err(|violation| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            range,
            lower_connection_violation_message(violation),
        )
    })?;
    if let Some(ResolvedPortContract::Signal { support, clock, .. }) = definitions.first()
        && definitions.iter().skip(1).any(|contract| !matches!(contract, ResolvedPortContract::Signal { support: candidate_support, clock: candidate_clock, .. } if candidate_support == support && candidate_clock == clock)) {
            return Err(source_error(codes::LANGUAGE_TYPE_ERROR, file, range,
                "signal Connection requires exact matching support and clock activation"));
        }
    connected_ports.extend(members);
    Ok((ConnectionDef::new(id, semantics), ports))
}

fn resolved_scalar_port_contract(
    contract: &ResolvedPortContract,
) -> ScalarPortContract<Id<kinds::Domain>> {
    match contract {
        ResolvedPortContract::Signal {
            direction,
            value_type,
            ..
        } => ScalarPortContract::Signal {
            direction: match direction {
                SignalDirectionSyntax::Input => SignalDirection::Input,
                SignalDirectionSyntax::Output => SignalDirection::Output,
            },
            value_type: value_type.clone(),
        },
        ResolvedPortContract::ScalarPhysical { domain, .. } => {
            ScalarPortContract::ScalarPhysical { nominal: *domain }
        }
        ResolvedPortContract::BoundaryPhysical { connector, .. } => {
            ScalarPortContract::ScalarPhysical {
                nominal: *connector,
            }
        }
    }
}

fn lower_connection_violation_message(violation: ScalarConnectionViolation) -> &'static str {
    match violation {
        ScalarConnectionViolation::TooFewPorts { .. } => "Connection requires at least two Ports",
        ScalarConnectionViolation::SignalDirections { .. } => {
            "signal Connection requires exactly one output and one or more inputs"
        }
        ScalarConnectionViolation::SignalTypeMismatch => {
            "signal Connection requires dimension-matched inputs with compatible scalar domains, shapes and frames"
        }
        ScalarConnectionViolation::MixedConservingFamilies => {
            "conserving Connection cannot mix signal and scalar physical Ports"
        }
        ScalarConnectionViolation::PhysicalNominalMismatch => {
            "conserving Connection requires scalar physical Ports on the exact same nominal Domain"
        }
    }
}
