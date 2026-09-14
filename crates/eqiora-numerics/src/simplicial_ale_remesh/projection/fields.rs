//! Exact Field ownership at the existing single-motion projection boundary.
use super::*;
use crate::region_assembly::mapping::FieldDof;

#[derive(Clone, Copy)]
pub(super) struct ProjectionFields {
    policy: P1HarmonicMeshMotionPolicy,
    pub(super) fluid_velocity: Id<kinds::Field>,
    pub(super) solid_velocity: Id<kinds::Field>,
    pressure: Id<kinds::Field>,
    displacement: Id<kinds::Field>,
}

impl ProjectionFields {
    pub(super) fn admit(
        source: &FinalizedResolvedFixedTopologyAleFsi<2>,
        target: &FixedReferenceFsiPartition<2>,
        target_motion: &P1HarmonicMeshMotionAction<2>,
    ) -> Result<Self, Diagnostic> {
        let policy = source.motion().policy();
        let layout = source.layout();
        let fluid_velocity = layout
            .velocity_field(policy.fluid_domain().erase())?
            .downcast()
            .expect("typed velocity Field");
        let solid_velocity = layout
            .velocity_field(policy.solid_domain().erase())?
            .downcast()
            .expect("typed rate Field");
        let pressure = layout
            .pressure_field(policy.fluid_domain().erase())
            .ok_or_else(|| {
                super::super::invalid("remesh motion Domain has no admitted constraint Field")
            })?
            .downcast()
            .expect("typed constraint Field");
        let displacement = policy.solid_displacement();
        if layout.state_field(policy.solid_domain().erase()) != Some(displacement.erase())
            || layout.state_rate(displacement.erase())? != solid_velocity.erase()
            || target_motion.policy() != policy
        {
            return Err(super::super::invalid(
                "remesh source and target differ in exact motion/state ownership",
            ));
        }
        let domains =
            BTreeSet::from([policy.fluid_domain().erase(), policy.solid_domain().erase()]);
        if source
            .partition()
            .domains()
            .map(|domain| domain.erase())
            .collect::<BTreeSet<_>>()
            != domains
            || target
                .domains()
                .map(|domain| domain.erase())
                .collect::<BTreeSet<_>>()
                != domains
            || source.partition().quotients().collect::<Vec<_>>()
                != target.quotients().collect::<Vec<_>>()
        {
            return Err(super::super::invalid(
                "remesh requires the complete exact single-motion Domain/Connection inventory",
            ));
        }
        let expected_endpoints = BTreeSet::from([
            (policy.fluid_domain().erase(), fluid_velocity.erase()),
            (policy.solid_domain().erase(), solid_velocity.erase()),
        ]);
        if source.partition().quotients().any(|trace| {
            trace.connection() != policy.interface()
                || trace
                    .endpoints()
                    .map(|endpoint| (endpoint.domain().erase(), endpoint.field().erase()))
                    .into_iter()
                    .collect::<BTreeSet<_>>()
                    != expected_endpoints
        }) {
            return Err(super::super::invalid(
                "remesh quotient differs from exact motion velocity endpoints",
            ));
        }
        Ok(Self {
            policy,
            fluid_velocity,
            solid_velocity,
            pressure,
            displacement,
        })
    }
}

pub(super) struct SourceCoefficients {
    pub(super) velocity: Vec<[f64; COMPONENTS]>,
    pub(super) bubbles: BTreeMap<CellId, [f64; COMPONENTS]>,
    pub(super) pressure: Vec<f64>,
    pub(super) displacement: Vec<[f64; COMPONENTS]>,
}

pub(super) fn source_coefficients(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    state: &FixedReferenceFsiState<2>,
    fields: ProjectionFields,
) -> Result<SourceCoefficients, Diagnostic> {
    let policy = fields.policy;
    let expected = BTreeMap::from([
        (fields.fluid_velocity.erase(), policy.fluid_domain().erase()),
        (fields.solid_velocity.erase(), policy.solid_domain().erase()),
        (fields.pressure.erase(), policy.fluid_domain().erase()),
        (fields.displacement.erase(), policy.solid_domain().erase()),
    ]);
    if state
        .fields
        .iter()
        .map(|(&id, field)| (id, field.domain))
        .collect::<BTreeMap<_, _>>()
        != expected
    {
        return Err(super::super::invalid(
            "remesh must consume every exact physical Field on its admitted Domain",
        ));
    }
    let mut velocity = vec![None; mesh.vertices().len()];
    for (field, domain) in [
        (fields.fluid_velocity, policy.fluid_domain()),
        (fields.solid_velocity, policy.solid_domain()),
    ] {
        let values = state.vector_vertices(field)?;
        if !values.keys().copied().eq(partition
            .domain_vertices(domain)
            .expect("admitted Domain")
            .iter()
            .copied())
        {
            return Err(super::super::invalid(
                "remesh velocity differs from exact Domain vertex support",
            ));
        }
        for (vertex, value) in values {
            let entry = &mut velocity[vertex.index()];
            if entry.is_some_and(|old| old != value) {
                return Err(super::super::invalid(
                    "remesh physical velocity violates the exact Connection quotient",
                ));
            }
            *entry = Some(value);
        }
    }
    let velocity = velocity
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| super::super::invalid("remesh velocity omits supporting vertices"))?;
    let bubbles = state
        .vector_entities(fields.fluid_velocity, DIMENSION)?
        .into_iter()
        .map(|(entity, value)| (CellId::new(entity.index()), value))
        .collect::<BTreeMap<_, _>>();
    if !bubbles.keys().copied().eq(partition
        .domain_cells(policy.fluid_domain())
        .expect("admitted Domain")
        .iter()
        .copied())
    {
        return Err(super::super::invalid(
            "remesh MINI bubbles differ from exact Domain cell support",
        ));
    }
    let mut displacement = vec![[0.0; COMPONENTS]; mesh.vertices().len()];
    let values = state.vector_vertices(fields.displacement)?;
    if !values.keys().copied().eq(partition
        .domain_vertices(policy.solid_domain())
        .expect("admitted Domain")
        .iter()
        .copied())
    {
        return Err(super::super::invalid(
            "remesh displacement differs from exact driver support",
        ));
    }
    for (vertex, value) in values {
        displacement[vertex.index()] = value;
    }
    let pressure = partition
        .domain_vertices(policy.fluid_domain())
        .expect("admitted Domain")
        .iter()
        .map(|vertex| {
            state.fields[&fields.pressure.erase()]
                .coefficients
                .get(&FieldDof {
                    field: fields.pressure.erase(),
                    entity: MeshEntity::new(0, vertex.index()),
                    slot: 0,
                    component: 0,
                })
                .copied()
                .ok_or_else(|| super::super::invalid("remesh constraint Field omits exact vertex"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SourceCoefficients {
        velocity,
        bubbles,
        pressure,
        displacement,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn target_state(
    mesh: &SimplicialMesh,
    source: &FixedReferenceFsiState<2>,
    partition: &FixedReferenceFsiPartition<2>,
    fields: ProjectionFields,
    velocity: &[[f64; COMPONENTS]],
    bubbles: &BTreeMap<CellId, [f64; COMPONENTS]>,
    pressure: &[f64],
    displacement: &[[f64; COMPONENTS]],
) -> Result<FixedReferenceFsiState<2>, Diagnostic> {
    let mut state = source.clone();
    for (&id, field) in &mut state.fields {
        field.coefficients.clear();
        let domain = field.domain.downcast().expect("typed Domain");
        let vertices = partition
            .domain_vertices(domain)
            .ok_or_else(|| super::super::invalid("remesh target omits exact Field Domain"))?;
        for (position, vertex) in vertices.iter().enumerate() {
            let values = if id == fields.pressure.erase() {
                vec![pressure[position]]
            } else if id == fields.displacement.erase() {
                displacement[vertex.index()].to_vec()
            } else {
                velocity[vertex.index()].to_vec()
            };
            for (component, value) in values.into_iter().enumerate() {
                field.coefficients.insert(
                    FieldDof {
                        field: id,
                        entity: MeshEntity::new(0, vertex.index()),
                        slot: 0,
                        component,
                    },
                    value,
                );
            }
        }
        if id == fields.fluid_velocity.erase() {
            for (&cell, values) in bubbles {
                for (component, &value) in values.iter().enumerate() {
                    field.coefficients.insert(
                        FieldDof {
                            field: id,
                            entity: MeshEntity::new(DIMENSION, cell.index()),
                            slot: 0,
                            component,
                        },
                        value,
                    );
                }
            }
        }
        if field.coefficients.values().any(|value| !value.is_finite()) {
            return Err(super::super::invalid(
                "remesh recovered nonfinite physical Field",
            ));
        }
    }
    // Revalidate complete exact supports and shared physical traces before publication.
    source_coefficients(mesh, partition, &state, fields)?;
    Ok(state)
}

#[cfg(test)]
#[path = "fields_tests.rs"]
mod tests;
