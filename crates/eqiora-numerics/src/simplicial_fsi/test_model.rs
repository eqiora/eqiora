//! Geometry-authored test models shared by the existing simplicial FSI consumers.

use std::collections::BTreeSet;
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_compiler::{CompiledModel, StaticBindingValue};
use eqiora_core::{DimExponents, DynQuantity};
use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore};
use eqiora_realization::*;
use eqiora_schema::kernel::{ConnectionSemantics, KernelNode};
use eqiora_sem::KernelProgram;
use eqiora_solver::{AlgebraicBlock, LinearOperatorProperties, SolverPlan};

use super::FixedReferenceFsiStepConfig;

pub(crate) mod polyhedra;

pub(crate) struct AuthoredFsiModel {
    pub(crate) program: KernelProgram,
    pub(crate) plan: CoupledFieldwiseRealizationPlan,
}

/// The caller authenticates the actual mesh and Region membership against this
/// Geometry before binding the returned Model/Plan to any numerical layout.
pub(crate) fn authored_model<const D: usize>(
    geometry: &CanonicalGeometryV1,
    supports: [[&NamedEntitySet; 2]; 3],
    mesh: MeshArtifactReference,
    config: FixedReferenceFsiStepConfig<D>,
    solver: SolverPlan,
    ale: bool,
) -> AuthoredFsiModel {
    let [parents, outer, contact] = supports;
    let source = source(D, ale);
    let material = config.material();
    let (fluid_domain, fluid_velocity, fluid_viscosity) = material
        .viscosity_entries()
        .next()
        .expect("fixture owns one viscous Field");
    let fluid_density = material
        .density_entries()
        .find(|(domain, field, _)| *domain == fluid_domain && *field == fluid_velocity)
        .map(|(_, _, value)| value)
        .expect("fixture viscous Field owns density");
    let (solid_domain, _, solid_elasticity) = material
        .elasticity_entries()
        .next()
        .expect("fixture owns one elastic state");
    let solid_density = material
        .density_entries()
        .find(|(domain, _, _)| *domain == solid_domain)
        .map(|(_, _, value)| value)
        .expect("fixture solid Domain owns a rate density");
    let quantity =
        |value, powers| DynQuantity::new(value, DimExponents::from_integers(powers).unwrap());
    let parameters = [
        (
            "fluid_density",
            quantity(fluid_density, [1, -3, 0, 0, 0, 0, 0]),
        ),
        (
            "fluid_viscosity",
            quantity(fluid_viscosity, [1, -1, -1, 0, 0, 0, 0]),
        ),
        (
            "solid_density",
            quantity(solid_density, [1, -3, 0, 0, 0, 0, 0]),
        ),
        (
            "solid_mu",
            quantity(solid_elasticity.shear_modulus(), [1, -1, -2, 0, 0, 0, 0]),
        ),
        (
            "solid_lambda",
            quantity(
                solid_elasticity.first_lame_parameter(),
                [1, -1, -2, 0, 0, 0, 0],
            ),
        ),
        ("zero_pressure", quantity(0.0, [1, -1, -2, 0, 0, 0, 0])),
    ]
    .map(|(name, value)| {
        (
            name,
            eqiora_lang::SourceAstFactory::value_literal(
                &eqiora_core::ValueLiteral::try_from(value).unwrap(),
                None,
                eqiora_lang::TextRange::default(),
                |_| None,
                |_| None,
            )
            .unwrap(),
        )
    });
    let mut bindings = Vec::new();
    for (index, names) in [
        ["fluid", "fluid_outer", "fluid_contact"],
        ["solid", "solid_outer", "solid_contact"],
    ]
    .iter()
    .enumerate()
    {
        bindings.push((
            names[0],
            StaticBindingValue::GeometrySupport {
                geometry,
                selection: parents[index],
                parent: None,
            },
        ));
        for (name, selection) in [(names[1], outer[index]), (names[2], contact[index])] {
            bindings.push((
                name,
                StaticBindingValue::GeometrySupport {
                    geometry,
                    selection,
                    parent: Some(parents[index]),
                },
            ));
        }
    }
    bindings.extend(
        parameters
            .iter()
            .map(|(name, value)| (*name, StaticBindingValue::Expression(value))),
    );
    let compiled =
        CompiledModel::compile_selected("authored-fsi.eqi", &source, "FixtureFsi", &bindings)
            .unwrap();
    let (transaction, model, symbols) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[geometry]).unwrap();
    let fluid = symbols.get("fluid").unwrap().downcast().unwrap();
    let solid = symbols.get("solid").unwrap().downcast().unwrap();
    // compile_selected instantiates the selected Component as `definition`;
    // support bindings remain root Model declarations.
    let vf = symbols
        .get("definition.fluid_velocity")
        .unwrap()
        .downcast()
        .unwrap();
    let pressure = symbols
        .get("definition.fluid_pressure")
        .unwrap()
        .downcast()
        .unwrap();
    let vs = symbols
        .get("definition.solid_velocity")
        .unwrap()
        .downcast()
        .unwrap();
    let displacement = symbols
        .get("definition.solid_displacement")
        .unwrap()
        .downcast()
        .unwrap();
    let ports = BTreeSet::from([
        symbols.get("definition.fluid_interface").unwrap(),
        symbols.get("definition.solid_interface").unwrap(),
    ]);
    let connections = program
        .nodes()
        .filter_map(|node| {
            let KernelNode::Connection(connection) = node else {
                return None;
            };
            let actual = program
                .edges()
                .iter()
                .filter(|edge| {
                    edge.from() == connection.id().erase() && edge.kind() == EdgeKind::Connects
                })
                .map(|edge| edge.to())
                .collect::<BTreeSet<_>>();
            (actual == ports && connection.semantics() == ConnectionSemantics::Conserving)
                .then_some(connection.id())
        })
        .collect::<Vec<_>>();
    assert_eq!(connections.len(), 1);
    let quotient = ConformingTraceQuotient::new(
        connections[0],
        TraceFieldEndpoint::new(fluid, vf),
        TraceFieldEndpoint::new(solid, vs),
    )
    .unwrap();
    let pair = BackwardEulerStatePair::new(
        symbols
            .get("definition.solid_kinematics")
            .unwrap()
            .downcast()
            .unwrap(),
        displacement,
        vs,
    )
    .unwrap();
    let p1 = Space::continuous_lagrange(NonZeroU16::MIN);
    let scale = config.scale();
    let length =
        PositivePhysicalScale::new(quantity(scale.length(), [0, 1, 0, 0, 0, 0, 0])).unwrap();
    let velocity =
        PositivePhysicalScale::new(quantity(scale.velocity(), [0, 1, -1, 0, 0, 0, 0])).unwrap();
    let pressure_scale =
        PositivePhysicalScale::new(quantity(scale.pressure(), [1, -1, -2, 0, 0, 0, 0])).unwrap();
    // Boundary power per out-of-plane measure: P U L^(D-1).
    let weak = PositivePhysicalScale::new(quantity(
        scale.pressure() * scale.velocity() * scale.length().powi(D as i32 - 1),
        [1, D as i32 - 1, -3, 0, 0, 0, 0],
    ))
    .unwrap();
    let spatial = CoupledFieldwiseSpatialDiscretization::new(
        length,
        [
            DomainFieldDiscretization::new(
                fluid,
                [
                    FieldSpaceBinding::new(vf, Space::simplex_p1_bubble()),
                    FieldSpaceBinding::new(pressure, p1),
                ],
                [],
            )
            .unwrap(),
            DomainFieldDiscretization::new(solid, [FieldSpaceBinding::new(vs, p1)], []).unwrap(),
        ],
        [quotient],
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::ImportedSimplicial { artifact: mesh },
            QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension: NonZeroUsize::new(D).unwrap(),
                points_per_axis: NonZeroUsize::new(if ale { D + 4 } else { D + 2 }).unwrap(),
            },
        ),
    )
    .unwrap();
    let time = BackwardEulerStep::new(
        quantity(config.time_step(), [0, 0, 1, 0, 0, 0, 0]),
        [BackwardEulerStateBinding::new(pair, p1, length)],
    )
    .unwrap();
    let scaling = SymmetricCongruenceScaling::new(
        [
            AlgebraicBlockScale::new(AlgebraicBlock::Field(vf), velocity),
            AlgebraicBlockScale::new(AlgebraicBlock::Field(pressure), pressure_scale),
            AlgebraicBlockScale::new(AlgebraicBlock::Field(vs), velocity),
        ],
        weak,
    )
    .unwrap();
    let plan = CoupledFieldwiseRealizationPlan::new(
        spatial,
        time,
        scaling,
        if ale {
            LinearOperatorProperties::General
        } else {
            LinearOperatorProperties::SymmetricIndefinite
        },
        solver,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    AuthoredFsiModel { program, plan }
}

fn source(dimension: usize, ale: bool) -> String {
    let original = include_str!("../../../../examples/fixed-reference-fsi.eqi");
    let mut source = original
        .lines()
        .filter(|line| {
            !line.contains("support fluid_y_")
                && !line.contains("support solid_y_")
                && !line.contains("relation fluid_y_")
                && !line.contains("relation solid_y_")
                && !line.trim_start().starts_with("//")
        })
        .collect::<Vec<_>>()
        .join("\n")
        .replace("FixedReferenceFsi2d", "FixtureFsi")
        .replace("fluid_x_lower", "fluid_outer")
        .replace("fluid_x_upper", "fluid_contact")
        .replace("solid_x_lower", "solid_contact")
        .replace("solid_x_upper", "solid_outer")
        .replace(
            "ambient_dimension = 2",
            &format!("ambient_dimension = {dimension}"),
        )
        .replace(", 2>", &format!(", {dimension}>"));
    if ale {
        source = format!("public operator outer_product(input left: spatial[1], input right: spatial[1]): spatial[2] = component(left, 0) * component(right, 1);\n{}", source.replace(
            "fluid_density * derivative(fluid_velocity)",
            "fluid_density * derivative(fluid_velocity) + div(fluid_density * outer_product(left = fluid_velocity, right = fluid_velocity))"));
    }
    source
}

pub(crate) fn planar_layout(
    region: &eqiora_geometry::PlanarRegion,
    mesh: &eqiora_meshing::SimplicialMesh,
    partition: &super::FixedReferenceFsiPartition<2>,
    boundary: &super::FixedReferenceFsiBoundary<2>,
    config: FixedReferenceFsiStepConfig<2>,
    solver: SolverPlan,
    ale: bool,
) -> super::layout::FsiLayout<2> {
    use eqiora_artifact::{
        GeometryDefinitionV1, GeometryMeshCorrespondenceEnvelopeV1, SimplicialMeshEnvelopeV1,
    };
    let geometry = CanonicalGeometryV1::from_region(region).unwrap();
    let definition = GeometryDefinitionV1::from_region(region);
    let mesh_artifact = SimplicialMeshEnvelopeV1::from_mesh(mesh).unwrap();
    let correspondence =
        GeometryMeshCorrespondenceEnvelopeV1::from_region(&definition, &mesh_artifact).unwrap();
    correspondence
        .validate_against_region(&definition, &mesh_artifact)
        .unwrap();
    for (name, actual) in [
        ("fluid", partition.fluid_cells()),
        ("solid", partition.solid_cells()),
    ] {
        let expected = correspondence
            .region_entity_set_entities(&definition, name)
            .unwrap()
            .into_iter()
            .map(|entity| entity.index())
            .collect::<BTreeSet<_>>();
        assert_eq!(expected, actual.iter().map(|cell| cell.index()).collect());
    }
    let expected = correspondence
        .region_entity_set_entities(&definition, "fluid_contact")
        .unwrap()
        .into_iter()
        .map(|entity| entity.index())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        expected,
        partition
            .interface_facets()
            .iter()
            .map(|facet| facet.index())
            .collect()
    );
    let model = authored_model(
        &geometry,
        [
            [
                geometry.entity_set("fluid").unwrap(),
                geometry.entity_set("solid").unwrap(),
            ],
            [
                geometry.entity_set("fluid_outer").unwrap(),
                geometry.entity_set("solid_outer").unwrap(),
            ],
            [
                geometry.entity_set("fluid_contact").unwrap(),
                geometry.entity_set("solid_contact").unwrap(),
            ],
        ],
        MeshArtifactReference::from_sha256(mesh_artifact.digest().unwrap().sha256_bytes()),
        config,
        solver,
        ale,
    );
    super::layout::FsiLayout::bind(&model.program, &model.plan, mesh, partition, boundary).unwrap()
}

pub(crate) fn adjacent_rectangles() -> eqiora_geometry::PlanarRegion {
    use eqiora_geometry::{PlanarFace, PlanarRegion};
    PlanarRegion::new(
        vec![
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        vec![
            PlanarFace::new(vec![0, 2, 3, 1], vec![]),
            PlanarFace::new(vec![2, 4, 5, 3], vec![]),
        ],
        vec![
            NamedEntitySet::new("fluid", 2, vec![0]),
            NamedEntitySet::new("solid", 2, vec![1]),
            NamedEntitySet::new("fluid_outer", 1, vec![0, 2, 3]),
            NamedEntitySet::new("solid_outer", 1, vec![4, 5, 6]),
            NamedEntitySet::new("fluid_contact", 1, vec![1]),
            NamedEntitySet::new("solid_contact", 1, vec![7]),
        ],
        1e-12,
    )
    .unwrap()
}
