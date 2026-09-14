use super::*;
mod geometry;

use crate::simplicial_ale_fsi::test_support::{
    material, material_for, motion_for_plan, partition_for_plan,
};
use crate::simplicial_fsi::{FixedReferenceFsiMaterial, FixedReferenceFsiScale};
use eqiora_assembly::LocalUnknown;
use eqiora_meshing::{CellId, MeshQualityGate};
use eqiora_realization::{NonlinearSolvePlan, Target};
use eqiora_solver::{LinearSolveRequest, LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, num::NonZeroUsize};
#[derive(Clone, Copy)]
struct Nodal {
    mesh: u8,
    vertex: usize,
    y: u64,
    gamma: u64,
}

fn parse_ramp_authority() -> [u64; 8001] {
    const BODY: &str = include_str!("ramp-authority-v1.txt");

    assert_eq!(
        BODY.as_bytes().last(),
        Some(&b'\n'),
        "RAMP authority must end with LF",
    );
    let mut words = [0; 8001];
    let mut lines = BODY.split_terminator('\n');
    for (expected_index, slot) in words.iter_mut().enumerate() {
        let line = lines.next().expect("RAMP authority row is missing");
        assert_eq!(line.len(), 21, "RAMP authority row width differs");
        let bytes = line.as_bytes();
        assert_eq!(bytes[4], b' ', "RAMP authority separator differs");
        assert!(
            bytes[..4].iter().all(u8::is_ascii_digit),
            "RAMP authority index is not decimal",
        );
        assert!(
            bytes[5..]
                .iter()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte)),
            "RAMP authority word is not lowercase hexadecimal",
        );
        assert_eq!(
            line[..4]
                .parse::<usize>()
                .expect("RAMP authority index must parse"),
            expected_index,
            "RAMP authority index is out of sequence",
        );
        *slot = u64::from_str_radix(&line[5..], 16).expect("RAMP authority word must parse as u64");
    }
    assert!(lines.next().is_none(), "RAMP authority has extra rows");
    words
}

static RAMP: std::sync::LazyLock<[u64; 8001]> = std::sync::LazyLock::new(parse_ramp_authority);
const NODAL: [Nodal; 45] = [
    Nodal {
        mesh: 0,
        vertex: 0,
        y: 0,
        gamma: 0,
    },
    Nodal {
        mesh: 0,
        vertex: 1,
        y: 0x3fb17e4b17e4b194,
        gamma: 0x3ffaaaaaaaaaaac5,
    },
    Nodal {
        mesh: 0,
        vertex: 2,
        y: 0x3fc17e4b17e4b18e,
        gamma: 0x400555555555555f,
    },
    Nodal {
        mesh: 0,
        vertex: 3,
        y: 0x3fca3d70a3d70a43,
        gamma: 0x4008000000000000,
    },
    Nodal {
        mesh: 0,
        vertex: 4,
        y: 0x3fd17e4b17e4b180,
        gamma: 0x4005555555555553,
    },
    Nodal {
        mesh: 0,
        vertex: 5,
        y: 0x3fd5dddddddddddf,
        gamma: 0x3ffaaaaaaaaaaaa5,
    },
    Nodal {
        mesh: 0,
        vertex: 6,
        y: 0x3fda3d70a3d70a3d,
        gamma: 0x3cc9c18f9c18f9c1,
    },
    Nodal {
        mesh: 1,
        vertex: 0,
        y: 0,
        gamma: 0,
    },
    Nodal {
        mesh: 1,
        vertex: 1,
        y: 0x3fb17e4b17e4b194,
        gamma: 0x3ffaaaaaaaaaaac5,
    },
    Nodal {
        mesh: 1,
        vertex: 2,
        y: 0x3fc17e4b17e4b18e,
        gamma: 0x400555555555555f,
    },
    Nodal {
        mesh: 1,
        vertex: 3,
        y: 0x3fca3d70a3d70a43,
        gamma: 0x4008000000000000,
    },
    Nodal {
        mesh: 1,
        vertex: 4,
        y: 0x3fd17e4b17e4b180,
        gamma: 0x4005555555555553,
    },
    Nodal {
        mesh: 1,
        vertex: 5,
        y: 0x3fd5dddddddddddf,
        gamma: 0x3ffaaaaaaaaaaaa5,
    },
    Nodal {
        mesh: 1,
        vertex: 6,
        y: 0x3fda3d70a3d70a3d,
        gamma: 0x3cc9c18f9c18f9c1,
    },
    Nodal {
        mesh: 1,
        vertex: 539,
        y: 0x3fa17e4b17e4b194,
        gamma: 0x3fed555555555576,
    },
    Nodal {
        mesh: 1,
        vertex: 542,
        y: 0x3fba3d70a3d70a58,
        gamma: 0x400200000000000c,
    },
    Nodal {
        mesh: 1,
        vertex: 545,
        y: 0x3fc5dddddddddde8,
        gamma: 0x4007555555555558,
    },
    Nodal {
        mesh: 1,
        vertex: 549,
        y: 0x3fce9d0369d036a2,
        gamma: 0x4007555555555554,
    },
    Nodal {
        mesh: 1,
        vertex: 553,
        y: 0x3fd3ae147ae147b0,
        gamma: 0x4001fffffffffffc,
    },
    Nodal {
        mesh: 1,
        vertex: 556,
        y: 0x3fd80da740da740e,
        gamma: 0x3fed555555555551,
    },
    Nodal {
        mesh: 2,
        vertex: 0,
        y: 0,
        gamma: 0,
    },
    Nodal {
        mesh: 2,
        vertex: 1,
        y: 0x3fb17e4b17e4b194,
        gamma: 0x3ffaaaaaaaaaaac5,
    },
    Nodal {
        mesh: 2,
        vertex: 2,
        y: 0x3fc17e4b17e4b18e,
        gamma: 0x400555555555555f,
    },
    Nodal {
        mesh: 2,
        vertex: 3,
        y: 0x3fca3d70a3d70a43,
        gamma: 0x4008000000000000,
    },
    Nodal {
        mesh: 2,
        vertex: 4,
        y: 0x3fd17e4b17e4b180,
        gamma: 0x4005555555555553,
    },
    Nodal {
        mesh: 2,
        vertex: 5,
        y: 0x3fd5dddddddddddf,
        gamma: 0x3ffaaaaaaaaaaaa5,
    },
    Nodal {
        mesh: 2,
        vertex: 6,
        y: 0x3fda3d70a3d70a3d,
        gamma: 0x3cc9c18f9c18f9c1,
    },
    Nodal {
        mesh: 2,
        vertex: 539,
        y: 0x3fa17e4b17e4b194,
        gamma: 0x3fed555555555576,
    },
    Nodal {
        mesh: 2,
        vertex: 542,
        y: 0x3fba3d70a3d70a58,
        gamma: 0x400200000000000c,
    },
    Nodal {
        mesh: 2,
        vertex: 545,
        y: 0x3fc5dddddddddde8,
        gamma: 0x4007555555555558,
    },
    Nodal {
        mesh: 2,
        vertex: 549,
        y: 0x3fce9d0369d036a2,
        gamma: 0x4007555555555554,
    },
    Nodal {
        mesh: 2,
        vertex: 553,
        y: 0x3fd3ae147ae147b0,
        gamma: 0x4001fffffffffffc,
    },
    Nodal {
        mesh: 2,
        vertex: 556,
        y: 0x3fd80da740da740e,
        gamma: 0x3fed555555555551,
    },
    Nodal {
        mesh: 2,
        vertex: 2057,
        y: 0x3f917e4b17e4b194,
        gamma: 0x3fdeaaaaaaaaaacf,
    },
    Nodal {
        mesh: 2,
        vertex: 2060,
        y: 0x3faa3d70a3d70a5e,
        gamma: 0x3ff5000000000016,
    },
    Nodal {
        mesh: 2,
        vertex: 2061,
        y: 0x3fb5ddddddddddf6,
        gamma: 0x3fffaaaaaaaaaac4,
    },
    Nodal {
        mesh: 2,
        vertex: 2064,
        y: 0x3fbe9d0369d036ba,
        gamma: 0x4003d55555555560,
    },
    Nodal {
        mesh: 2,
        vertex: 2065,
        y: 0x3fc3ae147ae147bb,
        gamma: 0x4006800000000006,
    },
    Nodal {
        mesh: 2,
        vertex: 2069,
        y: 0x3fc80da740da7416,
        gamma: 0x4007d55555555557,
    },
    Nodal {
        mesh: 2,
        vertex: 2070,
        y: 0x3fcc6d3a06d3a072,
        gamma: 0x4007d55555555555,
    },
    Nodal {
        mesh: 2,
        vertex: 2074,
        y: 0x3fd0666666666668,
        gamma: 0x40067fffffffffff,
    },
    Nodal {
        mesh: 2,
        vertex: 2075,
        y: 0x3fd2962fc962fc98,
        gamma: 0x4003d55555555553,
    },
    Nodal {
        mesh: 2,
        vertex: 2078,
        y: 0x3fd4c5f92c5f92c8,
        gamma: 0x3fffaaaaaaaaaaa2,
    },
    Nodal {
        mesh: 2,
        vertex: 2079,
        y: 0x3fd6f5c28f5c28f6,
        gamma: 0x3ff4ffffffffffff,
    },
    Nodal {
        mesh: 2,
        vertex: 2082,
        y: 0x3fd9258bf258bf26,
        gamma: 0x3fdeaaaaaaaaaa9f,
    },
];
const EDGES: [&[(usize, usize)]; 3] = [
    &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
    &[
        (0, 539),
        (1, 539),
        (1, 542),
        (2, 542),
        (2, 545),
        (3, 545),
        (3, 549),
        (4, 549),
        (4, 553),
        (5, 553),
        (5, 556),
        (6, 556),
    ],
    &[
        (0, 2057),
        (1, 2060),
        (1, 2061),
        (2, 2064),
        (2, 2065),
        (3, 2069),
        (3, 2070),
        (4, 2074),
        (4, 2075),
        (5, 2078),
        (5, 2079),
        (6, 2082),
        (539, 2057),
        (539, 2060),
        (542, 2061),
        (542, 2064),
        (545, 2065),
        (545, 2069),
        (549, 2070),
        (549, 2074),
        (553, 2075),
        (553, 2078),
        (556, 2079),
        (556, 2082),
    ],
];
fn dyadic(word: u64) -> (u128, i32) {
    assert_eq!(word >> 63, 0);
    let e = ((word >> 52) & 0x7ff) as i32;
    let m = word & ((1_u64 << 52) - 1);
    if e == 0 {
        (u128::from(m), -1074)
    } else {
        (u128::from((1_u64 << 52) | m), e - 1075)
    }
}
fn rn(n: u128, e: i32) -> u64 {
    if n == 0 {
        return 0;
    }
    let shift = ((127 - n.leading_zeros() as i32) - 52).max(0) as u32;
    let mut rounded = n >> shift;
    if shift > 0 {
        let rem = n & ((1_u128 << shift) - 1);
        let half = 1_u128 << (shift - 1);
        if rem > half || (rem == half && rounded & 1 == 1) {
            rounded += 1;
        }
    }
    let carry = rounded == 1_u128 << 53;
    if carry {
        rounded >>= 1;
    }
    let exponent = e + shift as i32 + 52 + i32::from(carry) + 1023;
    assert!((1..2047).contains(&exponent));
    ((exponent as u64) << 52) | rounded as u64 & ((1_u64 << 52) - 1)
}
fn product(a: u64, b: u64) -> u64 {
    let (a, ae) = dyadic(a);
    let (b, be) = dyadic(b);
    rn(a * b, ae + be)
}
fn half(word: u64) -> u64 {
    let (n, e) = dyadic(word);
    rn(n, e - 1)
}
fn fraction(word: u64) -> (u128, u128) {
    let (mut n, mut e) = dyadic(word);
    let z = n.trailing_zeros().min((-e).max(0) as u32);
    n >>= z;
    e += z as i32;
    assert!(e <= 0);
    (n, 1_u128 << (-e as u32))
}
fn sha(body: &str) -> String {
    Sha256::digest(body.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn authority_bodies() -> (String, String) {
    let ramp = RAMP
        .iter()
        .enumerate()
        .map(|(i, w)| format!("{i:04} {w:016x}\n"))
        .collect();
    let nodal = NODAL
        .iter()
        .map(|r| format!("M{} {} {:016x} {:016x}\n", r.mesh, r.vertex, r.y, r.gamma))
        .collect();
    (ramp, nodal)
}
fn values(level: u8, ramp: usize) -> Vec<(VertexId, [f64; 2])> {
    NODAL
        .iter()
        .filter(|r| r.mesh == level)
        .map(|r| {
            (
                VertexId::new(r.vertex),
                [f64::from_bits(product(RAMP[ramp], r.gamma)), 0.0],
            )
        })
        .collect()
}
type OracleMesh = (
    SimplicialMesh,
    Vec<(MeshEntity, AleFsiExteriorFacetDisposition)>,
    eqiora_geometry::PlanarRegion,
);

fn oracle_mesh(level: u8) -> Result<OracleMesh, Diagnostic> {
    let count = [539, 2057, 8030][level as usize];
    let mut rows = NODAL.iter().filter(|r| r.mesh == level).collect::<Vec<_>>();
    rows.sort_by_key(|r| r.y);
    let inlet = rows.iter().map(|r| r.vertex).collect::<BTreeSet<_>>();
    let center = (0..count).find(|i| !inlet.contains(i)).expect("center");
    let apex = (center + 1..count)
        .find(|i| !inlet.contains(i))
        .expect("solid apex");
    let mut points = vec![vec![0.0, 0.0]; count];
    for r in &rows {
        points[r.vertex] = vec![0.0, f64::from_bits(r.y)];
    }
    points[center] = vec![1.0, 0.2];
    let outer = (0..count)
        .filter(|i| !inlet.contains(i) && *i != center && *i != apex)
        .collect::<Vec<_>>();
    for (i, v) in outer.iter().copied().enumerate() {
        let p = std::f64::consts::FRAC_PI_2
            - std::f64::consts::PI * i as f64 / (outer.len() - 1) as f64;
        points[v] = vec![1.0 + p.cos(), 0.2 + 0.2 * p.sin()];
    }
    let boundary = rows
        .iter()
        .map(|r| r.vertex)
        .chain(outer)
        .collect::<Vec<_>>();
    let split = rows.len() + (boundary.len() - rows.len()) / 2;
    let a = boundary[split];
    let b = boundary[(split + 1) % boundary.len()];
    points[apex] = vec![
        0.45 * (points[a][0] + points[b][0]) + 0.1 * points[center][0],
        0.45 * (points[a][1] + points[b][1]) + 0.1 * points[center][1],
    ];
    let mut cells = Vec::with_capacity(boundary.len() + 2);
    for i in 0..boundary.len() {
        let a = boundary[i];
        let b = boundary[(i + 1) % boundary.len()];
        if i == split {
            cells.push(vec![center, b, apex]);
            cells.push(vec![center, apex, a]);
        } else {
            cells.push(vec![center, b, a]);
        }
    }
    cells.push(vec![apex, b, a]);
    let region = geometry::authored_region(&points, &boundary, split, apex);
    let mesh = SimplicialMesh::new(2, points, cells, MeshQualityGate::new(1e-15)?)?;
    let roles = exterior_facets(&mesh)?
        .into_iter()
        .map(|facet| {
            let v = mesh.entity_vertices(facet).expect("facet");
            let pair = (
                v[0].index().min(v[1].index()),
                v[0].index().max(v[1].index()),
            );
            let essential = EDGES[level as usize]
                .iter()
                .any(|&(a, b)| (a.min(b), a.max(b)) == pair);
            (
                facet,
                if essential {
                    AleFsiExteriorFacetDisposition::EssentialVelocity
                } else {
                    AleFsiExteriorFacetDisposition::NaturalOutflow
                },
            )
        })
        .collect();
    Ok((mesh, roles, region))
}
fn partition(
    mesh: &SimplicialMesh,
    plan: &eqiora_realization::CoupledFieldwiseRealizationPlan,
) -> Result<FixedReferenceFsiPartition<2>, Diagnostic> {
    let solid = mesh.cells().len() - 1;
    Ok(partition_for_plan(
        mesh,
        (0..solid).map(CellId::new).collect(),
        vec![CellId::new(solid)],
        plan,
    ))
}
fn solver_plan(algorithm: LinearSolver) -> Result<SolverPlan, Diagnostic> {
    SolverPlan::new(algorithm, 1e-12, 1e-14, NonZeroUsize::new(32).expect("cap"))
}
fn step_plan(dt: f64) -> Result<AleFsiStepPlan<2>, Diagnostic> {
    step_plan_with_material(dt, material())
}
fn step_plan_with_material(
    dt: f64,
    material: FixedReferenceFsiMaterial<2>,
) -> Result<AleFsiStepPlan<2>, Diagnostic> {
    AleFsiStepPlan::new(
        dt,
        material,
        FixedReferenceFsiScale::new(1.0, 2.0, 1.0)?,
        Default::default(),
        NonlinearSolvePlan::new(1e-10, 1e-12, NonZeroUsize::new(8).expect("cap"), 4)?,
        solver_plan(LinearSolver::BiConjugateGradientStabilized)?,
        Target::HostCpu {
            threads: NonZeroUsize::new(1).expect("width"),
        },
    )
}
fn zero_state(
    mesh: &SimplicialMesh,
    part: &FixedReferenceFsiPartition<2>,
    motion: &P1HarmonicMeshMotionAction<2>,
    model: &crate::simplicial_fsi::test_model::AuthoredFsiModel,
) -> Result<AleFsiState<2>, Diagnostic> {
    let physical = crate::simplicial_fsi::test_model::exact_state(
        &model.program,
        &model.plan,
        mesh,
        part,
        |_field, _entity, _component| 0.0,
    );
    AleFsiState::new(0.0, mesh, part, motion, physical)
}
fn fixed(unknown: LocalUnknown, expected: u64) -> bool {
    matches!(unknown,LocalUnknown::Fixed(v) if v.to_bits()==expected)
}
#[allow(clippy::too_many_arguments)]
fn admits_raw(
    bytes: usize,
    vertices: [usize; 3],
    edges: [usize; 3],
    cells: [usize; 3],
    ramp: usize,
    nodal: usize,
    members: usize,
    steps: usize,
) -> bool {
    bytes <= 747_943
        && vertices.iter().zip([539, 2057, 8030]).all(|(a, b)| *a <= b)
        && edges.iter().zip([1518, 5973, 23694]).all(|(a, b)| *a <= b)
        && cells.iter().zip([1024, 4096, 16384]).all(|(a, b)| *a <= b)
        && ramp <= 8001
        && nodal <= 45
        && members <= 5
        && steps <= 30_000
}
#[test]
fn fsi3_p1_inlet_trace_oracle_v1() -> Result<(), Diagnostic> {
    let (ramp_body, nodal_body) = authority_bodies();
    assert_eq!(
        (RAMP.len(), ramp_body.len(), sha(&ramp_body)),
        (
            8001,
            176_022,
            "c196074aac01a4b60cb80a285efcd216444c75c8ce7308e87d0eaf243a56823a".into()
        )
    );
    assert_eq!(
        (NODAL.len(), nodal_body.len(), sha(&nodal_body)),
        (
            45,
            1815,
            "64d4c28d6e40858fdad2b27368eb84ff6ff3fd3dd2a9e3da638827b0d9ed8875".into()
        )
    );
    assert_eq!(
        (fraction(0x3e7f0c0e1fbb7b5c), fraction(0x3e6f0c0e1fbb7b5c)),
        (
            (2184744769871575, 18889465931478580854784),
            (2184744769871575, 37778931862957161709568)
        )
    );
    assert_eq!(
        (fraction(0x3e9f0c0e0ba690c2), fraction(0x3e8f0c0e0ba690c2)),
        (
            (4369489371285601, 9444732965739290427392),
            (4369489371285601, 18889465931478580854784)
        )
    );
    assert_eq!(
        (fraction(0x3ebf0c0dbb52e6be), fraction(0x3eaf0c0dbb52e6be)),
        (
            (4369488697455455, 2361183241434822606848),
            (4369488697455455, 4722366482869645213696)
        )
    );
    assert_eq!(
        (fraction(0x3e713f9611a10bb6), fraction(0x3e613f9611a10bb6)),
        (
            (2427494188746203, 37778931862957161709568),
            (2427494188746203, 75557863725914323419136)
        )
    );
    let members = [
        ("M0T2", 0, 2, 1, 1, 1.0 / 4000.0),
        ("M1T2", 1, 2, 1, 1, 1.0 / 4000.0),
        ("M2T0", 2, 0, 4, 4, 1.0 / 1000.0),
        ("M2T1", 2, 1, 2, 2, 1.0 / 2000.0),
        ("M2T2", 2, 2, 1, 1, 1.0 / 4000.0),
    ];
    let mesh_data = (0..3).map(oracle_mesh).collect::<Result<Vec<_>, _>>()?;
    let seed = step_plan(1.0 / 4000.0)?;
    let models = mesh_data
        .iter()
        .map(|(mesh, _, region)| {
            crate::simplicial_fsi::test_model::planar_model(
                region,
                mesh,
                seed.fixed_reference_config().clone(),
                seed.linear_solver(),
                true,
            )
        })
        .collect::<Vec<_>>();
    let parts = mesh_data
        .iter()
        .zip(&models)
        .map(|((m, _, _), model)| partition(m, &model.plan))
        .collect::<Result<Vec<_>, _>>()?;
    let motions = (0..3)
        .map(|i| {
            Ok(motion_for_plan(
                &mesh_data[i].0,
                &parts[i],
                &models[i].plan,
                LinearSolveRequest::new(
                    &REFERENCE_LINEAR_SOLVER,
                    solver_plan(LinearSolver::ConjugateGradient)?,
                ),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Topology, Fields, scales and quotient are constant across this schedule;
    // only the boundary endpoint values change at each prepared action.
    let base_layouts = mesh_data
        .iter()
        .zip(&parts)
        .zip(&models)
        .map(|(((mesh, _, _), part), model)| {
            crate::simplicial_fsi::layout::FsiLayout::bind(
                &model.program,
                &model.plan,
                mesh,
                part,
                &FixedReferenceFsiBoundary::homogeneous_exterior(mesh)?,
            )
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    let mut body = String::new();
    let mut prepared_members = Vec::new();
    for &(name, level, schedule, stride, ramp, dt) in &members {
        let (mesh, roles, _) = &mesh_data[level as usize];
        let part = &parts[level as usize];
        let motion = &motions[level as usize];
        let previous = AleFsiBoundaryEndpointIdentity::new([level as u64, schedule, 0, 0], 0.0)?;
        let current = AleFsiBoundaryEndpointIdentity::new([level as u64, schedule, 1, stride], dt)?;
        let prepared = PreparedAleFsiBoundaryStep::canonical_p1(
            mesh,
            previous,
            current,
            values(level, 0),
            values(level, ramp),
            roles.clone(),
            2.0,
        )?;
        assert_eq!(
            (prepared.previous_endpoint(), prepared.current_endpoint()),
            (previous, current)
        );
        let fields = crate::simplicial_fsi::test_model::exact_fields(&models[level as usize].plan);
        let plan = step_plan_with_material(dt, material_for(fields))?;
        let layout = prepared.layout(&base_layouts[level as usize])?;
        let prior = zero_state(mesh, part, motion, &models[level as usize])?;
        let initial = prepared.reduce_initial_point(&prior, &plan, &layout)?;
        let primal = layout.reconstruct_primal(&initial)?;
        let direction = layout.reconstruct_direction(&vec![0.0; layout.reduced_size()])?;
        let state = prepared
            .reconstruct_current_state(mesh, part, motion, &prior, &initial, &plan, &layout)?;
        for row in NODAL.iter().filter(|r| r.mesh == level) {
            let g = product(RAMP[ramp], row.gamma);
            let q = half(g);
            for (component, expected) in [g, 0].into_iter().enumerate() {
                let quotient = if component == 0 { q } else { 0 };
                let key = crate::region_assembly::mapping::FieldDof {
                    field: fields.fluid_velocity.erase(),
                    entity: MeshEntity::new(0, row.vertex),
                    slot: 0,
                    component,
                };
                assert_eq!(
                    prepared.previous_physical()[row.vertex][component]
                        .expect("previous")
                        .to_bits(),
                    0
                );
                assert_eq!(
                    prepared.current_physical()[row.vertex][component]
                        .expect("current")
                        .to_bits(),
                    expected
                );
                assert_eq!(
                    prepared.current_quotient()[row.vertex][component]
                        .expect("quotient")
                        .to_bits(),
                    quotient
                );
                assert_eq!(primal[&key].to_bits(), quotient);
                assert_eq!(direction[&key].to_bits(), 0);
                assert_eq!(
                    state
                        .physical_state()
                        .vector_vertices(fields.fluid_velocity)?
                        [&eqiora_meshing::VertexId::new(row.vertex,)][component]
                        .to_bits(),
                    expected
                );
                assert_eq!((f64::from_bits(quotient) * 2.0).to_bits(), expected);
                body.push_str(&format!(
                    "{name} 0 1 {ramp} M{level} {} {} {expected:016x} {quotient:016x}\n",
                    row.vertex,
                    if component == 0 { "x" } else { "y" }
                ));
            }
            let pair = EDGES[level as usize]
                .iter()
                .copied()
                .find(|(a, b)| *a == row.vertex || *b == row.vertex)
                .expect("inlet edge");
            let cell = mesh
                .cells()
                .iter()
                .position(|c| c.contains(&pair.0) && c.contains(&pair.1))
                .expect("P1 cell");
            let vertices = mesh
                .entity_vertices(MeshEntity::new(2, cell))
                .expect("vertices");
            let map = layout.fluid_map(eqiora_meshing::CellId::new(cell), &vertices, true)?;
            let local = vertices
                .iter()
                .position(|v| v.index() == row.vertex)
                .expect("local");
            assert!(fixed(map.unknowns()[2 * local], q) && fixed(map.unknowns()[2 * local + 1], 0));
        }
        let observed = prepared.exterior_facets();
        assert_eq!(observed.len(), exterior_facets(mesh)?.len());
        assert_eq!(
            observed
                .iter()
                .map(|(f, _)| f.index())
                .collect::<BTreeSet<_>>()
                .len(),
            observed.len()
        );
        assert!(
            observed
                .iter()
                .any(|(_, d)| *d == AleFsiExteriorFacetDisposition::NaturalOutflow)
        );
        for &(a, b) in EDGES[level as usize] {
            let facet = observed
                .iter()
                .find(|(f, _)| {
                    let v = mesh.entity_vertices(*f).expect("facet");
                    [v[0].index(), v[1].index()].contains(&a)
                        && [v[0].index(), v[1].index()].contains(&b)
                })
                .expect("native inlet facet");
            assert_eq!(facet.1, AleFsiExteriorFacetDisposition::EssentialVelocity);
            assert_eq!((27.0_f64 * 0.25 * 0.75 * 0.0).to_bits(), 0);
        }
        prepared_members.push(prepared);
    }
    assert_eq!(
        (body.lines().count(), body.len(), sha(&body)),
        (
            190,
            10192,
            "b3390cfc2822828ac42e299ac793715b47f770257fa7fd6d8ee5940cfc9581c2".into()
        )
    );
    let mesh = &mesh_data[0].0;
    let part = &parts[0];
    let motion = &motions[0];
    let homogeneous_boundary = FixedReferenceFsiBoundary::homogeneous_exterior(mesh)?;
    let homogeneous = PreparedAleFsiBoundaryStep::homogeneous(
        mesh,
        &homogeneous_boundary,
        0.0,
        1.0 / 4000.0,
        2.0,
    )?;
    let fields = crate::simplicial_fsi::test_model::exact_fields(&models[0].plan);
    let plan = step_plan_with_material(1.0 / 4000.0, material_for(fields))?;
    let layout = homogeneous.layout(&base_layouts[0])?;
    let prior = zero_state(mesh, part, motion, &models[0])?;
    let initial = homogeneous.reduce_initial_point(&prior, &plan, &layout)?;
    let primal = layout.reconstruct_primal(&initial)?;
    let direction = layout.reconstruct_direction(&vec![0.0; layout.reduced_size()])?;
    let state = homogeneous
        .reconstruct_current_state(mesh, part, motion, &prior, &initial, &plan, &layout)?;
    for (v, row) in homogeneous.current_physical().iter().enumerate() {
        for c in 0..2 {
            if row[c].is_some() {
                let key = crate::region_assembly::mapping::FieldDof {
                    field: fields.fluid_velocity.erase(),
                    entity: MeshEntity::new(0, v),
                    slot: 0,
                    component: c,
                };
                assert_eq!(
                    (
                        row[c].expect("zero").to_bits(),
                        homogeneous.current_quotient()[v][c]
                            .expect("zero")
                            .to_bits(),
                        primal[&key].to_bits(),
                        direction[&key].to_bits(),
                        state
                            .physical_state()
                            .vector_vertices(fields.fluid_velocity)?
                            [&eqiora_meshing::VertexId::new(v)][c]
                            .to_bits()
                    ),
                    (0, 0, 0, 0, 0)
                );
            }
        }
    } /* Every ordinary positive above passes before these independent precommitted mutants. */
    let accepted = &prepared_members[0];
    let current_as_previous_physical = crate::simplicial_fsi::test_model::exact_state(
        &models[0].program,
        &models[0].plan,
        mesh,
        part,
        |field, entity, component| {
            if (field == fields.fluid_velocity || field == fields.solid_velocity)
                && entity.dimension() == 0
            {
                accepted.current_physical()[entity.index()][component].unwrap_or(0.0)
            } else {
                0.0
            }
        },
    );
    let current_as_previous =
        AleFsiState::new(0.0, mesh, part, motion, current_as_previous_physical)?;
    let chronology_error = accepted
        .validate_inputs(
            mesh,
            part,
            motion,
            &current_as_previous,
            &plan,
            &QuadratureRule::point(),
        )
        .expect_err("current trace must fail the previous-state chronology gate");
    assert!(
        format!("{chronology_error:?}")
            .contains("previous state differs from its prepared physical trace")
    );
    let stale_previous = AleFsiBoundaryEndpointIdentity::new([2, 2, 3, 3], 3.0 / 4000.0)?;
    let stale_current = AleFsiBoundaryEndpointIdentity::new([2, 2, 4, 4], 4.0 / 4000.0)?;
    let stale_error = advance_simplicial_ale_fsi_prepared_step(
        &mesh_data[2].0,
        &parts[2],
        &prepared_members[2],
        stale_previous,
        stale_current,
        &motions[2],
        &zero_state(&mesh_data[2].0, &parts[2], &motions[2], &models[2])?,
        &step_plan_with_material(
            1.0 / 1000.0,
            material_for(crate::simplicial_fsi::test_model::exact_fields(
                &models[2].plan,
            )),
        )?,
        &QuadratureRule::point(),
        &eqiora_assembly::ReferenceAssemblyBackend,
        &REFERENCE_LINEAR_SOLVER,
        &base_layouts[2],
    )
    .expect_err("equal-ramp stale member must fail the expected-identity gate before solve");
    assert!(
        format!("{stale_error:?}")
            .contains("prepared boundary does not match the requested endpoint identities")
    );
    let exact_trace_word_gate =
        |route: &'static str, observed: u64, expected: u64| -> Result<(), &'static str> {
            if observed == expected {
                Ok(())
            } else {
                Err(route)
            }
        };
    let reject_trace_route = |route: &'static str, observed: u64, expected: u64| {
        assert_ne!(
            observed, expected,
            "{route} mutant must alter the target word"
        );
        assert_eq!(exact_trace_word_gate(route, observed, expected), Err(route));
    };
    let rho = f64::from_bits(RAMP[1]);
    let target = NODAL[2];
    let y = f64::from_bits(target.y);
    let gamma = f64::from_bits(target.gamma);
    let expected = product(RAMP[1], target.gamma);
    let coarse_p1 = (f64::from_bits(NODAL[1].gamma) + f64::from_bits(NODAL[2].gamma)) * 0.5;
    reject_trace_route(
        "other-level P1 restriction",
        (rho * coarse_p1).to_bits(),
        product(RAMP[1], NODAL[15].gamma),
    );
    let profile_constant = 120_000.0 / 1_681.0;
    let height = 41.0 / 100.0;
    reject_trace_route(
        "analytic G",
        (rho * (profile_constant * y * (height - y))).to_bits(),
        expected,
    );
    let level_zero = NODAL.iter().filter(|row| row.mesh == 0);
    let count = level_zero.clone().count() as f64;
    let sum_y = level_zero
        .clone()
        .map(|row| f64::from_bits(row.y))
        .sum::<f64>();
    let sum_gamma = level_zero
        .clone()
        .map(|row| f64::from_bits(row.gamma))
        .sum::<f64>();
    let sum_yy = level_zero
        .clone()
        .map(|row| {
            let value = f64::from_bits(row.y);
            value * value
        })
        .sum::<f64>();
    let sum_yg = level_zero
        .map(|row| f64::from_bits(row.y) * f64::from_bits(row.gamma))
        .sum::<f64>();
    let fitted_slope = (count * sum_yg - sum_y * sum_gamma) / (count * sum_yy - sum_y * sum_y);
    let fitted_intercept = (sum_gamma - fitted_slope * sum_y) / count;
    reject_trace_route(
        "fitted trace",
        (rho * (fitted_intercept + fitted_slope * y)).to_bits(),
        expected,
    );
    reject_trace_route(
        "lifted trace",
        (rho * (gamma + y * (height - y))).to_bits(),
        expected,
    );
    let fma_word = (-(rho * profile_constant * y))
        .mul_add(y, rho * profile_constant * height * y)
        .to_bits();
    reject_trace_route("FMA evaluation", fma_word, expected);
    reject_trace_route(
        "reassociated evaluation",
        (((rho * profile_constant) * y) * (height - y)).to_bits(),
        expected,
    );
    let second_rounded_gamma = format!("{gamma:.12e}")
        .parse::<f64>()
        .expect("finite decimal trace word");
    reject_trace_route(
        "second rounded trace",
        (rho * second_rounded_gamma).to_bits(),
        expected,
    );
    let adjacent = accepted.current_physical()[3][0].expect("g").to_bits() + 1;
    assert_ne!(adjacent, product(RAMP[1], NODAL[3].gamma));
    let g = accepted.current_physical()[3][0].expect("g").to_bits();
    let q = accepted.current_quotient()[3][0].expect("q").to_bits();
    assert!(!fixed(LocalUnknown::Fixed(f64::from_bits(g)), q));
    assert!(!fixed(
        LocalUnknown::Free(eqiora_assembly::DofId::new(0)),
        q
    ));
    assert!(
        PreparedAleFsiBoundaryStep::canonical_p1(
            mesh,
            accepted.previous_endpoint(),
            accepted.current_endpoint(),
            values(0, 0),
            values(0, 1),
            accepted.exterior_facets().to_vec(),
            f64::from_bits(2.0_f64.to_bits() + 1)
        )
        .is_err()
    );
    let mut negative = values(0, 1);
    negative[0].1[1] = -0.0;
    assert!(
        PreparedAleFsiBoundaryStep::canonical_p1(
            mesh,
            accepted.previous_endpoint(),
            accepted.current_endpoint(),
            values(0, 0),
            negative,
            accepted.exterior_facets().to_vec(),
            2.0
        )
        .is_err()
    );
    assert!(!fixed(LocalUnknown::Fixed(-0.0), 0));
    assert!(
        PreparedAleFsiBoundaryStep::canonical_p1(
            mesh,
            accepted.current_endpoint(),
            accepted.previous_endpoint(),
            values(0, 1),
            values(0, 0),
            accepted.exterior_facets().to_vec(),
            2.0
        )
        .is_err()
    );
    let mut missing = accepted.exterior_facets().to_vec();
    missing.pop();
    assert!(
        PreparedAleFsiBoundaryStep::canonical_p1(
            mesh,
            accepted.previous_endpoint(),
            accepted.current_endpoint(),
            values(0, 0),
            values(0, 1),
            missing,
            2.0
        )
        .is_err()
    );
    let mut duplicate = accepted.exterior_facets().to_vec();
    duplicate.push(duplicate[0]);
    assert!(
        PreparedAleFsiBoundaryStep::canonical_p1(
            mesh,
            accepted.previous_endpoint(),
            accepted.current_endpoint(),
            values(0, 0),
            values(0, 1),
            duplicate,
            2.0
        )
        .is_err()
    );
    let mut conflict = accepted.exterior_facets().to_vec();
    let role = if conflict[0].1 == AleFsiExteriorFacetDisposition::EssentialVelocity {
        AleFsiExteriorFacetDisposition::NaturalOutflow
    } else {
        AleFsiExteriorFacetDisposition::EssentialVelocity
    };
    conflict.push((conflict[0].0, role));
    assert!(
        PreparedAleFsiBoundaryStep::canonical_p1(
            mesh,
            accepted.previous_endpoint(),
            accepted.current_endpoint(),
            values(0, 0),
            values(0, 1),
            conflict,
            2.0
        )
        .is_err()
    );
    let all_vertices = exterior_facets(mesh)?
        .iter()
        .flat_map(|f| mesh.entity_vertices(*f).expect("facet"))
        .map(|v| v.index())
        .collect::<BTreeSet<_>>();
    let all_values = all_vertices
        .into_iter()
        .map(|v| (VertexId::new(v), [0.0; 2]))
        .collect::<Vec<_>>();
    let all_essential = exterior_facets(mesh)?
        .into_iter()
        .map(|f| (f, AleFsiExteriorFacetDisposition::EssentialVelocity))
        .collect();
    let err = PreparedAleFsiBoundaryStep::canonical_p1(
        mesh,
        accepted.previous_endpoint(),
        accepted.current_endpoint(),
        all_values.clone(),
        all_values,
        all_essential,
        2.0,
    )
    .expect_err("outlet omission");
    assert!(format!("{err:?}").contains("explicit natural outflow"));
    assert!(admits_raw(
        747_943,
        [539, 2057, 8030],
        [1518, 5973, 23694],
        [1024, 4096, 16384],
        8001,
        45,
        5,
        30_000
    ));
    assert!(!admits_raw(
        747_944,
        [539, 2057, 8030],
        [1518, 5973, 23694],
        [1024, 4096, 16384],
        8001,
        45,
        5,
        30_000
    ));
    assert!(!admits_raw(
        747_943,
        [539, 2057, 8031],
        [1518, 5973, 23694],
        [1024, 4096, 16384],
        8001,
        45,
        5,
        30_000
    ));
    assert!(!admits_raw(
        747_943,
        [539, 2057, 8030],
        [1518, 5973, 23694],
        [1024, 4096, 16384],
        8002,
        45,
        5,
        30_000
    ));
    assert!(!admits_raw(
        747_943,
        [539, 2057, 8030],
        [1518, 5973, 23695],
        [1024, 4096, 16384],
        8001,
        45,
        5,
        30_000
    ));
    assert!(!admits_raw(
        747_943,
        [539, 2057, 8030],
        [1518, 5973, 23694],
        [1024, 4096, 16385],
        8001,
        45,
        5,
        30_000
    ));
    assert!(!admits_raw(
        747_943,
        [539, 2057, 8030],
        [1518, 5973, 23694],
        [1024, 4096, 16384],
        8001,
        46,
        5,
        30_000
    ));
    assert!(!admits_raw(
        747_943,
        [539, 2057, 8030],
        [1518, 5973, 23694],
        [1024, 4096, 16384],
        8001,
        45,
        6,
        30_000
    ));
    assert!(!admits_raw(
        747_943,
        [539, 2057, 8030],
        [1518, 5973, 23694],
        [1024, 4096, 16384],
        8001,
        45,
        5,
        30_001
    ));
    Ok(())
}
