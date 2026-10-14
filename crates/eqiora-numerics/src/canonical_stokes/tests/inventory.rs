use super::{SOURCE, TRANSIENT_NAVIER_STOKES_SOURCE, compile_program};
use crate::canonical_stokes::{inertial, navier_stokes, recognize};
use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_graph::{GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::{DomainDef, DomainKind, GeometryDigest, KernelNode};
use eqiora_sem::KernelProgram;

fn steady(program: &KernelProgram) -> Result<(), Diagnostic> {
    super::lower_steady_incompressible_stokes_cartesian_2d(program).map(|_| ())
}

fn transient(program: &KernelProgram) -> Result<(), Diagnostic> {
    super::lower_transient_incompressible_navier_stokes_cartesian::<2>(program).map(|_| ())
}

fn inertial(program: &KernelProgram) -> Result<(), Diagnostic> {
    let (domain, bounds) = recognize::unique_box_2d(program)?;
    inertial::lower_inertial_incompressible_newtonian_subdomain_2d_with_boundaries(
        program, domain, bounds, None,
    )
    .map(|_| ())
}

fn assert_denied(result: Result<(), Diagnostic>, message: &str) {
    let error = result.expect_err("invalid flow inventory must be rejected");
    assert_eq!(error.code(), codes::INVALID_SPATIAL_LOWERING);
    assert!(error.message().contains(message), "{error:?}");
}

#[test]
fn flow_families_preserve_field_relation_and_boundary_admission() {
    let inertial_source = TRANSIENT_NAVIER_STOKES_SOURCE.replace(
        "      + div(rho * outer_product(left = velocity, right = velocity))\n",
        "",
    );
    type Recognizer = fn(&KernelProgram) -> Result<(), Diagnostic>;
    for (source, recognize) in [
        (SOURCE, steady as Recognizer),
        (TRANSIENT_NAVIER_STOKES_SOURCE, transient as Recognizer),
        (inertial_source.as_str(), inertial as Recognizer),
    ] {
        recognize(&compile_program(source)).expect("ordinary flow family is admitted first");
        // A second pressure candidate cannot be silently ignored or chosen by name.
        let ambiguous = source.replace(
            "  parameter mu:",
            "  variable extra_pressure: kg / (m * s ^ 2) on fluid in h1;\n  parameter mu:",
        );
        assert_denied(recognize(&compile_program(&ambiguous)), "pressure");
        // Keep the relation count fixed so the duplicate reaches the continuity selector.
        let duplicate = source.replace(
            if source == SOURCE {
                "force_potential - load_scale * coordinate(0) / length_scale = 0;"
            } else {
                "force_potential - zero_pressure = 0;"
            },
            "div(velocity) = 0;",
        );
        assert_denied(recognize(&compile_program(&duplicate)), "exactly one `div(");
        let unrelated = source.replace(
            "  relation force on fluid {",
            "  relation unrelated on fluid { pressure = 0; }\n  relation force on fluid {",
        );
        assert_denied(recognize(&compile_program(&unrelated)), "Relation");
        let missing = source
            .replace(
                "  domain x_lower = boundary(fluid, axis = 0, side = lower);",
                "",
            )
            .replace(
                "  relation x_lower_zero on x_lower { trace(velocity) = 0; }",
                "",
            );
        assert_denied(
            recognize(&compile_program(&missing)),
            "missing Cartesian boundary",
        );
    }
}

fn geometry_program(source: &str) -> KernelProgram {
    let graph = eqiora_geometry::GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, 2.0], [-1.0, 1.0]).unwrap();
    let edges = rectangle.boundaries();
    let geometry = graph
        .build(
            &rectangle,
            &std::collections::BTreeMap::from([
                ("fluid".to_owned(), vec![rectangle.region().into()]),
                ("face-0-Lower".to_owned(), vec![edges[0].into()]),
                ("face-0-Upper".to_owned(), vec![edges[1].into()]),
                ("face-1-Lower".to_owned(), vec![edges[2].into()]),
                ("face-1-Upper".to_owned(), vec![edges[3].into()]),
            ]),
        )
        .unwrap();
    let mut compiled = eqiora_compiler::compile("geometry-flow.eqi", source).unwrap();
    let (original, model, _) = compiled.remove(0).into_parts();
    let mut transaction = Transaction::new("flow equations on named geometry supports");
    for op in original.ops() {
        let replacement = match op {
            Op::DefineKernelNode {
                node: KernelNode::Domain(domain),
            } => match domain.kind() {
                DomainKind::CartesianBox { .. } => Some(
                    DomainDef::geometry_region(
                        domain.id(),
                        GeometryDigest::new(geometry.digest_bytes()),
                        "fluid",
                    )
                    .unwrap(),
                ),
                DomainKind::CartesianBoundary { axis, side } => Some(
                    DomainDef::geometry_boundary(domain.id(), format!("face-{axis}-{side:?}"))
                        .unwrap(),
                ),
                _ => None,
            },
            _ => None,
        };
        transaction.push(replacement.map_or_else(
            || op.clone(),
            |domain| Op::DefineKernelNode {
                node: domain.into(),
            },
        ));
    }
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry]).unwrap()
}

#[test]
fn geometry_flow_recognition_preserves_closure_and_missing_support_diagnostics() {
    type Recognizer = fn(&KernelProgram) -> Result<(), Diagnostic>;
    for (source, recognize, context) in [
        (
            SOURCE,
            (|p| {
                recognize::recognize_steady_incompressible_stokes_geometry_mathematics(p)
                    .map(|_| ())
            }) as Recognizer,
            "closed 2D steady Stokes",
        ),
        (
            TRANSIENT_NAVIER_STOKES_SOURCE,
            navier_stokes::recognize_transient_incompressible_navier_stokes_geometry_mathematics
                as Recognizer,
            "closed Geometry-backed transient",
        ),
    ] {
        recognize(&geometry_program(source)).expect("named geometry flow is admitted first");
        let unused = source.replace(
            "  parameter mu:",
            "  parameter unused: m = 1;\n  parameter mu:",
        );
        assert_denied(recognize(&geometry_program(&unused)), context);
        let no_boundaries = source
            .lines()
            .filter(|line| !line.contains("= boundary(") && !line.contains("trace(velocity) = 0;"))
            .collect::<Vec<_>>()
            .join("\n");
        assert_denied(
            recognize(&geometry_program(&no_boundaries)),
            "boundary supports",
        );
    }
}
