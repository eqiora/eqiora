//! Imported fixed-SI thermal equations through the common Cartesian Q1 lifecycle.
#[path = "standard_thermal_package/robin.rs"]
mod robin;
#[path = "standard_thermal_package/transient.rs"]
mod transient;
use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use eqiora::artifact::{
    CartesianMeshCellsV2, GeometryMeshCorrespondenceEnvelopeV1, MeshProductionLineageEnvelopeV1,
    ModelEnvelope,
};
use eqiora::compiler::StaticBindingValue;
use eqiora::geometry::GeometryGraph;
use eqiora::package::{
    SourceCompilation, SourceCompilationError, SourceImportContext, compile_source,
};
use eqiora::solver::{
    LinearSolver, REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER, SolverPlan,
};
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonLinearRequest, CommonResult, CommonSolvePolicy,
    CommonSpatialPolicy, ResolvedCommonPlan,
};

const IMPORT: &str = "import Eqiora.Scalar.scalar as scalar;\n";
const SIGNATURE: &str = r#"
public component Main(
    support body: volume(ambient_dimension = 2),
    support left: boundary(parent = body),
    support right: boundary(parent = body),
    support bottom: boundary(parent = body),
    support top: boundary(parent = body)
) {
"#;

fn fixed(face: &str) -> String {
    format!(
        "instance {face}_value: scalar.FixedTemperature2d(body=body, face={face}, temperature=temperature, value=300[K]);"
    )
}

fn heated_square(conductivity: u32, heating: u32) -> String {
    format!(
        "{IMPORT}{SIGNATURE}
 variable temperature: K on body in h1;
 parameter conductivity: W/(m*K) = {conductivity};
 instance heat: scalar.SteadyHeat2d(body=body, temperature=temperature,
   conductivity=conductivity, heating={heating}[W/m^3]);
 {} }}",
        ["left", "right", "bottom", "top"].map(fixed).join("\n")
    )
}

fn slab(outward: i32) -> String {
    format!(
        "{IMPORT}{SIGNATURE}
 variable temperature: K on body in smooth;
 parameter conductivity: W/(m*K) = 2;
 instance heat: scalar.SteadyHeat2d(body=body, temperature=temperature,
   conductivity=conductivity, heating=0[W/m^3]);
 {}
 instance right_flux: scalar.OutwardHeatFlux2d(body=body, face=right,
   temperature=temperature, conductivity=conductivity, flux={outward}[W/m^2]);
 instance bottom_flux: scalar.OutwardHeatFlux2d(body=body, face=bottom,
   temperature=temperature, conductivity=conductivity, flux=0[W/m^2]);
 instance top_flux: scalar.OutwardHeatFlux2d(body=body, face=top,
   temperature=temperature, conductivity=conductivity, flux=0[W/m^2]);
 }}",
        fixed("left")
    )
}

fn fixture(
    source: &str,
) -> Result<(SourceCompilation, AuthenticatedCommonMesh), SourceCompilationError> {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, 1.0], [0.0, 1.0]).unwrap();
    let edges = rectangle.boundaries();
    let geometry = graph
        .build(
            &rectangle,
            &BTreeMap::from([
                ("body".to_owned(), vec![rectangle.region().into()]),
                ("left".to_owned(), vec![edges[0].into()]),
                ("right".to_owned(), vec![edges[1].into()]),
                ("bottom".to_owned(), vec![edges[2].into()]),
                ("top".to_owned(), vec![edges[3].into()]),
            ]),
        )
        .unwrap();
    let body = geometry.entity_set("body").unwrap();
    let bindings = ["body", "left", "right", "bottom", "top"].map(|name| {
        (
            name,
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set(name).unwrap(),
                parent: (name != "body").then_some(body),
            },
        )
    });
    let document = compile_source(
        "thermal.eqi",
        source,
        Some("Main"),
        &bindings,
        SourceImportContext::Bundled,
    )?;
    let policy = CartesianMeshCellsV2::new([2, 2]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(&geometry, [2, 2])
            .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &policy,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let owner =
        AuthenticatedCommonMesh::structured_cartesian(geometry, mesh, correspondence, production)
            .unwrap();
    Ok((document, owner))
}

fn resolve(source: &str) -> Result<ResolvedCommonPlan, String> {
    resolve_temporal(source, None)
}

fn resolve_temporal(source: &str, step: Option<f64>) -> Result<ResolvedCommonPlan, String> {
    let (document, owner) = fixture(source).map_err(|error| format!("{error:?}"))?;
    assert!(matches!(document, SourceCompilation::Packaged(_)));
    let model = ModelEnvelope::from_program(document.model().program()).unwrap();
    let linear = CommonLinearRequest::exact(
        SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(1000).unwrap(),
        )
        .unwrap()
        .with_reduction(eqiora::solver::ReductionPolicy::Reproducible),
        REFERENCE_SOLVER_PROVIDER,
    )
    .unwrap();
    ResolvedCommonPlan::resolve(
        &model,
        owner,
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(linear),
        None,
        step.map(eqiora_numerics::CommonTemporalPolicy::backward_euler)
            .transpose()
            .map_err(|error| error.to_string())?,
        &REFERENCE_LINEAR_SOLVER,
        None,
        None,
    )
    .map_err(|error| error.to_string())
}

fn run_replayed(source: &str) -> [f64; 9] {
    let plan = resolve(source).unwrap();
    let bytes = plan.to_bytes().unwrap();
    let replay = ResolvedCommonPlan::from_bytes(&bytes, &REFERENCE_LINEAR_SOLVER, &[]).unwrap();
    assert_eq!(replay.to_bytes().unwrap(), bytes);
    let result = replay
        .as_spatial()
        .unwrap()
        .run_result(None, &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let result_bytes = result.to_bytes().unwrap();
    let restored = CommonResult::from_bytes(&result_bytes, &replay).unwrap();
    assert_eq!(restored.to_bytes().unwrap(), result_bytes);
    let (_, values, _) = restored.field_block(0, 0).unwrap();
    let owner = replay.authenticated_mesh().unwrap();
    let mesh = owner.cartesian_mesh().unwrap().mesh();
    // The independent stencil below uses x-minor indices. Interpret result
    // coefficients through the actual Mesh topology instead of assuming its
    // canonical vertex order matches the certificate's local indexing.
    let mut ordered = [0.0; 9];
    assert_eq!(values.len(), ordered.len());
    for (vertex, value) in values.iter().enumerate() {
        let index = mesh
            .vertex_multi_index(eqiora::meshing::MeshEntity::new(0, vertex))
            .unwrap();
        ordered[index[0] + 3 * index[1]] = *value;
    }
    ordered
}

#[test]
fn ordinary_import_compiles_fixed_si_temperature_and_flux_components() {
    fixture(&heated_square(1, 1)).unwrap();
    fixture(&slab(2)).unwrap();
    fixture(&slab(-2)).unwrap();
}

#[test]
fn imported_heat_matches_the_independent_single_hat_balance_and_replays() {
    for (conductivity, heating) in [(1, 1), (1, 2), (2, 1)] {
        let values = run_replayed(&heated_square(conductivity, heating));
        // Four h=1/2 Q1 cells: K_ii=8*k/3, integral(q*phi_i)=q/4.
        // This is the exact discrete balance, not a continuum profile claim.
        let expected = 300.0 + 3.0 * f64::from(heating) / (32.0 * f64::from(conductivity));
        for (index, value) in values.iter().enumerate() {
            assert!(value.is_finite());
            assert!((value - if index == 4 { expected } else { 300.0 }).abs() <= 1e-10);
        }
        // Existing heated-body specimen budgets: residual divided by 8/3
        // and subtraction near 300 K give 1e-10 K and 3e-10 W/m.
        assert!(
            ((8.0 * f64::from(conductivity) / 3.0) * (values[4] - 300.0)
                - f64::from(heating) / 4.0)
                .abs()
                <= 3e-10
        );
    }
}

#[test]
fn imported_heat_flux_has_the_physical_outward_sign() {
    for outward in [-2, 2] {
        certify_slab(&run_replayed(&slab(outward)), -f64::from(outward) / 2.0);
    }
}

// Independent exact integer Q1 assembly, h=1/2, k=2. A=6K/k=3K.
// Centering at 300 K is exact by Sterbenz for this temperature interval.
// An independently measured residual certifies assembly and solver error together.
fn certify_slab(temperatures: &[f64; 9], slope: f64) {
    assert_eq!(slope.abs(), 1.0);
    assert!(
        temperatures
            .iter()
            .all(|t| t.is_finite() && (150.0..=600.0).contains(t))
    );
    let values = temperatures.map(|t| t - 300.0);
    let mut matrix = [[0_i32; 9]; 9];
    let local = [
        [4, -1, -2, -1],
        [-1, 4, -1, -2],
        [-2, -1, 4, -1],
        [-1, -2, -1, 4],
    ];
    for y in 0..2 {
        for x in 0..2 {
            let vertices = [
                x + 3 * y,
                x + 1 + 3 * y,
                x + 1 + 3 * (y + 1),
                x + 3 * (y + 1),
            ];
            for i in 0..4 {
                for j in 0..4 {
                    matrix[vertices[i]][vertices[j]] += local[i][j];
                }
            }
        }
    }
    let mut matrix_norm = 0.0_f64;
    let mut load_norm = 0.0_f64;
    let mut max_residual = 0.0_f64;
    let mut max_error = 0.0_f64;
    for y in 0..3 {
        assert_eq!(values[3 * y], 0.0);
        for x in 1..3 {
            let row = x + 3 * y;
            // physical outward q=-k*slope, conormal load=-q. Multiply
            // its boundary-hat integral by 3; endpoint weights are h/2.
            let load = if x == 2 {
                slope * if y == 1 { 3.0 } else { 1.5 }
            } else {
                0.0
            };
            let mut residual = -load;
            let mut row_norm = 0;
            for column in 0..9 {
                if column % 3 == 0 {
                    continue;
                }
                residual += f64::from(matrix[row][column]) * values[column];
                row_norm += matrix[row][column].abs();
            }
            matrix_norm = matrix_norm.max(f64::from(row_norm));
            load_norm = load_norm.max(load.abs());
            assert!(residual.is_finite(), "independent residual must be finite");
            max_residual = max_residual.max(residual.abs());
            max_error = max_error.max((values[row] - slope * (x as f64) / 2.0).abs());
        }
    }
    // phi=x(2-x)/(2k) and the M-matrix comparison principle give
    // ||K^-1||inf <=2/(k*h^2)=4; therefore ||A^-1||inf <=4/3.
    // R is a predeclared independent residual acceptance, not an Eqiora
    // output or solver-error assumption. All norm inputs come from A,b.
    let inverse_bound = (4.0_f64 / 3.0).next_up();
    let residual_limit = 1e-9;
    assert!(
        max_residual <= residual_limit,
        "independent residual {max_residual}"
    );
    // At most nine products/additions: gamma20 bounds their rounding.
    // ||theta_exact||inf=1; solve e<=B[R+gamma20*(||A||(1+e)+||b||)].
    let roundoff = f64::EPSILON / 2.0;
    let gamma = ((20.0 * roundoff) / (1.0 - 20.0 * roundoff).next_down()).next_up();
    let rounding = (gamma * (matrix_norm + load_norm).next_up()).next_up();
    let numerator = (inverse_bound * (residual_limit + rounding).next_up()).next_up();
    let feedback = (inverse_bound * (gamma * matrix_norm).next_up()).next_up();
    let denominator = (1.0 - feedback).next_down();
    assert!(denominator > 0.0);
    let bound = (numerator / denominator).next_up();
    assert!(
        max_error <= bound,
        "nodal error {max_error} exceeds {bound}"
    );
    // Nonnegative Q1 shape functions partition unity, so the bound also
    // holds everywhere in the square. Reversed sign differs by 2 K at x=1.
    assert!((values[2] + slope).abs() > 1.0);
}

#[test]
fn independent_slab_certificate_rejects_nonfinite_and_reversed_fields() {
    let exact = std::array::from_fn(|i| 300.0 + (i % 3) as f64 / 2.0);
    certify_slab(&exact, 1.0);
    assert!(std::panic::catch_unwind(|| certify_slab(&exact, -1.0)).is_err());
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut values = exact;
        values[4] = invalid;
        assert!(std::panic::catch_unwind(|| certify_slab(&values, 1.0)).is_err());
    }
}

#[test]
fn thermal_expression_units_and_regularity_reject_before_solving() {
    let good = heated_square(1, 1);
    for source in [
        good.replace("variable temperature: K", "variable temperature: 1"),
        good.replace("conductivity=conductivity", "conductivity=1"),
        good.replace("heating=1[W/m^3]", "heating=1"),
        good.replace("value=300[K]", "value=300"),
        slab(2).replace("flux=2[W/m^2]", "flux=2[K/m]"),
        slab(2).replace("in smooth", "in h1"),
        good.replace("ambient_dimension = 2", "ambient_dimension = 3"),
        good.replace("variable temperature:", "domain foreign = box(0,1,0,1); domain other = boundary(foreign,axis=0,side=upper); variable temperature:")
            .replace("face=right", "face=other"),
    ] {
        let error = fixture(&source).expect_err("invalid thermal binding must reject");
        assert!(error.diagnostics().is_some_and(|diagnostics| !diagnostics.is_empty()),
            "expected typed source diagnostics: {error}");
    }
}

#[test]
fn physical_heat_flux_requires_the_volume_conductivity() {
    let source = slab(2).replace(
        "temperature=temperature, conductivity=conductivity, flux=2[W/m^2]",
        "temperature=temperature, conductivity=3[W/(m*K)], flux=2[W/m^2]",
    );
    let error = resolve(&source).expect_err("incompatible constitutive coefficient must reject");
    assert!(error.contains("complete constitutive flux"), "{error}");
}
