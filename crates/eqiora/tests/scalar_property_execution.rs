use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use eqiora::api::ModelDocument;
use eqiora::artifact::{
    CartesianMeshCellsV2, GeometryMeshCorrespondenceEnvelopeV1, MeshProductionLineageEnvelopeV1,
    ModelEnvelope,
};
use eqiora::geometry::{CanonicalGeometryV1, GeometryGraph, PlanarTopologyHandle};
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageDependencyV1, PackageManifestV1, PackageReleaseV1, PackageSourcesV1,
    PackagedModelDocument, QualifiedName, ResolutionRecordV1, SourceFileV1,
    prepare_package_release_v1,
};
use eqiora::solver::REFERENCE_LINEAR_SOLVER;
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonLinearPlan, CommonResult, CommonSolvePolicy, CommonSpatialPolicy,
};

const VERSION: &str = "1.0.0";
const SOURCE_PATH: &str = "src/main.eqi";
const PROPERTY_PACKAGE: &str = "org.example.ScalarProperties";
const NORMALIZED_DIFFUSIVITY: f64 = 0.025;
const PARAMETERS: &[(&str, f64)] = &[
    ("wave_number", std::f64::consts::PI),
    ("source_scale", 1.0),
    ("boundary_offset", 0.0),
];

#[test]
fn typed_material_composition_runs_as_the_same_effective_multi_property_law() {
    let geometry = rectangle_geometry();
    let composed_source = material_source(true, 2, 4, false);
    let composed_release = release("org.example.ComposedDiffusion", &composed_source, &[]);
    let composed = compile_root_component(&composed_release, "ExecutableDiffusion", &geometry)
        .expect("material-composed Component compiles");
    let compile_parameters = parameter_expressions();
    let compile_bindings = spatial_bindings(&geometry, &compile_parameters);
    let direct = ModelDocument::compile_selected(
        "direct-material-law.eqi",
        &material_source(false, 2, 4, false),
        "ExecutableDiffusion",
        &compile_bindings,
    )
    .expect("direct multi-parameter Law compiles");

    let bindings = composed.property_bindings().collect::<Vec<_>>();
    assert_eq!(bindings.len(), 2);
    assert!(bindings.iter().all(|binding| binding.0.is_some()));
    let direct_property_source = composed_source.replace(
        "conductivity = ReferenceMaterial.conductivity, capacity = ReferenceMaterial.capacity",
        "conductivity = ConductivityValue, capacity = CapacityValue",
    );
    let direct_property_release = release(
        "org.example.ComposedDiffusion",
        &direct_property_source,
        &[],
    );
    let direct_property =
        compile_root_component(&direct_property_release, "ExecutableDiffusion", &geometry)
            .expect("direct property bindings compile");
    assert!(
        direct_property
            .property_bindings()
            .all(|binding| binding.0.is_none())
    );
    assert!(
        composed
            .model()
            .structurally_equivalent(direct_property.model())
            .unwrap()
    );
    assert!(composed.model().structurally_equivalent(&direct).unwrap());
    assert_same_scalar_result(
        &resolve_scalar(composed.model(), &geometry)
            .run_result(&REFERENCE_LINEAR_SOLVER)
            .expect("composed material runs"),
        &resolve_scalar(&direct, &geometry)
            .run_result(&REFERENCE_LINEAR_SOLVER)
            .expect("direct material Law runs"),
    );

    let reordered_source = material_source(true, 2, 4, true);
    let reordered_release = release("org.example.ComposedDiffusion", &reordered_source, &[]);
    assert_eq!(
        composed_release.package_identity().unwrap().semantic_digest,
        reordered_release
            .package_identity()
            .unwrap()
            .semantic_digest,
        "material property ordering must not alter canonical composition identity"
    );

    let compatible_source = material_source(true, 4, 8, false);
    let compatible_release = release("org.example.ComposedDiffusion", &compatible_source, &[]);
    let compatible = compile_root_component(&compatible_release, "ExecutableDiffusion", &geometry)
        .expect("second compatible exact material compiles");
    assert_same_scalar_result(
        &resolve_scalar(composed.model(), &geometry)
            .run_result(&REFERENCE_LINEAR_SOLVER)
            .expect("first compatible material runs"),
        &resolve_scalar(compatible.model(), &geometry)
            .run_result(&REFERENCE_LINEAR_SOLVER)
            .expect("second compatible material runs"),
    );
    assert_ne!(
        composed.model().digest().unwrap(),
        compatible.model().digest().unwrap(),
        "changing exact releases must change Model identity"
    );
}

#[test]
fn public_material_composition_crosses_an_exact_package_boundary() {
    let properties = release(
        "org.example.Materials",
        r#"
public property contract Conductivity(): 1 { derivatives value_only; }
public property contract Capacity(): 1 { derivatives value_only; }
property release ConductivityA: Conductivity {
  analytic { value = 2; source_unit: 1 = 1; } validity unconditional; outside reject; branch single;
  citation org.example.a; license spdx.CC0_1_0;
}
property release CapacityA: Capacity {
  analytic { value = 4; source_unit: 1 = 1; } validity unconditional; outside reject; branch single;
  citation org.example.a; license spdx.CC0_1_0;
}
public material composition MaterialA {
  property conductivity = ConductivityA;
  property capacity = CapacityA;
}
"#,
        &[],
    );
    let consumer = release(
        "org.example.MaterialConsumer",
        r#"
public component Law(
  property conductivity: props.Conductivity,
  property capacity: props.Capacity
) {
  relation law { conductivity / capacity = 0; }
}
public model Main() { instance law: Law(conductivity = props.MaterialA.conductivity, capacity = props.MaterialA.capacity); }
"#,
        &[("props", &properties)],
    );
    let resolution =
        ResolutionRecordV1::from_exact_releases(&consumer, std::slice::from_ref(&properties))
            .expect("material dependency lock");
    let mut store = InMemoryPackageStore::default();
    store.insert(&properties).expect("store materials");
    store.insert(&consumer).expect("store consumer");
    let compiled = PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("public composition compiles with private exact releases");
    assert_eq!(compiled.property_bindings().len(), 2);
    assert!(compiled.property_bindings().all(|binding| {
        binding
            .0
            .is_some_and(|composition| composition.ends_with("::MaterialA"))
    }));
}

#[test]
fn one_exact_release_runs_through_two_independent_common_scalar_consumers() {
    let geometry = rectangle_geometry();
    let properties = property_release(25, "org.example.measurement", "Diffusivity");

    for consumer in [Consumer::Potential, Consumer::Temperature] {
        let property = compile_property_consumer(&properties, consumer, &geometry)
            .expect("exact property-bound Component compiles against caller Geometry");
        let compile_parameters = parameter_expressions();
        let compile_bindings = spatial_bindings(&geometry, &compile_parameters);
        let direct = ModelDocument::compile_selected(
            "direct-scalar-consumer.eqi",
            &consumer.source(false),
            consumer.wrapper(),
            &compile_bindings,
        )
        .expect("direct Parameter Component compiles against the same Geometry");

        assert_eq!(property.property_bindings().len(), 1);
        let binding = property.property_bindings().next().unwrap();
        assert_eq!(binding.4, consumer.requirement());
        assert_eq!(
            binding
                .5
                .constant_value()
                .unwrap()
                .real_scalar_value()
                .unwrap()
                .value(),
            NORMALIZED_DIFFUSIVITY
        );
        assert_eq!(binding.6, "unconditional");
        assert_eq!(binding.7, "org.example.measurement");
        assert_eq!(binding.8, "spdx.CC0_1_0");
        assert!(property.model().structurally_equivalent(&direct).unwrap());

        let property_plan = resolve_scalar(property.model(), &geometry);
        let direct_plan = resolve_scalar(&direct, &geometry);
        assert_eq!(
            property.compilation().model_digest().to_hex(),
            property_plan.model_digest(),
            "the common Plan must retain the exact package compilation's Model identity"
        );
        assert_ne!(property_plan.identity(), direct_plan.identity());
        let property_result = property_plan
            .run_result(&REFERENCE_LINEAR_SOLVER)
            .expect("property Plan runs");
        let direct_result = direct_plan
            .run_result(&REFERENCE_LINEAR_SOLVER)
            .expect("direct Plan runs");
        assert_eq!(
            property_result.plan().model_digest(),
            property.compilation().model_digest().to_hex(),
            "the Result must retain the package compilation's Model lineage"
        );
        assert_eq!(
            property.property_bindings().next().unwrap().7,
            "org.example.measurement",
            "execution must not replace exact provenance with the normalized scalar"
        );
        assert_same_scalar_result(&property_result, &direct_result);
    }
}

#[test]
fn provenance_and_value_changes_keep_identity_and_execution_roles_distinct() {
    let geometry = rectangle_geometry();
    let baseline_release = property_release(25, "org.example.measurement", "Diffusivity");
    let provenance_release = property_release(25, "org.example.remeasurement", "Diffusivity");
    let value_release = property_release(50, "org.example.measurement", "Diffusivity");
    let reordered_release = reordered_property_release();

    let baseline = compile_property_consumer(&baseline_release, Consumer::Potential, &geometry)
        .expect("baseline property package compiles");
    let provenance = compile_property_consumer(&provenance_release, Consumer::Potential, &geometry)
        .expect("provenance variant compiles");
    let value = compile_property_consumer(&value_release, Consumer::Potential, &geometry)
        .expect("value variant compiles");
    let reordered = compile_property_consumer(&reordered_release, Consumer::Potential, &geometry)
        .expect("declaration-order variant compiles");
    assert_ne!(
        baseline.model().digest().unwrap(),
        provenance.model().digest().unwrap()
    );
    assert_ne!(
        baseline.model().digest().unwrap(),
        value.model().digest().unwrap()
    );

    let baseline_plan = resolve_scalar(baseline.model(), &geometry);
    let provenance_plan = resolve_scalar(provenance.model(), &geometry);
    let value_plan = resolve_scalar(value.model(), &geometry);
    let reordered_plan = resolve_scalar(reordered.model(), &geometry);
    assert_ne!(baseline_plan.identity(), provenance_plan.identity());
    assert_ne!(baseline_plan.identity(), value_plan.identity());

    let baseline_result = baseline_plan
        .run_result(&REFERENCE_LINEAR_SOLVER)
        .expect("baseline runs");
    let provenance_result = provenance_plan
        .run_result(&REFERENCE_LINEAR_SOLVER)
        .expect("provenance variant runs");
    let value_result = value_plan
        .run_result(&REFERENCE_LINEAR_SOLVER)
        .expect("value variant runs");
    let reordered_result = reordered_plan
        .run_result(&REFERENCE_LINEAR_SOLVER)
        .expect("declaration-order variant runs");
    assert_same_scalar_result(&baseline_result, &provenance_result);
    assert!(
        baseline
            .model()
            .structurally_equivalent(reordered.model())
            .unwrap(),
        "property declaration order must not alter the effective Law"
    );
    assert_same_scalar_result(&baseline_result, &reordered_result);
    assert_ne!(
        baseline_result.field_block(0, 0).unwrap().1,
        value_result.field_block(0, 0).unwrap().1,
        "a changed normalized coefficient must reach numerical execution"
    );

    let wrong_contract = property_release(25, "org.example.measurement", "OtherDiffusivity");
    assert!(
        compile_property_consumer(&wrong_contract, Consumer::Potential, &geometry).is_err(),
        "a nominally foreign release must fail before a Model or Plan is exposed"
    );
}

#[derive(Clone, Copy)]
enum Consumer {
    Potential,
    Temperature,
}

impl Consumer {
    const fn package(self) -> &'static str {
        match self {
            Self::Potential => "org.example.PotentialDiffusion",
            Self::Temperature => "org.example.TemperatureDiffusion",
        }
    }

    const fn core(self) -> &'static str {
        match self {
            Self::Potential => "PotentialDiffusion",
            Self::Temperature => "TemperatureDiffusion",
        }
    }

    const fn wrapper(self) -> &'static str {
        match self {
            Self::Potential => "ExecutablePotentialDiffusion",
            Self::Temperature => "ExecutableTemperatureDiffusion",
        }
    }

    const fn field(self) -> &'static str {
        match self {
            Self::Potential => "potential",
            Self::Temperature => "temperature",
        }
    }

    const fn requirement(self) -> &'static str {
        match self {
            Self::Potential => "diffusivity",
            Self::Temperature => "conductivity",
        }
    }

    fn source(self, property: bool) -> String {
        let coefficient_declaration = if property {
            format!("  property {}: props.Diffusivity,", self.requirement())
        } else {
            format!("  parameter {}: 1,", self.requirement())
        };
        let coefficient_binding = if property {
            format!("{} = props.ReferenceDiffusivity", self.requirement())
        } else {
            format!("{} = {NORMALIZED_DIFFUSIVITY}", self.requirement())
        };
        format!(
            r#"
public component {core}(
  support square: volume(ambient_dimension = 2),
  support x_lower: boundary(parent = square),
  support x_upper: boundary(parent = square),
  support y_lower: boundary(parent = square),
  support y_upper: boundary(parent = square),
{coefficient_declaration}
  parameter wave_number: 1 / m,
  parameter source_scale: 1 / m ^ 2,
  parameter boundary_offset: 1
) {{
  variable {field}: 1 on square in h1;
  relation balance on square {{
    -div({coefficient} * grad({field}))
      - source_scale * math.sin(wave_number * coordinate(0))
        * math.sin(wave_number * coordinate(1)) = 0;
  }}
  relation x_lower_value on x_lower {{ trace({field}) - boundary_offset = 0; }}
  relation x_upper_value on x_upper {{ trace({field}) - boundary_offset = 0; }}
  relation y_lower_value on y_lower {{ trace({field}) - boundary_offset = 0; }}
  relation y_upper_value on y_upper {{ trace({field}) - boundary_offset = 0; }}
}}

public component {wrapper}(
  support square: volume(ambient_dimension = 2),
  support x_lower: boundary(parent = square),
  support x_upper: boundary(parent = square),
  support y_lower: boundary(parent = square),
  support y_upper: boundary(parent = square),
  parameter wave_number: 1 / m,
  parameter source_scale: 1 / m ^ 2,
  parameter boundary_offset: 1
) {{
  instance equation: {core}(
    square = square,
    x_lower = x_lower,
    x_upper = x_upper,
    y_lower = y_lower,
    y_upper = y_upper,
    {coefficient_binding},
    wave_number = wave_number,
    source_scale = source_scale,
    boundary_offset = boundary_offset
  );
}}
"#,
            core = self.core(),
            wrapper = self.wrapper(),
            field = self.field(),
            coefficient = self.requirement(),
        )
    }
}

fn property_release(value: u32, citation: &str, contract: &str) -> PackageReleaseV1 {
    let source = format!(
        r#"
public property contract Diffusivity(): 1 {{ derivatives value_only; }}
public property contract OtherDiffusivity(): 1 {{ derivatives value_only; }}
public property release ReferenceDiffusivity: {contract} {{
  analytic {{ value = {value};
  source_unit: 1 = 1 / 1000; }}
  validity unconditional; outside reject; branch single;
  citation {citation};
  license spdx.CC0_1_0;
}}
"#
    );
    release(PROPERTY_PACKAGE, &source, &[])
}

fn reordered_property_release() -> PackageReleaseV1 {
    release(
        PROPERTY_PACKAGE,
        r#"
public property contract OtherDiffusivity(): 1 { derivatives value_only; }
public property contract Diffusivity(): 1 { derivatives value_only; }
public property release ReferenceDiffusivity: Diffusivity {
  analytic { value = 25;
  source_unit: 1 = 1 / 1000; }
  validity unconditional; outside reject; branch single;
  citation org.example.measurement;
  license spdx.CC0_1_0;
}
"#,
        &[],
    )
}

fn compile_property_consumer(
    properties: &PackageReleaseV1,
    consumer: Consumer,
    geometry: &CanonicalGeometryV1,
) -> Result<PackagedModelDocument, String> {
    let root = try_release(
        consumer.package(),
        &consumer.source(true),
        &[("props", properties)],
    )?;
    let resolution =
        ResolutionRecordV1::from_exact_releases(&root, std::slice::from_ref(properties))
            .map_err(|error| error.to_string())?;
    let mut store = InMemoryPackageStore::default();
    store
        .insert(properties)
        .map_err(|error| error.to_string())?;
    store.insert(&root).map_err(|error| error.to_string())?;
    let parameters = parameter_expressions();
    let bindings = spatial_bindings(geometry, &parameters);
    PackagedModelDocument::compile_selected(&store, &resolution, consumer.wrapper(), &bindings)
        .map_err(|error| error.to_string())
}

fn compile_root_component(
    root: &PackageReleaseV1,
    component: &str,
    geometry: &CanonicalGeometryV1,
) -> Result<PackagedModelDocument, String> {
    let resolution =
        ResolutionRecordV1::from_exact_releases(root, &[]).map_err(|error| error.to_string())?;
    let mut store = InMemoryPackageStore::default();
    store.insert(root).map_err(|error| error.to_string())?;
    let parameters = parameter_expressions();
    let bindings = spatial_bindings(geometry, &parameters);
    PackagedModelDocument::compile_selected(&store, &resolution, component, &bindings)
        .map_err(|error| error.to_string())
}

fn material_source(composed: bool, conductivity: u32, capacity: u32, reverse: bool) -> String {
    let coefficient_declarations = if composed {
        "  property conductivity: Conductivity,\n  property capacity: Capacity,"
    } else {
        "  parameter conductivity: 1,\n  parameter capacity: 1,"
    };
    let coefficient_bindings = if composed {
        "conductivity = ReferenceMaterial.conductivity, capacity = ReferenceMaterial.capacity"
            .to_owned()
    } else {
        format!("conductivity = {conductivity}, capacity = {capacity}")
    };
    let material = if composed {
        let properties = if reverse {
            "  property conductivity = ConductivityValue;\n  property capacity = CapacityValue;"
        } else {
            "  property capacity = CapacityValue;\n  property conductivity = ConductivityValue;"
        };
        format!(
            r#"public property contract Conductivity(): 1 {{ derivatives value_only; }}
public property contract Capacity(): 1 {{ derivatives value_only; }}
public property release ConductivityValue: Conductivity {{
  analytic {{ value = {conductivity}; source_unit: 1 = 1; }} validity unconditional; outside reject; branch single;
  citation org.example.measurement; license spdx.CC0_1_0;
}}
public property release CapacityValue: Capacity {{
  analytic {{ value = {capacity}; source_unit: 1 = 1; }} validity unconditional; outside reject; branch single;
  citation org.example.measurement; license spdx.CC0_1_0;
}}
public material composition ReferenceMaterial {{
{properties}
}}
"#
        )
    } else {
        String::new()
    };
    format!(
        r#"{material}
public component DiffusionLaw(
  support square: volume(ambient_dimension = 2),
  support x_lower: boundary(parent = square),
  support x_upper: boundary(parent = square),
  support y_lower: boundary(parent = square),
  support y_upper: boundary(parent = square),
  parameter wave_number: 1 / m,
  parameter source_scale: 1 / m ^ 2,
  parameter boundary_offset: 1,
{coefficient_declarations}
) {{
  variable potential: 1 on square in h1;
  relation balance on square {{
    -div((conductivity / capacity) * grad(potential))
      - source_scale * math.sin(wave_number * coordinate(0))
        * math.sin(wave_number * coordinate(1)) = 0;
  }}
  relation x_lower_value on x_lower {{ trace(potential) - boundary_offset = 0; }}
  relation x_upper_value on x_upper {{ trace(potential) - boundary_offset = 0; }}
  relation y_lower_value on y_lower {{ trace(potential) - boundary_offset = 0; }}
  relation y_upper_value on y_upper {{ trace(potential) - boundary_offset = 0; }}
}}

public component ExecutableDiffusion(
  support square: volume(ambient_dimension = 2),
  support x_lower: boundary(parent = square),
  support x_upper: boundary(parent = square),
  support y_lower: boundary(parent = square),
  support y_upper: boundary(parent = square),
  parameter wave_number: 1 / m,
  parameter source_scale: 1 / m ^ 2,
  parameter boundary_offset: 1
) {{
  instance equation: DiffusionLaw(
    square = square,
    x_lower = x_lower,
    x_upper = x_upper,
    y_lower = y_lower,
    y_upper = y_upper,
    {coefficient_bindings},
    wave_number = wave_number,
    source_scale = source_scale,
    boundary_offset = boundary_offset
  );
}}
"#
    )
}

fn release(
    name: &str,
    source: &str,
    dependencies: &[(&str, &PackageReleaseV1)],
) -> PackageReleaseV1 {
    try_release(name, source, dependencies).expect("prepare exact package release")
}

fn try_release(
    name: &str,
    source: &str,
    dependencies: &[(&str, &PackageReleaseV1)],
) -> Result<PackageReleaseV1, String> {
    let path = NormalizedRelativePath::parse(SOURCE_PATH).unwrap();
    let imports = dependencies
        .iter()
        .map(|(alias, release)| {
            let name = &release.package_identity().expect("package identity").name;
            format!("import {name}.main as {alias};")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let source = if imports.is_empty() {
        source.to_owned()
    } else {
        format!("{imports}\n{source}")
    };
    let requirements = dependencies
        .iter()
        .map(|(_, release)| PackageDependencyV1::new(release.package_identity().unwrap()))
        .collect();
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse(name).unwrap(),
        ExactVersion::parse(VERSION).unwrap(),
        requirements,
        vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
    )
    .unwrap();
    let sources = PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            path,
            BundleRoleV1::ModelSource,
            source.into_bytes(),
        )],
    )
    .unwrap();
    let exact_dependencies = dependencies
        .iter()
        .map(|(_, release)| (*release).clone())
        .collect::<Vec<_>>();
    prepare_package_release_v1(sources, &exact_dependencies).map_err(|error| error.to_string())
}

fn rectangle_geometry() -> CanonicalGeometryV1 {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, 1.0], [0.0, 1.0]).unwrap();
    let edges = rectangle.boundaries();
    graph
        .build(
            &rectangle,
            &BTreeMap::from([
                ("square".to_owned(), vec![rectangle.region().into()]),
                (
                    "x_lower".to_owned(),
                    vec![PlanarTopologyHandle::from(edges[0])],
                ),
                (
                    "x_upper".to_owned(),
                    vec![PlanarTopologyHandle::from(edges[1])],
                ),
                (
                    "y_lower".to_owned(),
                    vec![PlanarTopologyHandle::from(edges[2])],
                ),
                (
                    "y_upper".to_owned(),
                    vec![PlanarTopologyHandle::from(edges[3])],
                ),
            ]),
        )
        .unwrap()
}

fn resolve_scalar(document: &ModelDocument, geometry: &CanonicalGeometryV1) -> CommonLinearPlan {
    let cells = CartesianMeshCellsV2::new([6, 6]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(
            geometry,
            cells.cells().try_into().unwrap(),
        )
        .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &cells,
        geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let owner = AuthenticatedCommonMesh::structured_cartesian(
        geometry.clone(),
        mesh,
        correspondence,
        production,
    )
    .unwrap();
    let solver = CommonSolvePolicy::Linear(
        eqiora_numerics::CommonLinearRequest::exact(
            eqiora::solver::SolverPlan::new(
                eqiora::solver::LinearSolver::BiConjugateGradientStabilized,
                1.0e-10,
                1.0e-12,
                NonZeroUsize::new(10_000).unwrap(),
            )
            .unwrap()
            .with_preconditioner(eqiora::solver::PreconditionerPolicy::Identity)
            .with_reduction(eqiora::solver::ReductionPolicy::Reproducible),
            eqiora::solver::REFERENCE_SOLVER_PROVIDER,
        )
        .unwrap(),
    );
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    eqiora_numerics::ResolvedCommonPlan::resolve(
        &model,
        owner,
        CommonSpatialPolicy::Q1,
        solver,
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap()
    .as_linear()
    .cloned()
    .expect("fixture retains its admitted scalar Plan")
}

fn assert_same_scalar_result(left: &CommonResult, right: &CommonResult) {
    assert_eq!(left.family_name(), "linear");
    assert_eq!(right.family_name(), "linear");
    assert_eq!(left.field_count(), 1);
    assert_eq!(right.field_count(), 1);
    let left = left.field_block(0, 0).unwrap();
    let right = right.field_block(0, 0).unwrap();
    assert_eq!(left.0, right.0);
    assert_eq!(left.1, right.1);
    assert_eq!(left.2, right.2);
}

fn spatial_bindings<'a>(
    geometry: &'a CanonicalGeometryV1,
    parameters: &'a [(&'static str, eqiora::language::Expr)],
) -> Vec<(&'static str, eqiora::compiler::StaticBindingValue<'a>)> {
    use eqiora::compiler::StaticBindingValue;
    let square = geometry.entity_set("square").unwrap();
    let mut bindings = vec![(
        "square",
        StaticBindingValue::GeometrySupport {
            geometry,
            selection: square,
            parent: None,
        },
    )];
    for name in ["x_lower", "x_upper", "y_lower", "y_upper"] {
        bindings.push((
            name,
            StaticBindingValue::GeometrySupport {
                geometry,
                selection: geometry.entity_set(name).unwrap(),
                parent: Some(square),
            },
        ));
    }
    bindings.extend(
        parameters
            .iter()
            .map(|(name, value)| (*name, StaticBindingValue::Expression(value))),
    );
    bindings
}

fn parameter_expressions() -> Vec<(&'static str, eqiora::language::Expr)> {
    PARAMETERS
        .iter()
        .map(|(name, value)| {
            (
                *name,
                eqiora::language::SourceAstFactory::expression(
                    eqiora::language::ExprKind::Number(
                        eqiora::language::DecimalLiteral::from_f64(*value)
                            .expect("finite fixture literal"),
                    ),
                    eqiora::language::TextRange::new(0, 0),
                )
                .unwrap(),
            )
        })
        .collect()
}
