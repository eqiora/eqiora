//! Narrow external Gmsh producer for the admitted exact-cylinder mesh path.

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::process::CommandExt as _;

use eqiora::Diagnostic;
use eqiora::artifact::{
    GeometryMeshCorrespondenceEnvelopeV1, GmshMeshPolicyV1, SimplicialMeshEnvelopeV1,
};
use eqiora::diagnostic::codes;
use eqiora::geometry::CanonicalGeometryV1;
use eqiora::io::gmsh::{Msh41Policy, import_msh41};
use eqiora::meshing::MeshQualityGate;
#[cfg(unix)]
use rustix::process::{Pid, Signal, kill_process_group};

const GMSH_VERSION: &str = "4.15.2";
const GMSH_ENV: &str = "EQIORA_GMSH";
const VERSION_TIMEOUT: Duration = Duration::from_secs(5);
const PROCESS_TIMEOUT: Duration = Duration::from_secs(30);
const PROCESS_FILE_LIMIT_FLOOR_BYTES: usize = 1024 * 1024;
const FAILURE_DETAIL_LIMIT_BYTES: usize = 4096;
static SCRATCH_NONCE: AtomicU64 = AtomicU64::new(0);

pub(super) struct GeneratedGmshMesh {
    pub(super) provider_output: Vec<u8>,
    pub(super) mesh: SimplicialMeshEnvelopeV1,
    pub(super) correspondence: GeometryMeshCorrespondenceEnvelopeV1,
}

pub(super) fn generate(
    source: &CanonicalGeometryV1,
    maximum_boundary_error_m: f64,
    minimum_mean_ratio: f64,
    maximum_boundary_facets: usize,
    maximum_target_size: Option<f64>,
    sizing: GmshSizingReceipt,
    quality_gate: MeshQualityGate,
) -> Result<GeneratedGmshMesh, Diagnostic> {
    if sizing_receipt(
        source,
        maximum_boundary_error_m,
        minimum_mean_ratio,
        maximum_boundary_facets,
        maximum_target_size,
    )? != sizing
    {
        return Err(invalid_import(
            "Gmsh MeshPlan sizing differs from the exact Geometry and provider policy",
        ));
    }
    let executable = gmsh_executable()?;
    let scratch = ScratchDirectory::create()?;
    require_version(&executable, scratch.path())?;

    let geometry_path = scratch.path().join("mesh.geo");
    let mesh_path = scratch.path().join("mesh.msh");
    let stderr_path = scratch.path().join("generation.stderr");
    let geometry_script = geometry_script(source, sizing)?;
    fs::write(&geometry_path, geometry_script).map_err(|error| {
        invalid_import(format!("cannot write the Gmsh geometry input: {error}"))
    })?;

    let limits = Msh41Policy::mesh(2, quality_gate)?;
    let mut command = bounded_output_command(&executable, limits.max_bytes.saturating_add(1));
    command
        .arg("-2")
        .arg(&geometry_path)
        .arg("-o")
        .arg(&mesh_path)
        .args(["-format", "msh41", "-v", "2"])
        .stdout(Stdio::null())
        .stderr(Stdio::from(File::create(&stderr_path).map_err(
            |error| invalid_import(format!("cannot create Gmsh diagnostic output: {error}")),
        )?));
    let status = run_with_timeout(command, PROCESS_TIMEOUT)?;
    if !status.success() {
        return Err(invalid_import(process_failure(
            &format!("Gmsh {GMSH_VERSION}"),
            status,
            &stderr_path,
        )));
    }

    let bytes = read_bounded_output(
        &mesh_path,
        limits.max_bytes,
        "Gmsh input exceeds the configured byte limit",
        "Gmsh mesh output",
    )?;
    derive_generated(source, quality_gate, bytes)
}

fn derive_generated(
    source: &CanonicalGeometryV1,
    quality_gate: MeshQualityGate,
    provider_output: Vec<u8>,
) -> Result<GeneratedGmshMesh, Diagnostic> {
    let policy = Msh41Policy::ascii_with_entity_assignments(2, quality_gate)?;
    let mut assignments = BTreeMap::new();
    let mesh = import_msh41(&provider_output, policy, |dimension, tag, indices| {
        assignments.insert((dimension, tag), indices.to_vec());
    })?;
    if assignments.keys().copied().collect::<Vec<_>>()
        != [(1, 1), (1, 5), (1, 6), (1, 7), (1, 8), (2, 1)]
    {
        return Err(invalid_import(
            "generated Gmsh entity-tag inventory is not canonical",
        ));
    }
    let mut edge_facets: [Vec<usize>; 5] = std::array::from_fn(|_| Vec::new());
    for (tag, source_edge) in [(1_u32, 4_usize), (5, 2), (6, 1), (7, 3), (8, 0)] {
        edge_facets[source_edge] = assignments
            .get(&(1, tag))
            .expect("exact tag inventory checked")
            .clone();
    }
    let mesh = SimplicialMeshEnvelopeV1::from_mesh(&mesh)?;
    let correspondence =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_circular_hole_v2_mesh_assignments(
            source,
            &mesh,
            edge_facets.clone(),
        )?;
    Ok(GeneratedGmshMesh {
        provider_output,
        mesh,
        correspondence,
    })
}

fn gmsh_executable() -> Result<OsString, Diagnostic> {
    match std::env::var_os(GMSH_ENV) {
        Some(value) if !value.is_empty() => Ok(value),
        Some(_) => Err(invalid_import(format!("{GMSH_ENV} must not be empty"))),
        None => Ok(OsString::from("gmsh")),
    }
}

fn require_version(executable: &OsStr, scratch: &Path) -> Result<(), Diagnostic> {
    let output_path = scratch.join("version.txt");
    let stderr_path = scratch.join("version.stderr");
    let output = File::create(&output_path)
        .map_err(|error| invalid_import(format!("cannot create Gmsh version output: {error}")))?;
    let mut command = bounded_output_command(executable, 65);
    command
        .arg("--version")
        .stdout(Stdio::from(output))
        .stderr(Stdio::from(File::create(&stderr_path).map_err(
            |error| {
                invalid_import(format!(
                    "cannot create Gmsh version diagnostic output: {error}"
                ))
            },
        )?));
    let status = run_with_timeout(command, VERSION_TIMEOUT)?;
    if !status.success() {
        return Err(invalid_import(process_failure(
            "Gmsh version check",
            status,
            &stderr_path,
        )));
    }
    let stdout = read_bounded_output(
        &output_path,
        64,
        "Gmsh version output exceeds 64 bytes",
        "Gmsh version output",
    )?;
    let version = std::str::from_utf8(&stdout)
        .map_err(|_| invalid_import("Gmsh version output must be UTF-8"))?
        .trim();
    if version != GMSH_VERSION {
        return Err(invalid_import(format!(
            "automatic meshing requires Gmsh {GMSH_VERSION}; found {version:?}"
        )));
    }
    Ok(())
}

fn read_bounded_output(
    path: &Path,
    max_bytes: usize,
    over_limit: &str,
    description: &str,
) -> Result<Vec<u8>, Diagnostic> {
    let length = fs::metadata(path)
        .map_err(|error| invalid_import(format!("cannot inspect {description}: {error}")))?
        .len();
    if length > max_bytes as u64 {
        return Err(invalid_import(over_limit));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    File::open(path)
        .map_err(|error| invalid_import(format!("cannot read {description}: {error}")))?
        .take(max_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| invalid_import(format!("cannot read {description}: {error}")))?;
    if bytes.len() > max_bytes {
        return Err(invalid_import(over_limit));
    }
    Ok(bytes)
}

fn bounded_output_command(executable: &OsStr, max_bytes: usize) -> Command {
    #[cfg(unix)]
    {
        const FILE_LIMIT_BLOCK_BYTES: usize = 512;
        // The PyPI Gmsh CLI is a Python entry point. Leave enough process-wide
        // headroom for interpreter cache/support files while enforcing the
        // narrower semantic limit when the requested output is read below.
        let blocks = max_bytes
            .max(PROCESS_FILE_LIMIT_FLOOR_BYTES)
            .div_ceil(FILE_LIMIT_BLOCK_BYTES);
        let mut command = Command::new("/bin/sh");
        command
            .args([
                "-c",
                "ulimit -f \"$1\" || exit 126; shift; exec \"$@\"",
                "eqiora-gmsh",
            ])
            .arg(blocks.to_string())
            .arg(executable);
        command
    }
    #[cfg(not(unix))]
    {
        let _ = max_bytes;
        Command::new(executable)
    }
}

fn process_failure(label: &str, status: ExitStatus, stderr_path: &Path) -> String {
    let detail = fs::read(stderr_path)
        .ok()
        .map(|bytes| {
            let retained = &bytes[..bytes.len().min(FAILURE_DETAIL_LIMIT_BYTES)];
            String::from_utf8_lossy(retained).trim().to_owned()
        })
        .filter(|detail| !detail.is_empty());
    match detail {
        Some(detail) => format!("{label} failed with status {status}: {detail}"),
        None => format!("{label} failed with status {status}"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct GmshSizingReceipt {
    circle_segments: usize,
    straight_segments: [usize; 4],
    minimum_size_m: f64,
    maximum_size_m: f64,
    policy: GmshMeshPolicyV1,
}

impl GmshSizingReceipt {
    pub(super) const fn policy(self) -> GmshMeshPolicyV1 {
        self.policy
    }
}

pub(super) fn plan(
    source: &CanonicalGeometryV1,
    maximum_boundary_error_m: f64,
    minimum_mean_ratio: f64,
    maximum_boundary_facets: usize,
    maximum_target_size: Option<f64>,
) -> Result<GmshSizingReceipt, Diagnostic> {
    sizing_receipt(
        source,
        maximum_boundary_error_m,
        minimum_mean_ratio,
        maximum_boundary_facets,
        maximum_target_size,
    )
}

fn sizing_receipt(
    source: &CanonicalGeometryV1,
    maximum_boundary_error_m: f64,
    minimum_mean_ratio: f64,
    maximum_boundary_facets: usize,
    maximum_target_size: Option<f64>,
) -> Result<GmshSizingReceipt, Diagnostic> {
    let radius = source
        .circular_hole_radius_m()
        .ok_or_else(|| invalid_import("GmshMesher requires planar circular-hole Geometry v2"))?;
    let error = maximum_boundary_error_m;
    let raw = if error >= 2.0 * radius {
        8.0
    } else {
        let half_angle = (error / (2.0 * radius)).sqrt().asin();
        (std::f64::consts::PI / (2.0 * half_angle)).ceil().max(8.0)
    };
    if !raw.is_finite() || raw > usize::MAX as f64 {
        return Err(invalid_import(
            "Gmsh circular-boundary segment count exceeds the local work range",
        ));
    }
    let mut circle_segments = raw as usize;
    while circle_segments > 8 && sagitta_m(radius, circle_segments - 1) <= error {
        circle_segments -= 1;
    }
    while sagitta_m(radius, circle_segments) > error {
        circle_segments = circle_segments.checked_add(1).ok_or_else(|| {
            invalid_import("Gmsh circular-boundary segment count exceeds the local work range")
        })?;
    }
    if circle_segments > maximum_boundary_facets {
        return Err(invalid_import(format!(
            "Gmsh circular-boundary error requires {circle_segments} segments, exceeding the caller limit of {}",
            maximum_boundary_facets,
        )));
    }
    let minimum_size_m = 2.0 * radius * (std::f64::consts::PI / circle_segments as f64).sin();
    let maximum_size_m = maximum_target_size.unwrap_or(4.0 * radius);
    if !minimum_size_m.is_finite() || minimum_size_m <= 0.0 || !maximum_size_m.is_finite() {
        return Err(invalid_import(
            "Gmsh derived target size must be finite and positive",
        ));
    }
    if maximum_size_m < minimum_size_m {
        return Err(invalid_import(format!(
            "Gmsh maximum target size {maximum_size_m} m is smaller than the {minimum_size_m} m chord required by the circular-boundary error policy",
        )));
    }
    let [[x_min, x_max], [y_min, y_max]] = source
        .circular_hole_bounds()
        .ok_or_else(|| invalid_import("GmshMesher requires planar circular-hole Geometry v2"))?;
    let width = x_max - x_min;
    let height = y_max - y_min;
    let segments = |length: f64| -> Result<usize, Diagnostic> {
        let count = (length / maximum_size_m).ceil();
        if !count.is_finite() || count < 1.0 || count > usize::MAX as f64 {
            return Err(invalid_import(
                "Gmsh straight-boundary segment count exceeds the local work range",
            ));
        }
        Ok(count as usize)
    };
    Ok(GmshSizingReceipt {
        circle_segments,
        // Canonical source-edge order is x-lower, x-upper, y-lower, y-upper.
        straight_segments: [
            segments(height)?,
            segments(height)?,
            segments(width)?,
            segments(width)?,
        ],
        minimum_size_m,
        maximum_size_m,
        policy: if maximum_target_size.is_some() {
            GmshMeshPolicyV1::explicit(
                maximum_boundary_error_m,
                minimum_mean_ratio,
                maximum_boundary_facets,
                maximum_size_m,
            )?
        } else {
            GmshMeshPolicyV1::automatic(
                maximum_boundary_error_m,
                minimum_mean_ratio,
                maximum_boundary_facets,
                maximum_size_m,
            )?
        },
    })
}

fn sagitta_m(radius_m: f64, segments: usize) -> f64 {
    let sine = (std::f64::consts::PI / (2.0 * segments as f64)).sin();
    2.0 * radius_m * sine * sine
}

fn geometry_script(
    source: &CanonicalGeometryV1,
    sizing: GmshSizingReceipt,
) -> Result<String, Diagnostic> {
    let bounds = source
        .circular_hole_bounds()
        .ok_or_else(|| invalid_import("GmshMesher requires planar circular-hole Geometry v2"))?;
    let center = source
        .circular_hole_center()
        .ok_or_else(|| invalid_import("GmshMesher requires planar circular-hole Geometry v2"))?;
    let radius = source
        .circular_hole_radius_m()
        .ok_or_else(|| invalid_import("GmshMesher requires planar circular-hole Geometry v2"))?;
    CanonicalGeometryV1::decode_circular_hole_canonical(
        source.canonical_bytes(),
        eqiora::geometry::CanonicalGeometryLimits::default(),
    )
    .map_err(|_| invalid_import("GmshMesher requires planar circular-hole Geometry v2"))?;
    let mut script = String::from(
        "SetFactory(\"OpenCASCADE\");\n\
         General.NumThreads = 1;\n\
         Mesh.Algorithm = 6;\n\
         Mesh.ElementOrder = 1;\n\
         Mesh.Binary = 0;\n\
         Mesh.MshFileVersion = 4.1;\n\
         Mesh.RandomFactor = 0;\n\
         Mesh.SaveAll = 1;\n",
    );
    let [[x_min, x_max], [y_min, y_max]] = *bounds;
    for (tag, [x, y]) in [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]
    .into_iter()
    .enumerate()
    {
        writeln!(script, "Point({}) = {{{x:?}, {y:?}, 0}};", tag + 1)
            .expect("writing to String cannot fail");
    }
    writeln!(
        script,
        "Circle(1) = {{{:?}, {:?}, 0, {radius:?}}};",
        center[0], center[1]
    )
    .expect("writing to String cannot fail");
    for (tag, [start, end]) in [[1, 2], [2, 3], [3, 4], [4, 1]].into_iter().enumerate() {
        writeln!(script, "Line({}) = {{{start}, {end}}};", tag + 5)
            .expect("writing to String cannot fail");
    }
    writeln!(
        script,
        "Curve Loop(1) = {{-1}};\nCurve Loop(2) = {{5, 6, 7, 8}};\nPlane Surface(1) = {{2, 1}};"
    )
    .expect("writing to String cannot fail");

    writeln!(script, "Mesh.MeshSizeMin = {:?};", sizing.minimum_size_m)
        .expect("writing to String cannot fail");
    writeln!(script, "Mesh.MeshSizeMax = {:?};", sizing.maximum_size_m)
        .expect("writing to String cannot fail");
    writeln!(
        script,
        "Transfinite Curve {{1}} = {};",
        sizing.circle_segments + 1
    )
    .expect("writing to String cannot fail");
    // Gmsh counts nodes, rather than segments, on open transfinite curves.
    for (tag, source_edge) in [(5, 2), (6, 1), (7, 3), (8, 0)] {
        writeln!(
            script,
            "Transfinite Curve {{{tag}}} = {};",
            sizing.straight_segments[source_edge] + 1
        )
        .expect("writing to String cannot fail");
    }

    Ok(script)
}

fn run_with_timeout(mut command: Command, timeout: Duration) -> Result<ExitStatus, Diagnostic> {
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command
        .spawn()
        .map_err(|error| invalid_import(format!("cannot launch Gmsh: {error}")))?;
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| invalid_import(format!("cannot wait for Gmsh: {error}")))?
        {
            kill_descendants(&child);
            return Ok(status);
        }
        if Instant::now() >= deadline {
            terminate(&mut child);
            return Err(invalid_import(
                "Gmsh exceeded the 30 second execution limit",
            ));
        }
        thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(unix)]
fn kill_descendants(child: &Child) {
    if let Ok(raw_pid) = i32::try_from(child.id())
        && let Some(pid) = Pid::from_raw(raw_pid)
    {
        let _ = kill_process_group(pid, Signal::KILL);
    }
}

#[cfg(not(unix))]
fn kill_descendants(_child: &Child) {}

fn terminate(child: &mut Child) {
    kill_descendants(child);
    let _ = child.kill();
    let _ = child.wait();
}

struct ScratchDirectory {
    path: PathBuf,
}

impl ScratchDirectory {
    fn create() -> Result<Self, Diagnostic> {
        let parent = std::env::temp_dir();
        for _ in 0..16 {
            let nonce = SCRATCH_NONCE.fetch_add(1, Ordering::Relaxed);
            let clock = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path = parent.join(format!(
                "eqiora-gmsh-{}-{clock}-{nonce}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(invalid_import(format!(
                        "cannot create Gmsh scratch directory: {error}"
                    )));
                }
            }
        }
        Err(invalid_import(
            "cannot allocate a unique Gmsh scratch directory",
        ))
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn invalid_import(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_MESH_IMPORT, message)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use eqiora::geometry::GeometryGraph;

    use super::*;

    fn circular_hole(bounds: [[f64; 2]; 2], center: [f64; 2], radius: f64) -> CanonicalGeometryV1 {
        let graph = GeometryGraph::new();
        let rectangle = graph.rectangle(bounds[0], bounds[1]).unwrap();
        let circle = graph.circle(center, radius).unwrap();
        let region = graph.subtract(&rectangle, &circle).unwrap();
        let boundaries = region.boundaries();
        graph
            .build(
                &region,
                &BTreeMap::from([
                    ("fluid".to_owned(), vec![region.region().into()]),
                    ("inlet".to_owned(), vec![boundaries[0].into()]),
                    ("outlet".to_owned(), vec![boundaries[1].into()]),
                    (
                        "walls".to_owned(),
                        vec![boundaries[2].into(), boundaries[3].into()],
                    ),
                    ("cylinder".to_owned(), vec![boundaries[4].into()]),
                ]),
            )
            .unwrap()
    }

    fn sizing(
        source: &CanonicalGeometryV1,
        maximum_boundary_error_m: f64,
        minimum_mean_ratio: f64,
        maximum_boundary_facets: usize,
        maximum_target_size: Option<f64>,
    ) -> Result<GmshSizingReceipt, Diagnostic> {
        sizing_receipt(
            source,
            maximum_boundary_error_m,
            minimum_mean_ratio,
            maximum_boundary_facets,
            maximum_target_size,
        )
    }

    #[test]
    fn sizing_is_minimal_at_binary64_sagitta_boundary() {
        let source = circular_hole([[0.0, 2.2], [0.0, 0.41]], [0.2, 0.2], 0.05);
        let error = sagitta_m(0.05, 20);
        let receipt = sizing(&source, error, 1.0e-5, 50, None).unwrap();
        assert_eq!(receipt.circle_segments, 20);
        assert!(sagitta_m(0.05, receipt.circle_segments) <= error);
        assert!(sagitta_m(0.05, receipt.circle_segments - 1) > error);

        assert_eq!(
            sizing(
                &source,
                f64::from_bits(error.to_bits() - 1),
                1.0e-5,
                50,
                None,
            )
            .unwrap()
            .circle_segments,
            21
        );
    }

    #[test]
    fn sizing_uses_exact_nondefault_geometry_and_policy() {
        let source = circular_hole([[-1.0, 3.0], [-2.0, 1.0]], [0.5, -0.5], 0.2);
        let receipt = sizing(&source, 2.0e-3, 2.0e-5, 64, None).unwrap();
        assert!(receipt.circle_segments >= 8);
        assert_eq!(receipt.straight_segments[0], receipt.straight_segments[1]);
        assert_eq!(receipt.straight_segments[2], receipt.straight_segments[3]);
        assert!(receipt.straight_segments[2] > receipt.straight_segments[0]);

        let alternate = circular_hole([[-1.0, 4.0], [-2.0, 1.0]], [0.5, -0.5], 0.2);
        assert_ne!(
            receipt,
            sizing(&alternate, 2.0e-3, 2.0e-5, 64, None).unwrap()
        );
    }

    #[test]
    fn sizing_retains_automatic_and_explicit_target_ownership() {
        let source = circular_hole([[0.0, 2.2], [0.0, 0.41]], [0.2, 0.2], 0.05);
        let automatic = sizing(&source, 1.0e-4, 1.0e-5, 50, None).unwrap();
        assert_eq!(automatic.policy().maximum_boundary_error_m(), 1.0e-4);
        assert_eq!(automatic.policy().minimum_mean_ratio(), 1.0e-5);
        assert_eq!(automatic.policy().maximum_boundary_facets(), 50);
        assert_eq!(automatic.policy().maximum_target_size_m(), 0.2);
        assert!(!automatic.policy().maximum_target_size_is_explicit());

        let explicit = sizing(&source, 1.0e-4, 1.0e-5, 50, Some(0.05)).unwrap();
        assert_eq!(explicit.policy().maximum_target_size_m(), 0.05);
        assert!(explicit.policy().maximum_target_size_is_explicit());
        assert!(explicit.straight_segments[2] > automatic.straight_segments[2]);
        assert!(
            geometry_script(&source, explicit)
                .unwrap()
                .contains("Mesh.MeshSizeMax = 0.05;")
        );
    }

    #[test]
    fn sizing_rejects_policy_that_cannot_admit_the_circle() {
        let source = circular_hole([[0.0, 2.2], [0.0, 0.41]], [0.2, 0.2], 0.05);
        assert!(sizing(&source, 1.0e-6, 1.0e-5, 8, None).is_err());
    }

    #[test]
    fn sizing_rejects_target_below_required_boundary_chord() {
        let source = circular_hole([[0.0, 2.2], [0.0, 0.41]], [0.2, 0.2], 0.05);
        assert!(sizing(&source, 1.0e-4, 1.0e-5, 50, Some(0.001)).is_err());
    }
}
