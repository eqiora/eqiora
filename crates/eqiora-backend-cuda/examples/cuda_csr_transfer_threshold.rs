//! Produce one cold, transfer-inclusive CUDA CSR SpMV observation set.
//!
//! This executable is evidence tooling, not a runtime-selection benchmark API.

use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use eqiora_assembly::CsrMatrix;
use eqiora_backend_cuda::{CudaCsrActionResult, verify_csr_action_against};
use eqiora_device::{SparseActionPolicy, SparseActionTolerance};

const REFINEMENTS: [usize; 7] = [64, 128, 256, 512, 1024, 2048, 4096];
const WARMUPS: usize = 2;
const REPETITIONS: usize = 9;
const ABSOLUTE_TOLERANCE: f64 = 1.0e-11;
const RELATIVE_TOLERANCE: f64 = 1.0e-11;

#[derive(Debug)]
struct Observation {
    refinement: usize,
    rows: usize,
    nonzeros: usize,
    repetition: usize,
    order: &'static str,
    cpu_ns: u128,
    cuda_setup_ns: u128,
    cuda_h2d_ns: u128,
    cuda_action_ns: u128,
    cuda_d2h_ns: u128,
    cuda_phase_sum_ns: u128,
    cuda_execution_wall_ns: u128,
    cuda_verification_ns: u128,
    cuda_verified_wall_ns: u128,
    maximum_absolute_error: f64,
    maximum_scaled_error: f64,
    host_to_device_bytes: usize,
    device_to_host_bytes: usize,
    workspace_bytes: usize,
}

#[derive(Debug)]
struct TimedCudaAction {
    result: CudaCsrActionResult,
    execution_wall_ns: u128,
    verified_wall_ns: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HardwareIdentity {
    device_name: String,
    device_memory_bytes: u64,
    driver: i32,
    cusparse: i32,
    cudarc: &'static str,
    binding_toolkit: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct EnvironmentSeed<'a> {
    hardware: &'a HardwareIdentity,
    source_commit: &'a str,
    selected_device: &'a str,
    selected_cpu: usize,
    cpu_affinity_count: usize,
    gpu_operating_before: &'a GpuOperatingSnapshot,
    gpu_compute_process_count_before: usize,
    system_load_before: SystemLoad,
}

#[derive(Debug, Clone, PartialEq)]
struct GpuOperatingSnapshot {
    performance_state: String,
    temperature_celsius: f64,
    utilization_percent: f64,
    memory_used_mib: u64,
    power_draw_watts: f64,
    power_limit_watts: f64,
    sm_clock_mhz: u64,
    memory_clock_mhz: u64,
    maximum_sm_clock_mhz: u64,
    maximum_memory_clock_mhz: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SystemLoad {
    one_minute: f64,
    five_minutes: f64,
    fifteen_minutes: f64,
    runnable_processes: usize,
    total_processes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CpuFrequencyPolicy {
    governor: String,
    minimum_khz: u64,
    maximum_khz: u64,
}

#[derive(Debug)]
struct RefinementSummary {
    refinement: usize,
    rows: usize,
    nonzeros: usize,
    cpu_median_ns: u128,
    cuda_median_ns: u128,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("CUDA transfer-threshold measurement failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    if cfg!(debug_assertions) {
        return Err("measurement must be built with --release".to_owned());
    }
    let mut arguments = std::env::args_os().skip(1);
    let output = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "usage: cuda_csr_transfer_threshold <new-output-directory>".to_owned())?;
    if arguments.next().is_some() {
        return Err("measurement accepts exactly one output directory".to_owned());
    }
    if output.exists() {
        return Err(format!(
            "output directory {} already exists",
            output.display()
        ));
    }

    let visible_device = std::env::var("CUDA_VISIBLE_DEVICES")
        .map_err(|_| "CUDA_VISIBLE_DEVICES must select exactly one physical device".to_owned())?;
    let selected_device = visible_device.trim();
    if selected_device.is_empty() || selected_device.contains(',') {
        return Err("CUDA_VISIBLE_DEVICES must contain exactly one device selector".to_owned());
    }
    let gpu_operating_before = gpu_operating_snapshot(selected_device)?;
    let gpu_compute_process_count_before = gpu_compute_process_count(selected_device)?;
    let system_load_before = read_system_load()?;
    let affinity = first_prefixed_value("/proc/self/status", "Cpus_allowed_list")?;
    let (cpu_affinity_count, selected_cpu) = parse_cpu_list(&affinity)?;
    if cpu_affinity_count != 1 {
        return Err(format!(
            "measurement requires one pinned CPU, found affinity count {cpu_affinity_count}"
        ));
    }
    let source_commit = command_output("git", &["rev-parse", "HEAD"])?;
    if source_commit.len() != 40
        || !source_commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("git HEAD must resolve to a full lowercase commit SHA".to_owned());
    }
    let source_status = command_output("git", &["status", "--porcelain"])?;
    if !source_status.is_empty() {
        return Err("source worktree must be clean before measurement".to_owned());
    }

    let tolerance = SparseActionTolerance::new(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE)
        .map_err(|diagnostic| diagnostic.to_string())?;
    let mut observations = Vec::with_capacity(REFINEMENTS.len() * REPETITIONS);
    let mut hardware: Option<HardwareIdentity> = None;

    for refinement in REFINEMENTS {
        let matrix = poisson_five_point(refinement)?;
        let input = deterministic_input(matrix.columns());
        let reference = matrix
            .multiply(&input)
            .map_err(|diagnostic| diagnostic.to_string())?;
        let mut cpu_output = vec![0.0; matrix.rows()];

        for _ in 0..WARMUPS {
            matrix
                .multiply_into(&input, &mut cpu_output)
                .map_err(|diagnostic| diagnostic.to_string())?;
            black_box(&cpu_output);
            let result = measure_cuda(&matrix, &input, &reference, tolerance)?;
            admit_hardware(&mut hardware, &result.result)?;
            black_box(result.result.values());
        }

        for repetition in 0..REPETITIONS {
            let cpu_first = repetition % 2 == 0;
            let (cpu_ns, result) = if cpu_first {
                (
                    measure_cpu(&matrix, &input, &mut cpu_output)?,
                    measure_cuda(&matrix, &input, &reference, tolerance)?,
                )
            } else {
                let result = measure_cuda(&matrix, &input, &reference, tolerance)?;
                let cpu_ns = measure_cpu(&matrix, &input, &mut cpu_output)?;
                (cpu_ns, result)
            };
            accept_host_action(&reference, &cpu_output, tolerance)?;
            admit_hardware(&mut hardware, &result.result)?;
            observations.push(observation(
                refinement,
                repetition,
                if cpu_first { "cpu-first" } else { "cuda-first" },
                cpu_ns,
                &matrix,
                &result,
            )?);
        }
    }

    let hardware = hardware.ok_or_else(|| "measurement produced no CUDA samples".to_owned())?;
    let summaries = summarize(&observations)?;
    persist(
        &output,
        &observations,
        &summaries,
        EnvironmentSeed {
            hardware: &hardware,
            source_commit: &source_commit,
            selected_device,
            selected_cpu,
            cpu_affinity_count,
            gpu_operating_before: &gpu_operating_before,
            gpu_compute_process_count_before,
            system_load_before,
        },
    )
}

fn poisson_five_point(refinement: usize) -> Result<CsrMatrix<f64>, String> {
    let rows = refinement
        .checked_mul(refinement)
        .ok_or_else(|| "Poisson row count overflowed".to_owned())?;
    let boundary_couplings = refinement
        .checked_mul(4)
        .ok_or_else(|| "Poisson boundary count overflowed".to_owned())?;
    let nonzeros = rows
        .checked_mul(5)
        .and_then(|value| value.checked_sub(boundary_couplings))
        .ok_or_else(|| "Poisson nonzero count overflowed".to_owned())?;
    let offset_count = rows
        .checked_add(1)
        .ok_or_else(|| "Poisson row-offset count overflowed".to_owned())?;
    let mut row_offsets = Vec::with_capacity(offset_count);
    let mut column_indices = Vec::with_capacity(nonzeros);
    let mut values = Vec::with_capacity(nonzeros);
    row_offsets.push(0);
    for row in 0..rows {
        let x = row % refinement;
        let y = row / refinement;
        if y > 0 {
            column_indices.push(row - refinement);
            values.push(-1.0);
        }
        if x > 0 {
            column_indices.push(row - 1);
            values.push(-1.0);
        }
        column_indices.push(row);
        values.push(4.0);
        if x + 1 < refinement {
            column_indices.push(row + 1);
            values.push(-1.0);
        }
        if y + 1 < refinement {
            column_indices.push(row + refinement);
            values.push(-1.0);
        }
        row_offsets.push(column_indices.len());
    }
    if column_indices.len() != nonzeros {
        return Err(format!(
            "Poisson builder produced {} nonzeros, expected {nonzeros}",
            column_indices.len()
        ));
    }
    CsrMatrix::from_sorted_csr(rows, rows, row_offsets, column_indices, values)
        .map_err(|diagnostic| diagnostic.to_string())
}

fn deterministic_input(length: usize) -> Vec<f64> {
    (0..length)
        .map(|index| 1.0 + (index % 97) as f64 / 97.0)
        .collect()
}

fn measure_cpu(matrix: &CsrMatrix<f64>, input: &[f64], output: &mut [f64]) -> Result<u128, String> {
    let started = Instant::now();
    matrix
        .multiply_into(input, output)
        .map_err(|diagnostic| diagnostic.to_string())?;
    let elapsed = started.elapsed().as_nanos();
    black_box(output);
    Ok(elapsed)
}

fn measure_cuda(
    matrix: &CsrMatrix<f64>,
    input: &[f64],
    reference: &[f64],
    tolerance: SparseActionTolerance,
) -> Result<TimedCudaAction, String> {
    let started = Instant::now();
    let result = verify_csr_action_against(
        matrix,
        input,
        reference,
        0,
        SparseActionPolicy::BackendNative,
        tolerance,
    )
    .map_err(|diagnostic| diagnostic.to_string())?;
    let verified_wall_ns = started.elapsed().as_nanos();
    let execution_wall_ns = verified_wall_ns
        .checked_sub(result.evidence().timings().verification().as_nanos())
        .ok_or_else(|| "CUDA verification phase exceeded the outer call wall time".to_owned())?;
    Ok(TimedCudaAction {
        result,
        execution_wall_ns,
        verified_wall_ns,
    })
}

fn accept_host_action(
    reference: &[f64],
    actual: &[f64],
    tolerance: SparseActionTolerance,
) -> Result<(), String> {
    if reference.len() != actual.len()
        || reference
            .iter()
            .chain(actual)
            .any(|value| !value.is_finite())
    {
        return Err("host reference and timed action require equal finite vectors".to_owned());
    }
    for (index, (reference, actual)) in reference.iter().zip(actual).enumerate() {
        let error = (reference - actual).abs();
        let threshold = tolerance.threshold(*reference);
        if error > threshold {
            return Err(format!(
                "timed host action differs from its reference at row {index}: {error:e} > {threshold:e}"
            ));
        }
    }
    Ok(())
}

fn admit_hardware(
    admitted: &mut Option<HardwareIdentity>,
    result: &CudaCsrActionResult,
) -> Result<(), String> {
    let evidence = result.evidence();
    let versions = evidence.versions();
    let observed = HardwareIdentity {
        device_name: evidence.device().name().to_owned(),
        device_memory_bytes: evidence.device().total_memory_bytes().get(),
        driver: versions.driver(),
        cusparse: versions.cusparse(),
        cudarc: versions.cudarc(),
        binding_toolkit: versions.binding_toolkit(),
    };
    match admitted {
        Some(expected) if *expected != observed => {
            Err("CUDA hardware or library identity changed during measurement".to_owned())
        }
        Some(_) => Ok(()),
        None => {
            *admitted = Some(observed);
            Ok(())
        }
    }
}

fn observation(
    refinement: usize,
    repetition: usize,
    order: &'static str,
    cpu_ns: u128,
    matrix: &CsrMatrix<f64>,
    timed: &TimedCudaAction,
) -> Result<Observation, String> {
    let result = &timed.result;
    let evidence = result.evidence();
    let timings = evidence.timings();
    let cuda_phase_sum_ns = [
        timings.setup().as_nanos(),
        timings.host_to_device().as_nanos(),
        timings.solve().as_nanos(),
        timings.device_to_host().as_nanos(),
    ]
    .into_iter()
    .try_fold(0_u128, |total, phase| total.checked_add(phase))
    .ok_or_else(|| "CUDA phase-sum duration overflowed".to_owned())?;
    if cuda_phase_sum_ns > timings.total().as_nanos() {
        return Err("sequential CUDA phases exceed the observed total".to_owned());
    }
    if timed.execution_wall_ns < cuda_phase_sum_ns {
        return Err("CUDA execution wall time is shorter than its sequential phases".to_owned());
    }
    if timed.verified_wall_ns < timed.execution_wall_ns + timings.verification().as_nanos() {
        return Err("CUDA outer wall time does not contain execution and verification".to_owned());
    }
    Ok(Observation {
        refinement,
        rows: matrix.rows(),
        nonzeros: matrix.values().len(),
        repetition,
        order,
        cpu_ns,
        cuda_setup_ns: timings.setup().as_nanos(),
        cuda_h2d_ns: timings.host_to_device().as_nanos(),
        cuda_action_ns: timings.solve().as_nanos(),
        cuda_d2h_ns: timings.device_to_host().as_nanos(),
        cuda_phase_sum_ns,
        cuda_execution_wall_ns: timed.execution_wall_ns,
        cuda_verification_ns: timings.verification().as_nanos(),
        cuda_verified_wall_ns: timed.verified_wall_ns,
        maximum_absolute_error: evidence.maximum_absolute_error(),
        maximum_scaled_error: evidence.maximum_scaled_error(),
        host_to_device_bytes: evidence.transfers().host_to_device_bytes(),
        device_to_host_bytes: evidence.transfers().device_to_host_bytes(),
        workspace_bytes: evidence.workspace_bytes(),
    })
}

fn summarize(observations: &[Observation]) -> Result<Vec<RefinementSummary>, String> {
    let mut summaries = Vec::with_capacity(REFINEMENTS.len());
    for refinement in REFINEMENTS {
        let matching = observations
            .iter()
            .filter(|observation| observation.refinement == refinement)
            .collect::<Vec<_>>();
        if matching.len() != REPETITIONS {
            return Err(format!(
                "refinement {refinement} has {} samples, expected {REPETITIONS}",
                matching.len()
            ));
        }
        let first = matching[0];
        summaries.push(RefinementSummary {
            refinement,
            rows: first.rows,
            nonzeros: first.nonzeros,
            cpu_median_ns: median(matching.iter().map(|sample| sample.cpu_ns))?,
            cuda_median_ns: median(matching.iter().map(|sample| sample.cuda_execution_wall_ns))?,
        });
    }
    Ok(summaries)
}

fn median(values: impl IntoIterator<Item = u128>) -> Result<u128, String> {
    let mut values = values.into_iter().collect::<Vec<_>>();
    if values.is_empty() || values.len() % 2 == 0 {
        return Err("median requires a nonempty odd sample count".to_owned());
    }
    values.sort_unstable();
    Ok(values[values.len() / 2])
}

fn persist(
    output: &Path,
    observations: &[Observation],
    summaries: &[RefinementSummary],
    environment_seed: EnvironmentSeed<'_>,
) -> Result<(), String> {
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "output directory requires a UTF-8 final component".to_owned())?;
    let staging = parent.join(format!(".{name}.staging-{}", std::process::id()));
    if staging.exists() {
        return Err(format!(
            "staging directory {} already exists",
            staging.display()
        ));
    }
    let environment = render_environment(environment_seed)?;
    if let Err(error) = fs::create_dir_all(staging.join("observations"))
        .and_then(|_| fs::create_dir_all(staging.join("expected")))
    {
        let _ = fs::remove_dir_all(&staging);
        return Err(format!("cannot create staging output: {error}"));
    }

    let write_result = (|| {
        fs::write(
            staging.join("observations/repetitions.csv"),
            render_observations(observations),
        )?;
        fs::write(staging.join("observations/environment.json"), environment)?;
        fs::write(
            staging.join("expected/summary.json"),
            render_summary(summaries),
        )?;
        Ok::<(), std::io::Error>(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_dir_all(&staging);
        return Err(format!("cannot write measurement output: {error}"));
    }
    if let Err(error) = fs::rename(&staging, output) {
        let _ = fs::remove_dir_all(&staging);
        return Err(format!(
            "cannot publish measurement output atomically: {error}"
        ));
    }
    Ok(())
}

fn render_observations(observations: &[Observation]) -> String {
    let mut output = String::from(
        "refinement,rows,nonzeros,repetition,order,cpu_ns,cuda_setup_ns,cuda_h2d_ns,cuda_action_ns,cuda_d2h_ns,cuda_phase_sum_ns,cuda_execution_wall_ns,cuda_verification_ns,cuda_verified_wall_ns,maximum_absolute_error,maximum_scaled_error,host_to_device_bytes,device_to_host_bytes,workspace_bytes\n",
    );
    for sample in observations {
        writeln!(
            output,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.17e},{:.17e},{},{},{}",
            sample.refinement,
            sample.rows,
            sample.nonzeros,
            sample.repetition,
            sample.order,
            sample.cpu_ns,
            sample.cuda_setup_ns,
            sample.cuda_h2d_ns,
            sample.cuda_action_ns,
            sample.cuda_d2h_ns,
            sample.cuda_phase_sum_ns,
            sample.cuda_execution_wall_ns,
            sample.cuda_verification_ns,
            sample.cuda_verified_wall_ns,
            sample.maximum_absolute_error,
            sample.maximum_scaled_error,
            sample.host_to_device_bytes,
            sample.device_to_host_bytes,
            sample.workspace_bytes,
        )
        .expect("writing to a String cannot fail");
    }
    output
}

fn render_environment(seed: EnvironmentSeed<'_>) -> Result<String, String> {
    let rustc = command_output("rustc", &["--version", "--verbose"])?;
    let kernel = command_output("uname", &["-srmo"])?;
    let gpu_operating_after = gpu_operating_snapshot(seed.selected_device)?;
    let gpu_compute_process_count_after = gpu_compute_process_count(seed.selected_device)?;
    let cpu_model = first_prefixed_value("/proc/cpuinfo", "model name")?;
    let frequency_policy = cpu_frequency_policy(seed.selected_cpu)?;
    let system_load_after = read_system_load()?;
    let os_release = os_release_pretty_name()?;
    let rustflags_present = std::env::var("RUSTFLAGS").is_ok_and(|value| !value.trim().is_empty());
    let environment = serde_json::json!({
        "schema": "eqiora.cuda-csr-transfer-threshold.environment.v3",
        "source_commit": seed.source_commit,
        "source_clean": true,
        "profile": "release",
        "rustc": rustc,
        "rustflags_present": rustflags_present,
        "kernel": kernel,
        "os_release": os_release,
        "cpu_model": cpu_model,
        "cpu_affinity_count": seed.cpu_affinity_count,
        "cpu_frequency_policy": {
            "governor": frequency_policy.governor,
            "minimum_khz": frequency_policy.minimum_khz,
            "maximum_khz": frequency_policy.maximum_khz,
        },
        "system_load_before": {
            "one_minute": seed.system_load_before.one_minute,
            "five_minutes": seed.system_load_before.five_minutes,
            "fifteen_minutes": seed.system_load_before.fifteen_minutes,
            "runnable_processes": seed.system_load_before.runnable_processes,
            "total_processes": seed.system_load_before.total_processes,
        },
        "system_load_after": {
            "one_minute": system_load_after.one_minute,
            "five_minutes": system_load_after.five_minutes,
            "fifteen_minutes": system_load_after.fifteen_minutes,
            "runnable_processes": system_load_after.runnable_processes,
            "total_processes": system_load_after.total_processes,
        },
        "eqiora_device_ordinal": 0,
        "gpu_name": seed.hardware.device_name.as_str(),
        "gpu_memory_bytes": seed.hardware.device_memory_bytes,
        "gpu_operating_before": operating_snapshot_json(seed.gpu_operating_before),
        "gpu_operating_after": operating_snapshot_json(&gpu_operating_after),
        "gpu_compute_process_count_before": seed.gpu_compute_process_count_before,
        "gpu_compute_process_count_after": gpu_compute_process_count_after,
        "cuda_driver": seed.hardware.driver,
        "cusparse": seed.hardware.cusparse,
        "cudarc": seed.hardware.cudarc,
        "binding_toolkit": seed.hardware.binding_toolkit,
        "scalar": "f64",
        "index": "i64",
        "matrix": "2d-dirichlet-five-point-row-major-sorted-csr",
        "policy": "backend-native",
        "host_memory": "pageable",
        "device_residency": "fresh-per-sample",
        "cuda_comparison": "outer-verified-call-minus-recorded-reference-comparison",
        "reference": "precomputed-serial-host-action; both timed actions independently accepted",
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "warmups": WARMUPS,
        "repetitions": REPETITIONS,
        "refinements": REFINEMENTS,
    });
    serde_json::to_string_pretty(&environment)
        .map(|mut rendered| {
            rendered.push('\n');
            rendered
        })
        .map_err(|error| format!("cannot render environment observation: {error}"))
}

fn render_summary(summaries: &[RefinementSummary]) -> String {
    let durable_crossing = summaries.iter().enumerate().position(|(index, candidate)| {
        index + 1 < summaries.len()
            && candidate.cuda_median_ns < candidate.cpu_median_ns
            && summaries
                .iter()
                .skip_while(|summary| summary.refinement < candidate.refinement)
                .all(|summary| summary.cuda_median_ns < summary.cpu_median_ns)
    });
    let crossing = durable_crossing
        .map(|index| summaries[index].refinement.to_string())
        .unwrap_or_else(|| "null".to_owned());
    let outcome = if durable_crossing.is_some() {
        "durable-crossing-observed"
    } else {
        "no-crossing-in-measured-range"
    };
    let mut output = format!(
        concat!(
            "{{\n",
            "  \"schema\": \"eqiora.cuda-csr-transfer-threshold.summary.v2\",\n",
            "  \"comparison\": \"median-cold-cuda-execution-wall-vs-serial-host-action\",\n",
            "  \"outcome\": \"{}\",\n",
            "  \"durable_crossing_refinement\": {},\n",
            "  \"summaries\": [\n"
        ),
        outcome, crossing
    );
    for (index, summary) in summaries.iter().enumerate() {
        writeln!(
            output,
            "    {{\"refinement\":{},\"rows\":{},\"nonzeros\":{},\"cpu_median_ns\":{},\"cuda_median_ns\":{}}}{}",
            summary.refinement,
            summary.rows,
            summary.nonzeros,
            summary.cpu_median_ns,
            summary.cuda_median_ns,
            if index + 1 == summaries.len() { "" } else { "," }
        )
        .expect("writing to a String cannot fail");
    }
    output.push_str("  ]\n}\n");
    output
}

fn command_output(program: &str, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .map_err(|error| format!("cannot execute {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_owned())
        .map_err(|error| format!("{program} output is not UTF-8: {error}"))
}

fn first_prefixed_value(path: &str, prefix: &str) -> Result<String, String> {
    fs::read_to_string(path)
        .map_err(|error| format!("cannot read {path}: {error}"))?
        .lines()
        .find(|line| line.starts_with(prefix))
        .and_then(|line| line.split_once(':'))
        .map(|(_, value)| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{path} has no {prefix} entry"))
}

fn read_trimmed(path: &str) -> Result<String, String> {
    fs::read_to_string(path)
        .map(|value| value.trim().to_owned())
        .map_err(|error| format!("cannot read {path}: {error}"))
}

fn gpu_operating_snapshot(device: &str) -> Result<GpuOperatingSnapshot, String> {
    let output = command_output(
        "nvidia-smi",
        &[
            "--query-gpu=pstate,temperature.gpu,utilization.gpu,memory.used,power.draw,power.limit,clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory",
            "--format=csv,noheader,nounits",
            "-i",
            device,
        ],
    )?;
    parse_gpu_operating_snapshot(&output)
}

fn parse_gpu_operating_snapshot(value: &str) -> Result<GpuOperatingSnapshot, String> {
    let fields = value.split(',').map(str::trim).collect::<Vec<_>>();
    if fields.len() != 10 {
        return Err(format!(
            "GPU operating snapshot has {} fields, expected 10",
            fields.len()
        ));
    }
    Ok(GpuOperatingSnapshot {
        performance_state: nonempty_field(fields[0], "GPU performance state")?.to_owned(),
        temperature_celsius: parse_metric(fields[1], "GPU temperature")?,
        utilization_percent: parse_metric(fields[2], "GPU utilization")?,
        memory_used_mib: parse_metric(fields[3], "GPU memory use")?,
        power_draw_watts: parse_metric(fields[4], "GPU power draw")?,
        power_limit_watts: parse_metric(fields[5], "GPU power limit")?,
        sm_clock_mhz: parse_metric(fields[6], "GPU SM clock")?,
        memory_clock_mhz: parse_metric(fields[7], "GPU memory clock")?,
        maximum_sm_clock_mhz: parse_metric(fields[8], "GPU maximum SM clock")?,
        maximum_memory_clock_mhz: parse_metric(fields[9], "GPU maximum memory clock")?,
    })
}

fn gpu_compute_process_count(device: &str) -> Result<usize, String> {
    let output = command_output(
        "nvidia-smi",
        &[
            "--query-compute-apps=used_gpu_memory",
            "--format=csv,noheader,nounits",
            "-i",
            device,
        ],
    )?;
    let output = output.trim();
    if output.is_empty() || output.eq_ignore_ascii_case("No running processes found") {
        return Ok(0);
    }
    output.lines().try_fold(0usize, |count, line| {
        let _: u64 = parse_metric(line.trim(), "GPU process memory use")?;
        count
            .checked_add(1)
            .ok_or_else(|| "GPU compute process count overflowed".to_owned())
    })
}

fn read_system_load() -> Result<SystemLoad, String> {
    parse_system_load(&read_trimmed("/proc/loadavg")?)
}

fn parse_system_load(value: &str) -> Result<SystemLoad, String> {
    let fields = value.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 4 {
        return Err(format!(
            "/proc/loadavg has {} fields, expected at least 4",
            fields.len()
        ));
    }
    let (runnable_processes, total_processes) = fields[3]
        .split_once('/')
        .ok_or_else(|| "/proc/loadavg process counts require runnable/total".to_owned())?;
    Ok(SystemLoad {
        one_minute: parse_metric(fields[0], "one-minute system load")?,
        five_minutes: parse_metric(fields[1], "five-minute system load")?,
        fifteen_minutes: parse_metric(fields[2], "fifteen-minute system load")?,
        runnable_processes: parse_metric(runnable_processes, "runnable process count")?,
        total_processes: parse_metric(total_processes, "total process count")?,
    })
}

fn parse_cpu_list(value: &str) -> Result<(usize, usize), String> {
    let mut count = 0usize;
    let mut first = None;
    for segment in value.split(',') {
        let segment = nonempty_field(segment.trim(), "CPU affinity segment")?;
        let (start, end) = match segment.split_once('-') {
            Some((start, end)) => (
                parse_metric(start, "CPU affinity range start")?,
                parse_metric(end, "CPU affinity range end")?,
            ),
            None => {
                let cpu = parse_metric(segment, "CPU affinity CPU")?;
                (cpu, cpu)
            }
        };
        if end < start {
            return Err(format!("CPU affinity range {segment} is descending"));
        }
        first.get_or_insert(start);
        count = count
            .checked_add(end - start + 1)
            .ok_or_else(|| "CPU affinity count overflowed".to_owned())?;
    }
    first
        .map(|first| (count, first))
        .ok_or_else(|| "CPU affinity list is empty".to_owned())
}

fn cpu_frequency_policy(cpu: usize) -> Result<CpuFrequencyPolicy, String> {
    let root = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq");
    let governor = read_trimmed(&format!("{root}/scaling_governor"))?;
    let minimum_khz = parse_metric(
        &read_trimmed(&format!("{root}/scaling_min_freq"))?,
        "minimum CPU frequency",
    )?;
    let maximum_khz = parse_metric(
        &read_trimmed(&format!("{root}/scaling_max_freq"))?,
        "maximum CPU frequency",
    )?;
    Ok(CpuFrequencyPolicy {
        governor,
        minimum_khz,
        maximum_khz,
    })
}

fn os_release_pretty_name() -> Result<String, String> {
    let source = fs::read_to_string("/etc/os-release")
        .map_err(|error| format!("cannot read /etc/os-release: {error}"))?;
    source
        .lines()
        .find_map(|line| line.strip_prefix("PRETTY_NAME="))
        .map(|value| value.trim_matches('"').to_owned())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "/etc/os-release has no PRETTY_NAME".to_owned())
}

fn operating_snapshot_json(snapshot: &GpuOperatingSnapshot) -> serde_json::Value {
    serde_json::json!({
        "performance_state": snapshot.performance_state.as_str(),
        "temperature_celsius": snapshot.temperature_celsius,
        "utilization_percent": snapshot.utilization_percent,
        "memory_used_mib": snapshot.memory_used_mib,
        "power_draw_watts": snapshot.power_draw_watts,
        "power_limit_watts": snapshot.power_limit_watts,
        "sm_clock_mhz": snapshot.sm_clock_mhz,
        "memory_clock_mhz": snapshot.memory_clock_mhz,
        "maximum_sm_clock_mhz": snapshot.maximum_sm_clock_mhz,
        "maximum_memory_clock_mhz": snapshot.maximum_memory_clock_mhz,
    })
}

fn nonempty_field<'a>(value: &'a str, name: &str) -> Result<&'a str, String> {
    if value.is_empty() {
        Err(format!("{name} is empty"))
    } else {
        Ok(value)
    }
}

fn parse_metric<T>(value: &str, name: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|error| format!("invalid {name} {value:?}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_point_builder_has_the_declared_sorted_shape() {
        let matrix = poisson_five_point(3).unwrap();
        assert_eq!(matrix.rows(), 9);
        assert_eq!(matrix.values().len(), 33);
        for row in 0..matrix.rows() {
            let start = matrix.row_offsets()[row];
            let end = matrix.row_offsets()[row + 1];
            assert!(matrix.column_indices()[start..end].is_sorted());
        }
    }

    #[test]
    fn median_and_durable_crossing_are_not_cherry_pickable() {
        assert_eq!(median([5, 1, 3]).unwrap(), 3);
        assert!(median([1, 2]).is_err());

        let summaries = [
            RefinementSummary {
                refinement: 64,
                rows: 4096,
                nonzeros: 20_224,
                cpu_median_ns: 10,
                cuda_median_ns: 11,
            },
            RefinementSummary {
                refinement: 128,
                rows: 16_384,
                nonzeros: 81_408,
                cpu_median_ns: 20,
                cuda_median_ns: 19,
            },
            RefinementSummary {
                refinement: 256,
                rows: 65_536,
                nonzeros: 326_656,
                cpu_median_ns: 40,
                cuda_median_ns: 30,
            },
        ];
        let summary = render_summary(&summaries);
        assert!(summary.contains("\"durable_crossing_refinement\": 128"));

        let mut rebound = summaries;
        rebound[2].cuda_median_ns = 41;
        assert!(render_summary(&rebound).contains("\"durable_crossing_refinement\": null"));

        let only_last = [
            RefinementSummary {
                refinement: 64,
                rows: 4096,
                nonzeros: 20_224,
                cpu_median_ns: 10,
                cuda_median_ns: 11,
            },
            RefinementSummary {
                refinement: 128,
                rows: 16_384,
                nonzeros: 81_408,
                cpu_median_ns: 20,
                cuda_median_ns: 19,
            },
        ];
        assert!(render_summary(&only_last).contains("\"durable_crossing_refinement\": null"));
    }

    #[test]
    fn operating_state_is_structured_without_device_or_process_identity() {
        let snapshot =
            parse_gpu_operating_snapshot("P0, 42, 17, 2048, 81.5, 300.0, 2505, 9001, 3105, 10001")
                .unwrap();
        assert_eq!(snapshot.performance_state, "P0");
        assert_eq!(snapshot.temperature_celsius, 42.0);
        assert_eq!(snapshot.utilization_percent, 17.0);
        assert_eq!(snapshot.memory_used_mib, 2048);
        assert_eq!(snapshot.sm_clock_mhz, 2505);
        assert_eq!(snapshot.maximum_memory_clock_mhz, 10_001);

        let load = parse_system_load("1.25 2.50 3.75 4/321 987654").unwrap();
        assert_eq!(
            load,
            SystemLoad {
                one_minute: 1.25,
                five_minutes: 2.5,
                fifteen_minutes: 3.75,
                runnable_processes: 4,
                total_processes: 321,
            }
        );
        assert_eq!(parse_cpu_list("2,4-6").unwrap(), (4, 2));
    }
}
