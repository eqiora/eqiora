//! All native ownership and unsafe calls are confined to this module.
use crate::runtime::failure;
use eqiora_core::Diagnostic;
use eqiora_time::{
    DaeVariableKind, ImplicitDaeInitialization, ImplicitDaeProblem, ImplicitTimeSystem,
    InitialConditionPolicy, TimeHistoryStep, TimePlan,
};
use std::{
    ffi::{c_char, c_int, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    ptr, slice,
};
use sundials_sys as sys;
// Reject silently falling back to the binding crate's older bundled release.
const _: () = assert!(
    sys::SUNDIALS_VERSION_MAJOR == 7
        && sys::SUNDIALS_VERSION_MINOR == 8
        && sys::SUNDIALS_VERSION_PATCH == 0
);
const _: Option<sys::sunrealtype> = Some(0.0_f64);
// The sys wrapper omits sundials_version.h. This scalar-only function is stable
// in the exact admitted header release and executes before any native objects.
unsafe extern "C" {
    fn SUNDIALSGetVersionNumber(
        major: *mut c_int,
        minor: *mut c_int,
        patch: *mut c_int,
        label: *mut c_char,
        len: c_int,
    ) -> c_int;
}
struct Callbacks<'a> {
    system: &'a dyn ImplicitTimeSystem,
    n: usize,
    error: Option<Diagnostic>,
    panic: Option<Box<dyn std::any::Any + Send>>,
}
impl Callbacks<'_> {
    fn invoke(&mut self, action: impl FnOnce(&Self) -> Result<(), Diagnostic>) -> c_int {
        if self.error.is_some() {
            return -1;
        }
        match catch_unwind(AssertUnwindSafe(|| action(self))) {
            Ok(Ok(())) => 0,
            Ok(Err(error)) => {
                self.error = Some(error);
                -1
            }
            Err(payload) => {
                self.panic = Some(payload);
                self.error = Some(failure("IDA callback panicked"));
                -1
            }
        }
    }
}
unsafe extern "C" fn residual(
    t: f64,
    y: sys::N_Vector,
    v: sys::N_Vector,
    r: sys::N_Vector,
    data: *mut c_void,
) -> c_int {
    // SAFETY: IDA synchronously borrows this stable boxed userdata; it outlives IDAFree.
    let callbacks = unsafe { &mut *data.cast::<Callbacks<'_>>() };
    callbacks.invoke(|c| {
        // SAFETY: IDA supplies distinct live serial vectors of the session dimension.
        let (y, v, r) = unsafe {
            (
                slice::from_raw_parts(sys::N_VGetArrayPointer(y), c.n),
                slice::from_raw_parts(sys::N_VGetArrayPointer(v), c.n),
                slice::from_raw_parts_mut(sys::N_VGetArrayPointer(r), c.n),
            )
        };
        r.fill(f64::NAN);
        c.system.residual(t, y, v, r)?;
        finite(r)
    })
}
unsafe extern "C" fn jacobian(
    t: f64,
    cj: f64,
    y: sys::N_Vector,
    v: sys::N_Vector,
    _r: sys::N_Vector,
    m: sys::SUNMatrix,
    data: *mut c_void,
    _a: sys::N_Vector,
    _b: sys::N_Vector,
    _c: sys::N_Vector,
) -> c_int {
    // SAFETY: the registered userdata is the live, exclusively borrowed callback box.
    let callbacks = unsafe { &mut *data.cast::<Callbacks<'_>>() };
    callbacks.invoke(|c| {
        // SAFETY: these read-only serial vectors have the validated session dimension.
        let (y, v) = unsafe {
            (
                slice::from_raw_parts(sys::N_VGetArrayPointer(y), c.n),
                slice::from_raw_parts(sys::N_VGetArrayPointer(v), c.n),
            )
        };
        let mut direction = vec![0.; c.n];
        let mut rate_direction = vec![0.; c.n];
        let mut output = vec![f64::NAN; c.n];
        for column in 0..c.n {
            direction[column] = 1.;
            rate_direction[column] = cj;
            output.fill(f64::NAN);
            c.system
                .residual_jvp(t, y, v, &direction, &rate_direction, &mut output)?;
            finite(&output)?;
            // SAFETY: the owned n-by-n dense matrix has n contiguous entries per column;
            // the separate output vector contains exactly n initialized values.
            unsafe {
                ptr::copy_nonoverlapping(
                    output.as_ptr(),
                    sys::SUNDenseMatrix_Column(m, column as sys::sunindextype),
                    c.n,
                )
            };
            direction[column] = 0.;
            rate_direction[column] = 0.;
        }
        Ok(())
    })
}
fn finite(values: &[f64]) -> Result<(), Diagnostic> {
    if values.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(failure(
            "IDA callback returned non-finite or incomplete values",
        ))
    }
}
pub(crate) struct Session<'a> {
    ctx: sys::SUNContext,
    y: sys::N_Vector,
    v: sys::N_Vector,
    atol: sys::N_Vector,
    id: sys::N_Vector,
    dense: sys::N_Vector,
    time: f64,
    matrix: sys::SUNMatrix,
    linear: sys::SUNLinearSolver,
    mem: *mut c_void,
    callbacks: Box<Callbacks<'a>>,
}
impl<'a> Session<'a> {
    pub(crate) fn new(
        problem: &'a ImplicitDaeProblem<'_>,
        plan: &TimePlan,
    ) -> Result<Self, Diagnostic> {
        let (mut major, mut minor, mut patch, mut label) = (0, 0, 0, [0 as c_char; 64]);
        // SAFETY: all output pointers address initialized scalars or the 64-byte label.
        let code = unsafe {
            SUNDIALSGetVersionNumber(&mut major, &mut minor, &mut patch, label.as_mut_ptr(), 64)
        };
        if code != 0 || (major, minor, patch) != (7, 8, 0) || label[0] != 0 {
            return Err(failure(
                "IDA requires native SUNDIALS 7.8.0 without a release suffix",
            ));
        }
        let n = problem.dimension();
        let native_n = sys::sunindextype::try_from(n)
            .map_err(|_| failure("IDA dimension exceeds native index range"))?;
        if n.checked_mul(n)
            .and_then(|x| x.checked_mul(size_of::<f64>()))
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(failure("IDA dense matrix size overflows"));
        }
        let mut s = Self {
            ctx: ptr::null_mut(),
            y: ptr::null_mut(),
            v: ptr::null_mut(),
            atol: ptr::null_mut(),
            id: ptr::null_mut(),
            dense: ptr::null_mut(),
            time: plan.start_time(),
            matrix: ptr::null_mut(),
            linear: ptr::null_mut(),
            mem: ptr::null_mut(),
            callbacks: Box::new(Callbacks {
                system: problem.system(),
                n,
                error: None,
                panic: None,
            }),
        };
        // SAFETY: acquire checked native handles in dependency order. Every partial
        // acquisition is owned by s and released by Drop. Validated input slices
        // match n; userdata remains boxed until after IDAFree, with no escaping borrow.
        unsafe {
            let code = sys::SUNContext_Create(sys::SUN_COMM_NULL, &mut s.ctx);
            s.check(code, "create context")?;
            s.y = sys::N_VNew_Serial(native_n, s.ctx);
            nonnull(s.y)?;
            s.v = sys::N_VClone(s.y);
            nonnull(s.v)?;
            s.atol = sys::N_VClone(s.y);
            nonnull(s.atol)?;
            s.id = sys::N_VClone(s.y);
            nonnull(s.id)?;
            s.dense = sys::N_VClone(s.y);
            nonnull(s.dense)?;
            ptr::copy_nonoverlapping(
                problem.initial_state().as_ptr(),
                sys::N_VGetArrayPointer(s.y),
                n,
            );
            ptr::copy_nonoverlapping(
                problem.initial_derivative().as_ptr(),
                sys::N_VGetArrayPointer(s.v),
                n,
            );
            ptr::copy_nonoverlapping(
                plan.absolute_tolerances().as_ptr(),
                sys::N_VGetArrayPointer(s.atol),
                n,
            );
            for (i, kind) in problem.variable_kinds().iter().enumerate() {
                *sys::N_VGetArrayPointer(s.id).add(i) = if *kind == DaeVariableKind::Differential {
                    1.
                } else {
                    0.
                };
            }
            s.matrix = sys::SUNDenseMatrix(native_n, native_n, s.ctx);
            nonnull(s.matrix)?;
            s.linear = sys::SUNLinSol_Dense(s.y, s.matrix, s.ctx);
            nonnull(s.linear)?;
            s.mem = sys::IDACreate(s.ctx);
            nonnull(s.mem)?;
            let code = sys::IDAInit(s.mem, Some(residual), plan.start_time(), s.y, s.v);
            s.check(code, "initialize")?;
            let data = (&raw mut *s.callbacks).cast();
            let code = sys::IDASetUserData(s.mem, data);
            s.check(code, "set callback data")?;
            let code = sys::IDASVtolerances(s.mem, plan.relative_tolerance(), s.atol);
            s.check(code, "set tolerances")?;
            let code = sys::IDASetLinearSolver(s.mem, s.linear, s.matrix);
            s.check(code, "set linear solver")?;
            let code = sys::IDASetJacFn(s.mem, Some(jacobian));
            s.check(code, "set paired JVP")?;
            let code = sys::IDASetId(s.mem, s.id);
            s.check(code, "set variable partition")?;
            let code = sys::IDASetInitStep(s.mem, plan.initial_step());
            s.check(code, "set initial step")?;
            let code = sys::IDASetStopTime(
                s.mem,
                *plan
                    .output_times()
                    .last()
                    .expect("validated nonempty outputs"),
            );
            s.check(code, "set terminal time")?;
            if problem.initial_condition() == InitialConditionPolicy::SolveConsistent {
                // IC scaling is a solver control, independent of output cadence.
                let target = (plan.start_time() + plan.initial_step())
                    .min(*plan.output_times().last().expect("validated outputs"));
                if target <= plan.start_time() {
                    return Err(failure(
                        "IDA initial step cannot advance representable time",
                    ));
                }
                let code = sys::IDACalcIC(s.mem, sys::IDA_YA_YDP_INIT, target);
                s.check(code, "consistent initialization")?;
                let code = sys::IDAGetConsistentIC(s.mem, s.y, s.v);
                s.check(code, "read consistent pair")?;
            }
        }
        Ok(s)
    }
    fn check(&mut self, code: c_int, operation: &str) -> Result<(), Diagnostic> {
        if let Some(error) = self.callbacks.error.take() {
            // Dispose of user panic payloads only after returning from C. A payload
            // destructor can itself panic; contain it and retain the original diagnostic.
            if let Some(payload) = self.callbacks.panic.take()
                && let Err(secondary) = catch_unwind(AssertUnwindSafe(|| drop(payload)))
            {
                std::mem::forget(secondary);
            }
            return Err(error);
        }
        if code < 0 {
            Err(failure(format!(
                "IDA {operation} failed with status {code}"
            )))
        } else {
            Ok(())
        }
    }
    pub(crate) fn initialization(&self) -> Result<ImplicitDaeInitialization, Diagnostic> {
        // SAFETY: successful construction owns both live n-coordinate serial vectors.
        unsafe {
            ImplicitDaeInitialization::accepted(
                slice::from_raw_parts(sys::N_VGetArrayPointer(self.y), self.callbacks.n).to_vec(),
                slice::from_raw_parts(sys::N_VGetArrayPointer(self.v), self.callbacks.n).to_vec(),
            )
        }
    }
    pub(crate) fn time(&self) -> f64 {
        self.time
    }
    fn state(&self) -> Vec<f64> {
        // SAFETY: the session owns a live serial vector of the validated dimension.
        unsafe { slice::from_raw_parts(sys::N_VGetArrayPointer(self.y), self.callbacks.n).to_vec() }
    }
    fn interpolate(&mut self, time: f64) -> Result<Vec<f64>, Diagnostic> {
        // SAFETY: native memory and scratch vector are live; k=0 samples the
        // latest accepted native polynomial without changing its owned history.
        let code = unsafe { sys::IDAGetDky(self.mem, time, 0, self.dense) };
        self.check(code, "capture native dense output")?;
        // SAFETY: the private scratch serial vector contains n initialized entries.
        let values =
            unsafe { slice::from_raw_parts(sys::N_VGetArrayPointer(self.dense), self.callbacks.n) };
        finite(values)?;
        Ok(values.to_vec())
    }
    pub(crate) fn step(&mut self, horizon: f64) -> Result<TimeHistoryStep, Diagnostic> {
        let start = self.time;
        let start_state = self.state();
        let mut end = start;
        // SAFETY: all owned handles/userdata outlive this synchronous advance.
        let code = unsafe {
            sys::IDASolve(
                self.mem,
                horizon,
                &mut end,
                self.y,
                self.v,
                sys::IDA_ONE_STEP,
            )
        };
        self.check(code, "advance accepted step")?;
        if ![sys::IDA_SUCCESS, sys::IDA_TSTOP_RETURN].contains(&code)
            || !end.is_finite()
            || end <= start
            || end > horizon
        {
            return Err(failure("IDA returned an invalid accepted interval"));
        }
        let mut order = 0;
        // SAFETY: order is a live output scalar; memory owns the last accepted step.
        let code = unsafe { sys::IDAGetLastOrder(self.mem, &mut order) };
        self.check(code, "read accepted interpolation order")?;
        let degree = usize::try_from(order)
            .map_err(|_| failure("IDA returned a negative interpolation order"))?;
        let middle = self.interpolate(start + (end - start) * 0.5)?;
        let end_state = self.state();
        let step =
            TimeHistoryStep::capture(start, end, degree, start_state, middle, end_state, |time| {
                self.interpolate(time)
            })?;
        self.time = end;
        Ok(step)
    }
}
fn nonnull<T>(value: *mut T) -> Result<(), Diagnostic> {
    if value.is_null() {
        Err(failure("IDA native allocation failed"))
    } else {
        Ok(())
    }
}
impl Drop for Session<'_> {
    fn drop(&mut self) {
        // SAFETY: each non-null field has one owner. Release the solver before
        // its borrowed algebra objects, vectors and context; callbacks remain live.
        unsafe {
            if !self.mem.is_null() {
                sys::IDAFree(&mut self.mem);
            }
            if !self.linear.is_null() {
                sys::SUNLinSolFree(self.linear);
            }
            if !self.matrix.is_null() {
                sys::SUNMatDestroy(self.matrix);
            }
            for vector in [self.dense, self.id, self.atol, self.v, self.y] {
                if !vector.is_null() {
                    sys::N_VDestroy(vector);
                }
            }
            if !self.ctx.is_null() {
                sys::SUNContext_Free(&mut self.ctx);
            }
        }
    }
}
