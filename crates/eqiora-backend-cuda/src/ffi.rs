//! The complete unsafe cuSPARSE boundary.
//!
//! cudarc owns contexts, streams, and allocations. This private boundary loads
//! only the CUDA 12 Generic API symbols used by Eqiora and retains the library
//! for every handle and descriptor lifetime. The narrow symbol set originated
//! with cudarc 0.18's eager loader; cudarc 0.19 now loads symbols on use.

use std::ffi::c_void;
use std::fmt;
use std::mem::MaybeUninit;
use std::sync::Arc;

use cudarc::driver;
use cudarc::driver::result::DriverError;
use libloading::Library;

type Status = u32;
type Handle = *mut c_void;
type SparseMatrix = *mut c_void;
type DenseVector = *mut c_void;

const SUCCESS: Status = 0;

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
enum Operation {
    NonTranspose = 0,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
enum IndexType {
    Signed64 = 3,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
enum IndexBase {
    Zero = 0,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
enum DataType {
    Float64 = 1,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
enum SpmvAlgorithm {
    Default = 0,
    DeterministicCsr = 3,
}

type Create = unsafe extern "C" fn(*mut Handle) -> Status;
type Destroy = unsafe extern "C" fn(Handle) -> Status;
type SetStream = unsafe extern "C" fn(Handle, *mut c_void) -> Status;
type GetVersion = unsafe extern "C" fn(Handle, *mut i32) -> Status;
type CreateCsr = unsafe extern "C" fn(
    *mut SparseMatrix,
    i64,
    i64,
    i64,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    IndexType,
    IndexType,
    IndexBase,
    DataType,
) -> Status;
type DestroySparseMatrix = unsafe extern "C" fn(SparseMatrix) -> Status;
type CreateDenseVector =
    unsafe extern "C" fn(*mut DenseVector, i64, *mut c_void, DataType) -> Status;
type DestroyDenseVector = unsafe extern "C" fn(DenseVector) -> Status;
type DenseVectorSetValues = unsafe extern "C" fn(DenseVector, *mut c_void) -> Status;
type SpmvBufferSize = unsafe extern "C" fn(
    Handle,
    Operation,
    *const c_void,
    SparseMatrix,
    DenseVector,
    *const c_void,
    DenseVector,
    DataType,
    SpmvAlgorithm,
    *mut usize,
) -> Status;
type Spmv = unsafe extern "C" fn(
    Handle,
    Operation,
    *const c_void,
    SparseMatrix,
    DenseVector,
    *const c_void,
    DenseVector,
    DataType,
    SpmvAlgorithm,
    *mut c_void,
) -> Status;

/// Private FFI failure converted to a stable Eqiora diagnostic at the adapter
/// boundary.
pub(crate) struct FfiError(String);

impl fmt::Debug for FfiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("FfiError").field(&self.0).finish()
    }
}

struct Functions {
    create: Create,
    destroy: Destroy,
    set_stream: SetStream,
    get_version: GetVersion,
    create_csr: CreateCsr,
    destroy_sparse_matrix: DestroySparseMatrix,
    create_dense_vector: CreateDenseVector,
    destroy_dense_vector: DestroyDenseVector,
    dense_vector_set_values: DenseVectorSetValues,
    spmv_buffer_size: SpmvBufferSize,
    spmv: Spmv,
}

struct CusparseLibrary {
    functions: Functions,
    _library: Library,
}

impl fmt::Debug for CusparseLibrary {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CusparseLibrary { libcusparse.so.12 }")
    }
}

impl CusparseLibrary {
    fn load() -> Result<Arc<Self>, FfiError> {
        // SAFETY: the library is retained by the returned `Arc` for longer
        // than every copied symbol and object created through those symbols.
        let library = unsafe { Library::new("libcusparse.so.12") }
            .map_err(|error| FfiError(format!("could not load libcusparse.so.12: {error}")))?;
        let functions = Functions {
            create: load(&library, b"cusparseCreate\0")?,
            destroy: load(&library, b"cusparseDestroy\0")?,
            set_stream: load(&library, b"cusparseSetStream\0")?,
            get_version: load(&library, b"cusparseGetVersion\0")?,
            create_csr: load(&library, b"cusparseCreateCsr\0")?,
            destroy_sparse_matrix: load(&library, b"cusparseDestroySpMat\0")?,
            create_dense_vector: load(&library, b"cusparseCreateDnVec\0")?,
            destroy_dense_vector: load(&library, b"cusparseDestroyDnVec\0")?,
            dense_vector_set_values: load(&library, b"cusparseDnVecSetValues\0")?,
            spmv_buffer_size: load(&library, b"cusparseSpMV_bufferSize\0")?,
            spmv: load(&library, b"cusparseSpMV\0")?,
        };
        Ok(Arc::new(Self {
            functions,
            _library: library,
        }))
    }
}

pub(crate) fn probe_cusparse() -> Result<(), FfiError> {
    CusparseLibrary::load().map(|_| ())
}

fn load<T: Copy>(library: &Library, symbol: &'static [u8]) -> Result<T, FfiError> {
    // SAFETY: each `T` exactly matches the documented CUDA 12 Generic API C
    // signature. The owning library outlives the copied function pointer.
    unsafe { library.get::<T>(symbol) }
        .map(|value| *value)
        .map_err(|error| {
            let name = std::str::from_utf8(symbol)
                .unwrap_or("<non-UTF8 symbol>")
                .trim_end_matches('\0');
            FfiError(format!("missing cuSPARSE symbol {name}: {error}"))
        })
}

fn status(operation: &str, value: Status) -> Result<(), FfiError> {
    if value == SUCCESS {
        Ok(())
    } else {
        Err(FfiError(format!(
            "{operation} returned cuSPARSE status {value}"
        )))
    }
}

pub(crate) struct DeviceProperties {
    pub(crate) name: String,
    pub(crate) uuid: [u8; 16],
    pub(crate) total_memory_bytes: usize,
    pub(crate) compute_capability: (i32, i32),
}

pub(crate) fn device_properties(ordinal: u16) -> Result<DeviceProperties, DriverError> {
    let device = driver::result::device::get(i32::from(ordinal))?;
    let name = driver::result::device::get_name(device)?;
    let uuid = driver::result::device::get_uuid(device)?
        .bytes
        .map(|byte| byte as u8);
    // SAFETY: `device` was returned by `cuDeviceGet` above and remains valid
    // for these read-only driver queries.
    let (total_memory_bytes, compute_major, compute_minor) = unsafe {
        (
            driver::result::device::total_mem(device)?,
            driver::result::device::get_attribute(
                device,
                driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )?,
            driver::result::device::get_attribute(
                device,
                driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )?,
        )
    };
    Ok(DeviceProperties {
        name,
        uuid,
        total_memory_bytes,
        compute_capability: (compute_major, compute_minor),
    })
}

pub(crate) fn driver_version() -> Result<i32, DriverError> {
    let mut version = MaybeUninit::uninit();
    // SAFETY: CUDA writes one `c_int` to the valid out pointer. Driver loading
    // has already succeeded before this query is called.
    unsafe {
        driver::sys::cuDriverGetVersion(version.as_mut_ptr()).result()?;
        Ok(version.assume_init())
    }
}

#[derive(Debug)]
pub(crate) struct CusparseHandle {
    library: Arc<CusparseLibrary>,
    raw: Handle,
}

impl CusparseHandle {
    pub(crate) fn new(stream: driver::sys::CUstream) -> Result<Self, FfiError> {
        let library = CusparseLibrary::load()?;
        let mut raw = MaybeUninit::uninit();
        // SAFETY: `raw` is a valid out pointer and the library remains live.
        unsafe {
            status(
                "cusparseCreate",
                (library.functions.create)(raw.as_mut_ptr()),
            )?
        };
        // SAFETY: successful creation initialized one live handle.
        let raw = unsafe { raw.assume_init() };
        // SAFETY: the handle is live and the stream is owned by the same live
        // CUDA context for the full handle lifetime.
        if let Err(error) = unsafe {
            status(
                "cusparseSetStream",
                (library.functions.set_stream)(raw, stream.cast()),
            )
        } {
            // SAFETY: creation succeeded and ownership has not escaped.
            let _ = unsafe { (library.functions.destroy)(raw) };
            return Err(error);
        }
        Ok(Self { library, raw })
    }

    pub(crate) fn version(&self) -> Result<i32, FfiError> {
        let mut version = MaybeUninit::uninit();
        // SAFETY: the live handle writes one `c_int` to the valid out pointer.
        unsafe {
            status(
                "cusparseGetVersion",
                (self.library.functions.get_version)(self.raw, version.as_mut_ptr()),
            )?;
            Ok(version.assume_init())
        }
    }
}

impl Drop for CusparseHandle {
    fn drop(&mut self) {
        let raw = std::mem::replace(&mut self.raw, std::ptr::null_mut());
        if !raw.is_null() {
            // SAFETY: ownership is unique and `raw` is replaced first.
            let _ = unsafe { (self.library.functions.destroy)(raw) };
        }
    }
}

#[derive(Debug)]
struct CsrDescriptor {
    library: Arc<CusparseLibrary>,
    raw: SparseMatrix,
}

impl CsrDescriptor {
    #[allow(clippy::too_many_arguments)]
    fn new_f64_i64(
        library: Arc<CusparseLibrary>,
        rows: i64,
        columns: i64,
        nonzeros: i64,
        row_offsets: driver::sys::CUdeviceptr,
        column_indices: driver::sys::CUdeviceptr,
        values: driver::sys::CUdeviceptr,
    ) -> Result<Self, FfiError> {
        let mut raw = MaybeUninit::uninit();
        // SAFETY: the caller proves every pointer names a live allocation with
        // the validated CSR lengths and declared element types.
        unsafe {
            status(
                "cusparseCreateCsr",
                (library.functions.create_csr)(
                    raw.as_mut_ptr(),
                    rows,
                    columns,
                    nonzeros,
                    row_offsets as *mut c_void,
                    column_indices as *mut c_void,
                    values as *mut c_void,
                    IndexType::Signed64,
                    IndexType::Signed64,
                    IndexBase::Zero,
                    DataType::Float64,
                ),
            )?;
            Ok(Self {
                library,
                raw: raw.assume_init(),
            })
        }
    }
}

impl Drop for CsrDescriptor {
    fn drop(&mut self) {
        let raw = std::mem::replace(&mut self.raw, std::ptr::null_mut());
        if !raw.is_null() {
            // SAFETY: the descriptor is uniquely owned and no call is in
            // flight because the stream is synchronized before this drop.
            let _ = unsafe { (self.library.functions.destroy_sparse_matrix)(raw) };
        }
    }
}

#[derive(Debug)]
struct DenseVectorDescriptor {
    library: Arc<CusparseLibrary>,
    raw: DenseVector,
}

impl DenseVectorDescriptor {
    fn new_f64(
        library: Arc<CusparseLibrary>,
        elements: i64,
        values: driver::sys::CUdeviceptr,
    ) -> Result<Self, FfiError> {
        let mut raw = MaybeUninit::uninit();
        // SAFETY: `values` names a live `f64` allocation with the declared
        // number of elements and outlives this descriptor.
        unsafe {
            status(
                "cusparseCreateDnVec",
                (library.functions.create_dense_vector)(
                    raw.as_mut_ptr(),
                    elements,
                    values as *mut c_void,
                    DataType::Float64,
                ),
            )?;
            Ok(Self {
                library,
                raw: raw.assume_init(),
            })
        }
    }

    fn set_values(&self, values: driver::sys::CUdeviceptr) -> Result<(), FfiError> {
        // SAFETY: the descriptor is live and `values` names a live allocation
        // with the descriptor's fixed element count and scalar type.
        unsafe {
            status(
                "cusparseDnVecSetValues",
                (self.library.functions.dense_vector_set_values)(self.raw, values as *mut c_void),
            )
        }
    }
}

impl Drop for DenseVectorDescriptor {
    fn drop(&mut self) {
        let raw = std::mem::replace(&mut self.raw, std::ptr::null_mut());
        if !raw.is_null() {
            // SAFETY: the descriptor is uniquely owned and synchronized.
            let _ = unsafe { (self.library.functions.destroy_dense_vector)(raw) };
        }
    }
}

#[derive(Debug)]
pub(crate) struct SpmvPlan {
    matrix: CsrDescriptor,
    input: DenseVectorDescriptor,
    output: DenseVectorDescriptor,
    algorithm: SpmvAlgorithm,
    workspace_bytes: usize,
}

impl SpmvPlan {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        handle: &CusparseHandle,
        rows: i64,
        columns: i64,
        nonzeros: i64,
        row_offsets: driver::sys::CUdeviceptr,
        column_indices: driver::sys::CUdeviceptr,
        values: driver::sys::CUdeviceptr,
        input: driver::sys::CUdeviceptr,
        output: driver::sys::CUdeviceptr,
        deterministic: bool,
    ) -> Result<Self, FfiError> {
        let matrix = CsrDescriptor::new_f64_i64(
            handle.library.clone(),
            rows,
            columns,
            nonzeros,
            row_offsets,
            column_indices,
            values,
        )?;
        let input = DenseVectorDescriptor::new_f64(handle.library.clone(), columns, input)?;
        let output = DenseVectorDescriptor::new_f64(handle.library.clone(), rows, output)?;
        let algorithm = if deterministic {
            SpmvAlgorithm::DeterministicCsr
        } else {
            SpmvAlgorithm::Default
        };
        let alpha = 1.0_f64;
        let beta = 0.0_f64;
        let mut workspace_bytes = MaybeUninit::uninit();
        // SAFETY: every descriptor is live and shape-compatible, scalars have
        // the declared type, and the out pointer receives one size.
        unsafe {
            status(
                "cusparseSpMV_bufferSize",
                (handle.library.functions.spmv_buffer_size)(
                    handle.raw,
                    Operation::NonTranspose,
                    (&alpha as *const f64).cast(),
                    matrix.raw,
                    input.raw,
                    (&beta as *const f64).cast(),
                    output.raw,
                    DataType::Float64,
                    algorithm,
                    workspace_bytes.as_mut_ptr(),
                ),
            )?;
            Ok(Self {
                matrix,
                input,
                output,
                algorithm,
                workspace_bytes: workspace_bytes.assume_init(),
            })
        }
    }

    pub(crate) fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    pub(crate) fn apply(
        &self,
        handle: &CusparseHandle,
        workspace: driver::sys::CUdeviceptr,
        input: driver::sys::CUdeviceptr,
        output: driver::sys::CUdeviceptr,
        alpha: f64,
        beta: f64,
    ) -> Result<(), FfiError> {
        self.input.set_values(input)?;
        self.output.set_values(output)?;
        // SAFETY: descriptors and allocations remain live; the retained
        // workspace is at least the queried size; the caller keeps the current
        // vector allocations live and synchronizes before they are dropped.
        unsafe {
            status(
                "cusparseSpMV",
                (handle.library.functions.spmv)(
                    handle.raw,
                    Operation::NonTranspose,
                    (&alpha as *const f64).cast(),
                    self.matrix.raw,
                    self.input.raw,
                    (&beta as *const f64).cast(),
                    self.output.raw,
                    DataType::Float64,
                    self.algorithm,
                    workspace as *mut c_void,
                ),
            )
        }
    }
}
