use std::ffi::{CString, c_void};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

use eqiora_core::Diagnostic;
use hdf5_metno::file::FileAccess;
use hdf5_metno::{File, from_id, h5check};
use hdf5_metno_sys::h5f::{H5F_ACC_RDONLY, H5Fopen};
use hdf5_metno_sys::h5p::{H5Pset_file_image, H5Pset_vol};
use hdf5_metno_sys::h5vl::H5VL_NATIVE;

use crate::contract::invalid_hdf5;

static NEXT_FILE_IMAGE_NAME: AtomicU64 = AtomicU64::new(1);

pub(crate) fn open(bytes: &[u8]) -> Result<File, Diagnostic> {
    let mut builder = FileAccess::build();
    builder.core_filebacked(false);
    let fapl = builder
        .finish()
        .map_err(|error| native_error("cannot create Core VFD access policy", error))?;
    let logical_name = CString::new(logical_file_image_name()?)
        .map_err(|_| invalid_hdf5("internal HDF5 file-image name is invalid"))?;

    hdf5_metno::sync::sync(|| {
        // SAFETY: `fapl` is a live file-access property list, H5VL_NATIVE is
        // the initialized built-in connector, no connector info is required,
        // and the binding's recursive mutex is held.
        h5check(unsafe { H5Pset_vol(fapl.id(), *H5VL_NATIVE, ptr::null()) })
            .map_err(|error| native_error("cannot force the native HDF5 VOL", error))?;
        // SAFETY: H5Pset_file_image copies `bytes` before returning. The slice
        // remains valid for the complete call, the local FAPL is an owned HDF5
        // file-access property list, and the shared binding mutex is held.
        h5check(unsafe {
            H5Pset_file_image(
                fapl.id(),
                bytes.as_ptr().cast_mut().cast::<c_void>(),
                bytes.len(),
            )
        })
        .map_err(|error| native_error("cannot bind complete HDF5 file-image bytes", error))?;

        // SAFETY: the Core VFD has been initialized from a copied complete file
        // image, `logical_name` is NUL-terminated for this call, and the
        // returned positive identifier is transferred exactly once to `File`.
        let file_id = h5check(unsafe { H5Fopen(logical_name.as_ptr(), H5F_ACC_RDONLY, fapl.id()) })
            .map_err(|error| native_error("cannot open HDF5 file image", error))?;
        // SAFETY: `file_id` is the newly owned identifier returned by H5Fopen;
        // no other Rust handle owns it.
        unsafe { from_id::<File>(file_id) }
            .map_err(|error| native_error("cannot own HDF5 file-image handle", error))
    })
}

fn logical_file_image_name() -> Result<String, Diagnostic> {
    #[allow(deprecated, reason = "try_update requires Rust 1.95; MSRV is 1.89")]
    let ordinal = NEXT_FILE_IMAGE_NAME
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| invalid_hdf5("HDF5 logical file-image ordinal is exhausted"))?;
    Ok(format!("eqiora-file-image-{ordinal}.h5"))
}

fn native_error(context: &str, error: hdf5_metno::Error) -> Diagnostic {
    invalid_hdf5(format!("{context}: {error}"))
}
