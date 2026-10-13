use eqiora_core::Diagnostic;

use crate::xml::invalid_import;

pub(crate) fn allocate_vec<T>(capacity: usize, label: &str) -> Result<Vec<T>, Diagnostic> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| invalid_import(format!("{label} allocation failed")))?;
    Ok(values)
}

pub(crate) fn copy_slice<T: Copy>(source: &[T], label: &str) -> Result<Vec<T>, Diagnostic> {
    let mut copy = allocate_vec(source.len(), label)?;
    copy.extend_from_slice(source);
    Ok(copy)
}

pub(crate) fn copy_string(source: &str, label: &str) -> Result<String, Diagnostic> {
    let mut copy = String::new();
    copy.try_reserve_exact(source.len())
        .map_err(|_| invalid_import(format!("{label} allocation failed")))?;
    copy.push_str(source);
    Ok(copy)
}
