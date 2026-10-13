//! Edit one JSON value without reordering any untouched object or graph bytes.

use serde_json::value::RawValue;
use std::collections::BTreeMap;

pub fn edit(bytes: &[u8], pointer: &str, change: impl FnOnce(&str) -> String) -> Vec<u8> {
    let mut raw: &RawValue = serde_json::from_slice(bytes).unwrap();
    for segment in pointer.strip_prefix('/').unwrap().split('/') {
        raw = if raw.get().starts_with('[') {
            let array: Vec<&RawValue> = serde_json::from_str(raw.get()).unwrap();
            array[segment.parse::<usize>().unwrap()]
        } else {
            let object: BTreeMap<&str, &RawValue> = serde_json::from_str(raw.get()).unwrap();
            object[segment]
        };
    }
    let start = raw.get().as_ptr() as usize - bytes.as_ptr() as usize;
    let replacement = change(raw.get());
    assert_ne!(raw.get(), replacement, "mutation must change {pointer}");
    let mut changed = bytes.to_vec();
    changed.splice(start..start + raw.get().len(), replacement.bytes());
    changed
}

pub fn replace(bytes: &[u8], pointer: &str, value: &str) -> Vec<u8> {
    edit(bytes, pointer, |_| value.to_owned())
}

pub fn edit_array(
    bytes: &[u8],
    pointer: &str,
    change: impl FnOnce(&mut Vec<Box<RawValue>>),
) -> Vec<u8> {
    edit(bytes, pointer, |raw| {
        let mut array = serde_json::from_str(raw).unwrap();
        change(&mut array);
        serde_json::to_string(&array).unwrap()
    })
}
