use hdf5_metno::dataset::Layout;
use hdf5_metno::{Dataset, File, Group, LinkInfo, LinkType};

use eqiora_core::Diagnostic;

use crate::contract::{Hdf5ResolveLimits, invalid_hdf5};
use crate::native_inspect::{exact_scalar_type, exact_shape, require_internal_unfiltered_storage};

pub(crate) fn complete_file(file: &File, limits: Hdf5ResolveLimits) -> Result<(), Diagnostic> {
    let root_info = file
        .loc_info()
        .map_err(|error| native_error("cannot inspect HDF5 root", error))?;
    reject_attributes(root_info.num_attrs)?;
    require_one_hard_link(root_info.num_links)?;

    let root_group = file
        .as_group()
        .map_err(|error| native_error("cannot own HDF5 root group", error))?;
    let mut pending = Vec::new();
    pending.try_reserve_exact(1).map_err(|error| {
        invalid_hdf5(format!(
            "cannot reserve bounded HDF5 group audit storage: {error}",
        ))
    })?;
    pending.push(root_group);

    let mut state = AuditState {
        limits,
        links: 0,
        objects: 1,
        datasets: 0,
        total_name_bytes: 0,
        total_declared_values: 0,
        work: 1,
        pending,
    };
    while let Some(group) = state.pending.pop() {
        state.audit_group(&group)?;
    }
    Ok(())
}

struct AuditState {
    limits: Hdf5ResolveLimits,
    links: usize,
    objects: usize,
    datasets: usize,
    total_name_bytes: usize,
    total_declared_values: usize,
    work: usize,
    pending: Vec<Group>,
}

impl AuditState {
    fn audit_group(&mut self, group: &Group) -> Result<(), Diagnostic> {
        let limits = self.limits;
        let mut batch = LinkBatch {
            entries: Vec::new(),
            links: self.links,
            total_name_bytes: self.total_name_bytes,
            work: self.work,
        };
        // Stop on the first admission failure while retaining its typed diagnostic.
        let failure = group
            .find_link(Default::default(), Default::default(), |name, info| {
                Ok(batch.try_admit(name, info, limits).err())
            })
            .map_err(|error| native_error("cannot enumerate HDF5 links", error))?;
        if let Some(error) = failure {
            return Err(error);
        }
        self.links = batch.links;
        self.total_name_bytes = batch.total_name_bytes;
        self.work = batch.work;

        for name in batch.entries {
            let info = group
                .loc_info_by_name(&name)
                .map_err(|error| native_error("cannot inspect hard-linked HDF5 object", error))?;
            reject_attributes(info.num_attrs)?;
            require_one_hard_link(info.num_links)?;
            self.objects = checked_add(self.objects, 1, "HDF5 object count")?;
            require_at_most(self.objects, self.limits.max_objects, "HDF5 object count")?;
            self.account_work(1)?;

            let location = group
                .open_by_token(info.token)
                .map_err(|error| native_error("cannot open audited HDF5 object token", error))?;
            match info.loc_type {
                hdf5_metno::LocationType::Group => {
                    let child = location
                        .as_group()
                        .map_err(|error| native_error("cannot own HDF5 group", error))?;
                    self.pending.try_reserve(1).map_err(|error| {
                        invalid_hdf5(format!(
                            "cannot reserve bounded HDF5 group audit storage: {error}",
                        ))
                    })?;
                    self.pending.push(child);
                }
                hdf5_metno::LocationType::Dataset => {
                    let dataset = location
                        .as_dataset()
                        .map_err(|error| native_error("cannot own HDF5 dataset", error))?;
                    self.audit_dataset(&dataset)?;
                }
                hdf5_metno::LocationType::NamedDatatype => {
                    return Err(invalid_hdf5("HDF5 v1 rejects committed or named datatypes"));
                }
                hdf5_metno::LocationType::TypeMap => {
                    return Err(invalid_hdf5("HDF5 v1 rejects map objects"));
                }
            }
        }
        Ok(())
    }

    fn audit_dataset(&mut self, dataset: &Dataset) -> Result<(), Diagnostic> {
        self.datasets = checked_add(self.datasets, 1, "HDF5 dataset count")?;
        require_at_most(
            self.datasets,
            self.limits.max_datasets,
            "HDF5 dataset count",
        )?;
        let dcpl = dataset
            .dcpl()
            .map_err(|error| native_error("cannot inspect HDF5 dataset creation policy", error))?;
        let layout = dcpl
            .get_layout()
            .map_err(|error| native_error("cannot inspect HDF5 dataset layout", error))?;
        if layout == Layout::Virtual {
            return Err(invalid_hdf5("HDF5 v1 rejects virtual datasets"));
        }
        require_internal_unfiltered_storage(&dcpl)?;
        exact_scalar_type(dataset)?;

        let shape = exact_shape(dataset, self.limits.max_rank)?;
        let mut platform_shape = Vec::new();
        platform_shape
            .try_reserve_exact(shape.len())
            .map_err(|error| {
                invalid_hdf5(format!(
                    "cannot reserve bounded platform HDF5 shape: {error}",
                ))
            })?;
        for extent in shape {
            platform_shape.push(
                usize::try_from(extent)
                    .map_err(|_| invalid_hdf5("HDF5 dataset extent exceeds usize"))?,
            );
        }
        let values = checked_product(&platform_shape, "HDF5 dataset scalar count")?;
        require_at_most(
            values,
            self.limits.max_dataset_values,
            "HDF5 dataset scalar count",
        )?;
        self.total_declared_values = checked_add(
            self.total_declared_values,
            values,
            "aggregate HDF5 declared scalar count",
        )?;
        require_at_most(
            self.total_declared_values,
            self.limits.max_total_declared_values,
            "aggregate HDF5 declared scalar count",
        )?;
        self.account_work(platform_shape.len().saturating_add(values.min(1)))
    }

    fn account_work(&mut self, amount: usize) -> Result<(), Diagnostic> {
        self.work = checked_add(self.work, amount, "HDF5 audit work")?;
        require_at_most(self.work, self.limits.max_audit_work, "HDF5 audit work")
    }
}

struct LinkBatch {
    entries: Vec<String>,
    links: usize,
    total_name_bytes: usize,
    work: usize,
}

impl LinkBatch {
    fn try_admit(
        &mut self,
        name: &str,
        link: LinkInfo,
        limits: Hdf5ResolveLimits,
    ) -> Result<(), Diagnostic> {
        if link.link_type != LinkType::Hard {
            return Err(invalid_hdf5(
                "HDF5 v1 rejects soft, external, and user-defined links",
            ));
        }
        if name.contains('\u{fffd}') || name.chars().any(char::is_control) {
            return Err(invalid_hdf5(
                "HDF5 link names must be valid bounded text without controls",
            ));
        }
        require_at_most(name.len(), limits.max_name_bytes, "HDF5 link-name bytes")?;
        self.total_name_bytes = checked_add(
            self.total_name_bytes,
            name.len(),
            "aggregate HDF5 link-name bytes",
        )?;
        require_at_most(
            self.total_name_bytes,
            limits.max_total_name_bytes,
            "aggregate HDF5 link-name bytes",
        )?;
        self.links = checked_add(self.links, 1, "HDF5 link count")?;
        require_at_most(self.links, limits.max_links, "HDF5 link count")?;
        self.work = checked_add(self.work, name.len().saturating_add(1), "HDF5 audit work")?;
        require_at_most(self.work, limits.max_audit_work, "HDF5 audit work")?;
        self.entries.try_reserve(1).map_err(|error| {
            invalid_hdf5(format!(
                "cannot reserve bounded HDF5 link audit storage: {error}",
            ))
        })?;
        let mut owned_name = String::new();
        owned_name.try_reserve_exact(name.len()).map_err(|error| {
            invalid_hdf5(format!("cannot reserve bounded HDF5 link name: {error}"))
        })?;
        owned_name.push_str(name);
        self.entries.push(owned_name);
        Ok(())
    }
}

fn require_one_hard_link(count: usize) -> Result<(), Diagnostic> {
    if count == 1 {
        Ok(())
    } else {
        Err(invalid_hdf5(
            "HDF5 v1 requires an acyclic hard-link tree without aliases",
        ))
    }
}

fn reject_attributes(count: usize) -> Result<(), Diagnostic> {
    if count == 0 {
        Ok(())
    } else {
        Err(invalid_hdf5(
            "HDF5 v1 rejects object attributes from its closed file profile",
        ))
    }
}

pub(crate) fn checked_product(values: &[usize], what: &str) -> Result<usize, Diagnostic> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| invalid_hdf5(format!("{what} overflows usize")))
    })
}

pub(crate) fn require_at_most(value: usize, limit: usize, what: &str) -> Result<(), Diagnostic> {
    if value <= limit {
        Ok(())
    } else {
        Err(invalid_hdf5(format!(
            "{what} {value} exceeds configured limit {limit}",
        )))
    }
}

fn checked_add(left: usize, right: usize, what: &str) -> Result<usize, Diagnostic> {
    left.checked_add(right)
        .ok_or_else(|| invalid_hdf5(format!("{what} overflows usize")))
}

fn native_error(context: &str, error: hdf5_metno::Error) -> Diagnostic {
    invalid_hdf5(format!("{context}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_iteration_preserves_admission_diagnostics_and_stops() {
        let file = File::with_options()
            .with_fapl(|options| options.core_filebacked(false))
            .create("bounded-link-audit.h5")
            .unwrap();
        for name in ["aa", "bb", "cc"] {
            file.create_group(name).unwrap();
        }
        let defaults = Hdf5ResolveLimits::default();
        for (limits, expected) in [
            (
                Hdf5ResolveLimits {
                    max_links: 1,
                    ..defaults
                },
                "HDF5 link count 2",
            ),
            (
                Hdf5ResolveLimits {
                    max_name_bytes: 1,
                    ..defaults
                },
                "HDF5 link-name bytes 2",
            ),
            (
                Hdf5ResolveLimits {
                    max_total_name_bytes: 1,
                    ..defaults
                },
                "aggregate HDF5 link-name bytes 2",
            ),
            (
                Hdf5ResolveLimits {
                    max_audit_work: 1,
                    ..defaults
                },
                "HDF5 audit work 4",
            ),
        ] {
            let error = complete_file(&file, limits).unwrap_err();
            assert!(error.message().starts_with(expected), "{error:?}");
        }
        complete_file(&file, defaults).unwrap();
    }
}
