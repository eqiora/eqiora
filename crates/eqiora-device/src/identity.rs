/// Stable Eqiora-owned identity for a device runtime adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeId(&'static str);

impl RuntimeId {
    /// Construct a namespaced compile-time runtime identity.
    #[must_use]
    pub const fn new(value: &'static str) -> Self {
        Self(value)
    }

    /// Namespaced identity.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

/// One device selected within a concrete runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceId {
    runtime: RuntimeId,
    ordinal: u16,
}

impl DeviceId {
    /// Construct a runtime-scoped device identity.
    #[must_use]
    pub const fn new(runtime: RuntimeId, ordinal: u16) -> Self {
        Self { runtime, ordinal }
    }

    /// Runtime that owns the device.
    #[must_use]
    pub const fn runtime(self) -> RuntimeId {
        self.runtime
    }

    /// Stable ordinal resolved in the deployment environment.
    #[must_use]
    pub const fn ordinal(self) -> u16 {
        self.ordinal
    }
}
