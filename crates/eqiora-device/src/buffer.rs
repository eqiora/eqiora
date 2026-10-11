use std::fmt::Debug;
use std::marker::PhantomData;
use std::num::{NonZeroU64, NonZeroUsize};

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, ScalarType};

use crate::DeviceId;

/// Stored element representation for device data and sparse structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceElementType {
    /// A numerical scalar participating in arithmetic.
    Scalar(ScalarType),
    /// A signed 64-bit sparse/topological index.
    SignedIndex64,
}

impl DeviceElementType {
    /// Size of one stored element in bytes.
    #[must_use]
    pub const fn byte_width(self) -> usize {
        match self {
            Self::Scalar(scalar) => scalar.byte_width(),
            Self::SignedIndex64 => size_of::<i64>(),
        }
    }
}

/// Rust element types admitted at the device-buffer boundary.
pub trait DeviceElement: Copy + Debug + Send + Sync + 'static {
    /// Corresponding Eqiora storage representation.
    const ELEMENT_TYPE: DeviceElementType;
}

impl DeviceElement for f32 {
    const ELEMENT_TYPE: DeviceElementType = DeviceElementType::Scalar(ScalarType::F32);
}

impl DeviceElement for f64 {
    const ELEMENT_TYPE: DeviceElementType = DeviceElementType::Scalar(ScalarType::F64);
}

impl DeviceElement for i64 {
    const ELEMENT_TYPE: DeviceElementType = DeviceElementType::SignedIndex64;
}

/// Runtime-local allocation identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferId {
    device: DeviceId,
    allocation: NonZeroU64,
}

impl BufferId {
    /// Construct an adapter-assigned allocation identity.
    #[must_use]
    pub const fn new(device: DeviceId, allocation: NonZeroU64) -> Self {
        Self { device, allocation }
    }

    /// Owning device.
    #[must_use]
    pub const fn device(self) -> DeviceId {
        self.device
    }

    /// Monotone allocation identity within one adapter process.
    #[must_use]
    pub const fn allocation(self) -> NonZeroU64 {
        self.allocation
    }
}

/// Shape and residency of one typed device allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBufferDescriptor<T: DeviceElement> {
    id: BufferId,
    elements: NonZeroUsize,
    element: PhantomData<T>,
}

impl<T: DeviceElement> DeviceBufferDescriptor<T> {
    /// Construct a nonempty typed device-buffer descriptor.
    #[must_use]
    pub const fn new(id: BufferId, elements: NonZeroUsize) -> Self {
        Self {
            id,
            elements,
            element: PhantomData,
        }
    }

    /// Allocation identity and residency.
    #[must_use]
    pub const fn id(self) -> BufferId {
        self.id
    }

    /// Number of stored elements.
    #[must_use]
    pub const fn elements(self) -> NonZeroUsize {
        self.elements
    }

    /// Storage representation implied by `T`.
    #[must_use]
    pub const fn element_type(self) -> DeviceElementType {
        T::ELEMENT_TYPE
    }
}

/// Typed ownership seam implemented by a concrete adapter allocation.
pub trait DeviceBuffer<T: DeviceElement>: Debug + Send + Sync {
    /// Eqiora-owned descriptor; vendor pointers never cross this seam.
    fn descriptor(&self) -> DeviceBufferDescriptor<T>;
}

/// Shape of one caller-owned host region participating in a transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostBufferDescriptor<T: DeviceElement> {
    elements: NonZeroUsize,
    element: PhantomData<T>,
}

impl<T: DeviceElement> HostBufferDescriptor<T> {
    /// Construct a nonempty host-region descriptor.
    #[must_use]
    pub const fn new(elements: NonZeroUsize) -> Self {
        Self {
            elements,
            element: PhantomData,
        }
    }

    /// Number of stored elements.
    #[must_use]
    pub const fn elements(self) -> NonZeroUsize {
        self.elements
    }
}

/// Explicit source or destination memory space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegion<T: DeviceElement> {
    /// Caller-owned host memory.
    Host(HostBufferDescriptor<T>),
    /// Adapter-owned allocation on one device.
    Device(DeviceBufferDescriptor<T>),
}

impl<T: DeviceElement> MemoryRegion<T> {
    fn elements(self) -> NonZeroUsize {
        match self {
            Self::Host(region) => region.elements(),
            Self::Device(region) => region.elements(),
        }
    }

    /// Resident device, if this endpoint is device memory.
    #[must_use]
    pub const fn device(self) -> Option<DeviceId> {
        match self {
            Self::Host(_) => None,
            Self::Device(region) => Some(region.id().device()),
        }
    }
}

/// Direction implied by two explicit memory regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Host to device.
    HostToDevice,
    /// Device to host.
    DeviceToHost,
    /// Device to device; peer admission remains an adapter capability.
    DeviceToDevice,
}

/// One validated movement of an equal typed region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferPlan<T: DeviceElement> {
    source: MemoryRegion<T>,
    destination: MemoryRegion<T>,
    direction: TransferDirection,
}

impl<T: DeviceElement> TransferPlan<T> {
    /// Construct an explicit non-host-only transfer.
    ///
    /// # Errors
    /// Returns `EQ0807` for different element counts, a host-to-host copy, or
    /// the same allocation as both endpoints.
    pub fn new(source: MemoryRegion<T>, destination: MemoryRegion<T>) -> Result<Self, Diagnostic> {
        if source.elements() != destination.elements() {
            return Err(invalid_transfer(
                "transfer endpoints must contain the same number of elements",
            ));
        }
        let direction = match (source, destination) {
            (MemoryRegion::Host(_), MemoryRegion::Host(_)) => {
                return Err(invalid_transfer(
                    "host-to-host copies are outside the device transfer contract",
                ));
            }
            (MemoryRegion::Host(_), MemoryRegion::Device(_)) => TransferDirection::HostToDevice,
            (MemoryRegion::Device(_), MemoryRegion::Host(_)) => TransferDirection::DeviceToHost,
            (MemoryRegion::Device(source), MemoryRegion::Device(destination)) => {
                if source.id() == destination.id() {
                    return Err(invalid_transfer(
                        "a device transfer requires distinct allocations",
                    ));
                }
                TransferDirection::DeviceToDevice
            }
        };
        Ok(Self {
            source,
            destination,
            direction,
        })
    }

    /// Transfer source.
    #[must_use]
    pub const fn source(self) -> MemoryRegion<T> {
        self.source
    }

    /// Transfer destination.
    #[must_use]
    pub const fn destination(self) -> MemoryRegion<T> {
        self.destination
    }

    /// Explicit movement direction.
    #[must_use]
    pub const fn direction(self) -> TransferDirection {
        self.direction
    }

    /// Number of elements moved.
    #[must_use]
    pub fn elements(self) -> NonZeroUsize {
        self.source.elements()
    }

    /// Number of bytes moved.
    ///
    /// # Errors
    /// Returns `EQ0807` if the byte count overflows `usize`.
    pub fn bytes(self) -> Result<usize, Diagnostic> {
        self.elements()
            .get()
            .checked_mul(T::ELEMENT_TYPE.byte_width())
            .ok_or_else(|| invalid_transfer("transfer byte count overflowed usize"))
    }
}

fn invalid_transfer(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}
