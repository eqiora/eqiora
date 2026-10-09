//! Authored continuum regularity on one exact spatial support.

/// A Field's continuum regularity, independent of its numerical representation.
///
/// Every assertion is local to the Field's own support. In particular, smooth
/// Fields on two adjacent volumes need not have matching one-sided traces.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpatialRegularity {
    /// No trace theorem has been asserted for this Field.
    #[default]
    Unspecified,
    /// Square-integrable components; no boundary trace is implied.
    L2,
    /// Square-integrable components and all first weak derivatives.
    H1,
    /// Square-integrable Cartesian vector and weak curl; tangential trace only.
    HCurl,
    /// Square-integrable Cartesian tensor and row-wise weak divergence.
    /// The normal trace contracts its last spatial axis.
    HDiv,
    /// Smooth up to the boundary of this exact support.
    Smooth,
}
