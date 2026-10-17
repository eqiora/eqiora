//! Content identity of one authored continuous geometry.
//!
//! [`eqiora_geometry::CanonicalGeometryV1`] owns the validated content,
//! canonical bytes, and domain-separated identity. This module supplies the
//! artifact admission budget and the repository-wide [`ArtifactDigest`] view.
//!
//! That split follows the one already drawn by
//! [`GeometryIdentityEnvelopeV1`](crate::GeometryIdentityEnvelopeV1), whose
//! entity handle is likewise a geometry-crate type. Geometry decides what a
//! shape is; this crate decides what naming one costs.

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_geometry::{CanonicalGeometryLimits, CanonicalGeometryV1, PlanarRegion};

use crate::{ArtifactDigest, JsonLimits};

/// Admission budgets for one authored planar or convex-polyhedral artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeometryDefinitionDecoderLimits {
    /// Common JSON byte and nesting admission.
    pub json: JsonLimits,
    /// Geometry-specific byte, entity, and membership budgets.
    pub geometry: CanonicalGeometryLimits,
}

impl Default for GeometryDefinitionDecoderLimits {
    fn default() -> Self {
        let geometry = CanonicalGeometryLimits::default();
        Self {
            json: JsonLimits {
                max_bytes: geometry.max_bytes,
                ..JsonLimits::default()
            },
            geometry,
        }
    }
}

/// A content-addressed authored geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct GeometryDefinitionV1 {
    inner: CanonicalGeometryV1,
}

impl GeometryDefinitionV1 {
    /// Wrap supported canonical Geometry with the ordinary artifact budgets.
    /// Currently admits straight-planar definitions and convex-polyhedral shells;
    /// circular and other source families return an unsupported-family error.
    /// # Errors
    /// Returns `EQ0901` for an unsupported family or exceeded admission budget.
    pub fn from_canonical(geometry: &CanonicalGeometryV1) -> Result<Self, Diagnostic> {
        Self::from_json(
            geometry.canonical_bytes(),
            GeometryDefinitionDecoderLimits::default(),
        )
    }

    /// Wrap one already canonical region as an addressable artifact.
    ///
    /// The region cannot be invalid, because [`PlanarRegion`] admits nothing
    /// invalid, so this cannot fail.
    #[must_use]
    pub fn from_region(region: &PlanarRegion) -> Self {
        // PlanarRegion has already rejected non-finite values, the only
        // content for which serde_json's finite-number serializer can fail.
        Self {
            inner: CanonicalGeometryV1::from_region(region)
                .expect("a validated planar region always has canonical JSON"),
        }
    }

    /// Decode externally supplied canonical geometry JSON under explicit
    /// syntax and geometry-work budgets.
    ///
    /// The decoded wire is revalidated by its straight-planar or convex-polyhedral
    /// Geometry owner and re-encoded. Other source families remain unsupported.
    /// The input is admitted only when those reconstructed bytes equal it
    /// exactly, so one geometry cannot acquire a second artifact identity.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, unknown, oversized, invalid, or
    /// noncanonical data.
    pub fn from_json(
        bytes: &[u8],
        limits: GeometryDefinitionDecoderLimits,
    ) -> Result<Self, Diagnostic> {
        limits.json.check(bytes)?;
        let metadata: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
            Diagnostic::error(
                codes::INVALID_ARTIFACT,
                format!("invalid Geometry definition JSON: {error}"),
            )
        })?;
        let inner = if metadata.get("schema").and_then(serde_json::Value::as_str)
            == Some("eqiora.convex-polyhedra-envelope/v1")
        {
            CanonicalGeometryV1::replay_canonical(bytes, limits.geometry)?
        } else {
            CanonicalGeometryV1::decode_canonical(bytes, limits.geometry)?
        };
        Ok(Self { inner })
    }

    /// Replay the validated region this artifact encodes.
    ///
    /// # Errors
    /// Returns `EQ0901` when the artifact is not a straight-planar region.
    pub fn region(&self) -> Result<PlanarRegion, Diagnostic> {
        self.inner.region().cloned().ok_or_else(|| {
            Diagnostic::error(
                codes::INVALID_ARTIFACT,
                "geometry definition artifact requires straight-edged planar geometry",
            )
        })
    }

    /// Canonical encoding of this geometry.
    ///
    /// # Errors
    /// This preserved signature cannot fail for a constructed artifact.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        Ok(self.inner.canonical_bytes().to_vec())
    }

    /// Domain-separated content identity of this exact geometry.
    ///
    /// # Errors
    /// This preserved signature cannot fail for a constructed artifact.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::from_sha256(self.inner.digest_bytes()))
    }

    /// Lower-layer canonical content consumed by later admission.
    #[must_use]
    pub const fn canonical(&self) -> &CanonicalGeometryV1 {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use eqiora_geometry::{EDGE_DIMENSION, FACE_DIMENSION, NamedEntitySet, PlanarFace};

    use super::*;

    fn square_with_hole() -> PlanarRegion {
        PlanarRegion::new(
            vec![
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.25, 0.25],
                [0.75, 0.25],
                [0.75, 0.75],
                [0.25, 0.75],
            ],
            vec![PlanarFace::new(vec![0, 1, 2, 3], vec![vec![4, 5, 6, 7]])],
            vec![
                NamedEntitySet::new("exterior", EDGE_DIMENSION, vec![0, 1, 2, 3]),
                NamedEntitySet::new("hole", EDGE_DIMENSION, vec![4, 5, 6, 7]),
                NamedEntitySet::new("fluid", FACE_DIMENSION, vec![0]),
            ],
            1.0e-9,
        )
        .expect("a square with a square hole is a region")
    }

    fn filled_square() -> PlanarRegion {
        PlanarRegion::new(
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            vec![PlanarFace::new(vec![0, 1, 2, 3], Vec::new())],
            vec![NamedEntitySet::new(
                "exterior",
                EDGE_DIMENSION,
                vec![0, 1, 2, 3],
            )],
            1.0e-9,
        )
        .expect("a square is a region")
    }

    #[test]
    fn one_region_has_one_digest() {
        assert_eq!(
            GeometryDefinitionV1::from_region(&square_with_hole())
                .digest()
                .unwrap(),
            GeometryDefinitionV1::from_region(&square_with_hole())
                .digest()
                .unwrap()
        );
    }

    #[test]
    fn filling_the_hole_changes_the_digest() {
        // The hole is part of what the geometry is, not decoration on it.
        assert_ne!(
            GeometryDefinitionV1::from_region(&square_with_hole())
                .digest()
                .unwrap(),
            GeometryDefinitionV1::from_region(&filled_square())
                .digest()
                .unwrap()
        );
    }

    #[test]
    fn an_artifact_replays_to_the_region_it_encodes() {
        let region = square_with_hole();
        let replayed = GeometryDefinitionV1::from_region(&region).region().unwrap();
        assert_eq!(replayed, region);
    }

    #[test]
    fn lower_identity_is_the_artifact_identity() {
        let artifact = GeometryDefinitionV1::from_region(&square_with_hole());
        assert_eq!(
            artifact.digest().unwrap().sha256_bytes(),
            artifact.canonical().digest_bytes()
        );
    }
}
