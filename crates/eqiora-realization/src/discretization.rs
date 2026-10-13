use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_core::{Diagnostic, DimExponents};
use serde::{Deserialize, Serialize};

use crate::invalid_realization;

/// Discrete function-space family and approximation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Space {
    family: SpaceFamily,
}

impl Space {
    /// Continuous nodal Lagrange space of a strictly positive order.
    #[must_use]
    pub const fn continuous_lagrange(order: NonZeroU16) -> Self {
        Self {
            family: SpaceFamily::ContinuousLagrange { order },
        }
    }

    /// Hierarchical simplex P1 basis enriched by one cell-interior bubble.
    #[must_use]
    pub const fn simplex_p1_bubble() -> Self {
        Self {
            family: SpaceFamily::SimplexP1Bubble,
        }
    }

    /// Minimum per-axis Duffy rule for basis products in an admitted affine simplex profile.
    /// Variable coefficients may require a higher integration order. Unsupported
    /// Space/dimension pairs return `None`; this does not select an execution method.
    #[must_use]
    pub fn minimum_simplex_duffy_points(self, dimension: NonZeroUsize) -> Option<NonZeroUsize> {
        let d = dimension.get();
        let points = match (self.family, d) {
            (SpaceFamily::ContinuousLagrange { order }, 2 | 3) if order == NonZeroU16::MIN => {
                (d + 2).div_ceil(2)
            }
            // Bubble degree d+1 gives squared degree 2(d+1); the Duffy
            // Jacobian adds d-1 to the largest transformed axis degree.
            (SpaceFamily::SimplexP1Bubble, 2 | 3) => (3 * d + 2).div_ceil(2),
            (SpaceFamily::TetrahedralEdge | SpaceFamily::TetrahedralFace, 3) => 3,
            _ => return None,
        };
        NonZeroUsize::new(points)
    }

    /// One cell-local constant degree of freedom.
    #[must_use]
    pub const fn cell_constant() -> Self {
        Self {
            family: SpaceFamily::CellConstant,
        }
    }

    /// Lowest-order tetrahedral edge element with oriented line-integral coefficients.
    #[must_use]
    pub const fn tetrahedral_edge() -> Self {
        Self {
            family: SpaceFamily::TetrahedralEdge,
        }
    }

    /// Lowest-order tetrahedral face element with oriented flux-integral coefficients.
    #[must_use]
    pub const fn tetrahedral_face() -> Self {
        Self {
            family: SpaceFamily::TetrahedralFace,
        }
    }

    /// Declared family.
    #[must_use]
    pub const fn family(self) -> SpaceFamily {
        self.family
    }

    /// Physical dimension of a coefficient functional applied to a Field.
    /// Scalar families retain Field units; oriented edge integrals multiply
    /// them by length, and oriented face flux integrals by area.
    /// Returns `None` if the resulting exact exponents exceed their bounds.
    #[must_use]
    pub fn coefficient_dimension(self, field_dimension: DimExponents) -> Option<DimExponents> {
        let power = match self.family {
            SpaceFamily::TetrahedralEdge => 1,
            SpaceFamily::TetrahedralFace => 2,
            _ => 0,
        };
        field_dimension.mul(DimExponents::from_integers([0, power, 0, 0, 0, 0, 0])?)
    }
}

/// Inspectable space family; it carries no field meaning or physical unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SpaceFamily {
    /// Globally continuous nodal Lagrange basis.
    ContinuousLagrange {
        /// Polynomial order.
        order: NonZeroU16,
    },
    /// Hierarchical simplex P1 basis plus one normalized cell bubble.
    SimplexP1Bubble,
    /// Cell-local piecewise constant basis.
    CellConstant,
    /// Lowest-order Nedelec first-kind tetrahedron. Each coefficient is the
    /// integral of the tangential Field along an oriented edge, not an average.
    /// Values use the covariant Piola map; coefficients carry Field units times length.
    TetrahedralEdge,
    /// Lowest-order Raviart–Thomas tetrahedron. Each coefficient is oriented
    /// normal flux through a face, not a point value or normalized average.
    /// Values use the contravariant Piola map; coefficients carry Field units times area.
    TetrahedralFace,
}

/// Spatial numerical method family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiscretizationMethod {
    /// Continuous Galerkin finite elements.
    ContinuousGalerkin,
    /// Conservative cell-centered finite volumes.
    CellCenteredFiniteVolume,
}

/// Topology/geometry family admitted by a complete realization path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MeshKind {
    /// Cartesian topology generated from canonical box bounds.
    GeneratedCartesian,
    /// Caller-supplied, content-addressed Cartesian topology.
    SuppliedCartesian,
    /// Content-addressed, fixed-connectivity affine simplex topology.
    ImportedAffineSimplicial,
}

/// Content identity of one independently versioned mesh artifact.
///
/// The Realization layer carries identity, not serialized coordinates or a
/// filesystem location. Artifact adapters reconstruct and validate the mesh
/// bytes before numerical lowering receives a typed mesh revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshArtifactReference([u8; 32]);

impl MeshArtifactReference {
    /// Construct from complete SHA-256 bytes.
    #[must_use]
    pub const fn from_sha256(value: [u8; 32]) -> Self {
        Self(value)
    }

    /// Complete SHA-256 bytes.
    #[must_use]
    pub const fn sha256(self) -> [u8; 32] {
        self.0
    }
}

/// Mesh selection owned by realization rather than by a semantic Domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeshPolicy {
    /// Generate a uniform mesh with this count on every topological axis.
    GeneratedUniform {
        /// Non-zero cells per axis.
        cells_per_axis: NonZeroUsize,
    },
    /// Use one caller-supplied Cartesian mesh artifact with exact axis counts.
    SuppliedCartesian {
        /// Content identity resolved by the artifact/control plane.
        artifact: MeshArtifactReference,
        /// Exact non-zero cell counts on the two Cartesian axes.
        cells: [NonZeroUsize; 2],
    },
    /// Use one caller-supplied one-dimensional Cartesian mesh artifact.
    SuppliedCartesian1d {
        /// Content identity resolved by the artifact/control plane.
        artifact: MeshArtifactReference,
        /// Exact non-zero cell count on the Cartesian axis.
        cells: [NonZeroUsize; 1],
    },
    /// Use one caller-supplied three-dimensional Cartesian mesh artifact.
    SuppliedCartesian3d {
        /// Content identity resolved by the artifact/control plane.
        artifact: MeshArtifactReference,
        /// Exact non-zero cell counts on the three Cartesian axes.
        cells: [NonZeroUsize; 3],
    },
    /// Use one independently versioned affine-simplex mesh artifact.
    ImportedSimplicial {
        /// Content identity resolved by the artifact/control plane.
        artifact: MeshArtifactReference,
    },
}

impl MeshPolicy {
    /// Mesh family required by this policy.
    #[must_use]
    pub const fn kind(self) -> MeshKind {
        match self {
            Self::GeneratedUniform { .. } => MeshKind::GeneratedCartesian,
            Self::SuppliedCartesian { .. }
            | Self::SuppliedCartesian1d { .. }
            | Self::SuppliedCartesian3d { .. } => MeshKind::SuppliedCartesian,
            Self::ImportedSimplicial { .. } => MeshKind::ImportedAffineSimplicial,
        }
    }
}

/// Explicit integration policy, never hidden inside a mesh or physics relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "WireQuadraturePolicy", into = "WireQuadraturePolicy")]
pub enum QuadraturePolicy {
    /// Tensor-product Gauss--Legendre points on each axis.
    GaussLegendre {
        /// Non-zero points on each axis.
        points_per_axis: NonZeroUsize,
    },
    /// Cell centroid rule used by the v0 finite-volume path.
    CellCentroid,
    /// Centroid rule on an affine simplex reference cell.
    SimplexCentroid,
    /// Dimension-explicit simplex rule obtained by a Duffy transform.
    ///
    /// The spatial dimension belongs to the policy so a realized tetrahedral
    /// rule cannot silently drift into a triangle rule, or vice versa.
    SimplexDuffyGaussLegendre {
        /// Non-zero spatial dimension of the reference simplex.
        spatial_dimension: NonZeroUsize,
        /// Non-zero Gauss--Legendre points on each Duffy coordinate.
        points_per_axis: NonZeroUsize,
    },
}

// Empty struct variants are intentional: Serde's internally tagged unit
// variants ignore extra fields even with deny_unknown_fields on the enum.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireQuadraturePolicy {
    GaussLegendre {
        points_per_axis: NonZeroUsize,
    },
    CellCentroid {},
    SimplexCentroid {},
    SimplexDuffyGaussLegendre {
        spatial_dimension: NonZeroUsize,
        points_per_axis: NonZeroUsize,
    },
}

impl From<QuadraturePolicy> for WireQuadraturePolicy {
    fn from(value: QuadraturePolicy) -> Self {
        match value {
            QuadraturePolicy::GaussLegendre { points_per_axis } => {
                Self::GaussLegendre { points_per_axis }
            }
            QuadraturePolicy::CellCentroid => Self::CellCentroid {},
            QuadraturePolicy::SimplexCentroid => Self::SimplexCentroid {},
            QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension,
                points_per_axis,
            } => Self::SimplexDuffyGaussLegendre {
                spatial_dimension,
                points_per_axis,
            },
        }
    }
}

impl From<WireQuadraturePolicy> for QuadraturePolicy {
    fn from(value: WireQuadraturePolicy) -> Self {
        match value {
            WireQuadraturePolicy::GaussLegendre { points_per_axis } => {
                Self::GaussLegendre { points_per_axis }
            }
            WireQuadraturePolicy::CellCentroid {} => Self::CellCentroid,
            WireQuadraturePolicy::SimplexCentroid {} => Self::SimplexCentroid,
            WireQuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension,
                points_per_axis,
            } => Self::SimplexDuffyGaussLegendre {
                spatial_dimension,
                points_per_axis,
            },
        }
    }
}

/// Method, mesh, and integration choices. Space remains a sibling contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Discretization {
    method: DiscretizationMethod,
    mesh: MeshPolicy,
    quadrature: QuadraturePolicy,
}

impl Discretization {
    /// Construct a discretization choice without applying method/space policy.
    #[must_use]
    pub const fn new(
        method: DiscretizationMethod,
        mesh: MeshPolicy,
        quadrature: QuadraturePolicy,
    ) -> Self {
        Self {
            method,
            mesh,
            quadrature,
        }
    }

    /// Numerical method.
    #[must_use]
    pub const fn method(self) -> DiscretizationMethod {
        self.method
    }

    /// Mesh policy.
    #[must_use]
    pub const fn mesh(self) -> MeshPolicy {
        self.mesh
    }

    /// Quadrature policy.
    #[must_use]
    pub const fn quadrature(self) -> QuadraturePolicy {
        self.quadrature
    }

    pub(crate) fn validate_space(self, space: Space) -> Result<(), Diagnostic> {
        self.validate_space_compatibility(space)?;
        if let QuadraturePolicy::SimplexDuffyGaussLegendre {
            spatial_dimension,
            points_per_axis,
        } = self.quadrature
            && !space
                .minimum_simplex_duffy_points(spatial_dimension)
                .is_some_and(|minimum| points_per_axis >= minimum)
        {
            return Err(invalid_realization(
                "simplex Duffy quadrature must integrate the selected Space basis products",
            ));
        }
        Ok(())
    }

    /// Structural compatibility, independent of equation-specific exactness requirements.
    pub(crate) fn validate_space_compatibility(self, space: Space) -> Result<(), Diagnostic> {
        match (self.method, space.family(), self.mesh, self.quadrature) {
            (
                DiscretizationMethod::ContinuousGalerkin,
                SpaceFamily::ContinuousLagrange { .. },
                MeshPolicy::GeneratedUniform { .. }
                | MeshPolicy::SuppliedCartesian { .. }
                | MeshPolicy::SuppliedCartesian1d { .. }
                | MeshPolicy::SuppliedCartesian3d { .. },
                QuadraturePolicy::GaussLegendre { .. },
            )
            | (
                DiscretizationMethod::CellCenteredFiniteVolume,
                SpaceFamily::CellConstant,
                MeshPolicy::GeneratedUniform { .. }
                | MeshPolicy::SuppliedCartesian { .. }
                | MeshPolicy::SuppliedCartesian1d { .. }
                | MeshPolicy::SuppliedCartesian3d { .. },
                QuadraturePolicy::CellCentroid | QuadraturePolicy::GaussLegendre { .. },
            ) => Ok(()),
            (
                DiscretizationMethod::ContinuousGalerkin,
                SpaceFamily::ContinuousLagrange { order },
                MeshPolicy::ImportedSimplicial { .. },
                QuadraturePolicy::SimplexCentroid,
            ) if order == NonZeroU16::MIN => Ok(()),
            (
                DiscretizationMethod::ContinuousGalerkin,
                _,
                MeshPolicy::ImportedSimplicial { .. },
                QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension, ..
                },
            ) if match space.family() {
                SpaceFamily::ContinuousLagrange { .. } | SpaceFamily::SimplexP1Bubble => true,
                SpaceFamily::TetrahedralEdge | SpaceFamily::TetrahedralFace => {
                    spatial_dimension.get() == 3
                }
                SpaceFamily::CellConstant => false,
            } =>
            {
                Ok(())
            }
            (DiscretizationMethod::ContinuousGalerkin, _, _, _) => Err(invalid_realization(
                "continuous Galerkin requires Cartesian/Gauss-Legendre, imported affine-simplex/P1-centroid, or an admitted simplex Space with matching dimension and Duffy quadrature",
            )),
            (DiscretizationMethod::CellCenteredFiniteVolume, _, _, _) => Err(invalid_realization(
                "cell-centered finite volume requires a generated or supplied Cartesian mesh, cell-constant space, and centroid or Gauss-Legendre quadrature",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadrature_wire_has_one_closed_dimension_explicit_owner() {
        for (policy, expected) in [
            (
                QuadraturePolicy::CellCentroid,
                r#"{"kind":"cell-centroid"}"#,
            ),
            (
                QuadraturePolicy::SimplexCentroid,
                r#"{"kind":"simplex-centroid"}"#,
            ),
            (
                QuadraturePolicy::GaussLegendre {
                    points_per_axis: NonZeroUsize::new(2).unwrap(),
                },
                r#"{"kind":"gauss-legendre","points_per_axis":2}"#,
            ),
            (
                QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension: NonZeroUsize::new(2).unwrap(),
                    points_per_axis: NonZeroUsize::new(5).unwrap(),
                },
                r#"{"kind":"simplex-duffy-gauss-legendre","spatial_dimension":2,"points_per_axis":5}"#,
            ),
            (
                QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension: NonZeroUsize::new(3).unwrap(),
                    points_per_axis: NonZeroUsize::new(7).unwrap(),
                },
                r#"{"kind":"simplex-duffy-gauss-legendre","spatial_dimension":3,"points_per_axis":7}"#,
            ),
        ] {
            assert_eq!(serde_json::to_string(&policy).unwrap(), expected);
            assert_eq!(
                serde_json::from_str::<QuadraturePolicy>(expected).unwrap(),
                policy
            );
        }
    }

    #[test]
    fn quadrature_wire_rejects_retired_unknown_and_invalid_counts() {
        for wire in [
            r#"{"kind":"triangle-duffy-gauss-legendre","points_per_axis":5}"#,
            r#"{"kind":"cell-centroid","unknown":1}"#,
            r#"{"kind":"simplex-centroid","unknown":1}"#,
            r#"{"kind":"unknown"}"#,
            r#"{"kind":"gauss-legendre","points_per_axis":0}"#,
            r#"{"kind":"gauss-legendre","points_per_axis":-1}"#,
            r#"{"kind":"gauss-legendre","points_per_axis":1.5}"#,
            r#"{"kind":"gauss-legendre","points_per_axis":18446744073709551616}"#,
            r#"{"kind":"gauss-legendre","points_per_axis":2,"points_per_axis":3}"#,
            r#"{"kind":"simplex-duffy-gauss-legendre","points_per_axis":5}"#,
            r#"{"kind":"simplex-duffy-gauss-legendre","spatial_dimension":0,"points_per_axis":5}"#,
            r#"{"kind":"simplex-duffy-gauss-legendre","spatial_dimension":2,"points_per_axis":0}"#,
            r#"{"kind":"simplex-duffy-gauss-legendre","spatial_dimension":2,"points_per_axis":5,"unknown":1}"#,
        ] {
            assert!(
                serde_json::from_str::<QuadraturePolicy>(wire).is_err(),
                "accepted {wire}"
            );
        }
    }
}
