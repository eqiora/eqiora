//! Persist the actual rule; numerical admission remains with QuadratureRule.
use super::*;
use eqiora_meshing::{QuadraturePoint, QuadratureRule, ReferenceCell, ReferenceCellFamily};

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Family {
    Point,
    Simplex,
    Hypercube,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Point {
    coordinates: Vec<f64>,
    weight: f64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Rule {
    family: Family,
    dimension: usize,
    polynomial_exactness: Option<usize>,
    points: Vec<Point>,
}

impl Rule {
    pub(super) fn encode(rule: &QuadratureRule) -> Self {
        Self {
            family: match rule.reference_cell().family() {
                ReferenceCellFamily::Point => Family::Point,
                ReferenceCellFamily::Simplex => Family::Simplex,
                ReferenceCellFamily::Hypercube => Family::Hypercube,
            },
            dimension: rule.reference_cell().dimension(),
            polynomial_exactness: rule.polynomial_exactness(),
            points: rule
                .points()
                .iter()
                .map(|point| Point {
                    coordinates: point.coordinates.clone(),
                    weight: point.weight,
                })
                .collect(),
        }
    }

    pub(super) fn decode(self) -> Result<QuadratureRule, Diagnostic> {
        let cell = match self.family {
            Family::Point if self.dimension == 0 => ReferenceCell::point(),
            Family::Point => return Err(invalid("point quadrature requires dimension zero")),
            Family::Simplex => ReferenceCell::simplex(self.dimension)?,
            Family::Hypercube => ReferenceCell::hypercube(self.dimension)?,
        };
        QuadratureRule::new(
            cell,
            self.polynomial_exactness,
            self.points
                .into_iter()
                .map(|point| QuadraturePoint {
                    coordinates: point.coordinates,
                    weight: point.weight,
                })
                .collect(),
        )
    }
}
