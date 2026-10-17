//! Exact binary-rational sign confirmation for authored polyhedral topology.
//!
//! Binary64 coordinates have bounded exponents and significands. No tolerance
//! changes a determinant's sign or makes a nonplanar facet planar.

use super::{edges, invalid};
use eqiora_core::Diagnostic;
use num_rational::BigRational;

type Point = [BigRational; 3];
const MAX_EXACT_PREDICATES: usize = 262_144;
pub(super) struct Predicates {
    points: Vec<Point>,
    remaining: usize,
}
impl Predicates {
    pub(super) fn new(vertices: &[[f64; 3]]) -> Self {
        Self {
            points: vertices
                .iter()
                .map(|point| {
                    point.map(|x| BigRational::from_float(x).expect("finite admitted coordinate"))
                })
                .collect(),
            remaining: MAX_EXACT_PREDICATES,
        }
    }
    fn charge(&mut self, n: usize) -> Result<(), Diagnostic> {
        self.remaining = self
            .remaining
            .checked_sub(n)
            .ok_or_else(|| invalid("polyhedral exact predicate budget exceeded"))?;
        Ok(())
    }
    fn axis(&self, ends: [usize; 4]) -> Point {
        cross(
            &sub(&self.points[ends[1]], &self.points[ends[0]]),
            &sub(&self.points[ends[3]], &self.points[ends[2]]),
        )
    }
    pub(super) fn facet(&mut self, polygon: &[usize]) -> Result<(), Diagnostic> {
        self.charge(polygon.len().saturating_mul(polygon.len() + 1))?;
        let axis = self.axis([polygon[0], polygon[1], polygon[0], polygon[2]]);
        let zero = BigRational::default();
        if axis.iter().all(|x| x == &zero) {
            return Err(invalid("polyhedral facet is exactly degenerate"));
        }
        let origin = &self.points[polygon[0]];
        for &id in polygon {
            if dot(&axis, &sub(&self.points[id], origin)) != zero {
                return Err(invalid("polyhedral facet is not exactly planar"));
            }
        }
        for (a, b) in edges(polygon) {
            let inward = cross(&axis, &sub(&self.points[b], &self.points[a]));
            for &id in polygon {
                if id != a
                    && id != b
                    && dot(&inward, &sub(&self.points[id], &self.points[a])) <= zero
                {
                    return Err(invalid(
                        "polyhedral facet is exactly degenerate, crossed, or nonconvex",
                    ));
                }
            }
        }
        Ok(())
    }
    pub(super) fn contains_point(
        &mut self,
        shell: &[Vec<usize>],
        point: [f64; 3],
    ) -> Result<bool, Diagnostic> {
        self.charge(shell.len())?;
        let point = point.map(|x| BigRational::from_float(x).expect("finite admitted point"));
        for polygon in shell {
            let normal = self.axis([polygon[0], polygon[1], polygon[0], polygon[2]]);
            if dot(&normal, &sub(&point, &self.points[polygon[0]])) > BigRational::default() {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub(super) fn convex_side(
        &mut self,
        polygon: &[usize],
        vertices: &[usize],
    ) -> Result<(), Diagnostic> {
        self.charge(vertices.len())?;
        let normal = self.axis([polygon[0], polygon[1], polygon[0], polygon[2]]);
        let origin = &self.points[polygon[0]];
        let zero = BigRational::default();
        let mut interior = false;
        for &id in vertices {
            let sign = dot(&normal, &sub(&self.points[id], origin));
            if sign > zero {
                return Err(invalid(
                    "polyhedral shell is nonconvex or has reversed parent-outward orientation",
                ));
            }
            interior |= sign < zero;
        }
        if !interior {
            return Err(invalid("polyhedral shell is exactly degenerate"));
        }
        Ok(())
    }
    pub(super) fn separated(
        &mut self,
        a: &[usize],
        b: &[usize],
        axis_ends: [usize; 4],
    ) -> Result<bool, Diagnostic> {
        self.charge(a.len() + b.len())?;
        let axis = self.axis(axis_ends);
        let zero = BigRational::default();
        if axis.iter().all(|x| x == &zero) {
            return Ok(false);
        }
        let project = |ids: &[usize]| {
            let mut values = ids.iter().map(|&id| dot(&axis, &self.points[id]));
            let first = values.next().expect("nonempty shell");
            values.fold((first.clone(), first), |(low, high), x| {
                (low.min(x.clone()), high.max(x))
            })
        };
        let (al, ah) = project(a);
        let (bl, bh) = project(b);
        Ok(ah <= bl || bh <= al)
    }
}
fn sub(a: &Point, b: &Point) -> Point {
    std::array::from_fn(|i| &a[i] - &b[i])
}
fn dot(a: &Point, b: &Point) -> BigRational {
    &a[0] * &b[0] + &a[1] * &b[1] + &a[2] * &b[2]
}
fn cross(a: &Point, b: &Point) -> Point {
    [
        &a[1] * &b[2] - &a[2] * &b[1],
        &a[2] * &b[0] - &a[0] * &b[2],
        &a[0] * &b[1] - &a[1] * &b[0],
    ]
}
