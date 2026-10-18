//! Symbolic coefficient comparisons; the original Data still owns evaluation and domains.
use super::*;
use num_complex::Complex;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::collections::{HashMap, HashSet};

type Scale = Complex<BigRational>;
type Sum<'a, S> = Vec<Product<'a, S>>;

#[derive(Clone)]
struct Product<'a, S: Coefficient> {
    scale: Scale,
    factors: Vec<(Factor<'a, S>, usize)>,
}

#[derive(Clone)]
enum Factor<'a, S: Coefficient> {
    Atom(&'a Data<S>),
    Sum(Arc<Sum<'a, S>>),
}

struct Normalizer<'a, S: Coefficient> {
    budget: u64,
    real_scales: bool,
    cache: HashMap<usize, Sum<'a, S>>,
}

impl<S: Coefficient> Data<S> {
    /// Exact algebraic comparison without sampling Parameters or erasing demanded inputs.
    pub(in crate::form_compiler) fn same_coefficient(&self, other: &Self) -> bool {
        if self == other {
            return true;
        }
        match (
            Normalizer::new(self, false).sum(self),
            Normalizer::new(other, false).sum(other),
        ) {
            (Some(a), Some(b)) => same_sum(&a, &b),
            _ => false,
        }
    }

    /// Orientation of explicit real scales, with bound inputs kept opaque.
    /// This is a row convention, not a positivity proof about Parameter values.
    pub(in crate::form_compiler) fn coefficient_orientation(&self) -> Option<i8> {
        // Complex phases are factors, not ordered scalars. A real row sign
        // applies equally to both retained coordinates of a complex Parameter.
        let sum = Normalizer::new(self, true).sum(self)?;
        let mut orientation = 0;
        for term in sum {
            if !term.scale.im.is_zero() {
                return None;
            }
            let sign = if term.scale.re.is_negative() { -1 } else { 1 };
            if orientation != 0 && orientation != sign {
                return None;
            }
            orientation = sign;
        }
        Some(orientation)
    }
}

impl<'a, S: Coefficient> Normalizer<'a, S> {
    fn new(root: &'a Data<S>, real_scales: bool) -> Self {
        let mut nodes = HashSet::new();
        let mut pending = vec![root];
        while let Some(data) = pending.pop() {
            if !nodes.insert(Arc::as_ptr(&data.0) as usize) {
                continue;
            }
            if let Node::Add(a, b) | Node::Mul(a, b) | Node::Div(a, b) = data.0.as_ref() {
                pending.extend([a, b]);
            }
        }
        // A binary64 literal needs at most 1076 numerator/denominator bits.
        // As in scalar IR additive projection, exact proof work grows with the
        // input DAG; repeated squaring must not allocate exponential integers.
        Self {
            budget: (nodes.len() as u64).saturating_mul(1076),
            real_scales,
            cache: HashMap::new(),
        }
    }

    fn sum(&mut self, data: &'a Data<S>) -> Option<Sum<'a, S>> {
        let key = Arc::as_ptr(&data.0) as usize;
        if let Some(sum) = self.cache.get(&key) {
            return Some(sum.clone());
        }
        let mut sum = match data.0.as_ref() {
            Node::Constant { value, .. } if self.real_scales && value.im() != 0.0 => atom(data),
            Node::Constant { value, .. } => vec![Product {
                scale: Scale::new(
                    BigRational::from_float(value.re())?,
                    BigRational::from_float(value.im())?,
                ),
                factors: Vec::new(),
            }],
            Node::Add(a, b) => {
                let mut sum = self.sum(a)?;
                for term in self.sum(b)? {
                    if let Some(existing) = sum
                        .iter_mut()
                        .find(|existing| same_factors(&existing.factors, &term.factors))
                    {
                        existing.scale += term.scale;
                    } else {
                        sum.push(term);
                    }
                }
                sum
            }
            Node::Mul(a, b) => {
                let left = self.sum(a)?;
                let right = self.sum(b)?;
                if left.is_empty() || right.is_empty() {
                    Vec::new()
                } else if let Some(scale) = constant(&left) {
                    scaled(right, scale)
                } else if let Some(scale) = constant(&right) {
                    scaled(left, scale)
                } else {
                    // Keep products of sums factored; do not expand a polynomial.
                    let mut left = product(left);
                    let right = product(right);
                    left.scale *= right.scale;
                    for (factor, count) in right.factors {
                        if let Some((_, existing)) = left
                            .factors
                            .iter_mut()
                            .find(|(other, _)| same_factor(other, &factor))
                        {
                            *existing = existing.checked_add(count)?;
                        } else {
                            left.factors.push((factor, count));
                        }
                    }
                    vec![left]
                }
            }
            Node::Div(a, b) => {
                let denominator = self.sum(b)?;
                if let Some(scale) = constant(&denominator).filter(|scale| !scale.is_zero()) {
                    scaled(self.sum(a)?, Scale::one() / scale)
                } else {
                    atom(data)
                }
            }
            _ => atom(data),
        };
        sum.retain(|term| !term.scale.is_zero());
        if sum.iter().any(|term| {
            [&term.scale.re, &term.scale.im]
                .into_iter()
                .any(|value| value.numer().bits() + value.denom().bits() > self.budget)
        }) {
            return None;
        }
        self.cache.insert(key, sum.clone());
        Some(sum)
    }
}

fn atom<S: Coefficient>(data: &Data<S>) -> Sum<'_, S> {
    vec![Product {
        scale: Scale::one(),
        factors: vec![(Factor::Atom(data), 1)],
    }]
}

fn constant<S: Coefficient>(sum: &Sum<'_, S>) -> Option<Scale> {
    match sum.as_slice() {
        [] => Some(Scale::zero()),
        [term] if term.factors.is_empty() => Some(term.scale.clone()),
        _ => None,
    }
}

fn scaled<S: Coefficient>(mut sum: Sum<'_, S>, scale: Scale) -> Sum<'_, S> {
    for term in &mut sum {
        term.scale *= &scale;
    }
    sum
}

fn product<S: Coefficient>(mut sum: Sum<'_, S>) -> Product<'_, S> {
    if sum.len() == 1 {
        sum.pop().expect("one term")
    } else {
        // Extract a deterministic real scale without expanding the sum. The
        // minimum nonzero coordinate magnitude commutes with term order and
        // scales uniformly under real multiplication.
        let mut scale = sum
            .iter()
            .flat_map(|term| [&term.scale.re, &term.scale.im])
            .filter(|value| !value.is_zero())
            .map(|value| value.abs())
            .min()
            .expect("nonzero terms");
        if sum
            .iter()
            .all(|term| term.scale.im.is_zero() && term.scale.re.is_negative())
        {
            scale = -scale;
        }
        let scale = Scale::new(scale, BigRational::zero());
        for term in &mut sum {
            term.scale /= &scale;
        }
        Product {
            scale,
            factors: vec![(Factor::Sum(Arc::new(sum)), 1)],
        }
    }
}

fn same_sum<S: Coefficient>(a: &Sum<'_, S>, b: &Sum<'_, S>) -> bool {
    a.len() == b.len()
        && a.iter().all(|a| {
            b.iter()
                .any(|b| a.scale == b.scale && same_factors(&a.factors, &b.factors))
        })
}

fn same_factors<S: Coefficient>(
    a: &[(Factor<'_, S>, usize)],
    b: &[(Factor<'_, S>, usize)],
) -> bool {
    a.len() == b.len()
        && a.iter()
            .all(|(a, n)| b.iter().any(|(b, m)| n == m && same_factor(a, b)))
}

fn same_factor<S: Coefficient>(a: &Factor<'_, S>, b: &Factor<'_, S>) -> bool {
    match (a, b) {
        (Factor::Sum(a), Factor::Sum(b)) => same_sum(a, b),
        (Factor::Atom(a), Factor::Atom(b)) => match (a.0.as_ref(), b.0.as_ref()) {
            (
                Node::Constant {
                    dimension: a,
                    value: c,
                },
                Node::Constant {
                    dimension: b,
                    value: d,
                },
            ) => a == b && c == d,
            (
                Node::Input {
                    dimension: a,
                    coordinate: c,
                    input: e,
                },
                Node::Input {
                    dimension: b,
                    coordinate: d,
                    input: f,
                },
            ) => a == b && e.same_binding(c, f, d),
            (Node::Pointwise(a), Node::Pointwise(b)) => a == b,
            (Node::Affine(a), Node::Affine(b)) => a == b,
            (Node::MapFactor(a), Node::MapFactor(b)) => a == b,
            (Node::Chart(a, c), Node::Chart(b, d)) => c == d && a.same_coefficient(b),
            (Node::Derivative(a, b), Node::Derivative(c, d))
            | (Node::Div(a, b), Node::Div(c, d)) => a.same_coefficient(c) && b.same_coefficient(d),
            (Node::Pow(a, n), Node::Pow(b, m)) => n == m && a.same_coefficient(b),
            (Node::Math(f, a), Node::Math(g, b)) => f == g && a.same_coefficient(b),
            (Node::Cos(a), Node::Cos(b))
            | (Node::SqrtDerivativeRoot(a), Node::SqrtDerivativeRoot(b)) => a.same_coefficient(b),
            _ => false,
        },
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_does_not_round_away_real_or_complex_product_differences() {
        let one = Data::constant(1, 1.0);
        let near_one =
            Data::constant(1, 1.0 + f64::EPSILON).multiply(Data::constant(1, 1.0 - f64::EPSILON));
        assert!(!near_one.same_coefficient(&one));
        let phase = Data::constant(1, Complex::new(1.0, f64::EPSILON));
        let conjugate = Data::constant(1, Complex::new(1.0, -f64::EPSILON));
        assert!(
            !phase
                .multiply(conjugate)
                .same_coefficient(&Data::constant(1, Complex::new(1.0, 0.0)))
        );
    }

    #[test]
    fn cancelled_coefficients_retain_demanded_primal_domains() {
        let undefined = Data::constant(1, 1.0).divide(Data::constant(1, 0.0));
        let cancelled = undefined
            .clone()
            .add(undefined.multiply(Data::constant(1, -1.0)));
        assert!(cancelled.same_coefficient(&Data::constant(1, 0.0)));
        assert!(cancelled.evaluate(&[0.0]).is_err());
    }

    #[test]
    fn shared_squaring_declines_unbounded_exact_proof_work() {
        let mut coefficient = Data::constant(1, 1.0 + f64::EPSILON);
        for _ in 0..32 {
            coefficient = coefficient.clone().multiply(coefficient);
        }
        assert_eq!(coefficient.coefficient_orientation(), None);
    }
}
