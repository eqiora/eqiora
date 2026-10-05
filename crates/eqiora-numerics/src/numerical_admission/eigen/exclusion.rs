use super::*;
use num_complex::Complex64;

/// Excluded directions of an authored spectral embedding, with separate
/// numerical tests for the original operator and metric nullspaces.
/// This record never infers physical gauge freedom from numerical nullity.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Exclusion {
    dimension: usize,
    projector: ValueLiteral,
    operator_defect: f64,
    metric_defect: f64,
}

impl Exclusion {
    pub(super) fn derive(
        source: &source::SourcePencil,
        request: CommonEigenRequest,
    ) -> Result<Option<Self>, Diagnostic> {
        let Some(projected) = &source.projected else {
            return Ok(None);
        };
        let map = &projected.embedding.map;
        let (from, to) = map.value_type().map_bases().expect("admitted map");
        let dimension = (to.extent() - from.extent()) as usize;
        if dimension == 0 {
            return Ok(None);
        }
        let projector = HermitianEigenproblem::excluded_metric_projector(
            &source.operator,
            &source.metric,
            map,
            request.normalization_tolerance(),
        )?;
        Ok(Some(Self {
            dimension,
            operator_defect: action_defect(&source.operator, &projector)?,
            metric_defect: action_defect(&source.metric, &projector)?,
            projector,
        }))
    }

    pub(super) fn description(&self) -> (&ValueLiteral, usize, f64, f64) {
        (
            &self.projector,
            self.dimension,
            self.operator_defect,
            self.metric_defect,
        )
    }
}

fn action_defect(matrix: &ValueLiteral, projector: &ValueLiteral) -> Result<f64, Diagnostic> {
    let n = matrix
        .value_type()
        .map_bases()
        .expect("square map")
        .0
        .extent() as usize;
    let coefficient = |v: &ValueLiteral, i| {
        let (re, im) = v.component(i).expect("typed shape");
        Complex64::new(re, im)
    };
    let scale = (0..n * n)
        .map(|i| coefficient(matrix, i))
        .fold(0_f64, |s, z| s.max(z.re.abs()).max(z.im.abs()));
    if scale == 0. {
        return Ok(0.);
    }
    let mut normalized = Vec::with_capacity(n * n);
    let (mut matrix_norm, mut projector_norm) = (0_f64, 0_f64);
    for i in 0..n * n {
        let original = coefficient(matrix, i);
        let z = original / scale;
        if (original.re != 0. && z.re == 0.) || (original.im != 0. && z.im == 0.) {
            return Err(invalid("excluded-space action scaling underflows binary64"));
        }
        matrix_norm = matrix_norm.hypot(z.norm());
        projector_norm = projector_norm.hypot(coefficient(projector, i).norm());
        normalized.push(z);
    }
    let mut residual = 0_f64;
    for row in 0..n {
        for column in 0..n {
            let z: Complex64 = (0..n)
                .map(|j| normalized[row * n + j] * coefficient(projector, j * n + column))
                .sum();
            residual = residual.hypot(z.norm());
        }
    }
    let defect = (residual / matrix_norm) / projector_norm;
    if !projector_norm.is_finite() || !defect.is_finite() {
        return Err(invalid(
            "excluded-space action verification produced nonfinite arithmetic",
        ));
    }
    Ok(defect)
}
