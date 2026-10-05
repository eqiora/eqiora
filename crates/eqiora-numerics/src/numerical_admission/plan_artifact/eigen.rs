use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireQuantity {
    value: f64,
    dimension: [(i32, i32); 7],
}

impl WireQuantity {
    fn from_native(value: DynQuantity) -> Self {
        Self {
            value: value.value(),
            dimension: value.dim().exponents(),
        }
    }
    fn to_native(self) -> Result<DynQuantity, Diagnostic> {
        Ok(DynQuantity::new(
            self.value,
            DimExponents::from_rationals(self.dimension)
                .ok_or_else(|| invalid("invalid spectral quantity dimension"))?,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireEigenRequest {
    algorithm: String,
    count: usize,
    target: Option<WireQuantity>,
    interval: Option<[WireQuantity; 2]>,
    residual_tolerance: f64,
    normalization_tolerance: f64,
}

impl WireEigenRequest {
    pub(super) fn from_native(request: CommonEigenRequest) -> Self {
        Self {
            algorithm: "dense-hermitian".to_owned(),
            count: request.count().get(),
            target: request.target().map(WireQuantity::from_native),
            interval: request
                .interval()
                .map(|values| values.map(WireQuantity::from_native)),
            residual_tolerance: request.residual_tolerance(),
            normalization_tolerance: request.normalization_tolerance(),
        }
    }
    pub(super) fn to_native(&self) -> Result<CommonEigenRequest, Diagnostic> {
        if self.algorithm != "dense-hermitian" {
            return Err(invalid("unsupported spectral algorithm"));
        }
        let mut request = CommonEigenRequest::dense(
            NonZeroUsize::new(self.count)
                .ok_or_else(|| invalid("spectral mode count must be positive"))?,
            self.residual_tolerance,
            self.normalization_tolerance,
        )?;
        if let Some(target) = self.target {
            request = request.with_target(target.to_native()?)?;
        }
        if let Some([left, right]) = self.interval {
            request = request.within_interval([left.to_native()?, right.to_native()?])?;
        }
        Ok(request)
    }
}
