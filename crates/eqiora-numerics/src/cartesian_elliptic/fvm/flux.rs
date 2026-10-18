//! One coefficient/data binding for primal fluxes and their Parameter directions.
use super::*;

type Diffusion<'a, S> = dyn Fn(usize, usize, usize, &[f64]) -> Result<S, Diagnostic> + Sync + 'a;
type Prescribed<'a, S> = dyn Fn(usize, usize, &[f64]) -> Result<S, Diagnostic> + Sync + 'a;
type Boundary<'a, S> = dyn Fn(usize, usize, BoundarySide, &[f64]) -> Result<CartesianBoundaryValue<S>, Diagnostic>
    + 'a;

#[derive(Clone, Copy)]
pub(super) struct FluxData<'a, S> {
    pub coefficient: &'a Diffusion<'a, S>,
    pub prescribed: &'a Prescribed<'a, S>,
    pub boundary: &'a Boundary<'a, S>,
}

#[derive(Clone, Copy)]
pub(crate) struct FvmFlux<'a, S> {
    pub(super) value: FluxData<'a, S>,
    pub(super) direction: Option<FluxData<'a, S>>,
}

impl<'a, S: Copy> FvmFlux<'a, S> {
    pub(crate) fn new(
        coefficient: &'a Diffusion<'a, S>,
        prescribed: &'a Prescribed<'a, S>,
        boundary: &'a Boundary<'a, S>,
    ) -> Self {
        Self {
            value: FluxData {
                coefficient,
                prescribed,
                boundary,
            },
            direction: None,
        }
    }

    pub(crate) fn with_direction(
        mut self,
        coefficient: &'a Diffusion<'a, S>,
        prescribed: &'a Prescribed<'a, S>,
        boundary: &'a Boundary<'a, S>,
    ) -> Self {
        self.direction = Some(FluxData {
            coefficient,
            prescribed,
            boundary,
        });
        self
    }

    pub(super) fn selected(&self) -> FluxData<'a, S> {
        self.direction.unwrap_or(self.value)
    }
}
