use super::*;

type SensitivityControl = (
    eqiora_core::TimeStateCoordinate,
    Id<kinds::Parameter>,
    eqiora_core::DynQuantity,
);

impl CommonOdePolicy {
    /// Bind a scalar Tsitouras tolerance to one admitted real first-order state.
    ///
    /// This shorthand does not broadcast across Fields, derivative coordinates,
    /// complex parts, or shaped values. The returned policy uses the same exact
    /// coordinates and positive finite controls as [`Self::new`].
    pub fn for_scalar_state(
        kernel: &KernelProgram,
        initial_step_s: f64,
        relative_tolerance: f64,
        absolute_tolerance: f64,
    ) -> Result<Self, Diagnostic> {
        let flow = events::flow(kernel, None)?;
        let program = FirstOrderProgram::lower(kernel, flow)?;
        let [coordinate] = program.state_coordinates() else {
            return Err(invalid(
                "scalar absolute_tolerance requires one real scalar first-order ODE state coordinate",
            ));
        };
        let Some(KernelNode::Field(field)) = kernel.node(coordinate.field().erase()) else {
            return Err(invalid("time coordinate has no exact Field type"));
        };
        if program.equation_class() != TimeEquationClass::ExplicitOde
            || coordinate.derivative_order() != 0
            || coordinate.component() != 0
            || coordinate.is_imaginary()
            || field.value_type().scalar_domain() != eqiora_core::ScalarDomain::Real
            || !field.value_type().shape().is_scalar()
        {
            return Err(invalid(
                "scalar absolute_tolerance requires one real scalar first-order ODE state coordinate",
            ));
        }
        Self::new(
            TimeMethod::Tsitouras45,
            initial_step_s,
            relative_tolerance,
            vec![(*coordinate, absolute_tolerance)],
        )
    }

    /// Check exact coordinate ownership and return tolerances in execution order.
    pub(crate) fn ordered_absolute_tolerances(
        &self,
        state_coordinates: &[eqiora_core::TimeStateCoordinate],
    ) -> Result<Vec<f64>, Diagnostic> {
        let requested = self.absolute_tolerances();
        if requested.len() != state_coordinates.len()
            || requested
                .iter()
                .any(|entry| !state_coordinates.contains(&entry.0))
        {
            return Err(invalid(
                "absolute tolerances must cover exactly the admitted Model state coordinates",
            ));
        }
        state_coordinates
            .iter()
            .map(|field| {
                requested
                    .iter()
                    .find(|entry| entry.0 == *field)
                    .map(|entry| entry.1)
                    .ok_or_else(|| invalid("time policy omitted one exact state coordinate"))
            })
            .collect()
    }

    /// Validate event controls before attaching them to an integration request.
    pub fn validate_event_controls(
        max_events: usize,
        guard_tolerances: Vec<(Id<kinds::Activation>, eqiora_core::DynQuantity)>,
    ) -> Result<(), Diagnostic> {
        let guard_tolerances = guard_tolerances
            .into_iter()
            .map(|(activation, quantity)| CommonGuardTolerance::new(activation, quantity))
            .collect::<Result<Vec<_>, _>>()?;
        CommonEventPolicy::new(max_events, guard_tolerances).map(|_| ())
    }

    /// Validate forward controls before attaching them to an integration request.
    pub fn validate_forward_sensitivity_controls(
        relative_tolerance: f64,
        absolute_tolerances: Vec<SensitivityControl>,
    ) -> Result<(), Diagnostic> {
        let absolute_tolerances = absolute_tolerances
            .into_iter()
            .map(|(field, parameter, quantity)| {
                CommonSensitivityTolerance::new(field, parameter, quantity)
            })
            .collect::<Result<Vec<_>, _>>()?;
        CommonForwardSensitivity::new(relative_tolerance, absolute_tolerances).map(|_| ())
    }

    /// Select explicit bounded canonical event enforcement.
    pub fn with_events(
        mut self,
        max_events: usize,
        guard_tolerances: Vec<(Id<kinds::Activation>, eqiora_core::DynQuantity)>,
    ) -> Result<Self, Diagnostic> {
        let guard_tolerances = guard_tolerances
            .into_iter()
            .map(|(activation, quantity)| CommonGuardTolerance::new(activation, quantity))
            .collect::<Result<Vec<_>, _>>()?;
        self.events = Some(CommonEventPolicy::new(max_events, guard_tolerances)?);
        Ok(self)
    }

    /// Exact event limits and unit-bearing guard tolerances.
    #[must_use]
    pub fn events(&self) -> Option<&CommonEventPolicy> {
        self.events.as_ref()
    }

    pub(crate) fn with_event_policy(mut self, policy: CommonEventPolicy) -> Self {
        self.events = Some(policy);
        self
    }

    /// Select explicit continuous forward Parameter derivatives.
    pub fn with_forward_sensitivities(
        mut self,
        relative_tolerance: f64,
        absolute_tolerances: Vec<SensitivityControl>,
    ) -> Result<Self, Diagnostic> {
        let absolute_tolerances = absolute_tolerances
            .into_iter()
            .map(|(field, parameter, quantity)| {
                CommonSensitivityTolerance::new(field, parameter, quantity)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.forward_sensitivities = Some(CommonForwardSensitivity::new(
            relative_tolerance,
            absolute_tolerances,
        )?);
        Ok(self)
    }

    /// Exact forward-derivative controls in their canonical coordinate identity.
    #[must_use]
    pub fn forward_sensitivities(&self) -> Option<&CommonForwardSensitivity> {
        self.forward_sensitivities.as_ref()
    }

    pub(crate) fn with_forward_sensitivity_policy(
        mut self,
        policy: CommonForwardSensitivity,
    ) -> Self {
        self.forward_sensitivities = Some(policy);
        self
    }
}

impl CommonOdePolicy {
    pub(crate) fn require_spatial_execution_controls(&self) -> Result<(), Diagnostic> {
        if self.events.is_some()
            || self.forward_sensitivities.is_some()
            || !self.conserved_norms.is_empty()
            || !self.hermitian_parameters.is_empty()
        {
            return Err(invalid(
                "spatial time execution does not yet implement event, sensitivity, norm or Hermitian controls",
            ));
        }
        Ok(())
    }
}
