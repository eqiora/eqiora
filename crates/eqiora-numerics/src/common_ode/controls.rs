use super::*;

impl CommonTsitouras45 {
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
        absolute_tolerances: Vec<(
            Id<kinds::Field>,
            Id<kinds::Parameter>,
            eqiora_core::DynQuantity,
        )>,
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
        absolute_tolerances: Vec<(
            Id<kinds::Field>,
            Id<kinds::Parameter>,
            eqiora_core::DynQuantity,
        )>,
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
