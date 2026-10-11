//! Cardinality algebra for compositional physical endpoints.
//!
//! A physical endpoint has two independent semantic slots: one constitutive
//! Relation owner and one conserving Connection membership. Component
//! definitions may export either slot unfilled; a closed Model must fill both.
//! This contract deliberately says nothing about scalar connection
//! compatibility, which is owned by `scalar_connection`.

/// Whether one physical-endpoint slot has been filled.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PhysicalSlot {
    /// The surrounding definition still owes this slot.
    #[default]
    Open,
    /// Exactly one declaration has filled this slot.
    Filled,
}

/// The two independently composable slots of one physical endpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PhysicalEndpointSlots {
    owner: PhysicalSlot,
    membership: PhysicalSlot,
}

impl PhysicalEndpointSlots {
    /// Start with both obligations open.
    #[must_use]
    pub const fn open() -> Self {
        Self {
            owner: PhysicalSlot::Open,
            membership: PhysicalSlot::Open,
        }
    }

    /// Current constitutive-Relation ownership slot.
    #[must_use]
    pub const fn owner(self) -> PhysicalSlot {
        self.owner
    }

    /// Current conserving-Connection membership slot.
    #[must_use]
    pub const fn membership(self) -> PhysicalSlot {
        self.membership
    }

    /// Fill the Relation-owner slot exactly once.
    ///
    /// # Errors
    /// Returns [`PhysicalClosureViolation::MultipleOwners`] when another
    /// Relation already owns the endpoint.
    pub fn fill_owner(&mut self) -> Result<(), PhysicalClosureViolation> {
        fill_slot(&mut self.owner, PhysicalClosureViolation::MultipleOwners)
    }

    /// Fill the Connection-membership slot exactly once.
    ///
    /// # Errors
    /// Returns [`PhysicalClosureViolation::MultipleMemberships`] when the
    /// endpoint already belongs to a conserving Connection.
    pub fn fill_membership(&mut self) -> Result<(), PhysicalClosureViolation> {
        fill_slot(
            &mut self.membership,
            PhysicalClosureViolation::MultipleMemberships,
        )
    }

    /// Require both endpoint obligations to be closed.
    ///
    /// # Errors
    /// Reports the first open slot in stable owner-then-membership order.
    pub const fn require_closed(self) -> Result<(), PhysicalClosureViolation> {
        match (self.owner, self.membership) {
            (PhysicalSlot::Open, _) => Err(PhysicalClosureViolation::MissingOwner),
            (_, PhysicalSlot::Open) => Err(PhysicalClosureViolation::MissingMembership),
            (PhysicalSlot::Filled, PhysicalSlot::Filled) => Ok(()),
        }
    }
}

/// Failure to preserve the physical-endpoint cardinality invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalClosureViolation {
    /// More than one Relation owns the endpoint.
    MultipleOwners,
    /// More than one Connection contains the endpoint.
    MultipleMemberships,
    /// A closed endpoint has no owning Relation.
    MissingOwner,
    /// A closed endpoint has no Connection membership.
    MissingMembership,
}

fn fill_slot(
    slot: &mut PhysicalSlot,
    violation: PhysicalClosureViolation,
) -> Result<(), PhysicalClosureViolation> {
    match slot {
        PhysicalSlot::Open => {
            *slot = PhysicalSlot::Filled;
            Ok(())
        }
        PhysicalSlot::Filled => Err(violation),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_independent_and_fill_exactly_once() {
        let mut slots = PhysicalEndpointSlots::open();
        assert_eq!(slots.owner(), PhysicalSlot::Open);
        assert_eq!(slots.membership(), PhysicalSlot::Open);
        slots.fill_owner().expect("owner closes once");
        assert_eq!(slots.membership(), PhysicalSlot::Open);
        assert_eq!(
            slots.fill_owner(),
            Err(PhysicalClosureViolation::MultipleOwners)
        );
        slots.fill_membership().expect("membership closes once");
        assert_eq!(slots.require_closed(), Ok(()));
        assert_eq!(
            slots.fill_membership(),
            Err(PhysicalClosureViolation::MultipleMemberships)
        );
    }

    #[test]
    fn closed_requirement_is_owner_then_membership_deterministic() {
        let mut slots = PhysicalEndpointSlots::open();
        assert_eq!(
            slots.require_closed(),
            Err(PhysicalClosureViolation::MissingOwner)
        );
        slots.fill_owner().expect("owner closes");
        assert_eq!(
            slots.require_closed(),
            Err(PhysicalClosureViolation::MissingMembership)
        );
    }
}
