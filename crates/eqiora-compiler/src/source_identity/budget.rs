//! Bounded source identity accounting and lexical reduction binder context.
use super::*;

pub(super) struct Budget {
    pub(super) limits: LocalSourceIdentityLimits,
    pub(super) total_members: usize,
    pub(super) expression_nodes: usize,
    pub(super) total_name_bytes: usize,
    pub(super) materialized_bytes: usize,
    pub(super) boundary_set_memberships: usize,
    pub(super) resolved_aliases: BTreeMap<String, ResolvedAliasTarget>,
    pub(super) reduction_binders: Vec<String>,
    pub(super) lexical_formals: BTreeMap<String, usize>,
    pub(super) operator_formals: BTreeMap<String, Vec<String>>,
}

impl Budget {
    pub(super) const fn new(limits: LocalSourceIdentityLimits) -> Self {
        Self {
            limits,
            total_members: 0,
            expression_nodes: 0,
            total_name_bytes: 0,
            materialized_bytes: 0,
            boundary_set_memberships: 0,
            resolved_aliases: BTreeMap::new(),
            reduction_binders: Vec::new(),
            operator_formals: BTreeMap::new(),
            lexical_formals: BTreeMap::new(),
        }
    }

    pub(super) fn with_resolved_aliases(
        limits: LocalSourceIdentityLimits,
        resolved_aliases: BTreeMap<String, ResolvedAliasTarget>,
    ) -> Self {
        Self {
            resolved_aliases,
            ..Self::new(limits)
        }
    }

    pub(super) fn account_members(
        &mut self,
        count: usize,
        container: &'static str,
    ) -> Result<(), Diagnostic> {
        if count > self.limits.max_members_per_container {
            return Err(source_identity_error(format!(
                "{container} has {count} members, exceeding the {} member limit",
                self.limits.max_members_per_container
            )));
        }
        self.total_members = self
            .total_members
            .checked_add(count)
            .ok_or_else(|| source_identity_error("source member count overflows usize"))?;
        if self.total_members > self.limits.max_total_members {
            return Err(source_identity_error(format!(
                "source unit exceeds the {} total member limit",
                self.limits.max_total_members
            )));
        }
        Ok(())
    }

    pub(super) fn check_connection_members(
        &self,
        count: usize,
        label: &'static str,
    ) -> Result<(), Diagnostic> {
        if count > self.limits.max_connection_members {
            return Err(source_identity_error(format!(
                "{label} has {count} members, exceeding the {} member limit",
                self.limits.max_connection_members
            )));
        }
        Ok(())
    }

    pub(super) fn account_boundary_set_members(&mut self, count: usize) -> Result<(), Diagnostic> {
        if count > self.limits.max_boundary_set_members {
            return Err(source_identity_error(format!(
                "complete-exterior binding has {count} members, exceeding the {} member limit",
                self.limits.max_boundary_set_members
            )));
        }
        self.boundary_set_memberships = self
            .boundary_set_memberships
            .checked_add(count)
            .ok_or_else(|| source_identity_error("BoundarySet membership count overflows usize"))?;
        if self.boundary_set_memberships > self.limits.max_total_boundary_set_memberships {
            return Err(source_identity_error(format!(
                "source unit exceeds the {} total BoundarySet membership limit",
                self.limits.max_total_boundary_set_memberships
            )));
        }
        Ok(())
    }

    pub(super) fn connection_set_limits(&self) -> ConnectionSetLimits {
        let defaults = ConnectionSetLimits::default();
        let max_memberships = self
            .limits
            .max_members_per_container
            .saturating_mul(self.limits.max_connection_members)
            .min(defaults.max_memberships);
        ConnectionSetLimits {
            max_fragments: self
                .limits
                .max_members_per_container
                .min(defaults.max_fragments),
            max_memberships,
            max_endpoints: max_memberships.min(defaults.max_endpoints),
            // Every maximal set consumes at least one source Connection item,
            // so the existing container budget is also its natural bound.
            max_sets: self.limits.max_members_per_container.min(defaults.max_sets),
            max_members_per_fragment: self
                .limits
                .max_connection_members
                .min(defaults.max_members_per_fragment),
            // A transitive union must not evade the existing per-Connection
            // member policy merely because it was written as small fragments.
            max_members_per_set: self
                .limits
                .max_connection_members
                .min(defaults.max_members_per_set),
        }
    }

    pub(super) fn account_expression(&mut self, depth: usize) -> Result<(), Diagnostic> {
        if depth > self.limits.max_expression_depth {
            return Err(source_identity_error(format!(
                "expression exceeds the {} level depth limit",
                self.limits.max_expression_depth
            )));
        }
        self.expression_nodes = self
            .expression_nodes
            .checked_add(1)
            .ok_or_else(|| source_identity_error("expression node count overflows usize"))?;
        if self.expression_nodes > self.limits.max_expression_nodes {
            return Err(source_identity_error(format!(
                "source unit exceeds the {} expression node limit",
                self.limits.max_expression_nodes
            )));
        }
        Ok(())
    }

    pub(super) fn account_name(&mut self, name: &str) -> Result<(), Diagnostic> {
        if name.is_empty() {
            return Err(source_identity_error(
                "source identity name must not be empty",
            ));
        }
        if name.len() > self.limits.max_name_bytes {
            return Err(source_identity_error(format!(
                "source name requires {} bytes, exceeding the {} byte limit",
                name.len(),
                self.limits.max_name_bytes
            )));
        }
        self.total_name_bytes = self
            .total_name_bytes
            .checked_add(name.len())
            .ok_or_else(|| source_identity_error("source name bytes overflow usize"))?;
        if self.total_name_bytes > self.limits.max_total_name_bytes {
            return Err(source_identity_error(format!(
                "source unit exceeds the {} total name byte limit",
                self.limits.max_total_name_bytes
            )));
        }
        Ok(())
    }

    pub(super) fn account_materialized_bytes(&mut self, count: usize) -> Result<(), Diagnostic> {
        self.materialized_bytes = self
            .materialized_bytes
            .checked_add(count)
            .ok_or_else(|| source_identity_error("intermediate source bytes overflow usize"))?;
        if self.materialized_bytes > self.limits.max_intermediate_bytes {
            return Err(source_identity_error(format!(
                "canonical source sorting exceeds the {} intermediate byte limit",
                self.limits.max_intermediate_bytes
            )));
        }
        Ok(())
    }
}
