use crate::ast::DomainSyntax;
use crate::cartesian::CartesianCoordinateSyntax;

use super::{AstConstructionError, validate_finite, validate_identifier};

pub(super) fn validate_domain_syntax(syntax: &DomainSyntax) -> Result<(), AstConstructionError> {
    match syntax {
        DomainSyntax::PhysicalInterface { boundaries } => {
            for boundary in boundaries {
                validate_identifier(boundary, "interface boundary")?;
            }
            if boundaries[0] == boundaries[1] {
                return Err(AstConstructionError::new(
                    "physical interface requires distinct boundaries",
                ));
            }
            Ok(())
        }
        DomainSyntax::Product { factors } => {
            if factors.is_empty() {
                return Err(AstConstructionError::new(
                    "a coordinate product requires at least one factor",
                ));
            }
            for factor in factors {
                validate_identifier(factor, "coordinate factor support")?;
            }
            Ok(())
        }
        DomainSyntax::CartesianBox(bounds) => {
            if bounds.is_empty() {
                return Err(AstConstructionError::new(
                    "a Cartesian box requires at least one coordinate pair",
                ));
            }
            for coordinate in bounds.iter().flat_map(|(lower, upper)| [lower, upper]) {
                match coordinate {
                    CartesianCoordinateSyntax::Fixed { value, .. } => {
                        validate_finite(*value, "Cartesian coordinate")?
                    }
                    CartesianCoordinateSyntax::Parameter { name, .. } => {
                        validate_identifier(name, "Cartesian coordinate Parameter")?
                    }
                }
            }
            Ok(())
        }
        DomainSyntax::Boundary { parent, .. } => validate_identifier(parent, "parent Domain"),
        DomainSyntax::ScalarPhysical {
            across_name,
            across_type,
            through_name,
            through_type,
        } => {
            validate_identifier(across_name, "across quantity")?;
            validate_identifier(through_name, "through quantity")?;
            if across_name == through_name {
                return Err(AstConstructionError::new(
                    "physical quantity names must be distinct",
                ));
            }
            super::value_type::validate_syntax(across_type)?;
            super::value_type::validate_syntax(through_type)
        }
    }
}
