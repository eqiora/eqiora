//! Formatting of retained physical Law terms.

use super::{format_expression, write_indent};
use crate::ast::{ConservationSyntax, RelationDecl};
use core::fmt::Write;

pub(super) fn format_probability(
    declaration: &RelationDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    write_indent(output, indent);
    write!(
        output,
        "distribution {}: ",
        declaration.comments.named(&declaration.name)
    )
    .expect("String write");
    match declaration.body() {
        crate::RelationBody::ProbabilityMass { outcomes, weights } => {
            output.push_str("normalize_mass(");
            super::value_type::format_value_type(outcomes, output);
            output.push_str(", weights = ");
            format_expression(weights, 0, output);
        }
        crate::RelationBody::ProbabilityDensity {
            domain,
            measure,
            density,
        } => {
            let measure = match measure {
                Some(eqiora_schema::kernel::ObservableMeasure::SphericalVolume) => {
                    "spherical_measure"
                }
                _ => "measure",
            };
            write!(output, "normalize_density({measure}({domain}), density = ")
                .expect("String write");
            format_expression(density, 0, output);
        }
        _ => unreachable!("probability formatter dispatch"),
    }
    output.push_str(");\n");
}

pub(super) fn format_noise(
    declaration: &RelationDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    use eqiora_schema::kernel::NoiseProcess;
    let crate::RelationBody::Noise {
        process,
        parameter,
        independent,
    } = declaration.body()
    else {
        unreachable!("noise formatter dispatch")
    };
    let (process, parameter_name) = match process {
        NoiseProcess::Wiener => ("wiener", "covariance"),
        NoiseProcess::Poisson => ("poisson", "intensity"),
    };
    write_indent(output, indent);
    write!(
        output,
        "noise {}: {process}({parameter_name} = ",
        declaration.comments.named(&declaration.name)
    )
    .expect("String write");
    format_expression(parameter, 0, output);
    if *independent {
        output.push_str(", independent = true");
    }
    output.push_str(");\n");
}

pub(super) fn format_stochastic(
    declaration: &RelationDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    use eqiora_schema::kernel::StochasticCalculus;
    let crate::RelationBody::Stochastic {
        target,
        calculus,
        drift,
        diffusions,
        jumps,
    } = declaration.body()
    else {
        unreachable!("stochastic formatter dispatch")
    };
    write_indent(output, indent);
    writeln!(
        output,
        "law {} for {} {{",
        declaration.comments.named(&declaration.name),
        target
    )
    .expect("String write");
    if let Some(calculus) = calculus {
        write_indent(output, indent + 2);
        writeln!(
            output,
            "calculus {};",
            match calculus {
                StochasticCalculus::Ito => "ito",
                StochasticCalculus::Stratonovich => "stratonovich",
            }
        )
        .expect("String write");
    }
    if let Some(drift) = drift {
        write_indent(output, indent + 2);
        output.push_str("drift ");
        format_expression(drift, 0, output);
        output.push_str(";\n");
    }
    for (kind, bindings) in [("diffusion", diffusions), ("jump", jumps)] {
        for (group, coefficient) in bindings {
            write_indent(output, indent + 2);
            write!(output, "{kind} {group} = ").expect("String write");
            format_expression(coefficient, 0, output);
            output.push_str(";\n");
        }
    }
    write_indent(output, indent);
    output.push_str("}\n");
}

pub(super) fn format_law(
    declaration: &RelationDecl,
    terms: &ConservationSyntax,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    write_indent(output, indent);
    writeln!(
        output,
        "law {} on {} {{",
        declaration.comments.named(&declaration.name),
        declaration
            .domain
            .as_deref()
            .expect("Law owns source support")
    )
    .expect("String write");
    for (name, value) in [
        ("storage", terms.storage()),
        ("flux", Some(terms.flux())),
        ("source", Some(terms.source())),
    ] {
        if let Some(value) = value {
            write_indent(output, indent + 2);
            write!(output, "{name} ").expect("String write");
            format_expression(value, 0, output);
            output.push_str(";\n");
        }
    }
    write_indent(output, indent);
    output.push_str("}\n");
}
