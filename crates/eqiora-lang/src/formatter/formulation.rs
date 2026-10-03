//! Canonical formatting for authored mathematical formulations.

use core::fmt::Write;

use crate::ast::formulation::{FormulationBinding, FormulationDecl};
use crate::ast::{ComponentDecl, VisibilitySyntax};

use super::{format_component_item, format_expression, write_indent};

pub(super) fn format_component(
    component: &ComponentDecl,
    output: &mut crate::formatter::comments::Output,
) {
    output.begin(&component.comments);
    if component.visibility == VisibilitySyntax::Public {
        output.push_str("public ");
    }
    write!(
        output,
        "component {}",
        component.comments.named(&component.name)
    )
    .expect("String write");
    super::signature::format_signature(&component.signature, output);
    output.push_str(" {\n");
    for item in &component.items {
        format_component_item(item, 2, output);
    }
    for formulation in &component.formulations {
        format_formulation(formulation, 2, output);
    }
    output.push_str("}\n");
    output.end();
}

pub(super) fn format_formulation(
    declaration: &FormulationDecl,
    indent: usize,
    output: &mut crate::formatter::comments::Output,
) {
    output.begin(&declaration.comments);
    write_indent(output, indent);
    output.push_str("form ");
    output.push_str(&declaration.name);
    writeln!(output, " for {} {{", declaration.relations.join(", ")).expect("String write");
    match &declaration.binding {
        FormulationBinding::Finite { name, trials } => {
            write_indent(output, indent + 2);
            writeln!(output, "finite {name}({});", trials.join(", ")).expect("String write");
        }
        FormulationBinding::WeakTests { tests } => {
            for (name, trial, zero_on, dimension) in tests {
                write_indent(output, indent + 2);
                write!(output, "test {name}: ").expect("String write");
                format_expression(dimension, 0, output);
                write!(output, " for {trial}").expect("String write");
                if !zero_on.is_empty() {
                    write!(output, " zero_on {}", zero_on.join(", ")).expect("String write");
                }
                output.push_str(";\n");
            }
        }
        FormulationBinding::Interval {
            name,
            lower,
            upper,
            domain,
        } => {
            write_indent(output, indent + 2);
            writeln!(output, "interval {name}({lower}, {upper}) on {domain};")
                .expect("String write");
        }
    }
    if let Some((field, conditions)) = &declaration.gauge {
        write_indent(output, indent + 2);
        writeln!(output, "gauge {field} {{").expect("String write");
        for (label, (left, right)) in ["reference", "compatibility"].into_iter().zip(conditions) {
            write_indent(output, indent + 4);
            write!(output, "{label} ").expect("String write");
            format_expression(left, 0, output);
            output.push_str(" = ");
            format_expression(right, 0, output);
            output.push_str(";\n");
        }
        write_indent(output, indent + 2);
        output.push_str("}\n");
    }
    for (left, right) in &declaration.equations {
        write_indent(output, indent + 2);
        format_expression(left, 0, output);
        output.push_str(" = ");
        format_expression(right, 0, output);
        output.push_str(";\n");
    }
    write_indent(output, indent);
    output.push_str("}\n");
    output.end();
}

#[cfg(test)]
mod tests {
    use crate::{format, parse};

    #[test]
    fn finite_gauge_retains_coordinate_order_and_both_conditions() {
        let source = "component Network() { form floating for first, second { finite voltage(v1,v2); gauge voltage { reference v1=offset; compatibility i1+i2=0; } g*(v1-v2)=i1; g*(v2-v1)=i2; } }";
        let first = parse("finite.eqi", source).into_document().unwrap();
        let formatted = format(&first);
        let second = parse("finite.eqi", &formatted).into_document().unwrap();
        assert_eq!(format(&second), formatted);
        let component = &second.components()[0];
        assert_eq!(
            component.formulation_binding("floating"),
            Some(&crate::FormulationBinding::Finite {
                name: "voltage".into(),
                trials: vec!["v1".into(), "v2".into()]
            })
        );
        assert_eq!(
            component.formulation_gauge("floating").unwrap().0,
            "voltage"
        );
        assert!(formatted.contains("reference v1 = offset;"));
        assert!(formatted.contains("compatibility i1 + i2 = 0;"));
        for binder in [
            "finite voltage();",
            "finite voltage(v1,);",
            "finite voltage(v1,v2)",
        ] {
            let malformed = source.replace("finite voltage(v1,v2);", binder);
            assert!(parse("finite.eqi", &malformed).into_document().is_err());
        }
    }

    #[test]
    fn dimensional_direction_survives_formatting() {
        let source = "component Elastic() { form virtual_work for balance { test eta:m for displacement zero_on fixed; integrate(body,eta*force)=0; } }";
        let first = parse("elastic.eqi", source).into_document().unwrap();
        let formatted = format(&first);
        assert!(formatted.contains("test eta: m for displacement zero_on fixed;"));
        let second = parse("elastic.eqi", &formatted).into_document().unwrap();
        assert_eq!(format(&second), formatted);
    }

    #[test]
    fn primal_form_has_one_canonical_roundtrip() {
        let source = "component D(support region:volume(ambient_dimension=2)) {variable u: 1 on region;relation balance on region{-div(grad(u))=f;}form weak for balance { test w: 1 for u zero_on surface;integrate(region,dot(grad(w),grad(u)))=integrate(region,w*f);}}";
        let first = parse("form.eqi", source).into_document().unwrap();
        let formatted = format(&first);
        let second = parse("form.eqi", &formatted).into_document().unwrap();

        assert_eq!(format(&second), formatted);
        assert!(
            formatted.contains("form weak for balance {\n    test w: 1 for u zero_on surface;\n")
        );
        assert!(
            formatted
                .contains("integrate(region, dot(grad(w), grad(u))) = integrate(region, w * f);")
        );
    }
    #[test]
    fn mixed_form_preserves_ordered_relations_tests_and_equalities() {
        let source = "component C() { form weak for momentum,incompressibility { test v:1 for velocity zero_on surface; test q:1 for pressure; integrate(body,dot(grad(v),grad(velocity)))=integrate(body,dot(v,force)); integrate(body,q*div(velocity))=integrate(body,0); } }";
        let first = parse("mixed.eqi", source).into_document().unwrap();
        let formatted = format(&first);
        let second = parse("mixed.eqi", &formatted).into_document().unwrap();
        assert_eq!(format(&second), formatted);
        let component = &second.components()[0];
        let (name, relations, equations, _) = component.formulations().next().unwrap();
        assert_eq!(relations, ["momentum", "incompressibility"]);
        assert_eq!(equations.len(), 2);
        let Some(crate::FormulationBinding::WeakTests { tests }) =
            component.formulation_binding(name)
        else {
            panic!("weak tests");
        };
        assert_eq!(
            tests
                .iter()
                .map(|(name, trial, boundaries, _)| (
                    name.as_str(),
                    trial.as_str(),
                    boundaries.clone()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("v", "velocity", vec!["surface".into()]),
                ("q", "pressure", vec![])
            ]
        );
        assert!(formatted.contains("test q: 1 for pressure;"));
    }
}
