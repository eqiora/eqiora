//! Whole-Model admission distinguishes population observations from realized paths.
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_schema::kernel::{KernelNode, ObservableDef, ObservableReduction, ProbabilityMoment};
use eqiora_sem::KernelProgram;

#[test]
fn stochastic_realizations_cannot_be_relabeled_as_population_moments() {
    let source = r#"enum Outcome {Low,High}
        model M() {
            domain physical = scalar_physical(across effort:1, through flow:1);
            distribution law:normalize_mass(Outcome, weights=[1,3]);
            noise arrivals:poisson(intensity=1[1/s], independent=true);
            state realization:1;
            initial {realization=0;}
            law evolution for realization {jump arrivals=1;}
            observable alias:1=realization;
            observable indirect:1=alias;
            observable observation:1=indirect;
            variable derived:1;
            variable unrelated:1;
            relation algebraic {derived=indirect; unrelated=2;}
            observable field_alias:1=derived;
            observable field_indirect:1=field_alias;
            observable deterministic:1=unrelated;
            port sender:signal output 1;
            port receiver:signal input 1;
            connect sender -> receiver;
            relation send {sender=realization;}
            observable signal_alias:1=receiver;
            port a:physical; port b:physical;
            connect a,b;
            relation physical_values {a.effort=realization; a.flow=0;}
            relation terminal {b.flow=0;}
            observable physical_alias:1=b.effort;
            observable deterministic_flux:1=b.flow;
            state triggered:1;
            initial {triggered=0;}
            event arrived=crossing(realization-0.5,direction=rising);
            relation reset at arrived {next(triggered)=1;}
            observable event_alias:1=triggered;
        }"#;
    let compiled = eqiora_compiler::compile("sample.eqi", source)
        .unwrap()
        .remove(0);
    let (original, model, symbols) = compiled.into_parts();
    let law = symbols.get("law").unwrap().downcast().unwrap();
    for selected in [
        "alias",
        "indirect",
        "observation",
        "field_alias",
        "field_indirect",
        "deterministic",
        "signal_alias",
        "physical_alias",
        "deterministic_flux",
        "event_alias",
    ] {
        for reduction in [
            ObservableReduction::Value,
            ObservableReduction::RandomVariable { distribution: law },
            ObservableReduction::ProbabilityMoment {
                distribution: law,
                moment: ProbabilityMoment::Expectation,
            },
        ] {
            let mut transaction = Transaction::new("sample provenance admission");
            let mut dependencies = Vec::new();
            for op in original.ops() {
                if let Op::DefineKernelNode {
                    node: KernelNode::Observable(value),
                } = op
                    && Some(value.id().erase()) == symbols.get(selected)
                {
                    transaction.push(Op::DefineKernelNode {
                        node: ObservableDef::new(
                            value.id(),
                            value.value_type().clone(),
                            value.expression().clone(),
                            reduction,
                        )
                        .unwrap()
                        .into(),
                    });
                    if reduction.distribution().is_some() {
                        dependencies.push(Op::Connect {
                            from: value.id().erase(),
                            to: law.erase(),
                            edge: EdgeKind::DependsOn,
                        });
                    }
                } else {
                    transaction.push(op.clone());
                }
            }
            for dependency in dependencies {
                transaction.push(dependency);
            }
            let mut store = InMemoryGraphStore::new();
            store.commit(transaction).unwrap();
            let admitted = KernelProgram::from_snapshot(&store.snapshot(), model);
            if reduction == ObservableReduction::Value
                || matches!(selected, "deterministic" | "deterministic_flux")
            {
                admitted.unwrap();
            } else {
                let errors = admitted.unwrap_err();
                assert!(
                    errors
                        .iter()
                        .any(|error| error.message().contains("realized stochastic State")),
                    "{selected}: {errors:?}"
                );
            }
        }
    }
}

#[test]
fn probability_and_process_admission_use_the_selected_parameter_revision() {
    use eqiora_compiler::{CompiledModel, StaticBindingValue};
    use eqiora_core::{DimExponents, DynQuantity, ValueLiteral};
    use eqiora_schema::kernel::AxisBounds;
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let source = r#"enum Outcome {A,B} model M(support position:interval(m)) {
        parameter weight:1=1;
        parameter density:1/m=1[1/m];
        parameter intensity:1/s=1[1/s];
        distribution finite:normalize_mass(Outcome,weights=[weight,0]);
        distribution uniform:normalize_density(measure(position),density=density);
        noise arrivals:poisson(intensity=intensity,independent=true);
    }"#;
    for (name, units) in [
        ("weight", [0, 0, 0, 0, 0, 0, 0]),
        ("density", [0, -1, 0, 0, 0, 0, 0]),
        ("intensity", [0, 0, -1, 0, 0, 0, 0]),
    ] {
        for replacement in [-1.0, 0.0, 2.0] {
            let compiled = CompiledModel::compile_selected(
                "revision.eqi",
                source,
                "M",
                &[(
                    "position",
                    StaticBindingValue::CoordinateInterval(
                        AxisBounds::new(
                            DynQuantity::new(0.0, length),
                            DynQuantity::new(3.0, length),
                        )
                        .unwrap(),
                    ),
                )],
            )
            .unwrap();
            let (transaction, model, symbols) = compiled.into_parts();
            let mut store = InMemoryGraphStore::new();
            store.commit(transaction).unwrap();
            KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
            let mut edit = Transaction::new("replace a probability Parameter value");
            edit.push(Op::SetValue {
                target: symbols.get(name).unwrap(),
                value: ValueLiteral::try_from(DynQuantity::new(
                    replacement,
                    DimExponents::from_integers(units).unwrap(),
                ))
                .unwrap(),
            });
            store.commit(edit).unwrap();
            let admitted = KernelProgram::from_snapshot(&store.snapshot(), model);
            if replacement > 0.0 || replacement == 0.0 && name == "intensity" {
                admitted.unwrap();
            } else {
                let errors = admitted.unwrap_err();
                assert!(
                    errors
                        .iter()
                        .any(|error| error.message().contains("nonnegative")
                            || error.message().contains("normalization")),
                    "{name}={replacement}: {errors:?}"
                );
            }
        }
    }
}
