//! Independent companion-row and discrete-adjoint checks from one authored State.
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, DynQuantity, Id, OntologyId, ScalarDomain, ValueType};
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore, Op, Transaction};
use eqiora_ir::{LinearizedRelation, RelationCotangent, RelationTangent};
use eqiora_runtime::{CpuProgram, FirstOrderProgram, GeneralImplicitProgram};
use eqiora_schema::kernel::{
    ActivationDef, ExprDagBuilder, FieldDef, FieldRole, KernelNode, ParameterDef, RelationDef,
    SymbolRef,
};
use eqiora_schema::{Model, ModelView};
use eqiora_sem::{KernelProgram, ReferenceConfig};
use eqiora_time::{
    ImplicitTimeSystem, ReferenceImplicitTimeBackend, TimeMethod, TimePlan, TimeSystem,
};
use std::num::NonZeroU32;

#[test]
fn one_authored_state_projects_to_displacement_and_velocity() {
    for parameter_mass in [false, true] {
        let (kernel, relation, field) = oscillator(parameter_mass);
        let cpu = CpuProgram::lower(&kernel).unwrap();
        if !parameter_mass {
            let system = FirstOrderProgram::lower(&cpu, relation).unwrap();
            assert_eq!(system.state_coordinates(), &[(field, 0), (field, 1)]);
            let initial = system
                .initialize(0.0, ReferenceConfig::new(0., 1.).unwrap())
                .unwrap();
            assert_eq!(initial.state(), &[1., 2.]);
            assert_eq!(initial.derivative(), &[2., -4.]);
            let mut output = [0.; 2];
            system.rhs(0., initial.state(), &mut output).unwrap();
            assert_eq!(output, [2., -4.]);
            system
                .rhs_jvp(0., initial.state(), &[11., 13.], &mut output)
                .unwrap();
            assert_eq!(output, [13., -44.]);
        } else {
            let system = GeneralImplicitProgram::lower(&cpu, relation).unwrap();
            assert_eq!(system.state_coordinates(), &[(field, 0), (field, 1)]);
            let initial = system
                .initialize(0.0, ReferenceConfig::new(0., 1.).unwrap())
                .unwrap();
            assert_eq!(initial.state(), &[1., 2.]);
            assert_eq!(initial.derivative(), &[2., -4.]);
            let mut output = [0.; 2];
            system
                .residual(0., &[3., 2.], &[5., 7.], &mut output)
                .unwrap();
            assert_eq!(output, [19., 3.]); // [a+4*x, x_dot-v]
            system
                .residual_jvp(
                    0.,
                    &[3., 2.],
                    &[5., 7.],
                    &[11., 13.],
                    &[17., 19.],
                    &mut output,
                )
                .unwrap();
            assert_eq!(output, [63., 4.]);

            let step = system
                .linearize_implicit_euler_step(0., 0.5, &[1., 0.], &[3., 2.])
                .unwrap();
            step.primal(&mut output).unwrap();
            assert_eq!(output, [16., 2.]);
            step.jvp(
                RelationTangent::Both {
                    unknown: &[11., 13.],
                    parameter: &[17., 19., 23.],
                },
                &mut output,
            )
            .unwrap();
            assert_eq!(output, [124., -25.]);
            let mut next = [0.; 2];
            let mut previous_and_mass = [0.; 3];
            step.vjp(
                &[5., 7.],
                RelationCotangent::Both {
                    unknown: &mut next,
                    parameter: &mut previous_and_mass,
                },
            )
            .unwrap();
            // G=[m*(v_next-v_prev)/h+4*x_next,
            //    (x_next-x_prev)/h-v_next], h=1/2,m=1.
            assert_eq!(next, [34., 3.]);
            assert_eq!(previous_and_mass, [-14., -10., 20.]);
            let problem = system.implicit_problem().unwrap();
            let plan = TimePlan::new(
                TimeMethod::ImplicitEuler,
                0.,
                0.001,
                1e-11,
                vec![1e-13; 2],
                vec![0.25, 0.5, 1.],
            )
            .unwrap();
            let solution = ReferenceImplicitTimeBackend::new()
                .solve(&problem, &plan)
                .unwrap();
            for (sample, time) in [0.25_f64, 0.5, 1.].into_iter().enumerate() {
                let values = solution.state(sample).unwrap();
                // x(0)=1,v(0)=2: x=cos(2t)+sin(2t),v=2cos(2t)-2sin(2t).
                // Scaled-state norm sqrt(2) gives BE error <=2*sqrt(2)*T*h.
                assert!((values[0] - (2. * time).cos() - (2. * time).sin()).abs() < 0.004);
                assert!(
                    (values[1] - 2. * (2. * time).cos() + 2. * (2. * time).sin()).abs() < 0.008
                );
            }
        }
    }
}

fn oscillator(parameter_mass: bool) -> (KernelProgram, Id<kinds::Relation>, Id<kinds::Field>) {
    let field = Id::new();
    let mass = Id::new();
    let relation = Id::new();
    let initial = Id::new();
    let activation = Id::new();
    let model = OntologyId::<Model>::new();
    let unit = DimExponents::DIMENSIONLESS;
    let per_second = DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap();
    let per_second2 = DimExponents::from_integers([0, 0, -2, 0, 0, 0, 0]).unwrap();
    let mut expression = ExprDagBuilder::new();
    let value = expression.symbol(SymbolRef::Field(field)).unwrap();
    let acceleration = expression
        .symbol(SymbolRef::Derivative(field, NonZeroU32::new(2).unwrap()))
        .unwrap();
    let mass_value = if parameter_mass {
        expression.symbol(SymbolRef::Parameter(mass)).unwrap()
    } else {
        expression.constant(DynQuantity::new(1., unit)).unwrap()
    };
    let inertia = expression.mul(mass_value, acceleration).unwrap();
    let stiffness = expression
        .constant(DynQuantity::new(4., per_second2))
        .unwrap();
    let spring = expression.mul(stiffness, value).unwrap();
    let residual = expression.add(inertia, spring).unwrap();
    let zero = expression
        .constant(DynQuantity::new(0., per_second2))
        .unwrap();
    let mut conditions = ExprDagBuilder::new();
    let x = conditions.symbol(SymbolRef::Field(field)).unwrap();
    let one = conditions.constant(DynQuantity::new(1., unit)).unwrap();
    let v = conditions
        .symbol(SymbolRef::Derivative(field, NonZeroU32::MIN))
        .unwrap();
    let two = conditions
        .constant(DynQuantity::new(2., per_second))
        .unwrap();
    let mut nodes = vec![
        KernelNode::from(FieldDef::new(
            field,
            ValueType::scalar(ScalarDomain::Real, unit).unwrap(),
            FieldRole::State,
        )),
        KernelNode::from(
            RelationDef::new(relation, expression.finish([residual, zero]).unwrap()).unwrap(),
        ),
        KernelNode::from(
            RelationDef::initial(initial, conditions.finish([x, one, v, two]).unwrap()).unwrap(),
        ),
        KernelNode::from(ActivationDef::continuous(activation)),
    ];
    if parameter_mass {
        nodes.push(ParameterDef::new(mass, DynQuantity::new(1., unit).try_into().unwrap()).into());
    }
    let members = nodes.iter().map(KernelNode::id).collect::<Vec<_>>();
    let mut transaction = Transaction::new("native higher-order oscillator");
    for node in nodes {
        transaction.push(Op::DefineKernelNode { node });
    }
    for owner in [relation, initial] {
        transaction.push(Op::Connect {
            from: owner.erase(),
            to: field.erase(),
            edge: EdgeKind::DependsOn,
        });
    }
    if parameter_mass {
        transaction.push(Op::Connect {
            from: relation.erase(),
            to: mass.erase(),
            edge: EdgeKind::DependsOn,
        });
    }
    transaction.push(Op::Connect {
        from: activation.erase(),
        to: relation.erase(),
        edge: EdgeKind::Activates,
    });
    transaction.push(Op::DefineOntologyView {
        view: ModelView::new(model, members, []).unwrap().into(),
    });
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        relation,
        field,
    )
}
