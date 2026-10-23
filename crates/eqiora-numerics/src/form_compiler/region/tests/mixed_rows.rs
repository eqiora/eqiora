use super::*;

#[test]
fn explicit_mixed_blocks_match_independent_mass_stiffness_and_rate_partials() {
    for (renamed, nonlinear) in [(false, false), (true, false), (false, true), (true, true)] {
        let source = "model Mixed() {
            domain body=box(0,1,0,1);
            state u:1 on body; variable v:1 on body;
            relation evolution on body {derivative(u)-2[m^2/s]*div(grad(v))=0;}
            relation auxiliary on body {v+3[m^2]*div(grad(u))=0;}
        }";
        // Rename identifiers only; keywords and relation labels are stable.
        let source = if renamed {
            "model Mixed() {
                domain body=box(0,1,0,1);
                variable alpha:1 on body; state zeta:1 on body;
                relation auxiliary on body {alpha+3[m^2]*div(grad(zeta))=0;}
                relation evolution on body {derivative(zeta)-2[m^2/s]*div(grad(alpha))=0;}
            }"
            .to_owned()
        } else {
            source.to_owned()
        };
        let source = if nonlinear {
            source
                .replace("v+3", "v-(u+u*u*u)+3")
                .replace("alpha+3", "alpha-(zeta+zeta*zeta*zeta)+3")
        } else {
            source
        };
        let (transaction, model, symbols) = compile("mixed-blocks.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let domain = symbols.get("body").unwrap();
        let u = symbols.get(if renamed { "zeta" } else { "u" }).unwrap();
        let v = symbols.get(if renamed { "alpha" } else { "v" }).unwrap();
        let tests = BTreeMap::from([
            (symbols.get("evolution").unwrap(), u),
            (symbols.get("auxiliary").unwrap(), v),
        ]);
        let roles = EquationRoles::derive_with_tests(&program, [domain], &tests).unwrap();
        let form =
            CompiledRegionForm::<f64>::derive_with_roles(&program, domain, 2, None, roles).unwrap();
        let (fields, rows) = form.si_bindings(|_| Ok(p1())).unwrap();
        let bound = form
            .bind(ReferenceCell::simplex(2).unwrap(), &fields, &rows, None)
            .unwrap();
        let offset = |id| {
            bound
                .fields()
                .iter()
                .find(|field| field.field == id)
                .unwrap()
                .range
                .start
        };
        let (ui, vi) = (offset(u), offset(v));
        let mut current = [0.; 6];
        let mut rates = [0.; 6];
        let uv = [1., 2., 4.];
        let vv = [3., -1., 2.];
        let udot = [2., 5., -2.];
        current[ui..ui + 3].copy_from_slice(&uv);
        current[vi..vi + 3].copy_from_slice(&vv);
        rates[ui..ui + 3].copy_from_slice(&udot);
        let prepared = bound
            .prepare_cell(&geometry(), &simplex_duffy_gauss_legendre(2, 3).unwrap())
            .unwrap();
        let mut reaction = [0.; 3];
        let mut reaction_jacobian = [[0.; 3]; 3];
        if nonlinear {
            // u=sum u_a lambda_a. Integrate u+u^3 and its exact variation
            // with the Dirichlet monomial identity, independently of quadrature/AD.
            for i in 0..3 {
                for (j, uj) in uv.iter().enumerate() {
                    reaction[i] += mass(i, j) * uj;
                    reaction_jacobian[i][j] += mass(i, j);
                }
                for (a, ua) in uv.iter().enumerate() {
                    for (b, ub) in uv.iter().enumerate() {
                        for (c, uc) in uv.iter().enumerate() {
                            reaction[i] += ua * ub * uc * moment(&[i, a, b, c]);
                        }
                        for (j, entry) in reaction_jacobian[i].iter_mut().enumerate() {
                            *entry += 3. * ua * ub * moment(&[i, j, a, b]);
                        }
                    }
                }
            }
        }
        // On the unit triangle, M_ij=(1+delta_ij)/24 and
        // K_ij=grad(phi_i).grad(phi_j)/2. Thus R=(M udot+2K v, M v-3K u).
        // Probe both independent partials and their implicit pullback, including zero blocks.
        for (current_weight, rate_weight) in [(1., 0.), (0., 1.), (2., -3.)] {
            let action = prepared
                .implicit_action(
                    &current,
                    Some(&rates),
                    Some(FieldPullback::Uniform {
                        current: Some(current_weight),
                        rate: Some(rate_weight),
                    }),
                )
                .unwrap();
            for (i, gi) in GRADIENT.iter().enumerate() {
                let (mut ru, mut rv) = (0., -reaction[i]);
                for (j, gj) in GRADIENT.iter().enumerate() {
                    let m = mass(i, j);
                    let k = (gi[0] * gj[0] + gi[1] * gj[1]) / 2.;
                    ru += m * udot[j] + 2. * k * vv[j];
                    rv += m * vv[j] - 3. * k * uv[j];
                    close(action.jacobian[(ui + i) * 6 + ui + j], rate_weight * m);
                    close(
                        action.jacobian[(ui + i) * 6 + vi + j],
                        current_weight * 2. * k,
                    );
                    close(
                        action.jacobian[(vi + i) * 6 + ui + j],
                        -current_weight * (3. * k + reaction_jacobian[i][j]),
                    );
                    close(action.jacobian[(vi + i) * 6 + vi + j], current_weight * m);
                }
                close(action.residual[ui + i], ru);
                close(action.residual[vi + i], rv);
            }
        }
    }
}

// Integral over the unit triangle of product(lambda_i^n_i) is
// product(n_i!)/(sum(n_i)+2)!. No numerical assembly output enters this reference.
fn moment(indices: &[usize]) -> f64 {
    let mut powers = [0; 3];
    for &index in indices {
        powers[index] += 1;
    }
    let factorial = |n: usize| (1..=n).fold(1., |value, next| value * next as f64);
    powers.into_iter().map(factorial).product::<f64>() / factorial(indices.len() + 2)
}
