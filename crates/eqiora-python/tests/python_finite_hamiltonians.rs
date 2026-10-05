use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyModule};
use std::path::Path;

#[test]
fn python_hamiltonian_contracts_execute_replay_and_reject_foreign_handles() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", &module)?;
        py.run(c_str!(r#"
import math
source = """space S=orthonormal(up,down); model M(){
 parameter hbar:J*s=1[J*s];
 parameter H:map<complex<J>,S,S>=linear_map(S,S,[[0,2[J]],[2[J],0]]);
 state psi:coordinates<complex<1>,S>;
 initial {psi=coordinates(S,[math.complex(1,0),0]);}
 relation flow {derivative(psi)=math.complex(0,-1)/hbar*apply(H,psi);}
 observable norm:1=math.real(pair(adjoint(psi),psi));
}"""
model = eqiora.compile(source=source)
psi, h = model.field("psi"), model.parameter("H")
base = eqiora.time.ImplicitMidpoint(step_s=0.01,relative_tolerance=1e-12,
    absolute_tolerances={(psi,0,c,i):1e-14 for c in range(2) for i in (False,True)})
policy = base.with_hermitian_parameter(h).with_conserved_norm(
    [psi], target=1.0,tolerance=1e-10,dimension=eqiora.Dimension())
plan = eqiora.resolve(model,temporal=policy)
original_bytes = plan.to_bytes()
plan = eqiora.Plan.from_bytes(original_bytes)
assert plan.to_bytes() == original_bytes
# Rehydrated policy must retain both declarations when reused by the resolver.
assert eqiora.resolve(model,temporal=plan.temporal).to_bytes() == original_bytes
initial = eqiora.State.initial(plan)
result = eqiora.run(plan,state=initial,until_s=0.2,output_times_s=(0.2,))
state = eqiora.State.from_result(plan,result,time_s=0.2)
state = eqiora.State.from_bytes(plan,state.to_bytes())
angle = 40*math.atan(0.01)
value = state.value(psi)
assert abs(value[0]-math.cos(angle)) < 1e-11
assert abs(value[1]+1j*math.sin(angle)) < 1e-11
assert abs(result.observe_terminal(model.observable("norm")).value-1) < 1e-11
# Contracts are immutable and alter exact Plan/State-space identity.
unconstrained = eqiora.resolve(model,temporal=base)
assert unconstrained.to_bytes() != original_bytes
foreign = eqiora.compile(source=source.replace("2[J]","3[J]"))
for invalid in (
    lambda: eqiora.resolve(model,temporal=base.with_hermitian_parameter(foreign.parameter("H"))),
    lambda: eqiora.resolve(model,temporal=base.with_conserved_norm([foreign.field("psi")],target=1.0,tolerance=1e-10,dimension=eqiora.Dimension())),
    lambda: eqiora.State.from_bytes(unconstrained,state.to_bytes()),
    lambda: base.with_conserved_norm([psi,psi],target=1.0,tolerance=1e-10,dimension=eqiora.Dimension()),
):
    try:
        invalid()
    except (eqiora.ValidationError,TypeError):
        pass
    else:
        raise AssertionError("foreign, repeated or incompatible contract was accepted")

# U is an explicit map between distinct bases; H'=U H U^H, P'=U P U^H.
# Each global phase uses exact multiplication by i, independently of U.
def transformed(rotated, imaginary_phase):
    basis = "T" if rotated else "S"
    h_expr = "compose(U,compose(H0,adjoint(U)))" if rotated else "H0"
    p_expr = "compose(U,compose(P0,adjoint(U)))" if rotated else "P0"
    seed_expr = "apply(U,seed0)" if rotated else "seed0"
    phase = "math.complex(0,1)" if imaginary_phase else "math.complex(1,0)"
    text = f"""space S=orthonormal(up,down); space T=orthonormal(plus,minus);
model M() {{
 parameter hbar:J*s=1[J*s]; parameter r:1=math.sqrt(0.5);
 parameter U:map<complex<1>,S,T>=linear_map(S,T,[[math.sqrt(0.5),math.sqrt(0.5)],[math.sqrt(0.5),-math.sqrt(0.5)]]);
 parameter H0:map<complex<J>,S,S>=linear_map(S,S,[[0,2[J]],[2[J],0]]);
 parameter P0:map<complex<1>,S,S>=linear_map(S,S,[[0,0],[0,1]]);
 parameter seed0:coordinates<complex<1>,S>=coordinates(S,[math.complex(1,0),0]);
 parameter H:map<complex<J>,{basis},{basis}>={h_expr};
 parameter P:map<complex<1>,{basis},{basis}>={p_expr};
 parameter seed:coordinates<complex<1>,{basis}>={phase}*{seed_expr};
 state psi:coordinates<complex<1>,{basis}>;
 initial {{psi=seed;}}
 relation flow {{derivative(psi)=math.complex(0,-1)/hbar*apply(H,psi);}}
 observable probability:1=math.real(pair(adjoint(psi),apply(P,psi)));
 observable norm:1=math.real(pair(adjoint(psi),psi));
 observable energy:J=math.real(pair(adjoint(psi),apply(H,psi)));
}}"""
    m = eqiora.compile(source=text)
    f = m.field("psi")
    temporal = eqiora.time.ImplicitMidpoint(step_s=0.01,relative_tolerance=1e-12,
        absolute_tolerances={(f,0,c,i):1e-14 for c in range(2) for i in (False,True)})
    temporal = temporal.with_hermitian_parameter(m.parameter("H")).with_hermitian_parameter(m.parameter("P"))
    temporal = temporal.with_conserved_norm([f],target=1.0,tolerance=1e-10,dimension=eqiora.Dimension())
    pl = eqiora.resolve(m,temporal=temporal)
    res = eqiora.run(pl,state=eqiora.State.initial(pl),until_s=0.2,output_times_s=(0.2,))
    st = eqiora.State.from_result(pl,res,time_s=0.2)
    expected = (math.cos(angle),-1j*math.sin(angle))
    if rotated:
        a,b = expected
        expected = ((a+b)/math.sqrt(2),(a-b)/math.sqrt(2))
    if imaginary_phase:
        expected = tuple(1j*x for x in expected)
    assert all(abs(a-b)<1e-11 for a,b in zip(st.value(f),expected))
    for name,value in (("probability",math.sin(angle)**2),("norm",1.0),("energy",0.0)):
        actual = res.observe_terminal(m.observable(name)).value
        assert abs(actual-value)<1e-11, (rotated,imaginary_phase,name,actual,value)
    return pl,st

variants = [transformed(b,p) for b in (False,True) for p in (False,True)]
assert len({st.to_bytes() for pl,st in variants}) == 4
for pl,st in variants[1:]:
    try:
        eqiora.State.from_bytes(variants[0][0],st.to_bytes())
    except eqiora.ValidationError:
        pass
    else:
        raise AssertionError("physical equivalence must not erase exact State provenance")

# Mutate only contract bytes, preserving the original identity and all other bytes.
import re
for forged in (
    original_bytes.replace(b'"target":1.0',b'"target":2.0',1),
    re.sub(rb'"hermitian_parameters":\[[^\]]*\]',b'"hermitian_parameters":[]',original_bytes),
):
    assert forged != original_bytes
    try:
        eqiora.Plan.from_bytes(forged)
    except eqiora.ValidationError:
        pass
    else:
        raise AssertionError("contract mutation retained an old Plan identity")
"#),Some(&locals),Some(&locals))
    })
}

fn public_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
    let package_directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/python/eqiora")
        .canonicalize()?;
    let locals = PyDict::new(py);
    locals.set_item("native", native.bind(py))?;
    locals.set_item("package_directory", package_directory.to_string_lossy())?;
    py.run(
        c_str!(
            r#"
import importlib.util
import pathlib
import sys

package_path = pathlib.Path(package_directory)
spec = importlib.util.spec_from_file_location(
    "eqiora",
    package_path / "__init__.py",
    submodule_search_locations=[str(package_path)],
)
assert spec is not None and spec.loader is not None
package = importlib.util.module_from_spec(spec)
sys.modules["eqiora"] = package
sys.modules["eqiora._eqiora"] = native
spec.loader.exec_module(package)
"#
        ),
        None,
        Some(&locals),
    )?;
    Ok(locals
        .get_item("package")?
        .expect("the package loader must bind eqiora")
        .cast_into::<PyModule>()?)
}
