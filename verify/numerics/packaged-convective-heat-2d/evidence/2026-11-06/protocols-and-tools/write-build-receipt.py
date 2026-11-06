from pathlib import Path
import hashlib,json,subprocess,datetime,zipfile
root=Path('/data/nk523/tmp/eqiora-scalar-convection'); repo=root/'wt'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
head=git('rev-parse','HEAD'); assert head=='d3fd0424b7eda8df921a365ae2f4e179c989b7c7'
assert not git('status','--porcelain')
subprocess.run(['git','merge-base','--is-ancestor','10a2bffe23a05497fdd2628c892fd1a93dc65084','HEAD'],cwd=repo,check=True)
installed=json.loads((root/'combined-installed-identity.json').read_text())
assert installed['installed_native_extension_matches_wheel'] and installed['source_shadowing_rejected']
assert all(v['matches_wheel'] for v in installed['native_dependencies'].values())
assert 'not found' not in installed['ldd']
controls=json.loads((root/'frozen-native-no-pde-controls.json').read_text()); assert all(v['exit_code']==0 for v in controls)
getters=json.loads((root/'combined-pre-run-getters.json').read_text()); assert len(getters['cases'])==4
assert '4 passed' in (root/'frozen-python-compile-offline.log').read_text()
assert '21 tests collected' in (root/'frozen-python-collection.log').read_text()
assert 'Finished' in (root/'final-source-clippy.log').read_text()
assert sha(repo/'verify/numerics/cartesian-robin-q1/reference/robin.py')=='200cc1ff5de8327d438506a75e4af59e03b188e730d4fb2b2b653ea06caa507d'
logs=['combined-native-build.log','frozen-native-build.log','frozen-boundary-source-check.log','frozen-native-no-pde-controls.log','final-source-clippy.log','combined-format.log','combined-final-docs.log','combined-oracle-pure.log','combined-wheel-build.log','combined-wheel-build-attempt2.log','combined-wheel-build-known-ida.log','combined-wheel-package-final.log','frozen-wheel-package.log','python-targeted-clean.log','frozen-wheel-fresh-link.log','frozen-wheel-install.log','frozen-installed-inspection.log','frozen-pre-run-getters.log','frozen-python-collection.log','frozen-python-compile-offline.log','combined-pre-run-getters.log','combined-pre-run-getters-corrected.log']
sources=['Cargo.lock','pyproject.toml','crates/eqiora/tests/standard_thermal_package.rs','crates/eqiora/tests/standard_thermal_package/robin.rs','crates/eqiora/tests/standard_thermal_package/robin/observations.rs','bindings/python/tests/test_convective_heat_observations.py','examples/heated-body/src/main.eqi','examples/heated-body/handwritten.eqi','examples/heated-body/run.py','examples/heated-body/eqiora.lock','crates/eqiora-api/packages/Eqiora.Scalar/package.json','crates/eqiora-api/packages/Eqiora.Scalar/src/scalar.eqi','crates/eqiora-api/packages/Eqiora.Scalar/README.md','verify/numerics/cartesian-robin-q1/reference/robin.py']
binary=root/'target/debug/deps/standard_thermal_package-ff1beb6dda49cefe'
receipt={'schema':1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'combined-build-and-no-pde-controls-pass-awaiting-scientific-run-review',
 'source_head':head,'producer_source_head':head,'wheel_build_source_head':head,'tracked_and_untracked_tree_clean':True,
 'approved_core_commit':'10a2bffe23a05497fdd2628c892fd1a93dc65084','approved_core_is_ancestor':True,
 'native_test_executable':str(binary),'native_test_executable_sha256':sha(binary),
 'wheel_and_installed_identity':installed,
 'toolchain':json.loads((root/'combined-build-toolchain.json').read_text()),
 'toolchain_initial_snapshot_source_note':'snapshot made at 4c39 before producer-only deferred-ID/support-alias fixes; final native and wheel outputs are rebuilt from source_head above',
 'patchelf':subprocess.check_output([str(root/'combined-venv/bin/patchelf'),'--version'],text=True).strip(),
 'build_environment':{'TMPDIR':str(root/'scratch'),'CARGO_TARGET_DIR':str(root/'target'),'CARGO_BUILD_JOBS':'2','cpu_affinity':[3,18],
 'profile':'dev','maturin':'1.15.0','pytest':'9.1.1','LIBCLANG_PATH':'/usr/lib/llvm-18/lib',
 'BINDGEN_EXTRA_CLANG_ARGS':'-isystem /usr/lib/gcc/x86_64-linux-gnu/13/include',
 'SUNDIALS_INCLUDE_DIR':'/data/nk523/.cache/eqiora/ida-owned-install/include',
 'SUNDIALS_LIBRARY_DIR':'/data/nk523/.cache/eqiora/ida-owned-install/lib',
 'LD_LIBRARY_PATH_during_build':'/data/nk523/.cache/eqiora/ida-owned-install/lib',
 'LD_LIBRARY_PATH_during_installed_controls':'unset; repaired wheel dependencies resolved from installed wheel'},
 'no_pde_controls':{'native_compile_only':'pass','native_source_and_hex_controls':controls,'targeted_clippy':'pass','format':'pass','documentation_contract':'pass','oracle_sha256':'pass','oracle_observation_pure_tests':7,'python_collection':21,'python_compile_and_moved_offline_cases':4,'python_model_field_observable_domain_reopen_cases':4,'installed_shared_libraries_match_wheel':len(installed['native_dependencies'])},
 'no_imported_scientific_runs':True,'source_sha256':{name:sha(repo/name) for name in sources},
 'logs':{name:{'path':str(root/name),'sha256':sha(root/name)} for name in logs},
 'preparation_attempt_history':[
 {'log':'combined-wheel-build.log','outcome':'failed: missing stddef.h before native build completion'},
 {'log':'combined-wheel-build-attempt2.log','outcome':'failed: older bundled Sundials correctly rejected by 7.8.0 guard'},
 {'log':'combined-wheel-build-known-ida.log','outcome':'native build passed; packaging failed: patchelf missing'},
 {'log':'combined-wheel-package-final.log','outcome':'4c39 wheel build and installed controls passed; preserved in wheels-4c39'},
 {'log':'combined-pre-run-getters.log','outcome':'no-run probe failed: reopened deferred Model requires exact Field ID'},
 {'log':'combined-pre-run-getters-corrected.log','outcome':'no-run probe failed: bound caller Domain uses unprefixed support alias'},
 {'log':'frozen-wheel-package.log','outcome':'cached repackage failed: prior repair mutated target extension SONAMEs; failed wheel retained'},
 {'log':'frozen-wheel-fresh-link.log','outcome':'passed after targeted clean of private eqiora-python outputs and fresh relink'}],
 'scope':'local development build and source/import/archive validation only; no release, performance, GPU, or imported numerical correctness claim'}
(root/'combined-build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'receipt':str(root/'combined-build-receipt.json'),'source':head,'native_executable_sha256':receipt['native_test_executable_sha256'],'wheel_sha256':installed['wheel_sha256'],'extension_sha256':installed['native_extension_sha256'],'native_libraries':len(installed['native_dependencies'])}))
