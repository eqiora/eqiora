#[test]
fn public_facade_release_identity_is_the_authored_workspace_version() {
    assert_eq!(eqiora::VERSION, env!("CARGO_PKG_VERSION"));
    assert_eq!(eqiora::VERSION, "0.1.0");
}
