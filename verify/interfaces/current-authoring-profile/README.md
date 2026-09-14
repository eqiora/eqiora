# Current authoring and replay conformance

This case fixes one current-only authoring profile. Rust
`ModelDocument::compile` and `compile_module`, Python `compile` with text or a `Module`, and
the browser Studio request schema all select the same current semantic vocabulary
without accepting a wire or codec argument from the user.

The [public compile schema](../../../crates/eqiora-api/schemas/compile-v2.schema.json)
declares the current Model and Transaction schemas. The registered Rust test
compares source authoring and native definition, checks a quantitative edit and
exact artifact replay, and exercises control-v2 compilation through the current
owner. Schema identifiers are read from that public contract, not a second
version mapping in this case.

Installed-wheel Python tests separately check text/Module compilation and replay against
the same schema. Studio's TypeScript tests check only client constants and rejection. That
companion check runs in the browser client gate and adds no compile capability to
this registered Rust case.

Run:

```bash
cargo test --locked -p eqiora --test current_authoring_profile
cargo run --locked -p eqiora-verify -- run --case interfaces.current-authoring-profile
```
