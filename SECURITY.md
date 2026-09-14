# Security policy

Eqiora is pre-1.0 research software. It must not be used as a safety control,
certification authority, or sole basis for an engineering decision.

## Report a vulnerability

Do not disclose a suspected vulnerability in a public issue. Use
[GitHub private vulnerability reporting](https://github.com/nkiyohara/eqiora/security/advisories/new)
so the maintainer can investigate without exposing users before a remedy is
available.

Include the affected Eqiora version or commit, environment, minimal
reproduction, likely impact, and any known mitigation. We aim to acknowledge a
report within three working days and provide an initial assessment within ten
working days. These are response targets, not guarantees.

Do not include credentials, private infrastructure details, or personal data
beyond what is necessary to reproduce the problem.

## Supported versions

Security fixes target the latest published release and current `main`. Older
releases may be yanked when leaving them available
would materially mislead or endanger users. A correction is published under a
new version; artifacts under an existing version are never replaced. See the
[Python release policy](docs/development/python-release-policy.md).

## Scope

Security-sensitive areas include untrusted project and mesh parsing, package
resolution, artifact integrity, Studio IPC, FFI and unsafe Rust boundaries,
generated code, accelerator adapters, and release infrastructure. Scientific
correctness defects that do not cross a security boundary still belong in the
public issue tracker with a falsifying model whenever safe to disclose.
