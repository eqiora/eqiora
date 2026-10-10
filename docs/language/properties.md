# Property contracts and releases

These declarations use the existing contract/release owner. Bounded real analytic and
one-dimensional exact table releases execute through source, package, Python and current
Model replay. The [data-backed specimen](data-backed-property.md) derives the table and two
consumer laws independently; its complete target-language consumer surface remains broader
than the implemented scalar and Parameter-bound Component forms.

## Contract

```eqiora
property contract Conductivity(input temperature: K): W / (m * K) {
  derivatives first_open_intervals;
}
```

A contract has a typed independent-input signature and result type. `derivatives` is a
required closed profile, `value_only`, `first_partials`, or `first_open_intervals`.
It specifies what a consumer may request and what domain information a release must provide.
It does not manufacture smoothness from an annotation. The initial contracts are pure and
memoryless; adding history, uncertainty, or another independent input changes the contract.

Contract and release names permit notation immediately after the name. Top-level exported
declarations use the existing `public` visibility modifier. Body privacy and exact imported
identity use the same module rules as components. An output's units alone do not establish
contract compatibility.

## Table release

```eqiora
property release SyntheticConductivity: Conductivity {
  table {
    data conductivity_samples;
    axis temperature: K;
    value conductivity: W / (m * K);
    interpolation piecewise_affine;
    preprocessing identity;
    missing reject;
    knot_derivative reject;
    endpoint_derivative reject;
  }
  validity temperature in [300 [K], 360 [K]];
  outside reject;
  branch single;
  citation synthetic_definition;
  license repository_license;
}
```

`conductivity_samples`, `synthetic_definition`, and `repository_license` are exact package-owned
asset/provenance references supplied by the accepted release closure. They resolve in their
typed asset roles, not the ordinary value namespace. The specimen package binds the first to
the two-column, three-row data described in the consumer page, the second to that synthetic
derivation, and the third to the actual repository license. They are not filenames searched
at execution time, URLs fetched during compilation, or magic globally registered names.

Array assets use the exact bundle role `resolved_array` and paths
`data/<NamePath segments>.json`: `data/curves/Cp.json` binds `curves.Cp` in
its owning package. Local projects inventory that directory under the same bounded,
no-follow file rules as their source tree. The files contain the existing canonical
resolved-array JSON; only verified finite binary64 payloads enter compilation.
The exact bundle identity retains each complete path, role, and file payload.

Attribution assets use the existing `documentation` bundle role and the exact
`docs/<NamePath segments>.md` mapping. `README.md` is ordinary documentation and
has no typed attribution name. Citation and license references retain their owning
package, typed path and role, and SHA-256 of the complete UTF-8 payload already
authenticated by the exact bundle. Only referenced payloads affect release meaning.

The existing package/artifact owner validates content identity, data shape, and exact closure
membership before accepting the release. Missing or foreign references reject. This syntax
adds a typed reference at that owner, not a second artifact loader or generic source `data`
declaration. A package must actually carry those references and assets before this release
can be compiled or published; the names in the example alone do not supply the data.

The table body has exactly one occurrence of every displayed child. `axis` binds the exact
contract input and checks the named data axis; `value` identifies the result column. The
initial profile is one real scalar axis and one real scalar output. It does not infer multiple
axes or complex interpolation coordinates from array shape. Source units normalize through
the shared quantity owner, and the accepted release retains the original unit/preprocessing
meaning rather than just the resulting numbers.

`piecewise_affine` means the unique line joining adjacent tabulated values, with exact knot
values and the explicit derivative policies above. `identity` preprocessing changes nothing;
`missing reject` permits no filling. The closed validity interval and `outside reject` are
required. `branch single` selects this single-branch profile; it is not evidence of a physical
phase or an arbitrary string naming a material.

The release's data coverage must contain the declared validity interval. The example's interval
is exactly its endpoint range. A derivative-required consumer must respect the exposed open
segments; neither consumer nor numerical provider can choose an undocumented knot slope.

## Analytic release

The same release owner accepts an analytic definition instead of a table:

```eqiora
property release LinearConductivity: Conductivity {
  analytic {
    value = 10 [W / (m * K)]
      + 0.2 [W / (m * K^2)] * (temperature - 300 [K]);
  }
  validity temperature in [300 [K], 320 [K]];
  outside reject;
  branch single;
  citation synthetic_definition;
  license repository_license;
}
```

The contract formals are in lexical scope inside `analytic`; a caller's same-named local
cannot capture them. Exactly one of `analytic` and `table` is required. An analytic body
contains one pure `value = expression;`; operators and derivatives use the common expression
graph. A constant-output release still retains its declared inputs and is not automatically
the same contract as a parameter-like constant.

An application may take an evolving real scalar State as an explicit input. Each evaluation
uses that State at the current operating point and checks the declared validity guard.
The property remains memoryless: it acquires no private history or update callback. Focused
product tests compare two heated conductivity consumers with the directly written polynomial
at every reference-integrator sample, including exact offline packages and Model replay.
The formal-input partial holds the other inputs fixed; it is not a total trajectory sensitivity.
Near a validity boundary, reference Newton evaluation may use the inward finite-difference
probe. This does not supply an analytic derivative at the exact comparison boundary: if
initial regularity demands that derivative, initialization still rejects.

This analytic release agrees with the first affine segment of the table on their shared
value domain. It has its own validity and content/provenance identity; matching samples do
not collapse exact releases. Absolute-temperature subtraction, when admitted, uses the
absolute/difference quantity contract rather than applying affine offsets as a multiplicative
unit conversion.

## Grammar and rejection

```text
property-contract = ["public"] "property" "contract" name [notation]
                    "(" input-signature ")" ":" type "{" derivative-profile "}"
property-release = ["public"] "property" "release" name [notation] ":" qualified-name
                   "{" (analytic-body | table-body) validity outside branch citation license "}"
```

Release children use the displayed order for canonical emission; source resolution does not
use their order as an evaluation sequence. Duplicate, missing, or unknown children reject.
Validity expressions refer to the contract's exact independent bindings, not runtime policy.
The initial bounded profile rejects unlisted interpolation, preprocessing, branch, extrapolation,
and derivative modes rather than retaining an open attribute bag for future features.

Changing any data, preprocessing, validity, branch, or interpolation choice changes release
meaning. Substituting a conforming numerical execution provider does not change those choices.
Provider identity remains execution provenance, never a source callback inside the release.
