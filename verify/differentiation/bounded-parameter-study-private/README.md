# Ordered map composition falsifiers

This companion is required with
[`differentiation.bounded-parameter-study`](../bounded-parameter-study/README.md).
It invokes the production-private evaluator and membership admission with independently
accepted members, then injects concrete composition failures.

The exact aggregate test is
`evaluation_map::tests::registered_composition_oracle_executes_all_private_falsifiers`
in `eqiora-api`. It invokes the bounded scheduler and recomputation falsifiers as well as the
serial profile. It proves one call per dispatched request position, including duplicates,
rejects foreign Model/Plan/method/input-order/point members, and rejects missing, inserted or
wrongly reordered members. Declared equal-point occurrences remain valid.

After an observed failure no new tasks are dispatched; already in-flight workers finish
independent acceptance, preserving successful holes by index and original failure diagnostics.
Cancellation names the next undispatched occurrence. Empty completion never polls; final
completion wins; a numerical failure stays failed when cancellation also arrives. Partial
membership cannot be constructed into a complete collection.

Structural input and checked storage-estimate admission precede numerical execution.
The resource probes include overflow, an exact admitted byte limit, one byte below the limit,
and inventories larger than the removed historical 64-point bound.

Condition-variable ordering and counters deterministically force unequal completion order
and observe the admitted concurrent evaluator peak. The production ordering boundary reports
resident accepted payload counts: each `Recompute` chunk reaches its admitted extent and then
zero. Doubling request count charges additional full membership metadata without doubling the
fixed live numerical chunk. These observations make no wall-clock or allocator-memory claim.

Input/output mutation and deliberate previous-buffer substitution cannot retarget another
member. Negative diffusion at a chunk edge leaves other in-flight members independently
usable. Full frozen sample identities survive coupled equal-valued inputs and duplicate IDs;
foreign Program/input order rejects. A substituted original acceptance evidence makes recomputation and
mapped products fail, rather than silently changing accepted lineage.

The previous immutable-oracle/precommitment wording described a superseded contract. This
case is migrated in place with the current API, keeps pointwise scientific evidence unchanged,
and makes no new numerical tolerance or solver claim.

```bash
mise run affected -- \
  --case differentiation.bounded-parameter-study \
  --case differentiation.bounded-parameter-study-private
```
