# Studio run lifecycle status

The Studio-controlled `ReferenceRunPlan`/`ReferenceRunOutcome` workflow has been removed. Studio
no longer has its own application-specific solver lifecycle, and this page makes no promise about
those removed types.

The product lifecycle is Model-first and independent of the physics:

```text
.eqi -> compile(Geometry) -> resolve(Model, Mesh, typed policies) -> Plan
     -> run(Plan) / submit(Plan) -> Result or State
```

Studio may present artifacts produced through this lifecycle, but it must not infer physics,
rebuild numerical settings, or introduce a second Plan/Run identity. A future Studio execution
interface needs new evidence against the common lifecycle; the removed reference-run API will not
come back.
