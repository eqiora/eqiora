# Studio: complete modeling and analysis UX

[Capability matrix](../capability-matrix.md) · [legend](../capability-matrix.md#how-to-read-the-matrix)

Studio is currently a browser preview of one fixed example. It displays Relations and a CAD projection but cannot compile, run, edit CAD, export, or save. The table lists what exists and what is planned.

| Capability | C | X | V | M | Current implementation and planned extensions |
|---|:---:|:---:|:---:|:---:|---|
| Desktop application shell | ⬜ | ⬜ | ⬜ | ⬜ | The GTK3/Tauri shell was retired before 0.1.0 because its runtime dependency closure could not satisfy repository security policy. |
| Typed workflow/command registry | ✅ | ✅ | 🟨 | ⬜ | A single browser registry defines the Relations view, the fixed CAD projection, command scope, disabled reasons, and focus targets. Unit and hosted browser tests cover registry identity and accessibility. Execution, CAD authoring, export, project navigation, plugins, and result workspaces are not implemented. |
| Least-privilege native boundary | ⬜ | ⬜ | ⬜ | ⬜ | Studio has no native runtime or IPC boundary. A future native runtime must use the public facade and only approved platform dependencies. |
| Project/workspace browser | ⬜ | ⬜ | ⬜ | ⬜ | Browser Studio has no project, package, or artifact navigation surface. |
| Source editor | 🟨 | ✅ | 🟨 | ⬜ | A plain source text area with source-span navigation over the fixed preview document. It accepts only the checked-in example; compiler, package workspace, formatting, language-server, and save integration are not implemented. |
| Syntax highlighting | ⬜ | ⬜ | ⬜ | ⬜ | The browser Studio source textarea does not consume the repository TextMate grammar. |
| Completion and signature help | ⬜ | ⬜ | ⬜ | ⬜ | Requires incremental analysis/LSP |
| Hover, go-to-definition, references | ⬜ | ⬜ | ⬜ | ⬜ | Browser Studio does not connect to the language-server navigation surface. |
| Rename and refactoring | ⬜ | ⬜ | ⬜ | ⬜ | Must produce canonical transactions/source edits atomically |
| Quick fixes/code actions | ⬜ | ⬜ | ⬜ | ⬜ | Diagnostic-linked safe transformations remain |
| Semantic outline | ✅ | ✅ | 🟨 | ⬜ | Shows runtime-validated entities from the fixed preview. Live compilation and large-model navigation are not implemented. |
| Relation graph view | ✅ | ✅ | 🟨 | ⬜ | Accessible graph of the fixed preview. Live models, large-graph layout, and editing are not implemented. |
| Block diagram editor | 🟨 | ⬜ | ⬜ | ⬜ | Full port-aware create/connect/reroute/edit UX remains |
| Acausal schematic editor | ⬜ | ⬜ | ⬜ | ⬜ | Conserving connectors, junctions, component symbols, and diagnostics remain |
| Statechart editor | ⬜ | ⬜ | ⬜ | ⬜ | Hierarchy, parallel regions, transitions, events, and simulation animation remain |
| Bounded CAD geometry workspace | ✅ | ✅ | 🟨 | ⬜ | A preview workspace shows one fixed box with an accessible SVG viewport, Domain table, inspector and shared selection; TypeScript and Playwright tests cover selection, keyboard use and accessibility. Not yet: real CAD, CAD authoring, export, Plan replay; [`geometry.cad-semantic-selection-box`](../../verify/geometry/cad-semantic-selection-box/README.md), [RFC 0052](../../rfcs/0052-cad-semantic-selection.md) |
| Geometry viewer | 🟨 | ✅ | 🟨 | ⬜ | The bounded box workspace has fixed accessible 3D presentation and semantic selection. Robust navigation, large geometry, sections, clipping, measurements, and general topology remain |
| Parametric CAD editor | ⬜ | ⬜ | ⬜ | ⬜ | The bounded workspace presents one already-closed design; interactive sketch/feature/constraint editing, history, regeneration controls, and repair remain |
| Mesh viewer | ⬜ | ⬜ | ⬜ | ⬜ | Bulk mesh transfer/rendering, quality overlays, selections, and huge-data LOD remain |
| Mesh generation workflow | ⬜ | ⬜ | ⬜ | ⬜ | Sizing, methods, preview, diagnostics, adaptivity, and provenance remain |
| Physics setup workflow | ⬜ | ⬜ | ⬜ | ⬜ | Browser Studio has no physics setup workflow. |
| Typed property inspector | ✅ | ✅ | 🟨 | ⬜ | Real scalar Parameters can be edited locally in the fixed preview. Committing edits as transactions, nested values, units, expressions, and bulk edits are not implemented. |
| Material editor | ⬜ | ⬜ | ⬜ | ⬜ | Curves, tensors, uncertainty, citations, inheritance, and validation remain |
| Named-selection editor | 🟨 | ✅ | 🟨 | ⬜ | In the fixed CAD preview, the viewport and the keyboard-accessible table share one local selection and inspector. Selection replay, query authoring, reusable definitions, and saving are not implemented. |
| Realization/solver setup | ⬜ | ⬜ | ⬜ | ⬜ | Browser Studio has no realization or solver workflow. The Rust and Python Plan interfaces are listed in their own sections. |
| Capability preview | ⬜ | ⬜ | ⬜ | ⬜ | Browser Studio does not replay realization capability decisions. |
| Device and remote resource picker | ⬜ | ⬜ | ⬜ | ⬜ | General CPU/GPU/MPI/cloud selection and admission diagnostics remain |
| Run lifecycle | ⬜ | ⬜ | ⬜ | ⬜ | Browser Studio does not execute or submit Plans. The shared Rust and Python lifecycle is indexed under solver execution. |
| Immutable exact-cylinder analysis example | ⬜ | ⬜ | ⬜ | ⬜ | The browser Studio does not expose the Python exact-cylinder workflow. |
| Immutable packaged DC-drive trajectory example | ⬜ | ⬜ | ⬜ | ⬜ | The browser Studio does not expose the registered Rust DC-drive workflow. |
| Immutable mixed-boundary structural example | ⬜ | ⬜ | ⬜ | ⬜ | The browser Studio does not expose the Python mixed-boundary structural workflow. |
| Immutable fixed-reference FSI example | ⬜ | ⬜ | ⬜ | ⬜ | The browser Studio does not expose the Python fixed-reference FSI workflow. |
| Progress reporting | ⬜ | ⬜ | ⬜ | ⬜ | Browser Studio has no execution progress channel. |
| Safe cancellation | ⬜ | ⬜ | ⬜ | ⬜ | Browser Studio has no active execution to cancel. |
| Pause/resume/checkpoint controls | ⬜ | ⬜ | ⬜ | ⬜ | Browser Studio has no execution lifecycle controls. |
| Solver monitor | ⬜ | ⬜ | ⬜ | ⬜ | Residual histories, events, nonlinear steps, and live diagnostics remain absent from browser Studio. |
| Scalar trajectory table | ⬜ | ⬜ | ⬜ | ⬜ | Browser Studio has no trajectory result workspace. |
| Time-series chart | ⬜ | ⬜ | ⬜ | ⬜ | Browser Studio has no trajectory result workspace. |
| 2D field visualization | ⬜ | ⬜ | ⬜ | ⬜ | Browser Studio has no result workspace. |
| 3D field visualization | ⬜ | ⬜ | ⬜ | ⬜ | GPU rendering, volume/surface modes, clipping, iso-surfaces, and LOD remain |
| Animation | ⬜ | ⬜ | ⬜ | ⬜ | Time/mode/deformation playback, synchronized plots, and export remain |
| Derived result expressions | ⬜ | ⬜ | ⬜ | ⬜ | Unit-checked postprocessing on immutable result fields remains |
| Probes, paths, surfaces, and integrals | ⬜ | ⬜ | ⬜ | ⬜ | Reproducible result queries and selection provenance remain |
| Comparison of runs | ⬜ | ⬜ | ⬜ | ⬜ | Browser Studio has no run comparison surface. |
| Parameter-sweep dashboard | ⬜ | ⬜ | ⬜ | ⬜ | Experiment definition, scheduling, filtering, plots, and provenance remain |
| Optimization dashboard | ⬜ | ⬜ | ⬜ | ⬜ | Design variables, constraints, convergence, geometry history, and tradeoffs remain |
| Uncertainty dashboard | ⬜ | ⬜ | ⬜ | ⬜ | Distributions, correlations, ensembles, sensitivity, and confidence remain |
| Diagnostic center | ✅ | ✅ | 🟨 | ⬜ | Source-span navigation; cross-domain causal chains and remediation remain |
| Model debugger | ⬜ | ⬜ | ⬜ | ⬜ | Breakpoints, signal/state inspection, event trace, algebraic loops, and stepping remain |
| Equation browser | 🟨 | 🟨 | 🟨 | ⬜ | Relation inspection exists; lowered equations, substitutions, causality, and sparsity remain |
| Dependency/provenance explorer | 🟨 | 🟨 | 🟨 | ⬜ | Digests are visible in bounded flows; navigable artifact lineage remains |
| Undo/redo | ✅ | ✅ | 🟨 | ⬜ | Bounded revision navigation; portable persistent history remains |
| Autosave and crash recovery | 🟨 | 🟨 | 🟨 | ⬜ | Local workspace state exists; robust recovery and conflict handling remain |
| Command palette | ✅ | ✅ | ✅ | ⬜ | Searchable dialog for the main operations and example fixtures, driven by the same registry and availability rules as the navigation and header actions |
| Keyboard-only operation | ✅ | ✅ | ✅ | ⬜ | Current vertical slices tested; complete future surfaces must preserve it |
| Screen-reader semantics | ✅ | ✅ | ✅ | ⬜ | Current automated/interaction evidence; complex diagrams and 3D alternatives remain |
| WCAG 2.2 accessibility | 🟨 | ✅ | ✅ | ⬜ | Bounded surfaces pass evidence; whole-product conformance remains |
| Theme and high-contrast modes | 🟨 | 🟨 | 🟨 | ⬜ | Complete token system and visualization-safe palettes remain |
| Localization | 🟨 | 🟨 | 🟨 | ⬜ | Workflow and command labels, descriptions, and remediation reasons come from a typed fallback catalog. The rest of the UI, diagnostics, unit and number formats, docs, locale loading, and bidirectional text are not localized |
| Extension/plugin system | ⬜ | ⬜ | ⬜ | ⬜ | Sandboxed UI/workflow extensions without semantic forks remain |
| Multi-user collaboration | ⬜ | ⬜ | ⬜ | ⬜ | Presence, comments, semantic merge, permissions, and audit remain |
| Review and approval workflow | ⬜ | ⬜ | ⬜ | ⬜ | Signed model/run review, baselines, deviations, and release gates remain |
| Report builder | ⬜ | ⬜ | ⬜ | ⬜ | Reproducible plots/tables/equations/citations tied to evidence remain |
| Export images/video/data | ⬜ | ⬜ | ⬜ | ⬜ | Not implemented. A general exporter, scene model, accessibility contract, and publication workflow are planned. |
| Offline-first operation | 🟨 | 🟨 | 🟨 | ⬜ | Local app works; packages, docs, licenses, and remote queues need explicit behavior |
| Windows packaging | ⬜ | ⬜ | ⬜ | ⬜ | Signed installer, GPU/MPI dependencies, update and crash reporting remain |
| macOS packaging | ⬜ | ⬜ | ⬜ | ⬜ | Universal/signing/notarization and graphics evidence remain |
| Linux packaging | ⬜ | ⬜ | ⬜ | ⬜ | Supported desktop and distributable formats remain. |
| Automatic updates | ⬜ | ⬜ | ⬜ | ⬜ | Signed channels, rollback, schema compatibility, and admin policy remain |
| UX telemetry by consent | ⬜ | ⬜ | ⬜ | ⬜ | Privacy-preserving opt-in diagnostics and local disablement remain |
| Nonblocking UI under long operations | ⬜ | ⬜ | ⬜ | ⬜ | Browser Studio performs no long-running model operation. |
| Large-model virtualized navigation | ⬜ | ⬜ | ⬜ | ⬜ | Million-node trees/graphs need incremental loading, search, and stable selection |
| Large-result level of detail | ⬜ | ⬜ | ⬜ | ⬜ | Server/GPU-side decimation, tiling, streaming, and fidelity indication remain |
| Unified cross-view selection | 🟨 | ✅ | 🟨 | ⬜ | The fixed CAD viewport and the Domain table share local selection state. Linking Model, Plan, Geometry, and results is not implemented. |
| Cross-filtering between plots and model | ⬜ | ⬜ | ⬜ | ⬜ | Brushing, linked probes, selections, and reproducible filters remain |
| Multi-document and split-view workspace | ⬜ | ⬜ | ⬜ | ⬜ | Independent panes, synchronized selection, comparison, and layout restoration remain |
| Persistent user layout | ✅ | ✅ | 🟨 | ⬜ | Workspace-only layout exists; multi-screen/version migration remains |
| Context-sensitive units and conversions | 🟨 | 🟨 | 🟨 | ⬜ | Coherent SI exists; display units, input conversion, plots, and locale remain |
| Inline equations and dimensional feedback | 🟨 | 🟨 | 🟨 | ⬜ | Relation text and diagnostics exist; rich math rendering and live shape/unit feedback remain |
| Preview before expensive mutation | ⬜ | ⬜ | ⬜ | ⬜ | Browser Studio cannot change a model. |
| Explain-why diagnostics | 🟨 | 🟨 | 🟨 | ⬜ | Structured errors exist; causal explanations, violated assumptions, and remedies remain |
| Contextual theory and help | ⬜ | ⬜ | ⬜ | ⬜ | Versioned equations, assumptions, examples, citations, and direct links remain |
| Guided workflow templates | ⬜ | ⬜ | ⬜ | ⬜ | Expert-authored templates must stay transparent and produce ordinary canonical models |
| Beginner/expert progressive disclosure | 🟨 | 🟨 | 🟨 | ⬜ | Current surfaces are small; complete adaptive complexity without hidden defaults remains |
| Reversible destructive actions | ✅ | ✅ | 🟨 | ⬜ | Revision-based edits exist; delete/import/remesh/package and artifact retention remain |
| Crash-safe model editing | 🟨 | 🟨 | 🟨 | ⬜ | Workspace persistence exists; journaled canonical transactions and recovery testing remain |
| Deterministic visual regression tests | 🟨 | 🟨 | 🟨 | ⬜ | Automated accessibility exists; stable visual baselines across platforms remain |
| Interaction performance budgets | ⬜ | ⬜ | ⬜ | ⬜ | Explicit latency/frame/memory budgets and regression gates remain |
| User research and usability evidence | ⬜ | ⬜ | ⬜ | ⬜ | Task studies, accessibility testing, telemetry-by-consent, and issue linkage remain |

