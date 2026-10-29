# Code generation, embedded, and real-time execution

[Capability matrix](../capability-matrix.md) · [legend](../capability-matrix.md#how-to-read-the-matrix)

| Capability | C | X | V | M | Current implementation and planned extensions |
|---|:---:|:---:|:---:|:---:|---|
| Canonical execution schedule separate from clocks | 🟨 | 🟨 | ⬜ | ⬜ | The sampled-drive case shows that changing the sample period changes the package and Model identity, while changing only the host topology keeps the Model and changes the Run identity. Deployment priorities, deadlines, task lowering, and real-time scheduling are future work; [`hybrid.packaged-dc-motor-controller`](../../verify/hybrid/packaged-dc-motor-controller/README.md) |
| Fixed-step simulation | ✅ | ✅ | ✅ | ⬜ | Reference implicit Euler and periodic semantics; production breadth remains |
| Deterministic discrete scheduler | 🟨 | 🟨 | 🟨 | ⬜ | Exact clocks exist; full priorities, offsets, overruns, and rate transitions remain |
| C code generation | ⬜ | ⬜ | ⬜ | ⬜ | Lowered portable IR, runtime, traceability, and equivalence evidence remain |
| Rust code generation | ⬜ | ⬜ | ⬜ | ⬜ | Static allocation, no-std policy, and equivalence evidence remain |
| LLVM/native compiled execution | 🟨 | 🟨 | 🟨 | ⬜ | Rust CPU lowering exists; optimized generated kernels remain |
| Static memory allocation | ⬜ | ⬜ | ⬜ | ⬜ | Bounded memory plans and rejection of dynamic features remain |
| Fixed-point code generation | ⬜ | ⬜ | ⬜ | ⬜ | Bit-true types, scaling analysis, saturation, and equivalence remain |
| Multirate code generation | ⬜ | ⬜ | ⬜ | ⬜ | Tasks, communication buffers, and deterministic data transfer remain |
| RTOS task mapping | ⬜ | ⬜ | ⬜ | ⬜ | Priority, period, deadline, affinity, and generated integration remain |
| Worst-case execution-time evidence | ⬜ | ⬜ | ⬜ | ⬜ | Target-specific measurement/analysis and assumption tracking remain |
| Processor-in-the-loop | ⬜ | ⬜ | ⬜ | ⬜ | Transport, synchronization, trace, and equivalence remain |
| Software-in-the-loop | ⬜ | ⬜ | ⬜ | ⬜ | Generated/reference conformance and automated scenarios remain |
| Hardware-in-the-loop | ⬜ | ⬜ | ⬜ | ⬜ | Real-time I/O, target deployment, pacing, fault injection, and safety remain |
| Rapid control prototyping | ⬜ | ⬜ | ⬜ | ⬜ | Parameter tuning, instrumentation, deterministic target execution remain |
| AUTOSAR integration | ⬜ | ⬜ | ⬜ | ⬜ | Data types, runnables, scheduling, interfaces, and ARXML remain |
| Functional mock-up export | ⬜ | ⬜ | ⬜ | ⬜ | FMI Scheduled Execution/Co-Simulation packaging and conformance remain |
| Traceability to generated code | ⬜ | ⬜ | ⬜ | ⬜ | Bidirectional model/source/object mapping remains |
| Coverage instrumentation | ⬜ | ⬜ | ⬜ | ⬜ | Model, decision, condition, and generated-code coverage remain |
| Safety-standard qualification path | ⬜ | ⬜ | ⬜ | ⬜ | Tool confidence, qualified workflow evidence, and documentation remain |

