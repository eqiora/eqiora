# References

- The native primal/JVP/VJP and centered finite-difference reference is
  [`interfaces.python-differentiation`](../../python-differentiation/README.md).
- PyTorch's supported custom-operator path is `torch.library.custom_op` with
  `register_fake`, `register_autograd`, `opcheck`, and `gradcheck`.
- PyTorch 2.14's versioned DLPack consumer composes with Eqiora's immutable
  copy-on-export `Array` contract.
