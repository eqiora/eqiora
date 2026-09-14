# Expected result

The gate accepts exactly one source distribution and four Linux x86-64 wheels
for ordinary-GIL CPython 3.11 through 3.14. The completed v5 manifest binds the
clean source revision, five artifact hashes, exact wheel family, NumPy floor,
and the base, typing, PyTorch, JAX, and Matplotlib profile checks.

Preparation writes only the immutable distribution family. Finalization runs
the installed-wheel profiles without rebuilding the family and writes exactly
one manifest outside it. Any missing profile, substituted byte, extra family
member, or family mutation fails closed.

The release candidate has no notebook-host profile, browser acquisition,
frontend-host receipt, or second metadata artifact. Colab remains a separately
tested example surface and is not part of distribution admission.
