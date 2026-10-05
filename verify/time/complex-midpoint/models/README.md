# Model

At frequency 1/s, z' = i z, z(0) = (1,2), u' = -v, v' = u,
(u(0),v(0)) = (3,0). State values are dimensionless and time is measured in seconds.

The dense-mass variant premultiplies the two complex equations by
M = [[2,1],[1,2]]. Its eigenvalues are 1 and 3, hence it is invertible and retains
the same exact continuous trajectories. It is not a diagonal-mass shortcut.

Both models use the ordinary source compiler and simultaneous initial equations.
The test contains the complete source; there is no second model inventory.
