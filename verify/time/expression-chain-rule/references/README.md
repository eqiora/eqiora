# Independent analytic derivation

For f=q²y, the total derivative is 2*q*y*qdot+q²*ydot. At q=3, y=5,
qdot=2 and ydot=3, the two distinct contributions are 60 and 27, giving 87.
The derivative of q² is 12. At t=4, d(t*q)/dt=q+t*qdot=11. A fixed
Parameter has no physical-time contribution, so its polynomial has typed derivative zero.

The storage equation d(q²)/dt=2 and q(0)=1 imply q(t)=sqrt(1+2t) on the
positive branch, independently of either authored implementation. Here q'=1/q and
|q''|=1/q³<=1. On this branch f(q)=1/q is decreasing, so implicit Euler's
error recurrence is contractive. Each local defect is bounded by h²/2, giving an
absolute global bound t*h/2. For h=0.001 and 0<=t<=1, the chosen absolute
acceptance threshold 0.001 exceeds that 0.0005 discretization bound and allows the
separately prescribed numerical residual tolerances. No tolerance is fitted to output.
