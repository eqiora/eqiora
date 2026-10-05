# Independent acceptance criteria

For y'=i y, implicit midpoint gives
(1-i h/2)y_(n+1)=(1+i h/2)y_n. The multiplier has modulus one and angle
2 atan(h/2). At T=1 and N=1/h, the discrete angle is 2 N atan(h/2).
The real oscillator has exactly the same rotation matrix. Initial amplitudes
1, 2 and 3 multiply these formulas; no normalization is an algorithmic step.

For h in {0.1,0.05}, the alternating atan series gives positive phase lag
between h²/13 and h²/12. Halving h yields an error ratio between 3.95 and 4.05;
this interval contains the series prediction with substantial binary64 margin.
Discrete phase/amplitude and restart comparisons allow 1e-11 absolute error,
well below either truncation error. This margin exceeds accumulated binary64
roundoff for the six-coordinate, at-most-20-step fixture and Newton controls
(relative 1e-12, absolute 1e-14), without hiding phase convergence.

The collocation polynomial at the first step midpoint is (y_0+y_1)/2 and has
amplitude 1/sqrt(1+h²/4) for the unit channel. It must remain below 0.9999 for
both selected steps. Projecting samples back to a unit circle fails this test.

Intermediate output times must leave the accepted history and final state
exactly unchanged. Restart at the accepted t=0.5 boundary must reproduce the
uninterrupted t=1 state within 1e-11. These statements concern this method and
fixture, not exact continuous physics at a finite step or arbitrary restarts.
