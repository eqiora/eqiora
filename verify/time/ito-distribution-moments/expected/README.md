`campaign.json` fixes the complete sample design, grid counts and analytical
reference values before candidate execution. `../references/analytic.py`
reproduces it from Gaussian/lognormal identities without importing Eqiora.
Candidate means, variances and path errors are never copied into this file.
