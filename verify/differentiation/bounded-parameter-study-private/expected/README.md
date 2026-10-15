# Expected observations

Every dispatched occurrence is evaluated once, with no implicit retry. The serial profile
has maximum active depth one; the parallel profile never exceeds its declared worker cap.
Finished numerical payloads under Recompute reach at most the chunk extent and then zero
at ordered delivery boundaries. Full O(n) point, outcome and receipt metadata remains charged.

Failure/cancellation stops further dispatch, not acceptance of already in-flight members.
All outcomes keep original request indices, including successful holes. Empty and final
completion win over later cancellation. Missing, inserted, reordered, foreign or point-
substituted members cannot become complete. Recomputing a different original acceptance evidence fails.
Frozen generator/master/sample/coupling/lineage survives equal points and repeated sample IDs.
No numerical expected coefficient, tolerance, platform timing or heap-size claim is introduced.
