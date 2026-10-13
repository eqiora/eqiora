# Frozen Realization wire fixtures

- `realization-v1.json`: current scalar envelope

Each file is exact canonical JSON followed by one repository text newline. The
wire tests remove only that newline, decode with the corresponding exact
version, and require canonical re-encoding to reproduce every committed byte.
