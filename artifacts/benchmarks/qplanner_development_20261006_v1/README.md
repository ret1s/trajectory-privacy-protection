# Retained serial development prefix

This is an incomplete exploratory run on previously inspected native SUMO
families. It is retained for provenance, not a completed experiment or a fresh
test result.

Serial generation was interrupted to move the same protocol to a bounded,
three-process execution wrapper. The seven completed family/draw bundles were
copied byte-for-byte to `../qplanner_development_20261006_v2/`; the transfer
manifest there records their SHA-256 identities. Already created matching RNG
keys were retained locally outside the repository. SQLite ledgers were not
transferred. Interrupted blocks were regenerated using their existing key, and
no completed block was selected or replaced on the basis of its score.

Use the completed v2 evidence for development selection. This directory has no
completed generation manifest or attack selection and must not be reported as
an independent replication of v2. Private RNG keys and private ledgers are not
included.
