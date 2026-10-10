# Retained engineering review notes

- v1: cohort declaration failed because a copied source-pin filename did not exist. No cohort or defense score was produced. Preserve v1 freeze/failure; v2 was frozen before native generation.
- Before v2 freeze: long-session tests found epsilon-certificate outward rounding could cross an allocation by an ulp. Beta calibration now rounds downward until the reported upper bound fits the exact allowance.
- Independent validation: the first checker compared whole certificate records including event timestamps, although cached segment certificates are attached to different public times. The checker was corrected to compare certificate fields excluding t. No mechanism, source snapshot, sampled action or score changed.
- After all v2 scoring/independent review: the older snapshot calibration helper was tightened with exact binary rationals for tiny nonzero oscillation. The segment experiment never called that helper; its segment kernel already used exact-zero/outward calibration. postfreeze_maintenance.json records the differing hashes and unchanged utility-certificate AST. Replay uses the immutable experiment source.
- Utility gate failed for all six prototype arms. Retain every arm and opt-in status; no test-selected Gamma, sample edit, replacement draw or default-engine promotion.
