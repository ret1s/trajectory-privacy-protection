"""
DEPRECATED single-home averaging demo.

The single-home / single-seed / naive-sample-mean experiment was superseded by
the rigorous multi-home study, which uses the CORRECT (consistent MLE) adversary
and reports bootstrap confidence intervals over many real stay-point homes.
Reason: against the naive sample mean the exponential mechanisms look like they
"resist" averaging, but that is an estimator artefact (E[Z|x] != x); the MLE
adversary shows they do not. See:

    python3 -m experiments.run_averaging_multi

This shim forwards to it so no stale single-home numbers are produced.
"""
from experiments.run_averaging_multi import run

if __name__ == "__main__":
    run()
