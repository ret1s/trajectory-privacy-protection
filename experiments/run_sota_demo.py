"""Deprecated compatibility entry point for the dummy-generation benchmark.

Use ``python -m experiments.run_dummy_benchmark`` in new commands.  This
module deliberately re-exports the previous public helpers so existing review
scripts and old artifact recipes remain runnable during the migration.
"""

from experiments.run_dummy_benchmark import *  # noqa: F401,F403
from experiments.run_dummy_benchmark import parse_args, run


if __name__ == "__main__":
    run(parse_args())
