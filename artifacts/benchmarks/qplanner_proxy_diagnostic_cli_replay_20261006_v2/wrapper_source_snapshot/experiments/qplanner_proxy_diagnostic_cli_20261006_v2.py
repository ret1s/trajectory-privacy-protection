"""Relative-output CLI adapter for the immutable OLD-development diagnostic."""
import argparse
from pathlib import Path

from experiments import qplanner_proxy_diagnostic_20261006 as original

ROOT=original.ROOT
THIS='experiments/qplanner_proxy_diagnostic_cli_20261006_v2.py'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/original.OUT,
        help='New output directory; relative paths are resolved against the repository root')
    args=parser.parse_args()
    output=args.output if args.output.is_absolute() else ROOT/args.output
    output=output.resolve()
    original.run(output)
    original.save_new(output/'cli_wrapper_receipt.json',dict(
        schema='qplanner-old-development-diagnostic-cli-adapter-v2',
        wrapper_source=THIS,wrapper_sha256=original.sha(Path(__file__)),
        immutable_analysis_source=original.SOURCE,
        immutable_analysis_source_sha256=original.sha(ROOT/original.SOURCE),
        analysis_protocol_sha256=original.sha(output/'analysis_protocol.json'),
        diagnostic_sha256=original.sha(output/'diagnostic.json'),
        absolute_output_path=str(output),write_once_gate_preserved=True,
        fresh_data_or_scores_opened=False,
        scope='CLI path adapter only; all arithmetic and old-input provenance use the unchanged diagnostic v1'))


if __name__=='__main__':main()
