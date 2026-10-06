"""Additive tight export of the unchanged saved-readout scientific figure.

Preserves the original figure builder and earlier outputs. Tight bounding-box
export includes long forest labels; inputs, saved statistics and selection
remain unchanged. Both export and figure-builder source bytes are retained.
"""
import argparse
import json
from pathlib import Path

from experiments import plot_qplanner_response_depth_20261006 as figure


def render(development,output_dir,*,fresh=None):
    prepared,inputs=figure.load_inputs(development,fresh)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
    sources={}
    for name,source in (('export_source_snapshot.py',Path(__file__)),
                        ('figure_source_snapshot.py',Path(figure.__file__))):
        target=output_dir/name;target.write_bytes(source.read_bytes());sources[name]=figure.sha(target)
    fig=figure.plot(prepared);paths=[]
    for suffix in ('pdf','png'):
        path=output_dir/f'response_depth_utility_cost.{suffix}'
        fig.savefig(path,dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.18);paths.append(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    metadata=dict(schema='qplanner-response-depth-tight-scientific-export-v1',inputs=inputs,
        export_source_sha256=figure.sha(Path(__file__)),figure_builder_source_sha256=figure.sha(Path(figure.__file__)),
        exact_source_snapshots_sha256=sources,selected_depth=prepared['selected_depth'],
        development_depths=list(figure.DEPTHS),fresh_included=prepared['fresh'] is not None,
        units=dict(utility='percentage points:100 times saved rate differences',reply_cost='MiB:JSON bytes /2^20'),
        presentation_change='tight bounding box only; original builder, saved statistics and earlier figures preserved',
        intervals='saved95% paired whole-family intervals; no new fitting, scoring, selection or bootstrap',
        raw_coordinates_or_rng_keys_read=False,output_sha256={p.name:figure.sha(p) for p in paths})
    with (output_dir/'figure_provenance.json').open('x') as stream:
        json.dump(metadata,stream,indent=2,allow_nan=False);stream.write('\n')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development',type=Path,required=True)
    parser.add_argument('--fresh',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();render(args.development,args.output_dir,fresh=args.fresh)
    print('Saved tightly bounded scientific PDF/PNG with preserved sources:',args.output_dir)


if __name__=='__main__':main()
