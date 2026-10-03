"""Verify and record the guide's shared Geo-I evidence after report generation."""
from pathlib import Path
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    evidence = json.loads((OUT/'method_evidence.json').read_text())
    for name, digest in evidence['sources'].items():
        assert sha(ROOT/name) == digest, name
    names = ['report_explained.pdf', 'report_explained.tex', 'method_evidence.json',
             'geoi_configuration.tex', 'geoi_benchmark.tex', 'scenario_appendix.tex',
             'metrics_explained.tex', 'model_architecture.tex',
             'related_work_coverage.tex', 'related_work_coverage.json',
             'geoi_content.py', 'geoi_evidence.py', 'concise_presentation_content.py',
             'preparation_guide.tex', 'data_samples.json', 'report_case_labels.py',
             'figures/architecture_report_flow.pdf', 'prepare_guide_evidence.py']
    receipt = {'updated_on': '2026-10-03', 'new_model_runs': False,
               'purpose': 'Geo-I as main method; identical shared configuration, benchmark and 14-case appendix',
               'sources': {str((OUT/n).relative_to(ROOT)): sha(OUT/n) for n in names},
               'shared_benchmark_tables': evidence['presentation_tables'],
               'source_cases': evidence['source_cases'],
               'aggregation': evidence['aggregation'], 'cost_policy': evidence['cost_policy']}
    (OUT/'preparation_evidence.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    (OUT/'preparation_evidence.tex').write_text('% Geo-I tables are shared via geoi_configuration.tex and geoi_benchmark.tex.\n')
    print('Verified shared Geo-I configuration, 14 cases, v2 comparisons and paired component contrasts.')


if __name__ == '__main__':
    main()
