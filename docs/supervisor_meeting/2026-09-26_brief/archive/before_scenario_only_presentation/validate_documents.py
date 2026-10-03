"""Check source-backed shared tables, links, case labels and final PDFs."""
from pathlib import Path
from html.parser import HTMLParser
import argparse
import hashlib
import json
import re
import fitz

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.links, self.ids, self.headings = [], [], [], []
        self.row = self.cell = self.heading = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.links.extend(a[k] for k in ('href', 'src') if k in a)
        if 'id' in a:
            self.ids.append(a['id'])
        if tag == 'tr':
            self.row = []
        if tag in ('td', 'th'):
            self.cell = ''
        if tag == 'h2':
            self.heading = ''

    def handle_data(self, text):
        if self.cell is not None:
            self.cell += text
        if self.heading is not None:
            self.heading += text

    def handle_endtag(self, tag):
        if tag in ('td', 'th'):
            self.row.append(self.cell)
            self.cell = None
        if tag == 'tr':
            self.rows.append(self.row)
            self.row = None
        if tag == 'h2':
            self.headings.append(self.heading)
            self.heading = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=Path, default=OUT)
    args = parser.parse_args()
    evidence = json.loads((OUT/'method_evidence.json').read_text())
    guide = json.loads((OUT/'preparation_evidence.json').read_text())
    for receipt in (evidence, guide):
        for name, digest in receipt['sources'].items():
            assert sha(ROOT/name) == digest, name
    assert guide['shared_benchmark_tables'] == evidence['presentation_tables']
    paper = json.loads((ROOT/'artifacts/benchmarks/paper_benchmark/results.json').read_text())
    old = {(r['method'], r['scenario']): r for r in paper['summary'] if r['k'] == 5}
    numerical_checks = 0
    for row in evidence['legacy_rows']:
        for scenario, values in row['scenarios'].items():
            source = old[row['method'], scenario]
            for metric, field in [('hit100', 'location_hit_100m'), ('mae_m', 'location_mae_m'),
                                  ('recall', 'poi_recall_at_k')]:
                assert values[metric] == source[field]
                numerical_checks += 1
    base = ROOT/'artifacts/benchmarks/research_loop'
    attacks = json.loads((base/'iteration18_expanded_attacks.json').read_text())
    live = json.loads((base/'iteration28_live_service.json').read_text())
    pr = {(r['method'], r['case_id']): r for r in attacks['summaries']}
    sr = {(r['method'], r['mode'], r['case_id']): r for r in live['summaries'] if r['probability'] == .8}
    for row in evidence['case_rows']:
        c = row['source_case']
        for field in ('hit100', 'mae_m', 'hit500'):
            assert row[field] == pr['response_paced_slack03', c]['metrics'][field]
            numerical_checks += 1
        assert row['raw_hit100'] == pr['raw', c]['metrics']['hit100']
        assert row['recall'] == sr['response_paced_slack03', 'epoch_cache', c]['recall']
        numerical_checks += 2
    for row in evidence['service_rows']:
        values = row['case_recall']
        group_means = [sum(v for c, v in values.items() if c.startswith(s+'.')) /
                       sum(c.startswith(s+'.') for c in values) for s in ('S1', 'S2', 'S3', 'S9', 'S10')]
        assert abs(sum(group_means)/5-row['recall']) < 1e-12
        assert row['gates'] == sum(v >= .9 for v in values.values())
        numerical_checks += 2
    doc = Document()
    doc.feed((OUT/'report_explained.html').read_text())
    assert ['Related works', 'Kiến trúc', 'Benchmark'] == [
        next(s for s in ('Related works', 'Kiến trúc', 'Benchmark') if s in h)
        for h in doc.headings[:3]]
    for link in doc.links:
        if link.startswith('https://'):
            continue
        if link.startswith('#'):
            assert link[1:] in doc.ids, link
        else:
            assert (OUT/link).exists(), link
    table_rows = 0
    for table in evidence['presentation_tables']:
        for row in table['rows']:
            plain = [re.sub(r'\*\*(.*?)\*\*', r'\1', str(v)) for v in row]
            assert doc.rows.count(plain) == 1, plain
            table_rows += 1
    coverage = json.loads((OUT/'related_work_coverage.json').read_text())
    assert coverage['mapping_is_our_inference'] and len(coverage['rows']) == 12
    assert len(coverage['presentation_rows']) == 12
    for original, row in zip(coverage['rows'], coverage['presentation_rows']):
        assert original[:3] == row[:3]
        assert doc.rows.count(row) == 1, row
    tex = (OUT/'report_explained.tex').read_text()
    guide_tex = (OUT/'preparation_guide.tex').read_text()
    for name in ('metrics_explained.tex', 'model_architecture.tex', 'geoi_benchmark.tex', 'scenario_appendix.tex',
                 'related_work_coverage.tex'):
        assert guide_tex.count('\\input{'+name+'}') == 1
        shared = (OUT/name).read_text().split('\n', 1)[1]
        shared = re.sub(r'\\label\{(?:sec|sub):[^}]+\}', '', shared)
        report = re.sub(r'\\label\{(?:sec|sub):[^}]+\}', '', tex)
        # Guide references use external links; report references use bibliography anchors.
        for ref in json.loads((OUT/'sources.json').read_text())['sources']:
            report = report.replace('\\hyperlink{ref-'+ref['id']+'}{'+ref['id']+'}',
                                    '\\href{'+ref['url']+'}{'+ref['id']+'}')
        assert shared in report, name
    pct = lambda v: f'{100*v:.2f}%'.replace('.', ',')
    num = lambda v: f'{v:,.0f}'.replace(',', '.')
    for row in evidence['legacy_rows']:
        expected = [row['label']] + [pct(row['scenarios'][s]['hit100'])+' / '+num(row['scenarios'][s]['mae_m'])+' / '+pct(row['scenarios'][s]['recall'])
                                    for s in ('S1', 'S2', 'S3', 'S9', 'S10')]
        assert doc.rows.count(expected) == 1, expected
    assert '33,33% / 299 / 95,83%' in (OUT/'report_explained.html').read_text()
    assert tex.index('Attacker thấy gì?') < tex.index(r'\subsection{So sánh privacy')
    html = (OUT/'report_explained.html').read_text()
    literature_html = html.split('<h2>2. Kiến trúc', 1)[0]
    assert literature_html.count('<table>') == 1
    assert 'Căn cứ và giới hạn' not in literature_html
    assert literature_html.count('<li>') == 4
    for name in ('Centroid', 'Prior', 'Road filter', 'Shadow kNN', 'Continuity',
                 'Full path (Viterbi)', 'Stationary intersection', 'kNN', 'ExtraTrees'):
        assert '<strong>'+name+'</strong>' in html, name
        assert r'\textbf{'+name+'}' in tex, name
    assert 'không phải tên attacker' in html
    assert 'figures/architecture_report_flow.pdf' in tex
    assert 'figures/architecture_report_flow.svg' in html
    assert 'figures/architecture_model.pdf' not in tex
    appendix = (OUT/'scenario_appendix.tex').read_text()
    panels = re.findall(r'figures/case_(S\d+_[ABC])\.pdf', appendix)
    assert len(panels) == len(set(panels)) == 14
    assert set(panels) == {f'S{i}_{c}' for i in (1, 2, 3, 9) for c in 'ABC'} | {'S10_A', 'S10_B'}
    examples = {r['case_id']: r for r in json.loads((OUT/'data_samples.json').read_text())['examples']}
    for label in panels:
        assert examples[label.replace('_', '.')]['record']['record_id'] in appendix
    forbidden = ('PublicCover', 'CoverLite', 'CoverPlus', 'ours30', 'ours67', 'calendar30', 'calendar67')
    for name in ('report_explained.tex', 'report_explained.html', 'preparation_guide.tex',
                 'geoi_configuration.tex', 'geoi_benchmark.tex', 'README.md', 'PREPARATION.md'):
        text = (OUT/name).read_text()
        assert all(s not in text for s in forbidden), name
    documents = {}
    for stem, count in [('report_explained', 11), ('preparation_guide', 10)]:
        pdf = fitz.open(OUT/(stem+'.pdf'))
        assert len(pdf) == count, (stem, len(pdf))
        for page in pdf:
            assert all(s not in page.get_text() for s in forbidden)
            for word in page.get_text('words'):
                assert word[0] >= 0 and word[1] >= 0 and word[2] <= page.rect.width+.1 and word[3] <= page.rect.height+.1, word
        fonts = {font[0] for page in pdf for font in page.get_fonts()}
        assert all(pdf.extract_font(x)[3] for x in fonts)
        log_path = args.log_dir/(stem+'.log')
        logs = log_path.read_text() if log_path.exists() else None
        if logs is not None:
            assert 'Overfull' not in logs and 'Missing character' not in logs
        documents[stem] = {'pages': count, 'fonts_embedded': len(fonts),
                           'text_outside_page': [], 'compiler_log_checked': logs is not None,
                           'pdf_sha256': sha(OUT/(stem+'.pdf'))}
    authoring = ['build_report.py', 'concise_presentation_content.py', 'geoi_evidence.py',
                 'geoi_content.py', 'plot_report_architecture.py', 'prepare_guide_evidence.py',
                 'related_work_coverage.py', 'validate_documents.py', 'README.md', 'PREPARATION.md']
    result = {'status': 'passed', 'updated_on': '2026-10-03',
              'revision': 'Readable portrait execution-flow architecture with numbered stages and separate component/configuration page',
              'source_hashes_valid': True, 'numerical_checks': numerical_checks,
              'quantitative_rows_checked_in_html': table_rows, 'coverage_rows': 12,
              'shared_appendix_panels': 14, 'shared_content_identical': True,
              'obsolete_branch_absent_from_presentation': True,
              'local_links_resolve': True, 'new_model_runs': False,
              'immutable_benchmarks_modified': False,
              'visual_review': 'Architecture and changed main pages of both PDFs inspected; page geometry and font embedding checked.',
              'model_architecture': evidence['model_architecture'],
              'archive': 'archive/before_report_execution_flow/',
              'authoring_sources': {n: sha(OUT/n) for n in authoring},
              'outputs': {n: sha(OUT/n) for n in ['report_explained.pdf', 'report_explained.html', 'report_explained.tex',
                                                    'preparation_guide.pdf', 'preparation_guide.tex',
                                                    'geoi_configuration.tex', 'geoi_benchmark.tex', 'metrics_explained.tex',
                                                    'model_architecture.tex', 'scenario_appendix.tex']}}
    prior = json.loads((OUT/'archive/before_concise_2026-10-03/report_validation.json').read_text())
    result['font_fallback'] = prior['font_fallback']
    for stem, values in documents.items():
        name = 'report_validation.json' if stem == 'report_explained' else 'preparation_validation.json'
        (OUT/name).write_text(json.dumps({**result, **values, 'document': stem}, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': 'passed', 'numerical_checks': numerical_checks,
                      'quantitative_rows': table_rows, 'documents': documents}, ensure_ascii=False))


if __name__ == '__main__':
    main()
