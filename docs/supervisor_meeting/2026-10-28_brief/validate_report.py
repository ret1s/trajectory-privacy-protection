"""Authenticate source-derived tables and inspect the final PDF; no scoring."""
from datetime import date
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

import pdfplumber

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_report as build


def compact(value):
    return re.sub(r'\s+', '', value)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual-reviewed', action='store_true')
    args = parser.parse_args()
    tables, constants, bibliography, sources = build.generated()
    assert len(tables) == 14 and sources['main_release_pins_verified'] == 108
    assert (HERE/'report_constants.tex').read_text() == constants
    assert (HERE/'references.tex').read_text() == bibliography
    assert json.loads((HERE/'report_sources.json').read_text()) == sources
    for name, content in tables.items():
        assert (HERE/'tables'/name).read_text() == content, name
    assert digest(HERE/'figures/tradeoff.pdf') == sources['figure_sha256']
    receipt = build.read(build.BENCH/'validation.json')
    for name, value in receipt['files_sha256'].items():
        assert digest(build.BENCH/name) == value, name

    sections = sorted((HERE/'sections').glob('*.tex'))
    tex = (HERE/'report_explained.tex').read_text() + '\n'.join(p.read_text() for p in sections)
    assert re.findall(r'\\section\{([^}]+)\}', tex) == [
        'Experiments and Analysis', 'Introduction', 'Proposed Method', 'Related Work', 'Summary']
    for label in ('fig:system', 'fig:model', 'fig:tradeoff', 'sec:privacy', 'sec:utilityproof'):
        assert r'\label{'+label+'}' in tex, label
    for label in ('rem', 'belief', 'kernel', 'remproof', 'test', 'geosession', 'normalization',
                  'qbound', 'allowance', 'whole', 'postprocessing', 'minimum', 'bayes',
                  'gain', 'cache', 'utilitypath', 'transfer'):
        assert r'\label{eq:'+label+'}' in tex, label
    citations = set()
    for value in re.findall(r'\\cite\{([^}]+)\}', tex + ''.join(tables.values())):
        citations.update(value.split(','))
    assert citations == set(re.findall(r'\\bibitem\{([^}]+)\}', bibliography))
    literature = build.read(HERE/'literature_evidence.json')
    lo, hi = map(date.fromisoformat, literature['window'])
    assert len(literature['papers']) == 8
    assert all(lo <= date.fromisoformat(p['first_public_date']) <= hi for p in literature['papers'])

    expected = [
        ['Introduction', 'Đóng góp chính'], ['Related Work', 'TransProtect', 'SoK'],
        ['Proposed Method', 'System architecture', 'THIẾT BỊ TIN CẬY'],
        ['Model architecture', 'MÔ HÌNH GEOI-BQ', 'L60'],
        ['Algorithm specification', 'cache-029', 'POI cache'],
        ['Privacy analysis', 'Mệnh đề 1', 'Mệnh đề 2'],
        ['Mệnh đề 3', 'Mệnh đề 4', 'Bayes'], ['Utility analysis', 'Public floor'],
        ['Experiments and Analysis', 'Hit100', 'N/A'],
        ['Dataset construction', 'Comparison methods', 'ExtraTrees'],
        ['Main results', 'S10', '99,595'], ['Đánh đổi', 'Reply KB', '155,00'],
        ['Metrics gốc', 'Supporting analysis', '392.832'],
        ['Summary', 'References', '[10]'], ['Phụ lục A', 'CI95%'],
        ['Phụ lục B', 'Trung bình'], ['Phụ lục C', '99,711'], ['Phụ lục D', 'Bảng nối'],
    ]
    full = []
    with pdfplumber.open(HERE/'report_explained.pdf') as pdf:
        assert len(pdf.pages) == len(expected) == 18
        for index, (page, terms) in enumerate(zip(pdf.pages, expected), 1):
            text = page.extract_text() or ''
            full.append(text)
            assert len(text) > 700, ('Unexpected sparse page', index)
            for term in terms:
                assert compact(term) in compact(text), (index, term)
            for char in page.chars:
                assert char['x0'] >= 25 and char['x1'] <= page.width-25, ('Horizontal bounds', index, char['text'])
                assert char['top'] >= 10 and char['bottom'] <= page.height-15, ('Vertical bounds', index, char['text'])
                assert '\ufffd' not in char['text'], ('Replacement character', index)
    text = compact('\n'.join(full))
    for value in ('99,595', '99,224', '99,903', '2,437372', '70,91', '0,23', '0,188', '2104,6'):
        assert value in text, value
    log = (HERE/'report_explained.log').read_text()
    for bad in ('Overfull \\hbox', 'Overfull \\vbox', 'Missing character:',
                'undefined references', 'undefined citations', 'Citation `'):
        assert bad not in log, bad
    renders = []
    if args.visual_reviewed:
        renders = [build.ROOT/'build/paper_report_20261028'/f'qa-{i:02d}.png' for i in range(1, 19)]
        for path in renders:
            assert path.exists() and path.stat().st_mtime >= (HERE/'report_explained.pdf').stat().st_mtime, path
    files = [HERE/n for n in ('report_explained.pdf', 'report_explained.tex', 'README.md', 'build_report.py',
        'validate_report.py', 'requirements.txt', 'literature_evidence.json', 'references.tex',
        'report_constants.tex', 'report_sources.json', 'figures/tradeoff.pdf')]
    files += sections + sorted((HERE/'tables').glob('*.tex'))
    result = dict(schema='paper-comparison-report-validation-v1', date='2026-10-28',
        page_count=18, main_pages=14, appendix_pages=4, main_sections=5, generated_tables=14,
        citations=10, architecture_figures=2, scientific_tradeoff_figures=1,
        main_privacy_cells=30, main_release_pins_verified=108, source_tables_reproducible=True,
        all_arms_complete=True, own_model_resampled=False, report_fits_attackers=False,
        adaptations_not_official_reproductions=True, old_ablation_separate=True,
        text_bounds_verified=True, no_missing_glyphs_or_overfull_boxes=True,
        visual_qa_completed=args.visual_reviewed,
        visual_qa_scope='all 18 final pages, diagrams/arrows, tables, formulas and page breaks',
        implementation_certified_by_report=False, benchmark_sha256=sources['benchmark_sha256'],
        files_sha256={str(p.relative_to(HERE)):digest(p) for p in files},
        rendered_pages_sha256={p.name:digest(p) for p in renders})
    (HERE/'report_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('PASS:18 pages;14 tables;30 main privacy cells;108 pins; sources/citations/layout')


if __name__ == '__main__':
    main()
