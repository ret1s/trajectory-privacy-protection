"""Check report provenance, extracted numbers, page count and text bounds.

Visual QA is performed separately on rendered pages. This script does not
turn benchmark scores or kernel assumptions into mathematical certificates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import sys

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import build_report as build


def compact(text):
    return re.sub(r'\s+', '', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual-reviewed', action='store_true',
                        help='Record completed visual inspection of all rendered pages')
    args = parser.parse_args()
    subprocess.run([sys.executable, str(HERE/'build_report.py'), '--check'], check=True)
    tables, sources, sample, _ = build.generated()
    doc = fitz.open(HERE/'report_explained.pdf')
    assert len(doc) == 6, 'Expected six numbered sections/pages'
    titles = ['Phương pháp hiện tại', 'Chứng minh riêng tư', 'Suy luận theo scenario',
              'Ví dụ nhiều bước', 'Benchmark hiện tại', 'Diễn giải kết quả']
    for page, title in zip(doc, titles):
        assert compact(title) in compact(page.get_text()), 'Page/title mismatch'
        for block in page.get_text('dict')['blocks']:
            if block['type'] != 0:
                continue
            for line in block['lines']:
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    assert x0 >= 20 and x1 <= page.rect.width-20, ('Horizontal clipping', span['text'])
                    assert y0 >= 10 and y1 <= page.rect.height-15, ('Vertical clipping', span['text'])
                    assert '\ufffd' not in span['text'], 'Replacement glyph'
    text = compact(doc[4].get_text())
    # Every displayed number in the generated benchmark tables must appear
    # on the benchmark page, including all purposes and all three attack views.
    displayed = re.findall(r'\d+,\d+', tables.split(r'\newcommand{\TimelineRows}')[0])
    for value in displayed:
        assert value in text, 'Missing benchmark number: '+value
    for value in ('2,437372', '0,0553', '0,018', '55,53', '10,53'):
        assert value in compact(''.join(p.get_text() for p in doc)), value
    log = (HERE/'report_explained.log').read_text()
    assert 'Overfull \\hbox' not in log and 'Overfull \\vbox' not in log
    assert sources['derived']['utility_gate'] is True
    assert sources['benchmarks_rescored'] is False and sample['split'] == 'train'
    assert sources['evidence_cutoff'] == '2026-10-14' and sources['meeting_date'] == '2026-10-17'
    names = ['report_explained.pdf', 'report_explained.tex', 'presentation_script.md',
             'build_report.py', 'validate_report.py', 'report_sources.json',
             'report_tables.tex', 'sample_walkthrough.json', 'report_figures/sample_paths.pdf']
    report = dict(schema='supervisor-report-validation-v1', meeting_date=sources['meeting_date'],
        evidence_cutoff=sources['evidence_cutoff'], page_count=len(doc),
        source_pins_verified=True, benchmark_numbers_verified=True,
        page_titles_verified=True, no_overfull_boxes=True,
        text_bounds_verified=True, no_replacement_glyphs=True,
        no_benchmark_rescoring=True, sample_selection=sample['selection'],
        new_production_privacy_certificate_claimed=False,
        visual_qa_completed=args.visual_reviewed,
        visual_qa_scope='all six rendered pages; layout, formulas, tables, architecture and sample illustration',
        files_sha256={n: hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names})
    (HERE/'report_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('PASS: 6 pages, source-derived benchmark numbers, theorem constants, text bounds and TeX overflow checks')


if __name__ == '__main__': main()
