"""Authenticate thesis tables, citations and PDF without fitting or sampling."""
from pathlib import Path
import argparse
import hashlib
import json
import re

import pdfplumber
from pdfminer.pdftypes import resolve1

from experiments import export_thesis_consolidated_20261109 as build
from experiments.verify_geoi_bq_mathematics_20261014 import certificate

ROOT = build.ROOT
HERE = build.HERE
PDF = ROOT / 'artifacts/reports/graduation_thesis.pdf'
LOG = ROOT / 'build/thesis/main.log'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(text):
    return re.sub(r'\s+', '', text)


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual-reviewed', action='store_true')
    args = parser.parse_args()
    tables, constants, bibliography, sources = build.generated()
    assert len(tables) == 26
    for name, value in tables.items():
        assert (HERE/'generated'/name).read_text() == value, name
    assert (HERE/'generated/numbers.tex').read_text() == constants
    assert (HERE/'generated/references.tex').read_text() == bibliography
    assert read(HERE/'generated/sources.json') == sources
    assert sources['main_release_pins_verified'] == 108
    assert not sources['models_fitted'] and not sources['defense_outputs_regenerated']
    assert certificate() == read(ROOT/'docs/reviews/2026-10-14_mathematical_completion.json')

    visited = []

    def expand(path):
        assert path.exists(), path
        visited.append(path)
        return re.sub(r'\\input\{([^}]+)\}', lambda m: expand(HERE/(m[1]+'.tex')), path.read_text())

    tex = expand(HERE/'main.tex')
    chapters = re.findall(r'\\chapter\{([^}]+)\}', tex)
    assert chapters[:5] == [
        'Introduction -- Giới thiệu', 'Related Work -- Nghiên cứu liên quan',
        'Proposed Method -- Phương pháp đề xuất',
        'Experiments and Analysis -- Thực nghiệm và phân tích', 'Summary -- Kết luận']
    assert len(chapters) == 9, chapters
    labels = re.findall(r'\\label\{([^}]+)\}', tex)
    assert len(labels) == len(set(labels)), 'Duplicate label'
    references = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', tex)
    assert set(references) <= set(labels), set(references)-set(labels)
    used = set()
    for group in re.findall(r'\\cite\{([^}]+)\}', tex):
        used.update(group.split(','))
    defined = re.findall(r'\\bibitem\{([^}]+)\}', bibliography)
    assert len(defined) == len(set(defined)) == 17
    assert used == set(defined), (used-set(defined), set(defined)-used)
    assert len(re.findall(r'\\begin\{proposition\}', tex)) == 5
    assert len(re.findall(r'\\begin\{theorem\}', tex)) == 3
    assert all(r'\label{'+x+'}' in tex for x in ('fig:system','fig:model','fig:tradeoff',
                    'fig:future','fig:fusion','sec:privacy','sec:utility','thm:transcript'))
    assert all('current_' not in path.name for path in visited)
    assert {HERE/'generated'/name for name in tables} <= set(visited), 'Exported table omitted from thesis'
    assert 'models/' not in tex and '+thấp' not in tex and '+operatorname' not in tex

    texts, annot_count, toc_links, internal_links = [], 0, 0, 0
    with pdfplumber.open(PDF) as pdf:
        assert len(pdf.pages) >= 50, 'Unexpectedly missing thesis content'
        page_ids = {page.page_obj.pageid for page in pdf.pages}
        for index, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ''
            texts.append(text)
            assert len(compact(text)) >= 200, ('Unexpected blank/sparse page', index)
            annots = page.annots or []
            annot_count += len(annots)
            if index in (4,5,6):
                toc_links += len(annots)
            for annot in annots:
                action = resolve1(annot['data'].get('A', {}))
                if getattr(action.get('S'), 'name', None) == 'GoTo':
                    dest = action['D']
                    if isinstance(dest, bytes):
                        dest = resolve1(pdf.doc.get_dest(dest))
                    if isinstance(dest, dict):
                        dest = resolve1(dest['D'])
                    assert dest[0].objid in page_ids, ('Broken PDF destination', index, dest)
                    internal_links += 1
            for char in page.chars:
                assert char['x0'] >= 22 and char['x1'] <= page.width-22, ('Horizontal bound',index,char['text'])
                assert char['top'] >= 10 and char['bottom'] <= page.height-15, ('Vertical bound',index,char['text'])
                assert '\ufffd' not in char['text'], ('Missing glyph', index)
        pages = len(pdf.pages)
    assert toc_links >= 70, 'Missing linked contents entries'
    alltext = compact('\n'.join(texts))
    for value in ('99,595','99,224','99,903','2,437372','70,91','0,23','0,188',
                  '2104,6','47,50','50,83','568,9','392.832','S10','N/A'):
        assert value in alltext, value
    for term in ('Common support','Markov','S4','S5','S6','S7','S8'):
        assert compact(term).lower() in alltext.lower(), term
    log = LOG.read_text()
    for bad in ('Overfull \\hbox','Overfull \\vbox','Missing character:',
                'undefined references','undefined citations','Citation `','LaTeX Error:'):
        assert bad not in log, bad

    renders = []
    if args.visual_reviewed:
        for i in range(1,pages+1):
            path = ROOT/f'build/thesis/render/page-{i:02d}.png'
            assert path.exists(), path
            assert path.stat().st_mtime >= PDF.stat().st_mtime, ('Stale render', path)
            renders.append(path)
    files = set(visited) | {HERE/'README.md',HERE/'REVIEW_GUIDE.md',HERE/'requirements.txt',
        HERE/'references_foundations.json',HERE/'generated/sources.json',
        Path(__file__).resolve(), Path(build.__file__).resolve(),PDF}
    files |= set((HERE/'figures').glob('*.pdf'))
    result = dict(schema='consolidated-masters-thesis-validation-v1',date='2026-11-09',
        page_count=pages, main_chapters=5, appendix_chapters=4, citations=17,
        generated_tables=26, main_privacy_cells=30, additional_tasks_v1=66,
        joint_fusion_tasks_v2=24, main_release_pins_verified=108,
        main_comparison_is_paper_adaptations=True, adaptations_not_official_reproductions=True,
        benchmark_or_algorithm_changed=False, source_tables_reproducible=True,
        exact_mathematical_receipt_authenticated=True, implementation_certified=False,
        all_citations_used_and_resolved=True, labels_resolved=True,
        linked_contents_annotations=toc_links,pdf_annotations=annot_count,
        internal_pdf_destinations_verified=internal_links,
        text_bounds_verified=True,no_missing_glyphs_or_overfull_boxes=True,
        visual_qa_completed=args.visual_reviewed,
        files_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},
        rendered_pages_sha256={p.name:sha(p) for p in renders})
    (HERE/'validation_20261109.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'PASS: {pages} pages;5 chapters;26 tables;17 references;108 pins;math/PDF checks')


if __name__ == '__main__':
    main()
