"""Validate the four-page architecture/privacy report and source provenance.

Render and inspect every page before --visual-reviewed. Numeric checks neither
certify the floating-point Geo-I stack nor replace the mathematical proof.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import sys

import pdfplumber

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import build_report as build


def compact(text):return re.sub(r'\s+','',text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual-reviewed',action='store_true')
    args=parser.parse_args()
    subprocess.run([sys.executable,str(HERE/'build_report.py'),'--check'],check=True)
    _,sources=build.generated()
    tex=(HERE/'report_explained.tex').read_text()
    assert len(re.findall(r'\\section\{',tex))==2
    assert r'\input{report_tables}' not in tex and 'sample_paths.pdf' not in tex
    expected=[['Kiến trúc mô hình hiện tại','MÔ HÌNH TẠI THIẾT BỊ'],
              ['Đặc tả xác suất và chứng minh privacy','Chiều xuôi'],
              ['Cận Geo-I','Chiều ngược'],['Cận riêng của Q','Bayes','Bản thử cập nhật liên tục']]
    texts=[]
    with pdfplumber.open(HERE/'report_explained.pdf') as doc:
        assert len(doc.pages)==4,'Expected four pages with only two numbered sections'
        for page,titles in zip(doc.pages,expected):
            text=page.extract_text() or '';texts.append(text)
            for title in titles:assert compact(title) in compact(text),'Missing page title: '+title
            for char in page.chars:
                assert char['x0']>=25 and char['x1']<=page.width-25,('Horizontal clipping',char['text'])
                assert char['top']>=10 and char['bottom']<=page.height-15,('Vertical clipping',char['text'])
                assert '\ufffd' not in char['text'],'Replacement glyph'
    full=compact(''.join(texts))
    for value in ('0,24','0,01','0,23','2,437372','70,91','0,55','0,45'):
        assert value in full,'Missing source-derived/example constant: '+value
    for removed in ('Benchmark hiện tại','Suy luận theo scenario','Recall@','Ví dụ nhiều bước','0,0553'):
        assert compact(removed) not in full,'Removed section/metric remains: '+removed
    for label in ('eq:rem','eq:reuse','eq:belief','eq:score','eq:q','eq:remproof',
                  'eq:geosession','eq:budget','eq:postprocessing','eq:qbound',
                  'eq:allowance','eq:whole','eq:minimum','eq:bayes'):
        assert r'\label{'+label+'}' in tex,'Missing mechanism/proof step: '+label
    log=(HERE/'report_explained.log').read_text()
    assert 'Overfull \\hbox' not in log and 'Overfull \\vbox' not in log
    assert 'Missing character:' not in log
    assert not sources['main_release_changed'] and not sources['benchmarks_rescored']
    assert not sources['benchmark_results_in_report'] and not sources['utility_theorems_in_report']
    names=['report_explained.pdf','report_explained.tex','presentation_script.md','README.md',
           'build_report.py','validate_report.py','report_sources.json','privacy_constants.tex']
    result=dict(schema='supervisor-privacy-report-validation-v2',meeting_date='2026-10-17',
        evidence_cutoff='2026-10-14',page_count=4,numbered_sections=2,
        scope='architecture and privacy only',main_release_pins_verified=sources['main_release_pins_verified'],
        source_pins_verified=True,mechanism_and_proof_equations_present=14,
        outward_numeric_bounds_verified=True,page_titles_verified=True,text_bounds_verified=True,
        no_replacement_glyphs=True,no_overfull_boxes=True,no_benchmark_rescoring=True,
        no_benchmark_tables_or_utility_theorems=True,new_production_privacy_certificate_claimed=False,
        continual_trial_kept_separate=True,visual_qa_completed=args.visual_reviewed,
        visual_qa_scope='all four rendered pages: architecture, formulas, proof order, typography and margins',
        files_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names})
    (HERE/'report_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS: four pages, two sections,14 equations, source pins, numeric bounds and layout checks')


if __name__=='__main__':main()
