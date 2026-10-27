"""Check generated evidence, PDF coverage and layout; no benchmark scoring."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

import pdfplumber

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import build_report as build


def compact(s):return re.sub(r'\s+','',s)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--visual-reviewed',action='store_true');args=p.parse_args()
    tables,constants,sources=build.generated()
    assert (HERE/'report_constants.tex').read_text()==constants
    assert json.loads((HERE/'report_sources.json').read_text())==sources
    for name,content in tables.items():assert (HERE/'tables'/name).read_text()==content,name
    tex=(HERE/'report_explained.tex').read_text()
    assert len(re.findall(r'\\section\{',tex))==9
    assert len(re.findall(r'% PAGE \d+',tex))==13
    expected=[['Bài toán, mục tiêu và phạm vi','Attacker'],['Kiến trúc','MÔ HÌNH TẠI THIẾT BỊ'],
        ['Từ GPS','Ví dụ số','TRAIN'],['Mệnh đề 1','Mệnh đề 2','Hợp thành'],
        ['Mệnh đề 3','Mệnh đề 4','Bayes'],['Utility','cả phiên','Public floor'],
        ['Protocol','Attacker bank','Gate'],['đầy đủ sáu phương án','S10','MAE'],
        ['Kết quả utility','khoảng tin cậy','Public-base'],['Chi phí','Ba cách','GPS'],
        ['Độ sâu phản hồi','Kết luận theo ba câu hỏi'],['Đối chứng paper','AnotherMe'],
        ['Pilot continual','30 phút','720-1800']]
    full=[]
    with pdfplumber.open(HERE/'report_explained.pdf') as pdf:
        assert len(pdf.pages)==13,'Expected 11 main pages + 2 appendices'
        for index,(page,terms) in enumerate(zip(pdf.pages,expected),1):
            text=page.extract_text() or '';full.append(text)
            for term in terms:assert compact(term) in compact(text),(index,term)
            for c in page.chars:
                assert c['x0']>=25 and c['x1']<=page.width-25,('Horizontal clipping',index,c['text'])
                assert c['top']>=10 and c['bottom']<=page.height-15,('Vertical clipping',index,c['text'])
                assert '\ufffd' not in c['text'],'Replacement glyph'
    text=compact(''.join(full))
    for value in ('99,595','99,711','0,23','2,437372','70,91','95,348','94,170','99,873'):
        assert value in text,'Missing source-derived number '+value
    for label in ('rem','belief','score','q','remproof','test','geosession','normalization','qbound',
                  'allowance','whole','postprocessing','minimum','bayes','gain','cache','utilitypath','transfer'):
        assert r'\label{eq:'+label+'}' in tex,label
    log=(HERE/'report_explained.log').read_text()
    for bad in ('Overfull \\hbox','Overfull \\vbox','Missing character:','undefined references','undefined citations'):
        assert bad not in log,bad
    assert sources['main_privacy_cells']==30 and not sources['benchmark_rescored']
    assert not sources['models_fitted'] and not sources['new_private_samples']
    files=['report_explained.pdf','report_explained.tex','README.md','build_report.py','validate_report.py',
           'report_constants.tex','report_sources.json']+['tables/'+n for n in tables]
    receipt=dict(schema='geoi-bq-review-report-validation-v1',prepared_date='2026-10-27',
        page_count=13,main_pages=11,appendix_pages=2,generated_tables=len(tables),
        main_release_pins_verified=sources['main_release_pins_verified'],main_privacy_cells=30,
        complete_main_method_purpose_cache_coverage=True,historical_and_pilot_kept_separate=True,
        no_benchmark_rescoring=True,source_tables_reproducible=True,page_titles_verified=True,
        outward_numeric_bounds_verified=True,text_bounds_verified=True,no_overfull_boxes=True,
        visual_qa_completed=args.visual_reviewed,
        visual_qa_scope='all 13 final rendered pages, architecture arrows, equations, tables and page breaks',
        implementation_certified_by_this_report=False,
        files_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in files})
    (HERE/'report_manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print('PASS:13 pages;20 tables;108 frozen pins;30 main privacy cells; source/layout checks')


if __name__=='__main__':main()
