"""Validate supplement sources, task coverage and final PDF layout."""
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
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--visual-reviewed',action='store_true');args=parser.parse_args()
    tables,constants,sources=build.generated()
    assert len(tables)==11 and sources['main_pins_verified']==108
    assert json.loads((HERE/'report_sources.json').read_text())==sources
    assert (HERE/'report_constants.tex').read_text()==constants
    for name,text in tables.items():assert (HERE/'tables'/name).read_text()==text,name
    for folder in (build.B1,build.B2):
        for name,h in build.read(folder/'output_manifest.json')['files_sha256'].items():assert sha(folder/name)==h,name
    v2=build.read(build.B2/'validation.json')
    assert v2['checker_sha256']==sha(build.ROOT/'experiments/check_s8_fusion_20261109.py')
    assert v2['prior_verifier_failure_retained']
    assert (build.B2/'verification_failures/initial_gate.json').is_file()
    r=build.read(build.B1/'results.json');s=build.read(build.B2/'results.json')
    assert len(r['tasks'])==66 and len(s['tasks'])==24
    assert not r['failures'] and not s['failures']
    methods=set(build.METHODS)
    assert {t['method'] for t in r['tasks']}=={t['method'] for t in s['tasks']}==methods
    expected=[['Mục lục','Phụ lục D'],['Protocol chung','S4','47,50'],
        ['S5 và S6','50,83','100,00'],['S7 và S8','25.344','shared prefix'],
        ['Attacker mạnh hơn','V1','V2','SELECTION'],['Phụ lục A','Brier'],
        ['Phụ lục B','Log loss'],['Phụ lục C','ΔMAE'],['Phụ lục D','Gate kiểm tra đã sửa']]
    all_text=[]
    with pdfplumber.open(HERE/'report_explained.pdf') as pdf:
        assert len(pdf.pages)==len(expected)==9
        assert len(pdf.pages[0].annots)>=17,'Missing contents links'
        for i,(page,terms) in enumerate(zip(pdf.pages,expected),1):
            text=page.extract_text() or '';all_text.append(text)
            assert len(text)>700,('Sparse page',i)
            for term in terms:assert compact(term) in compact(text),(i,term)
            for char in page.chars:
                assert char['x0']>=25 and char['x1']<=page.width-25,('Horizontal clipping',i,char['text'])
                assert char['top']>=10 and char['bottom']<=page.height-15,('Vertical clipping',i,char['text'])
                assert '\ufffd' not in char['text'],('Replacement glyph',i)
    full=compact(''.join(all_text))
    for value in ('47,50','54,17','50,83','57,50','75,00','68,33','98,33','N/A'):
        assert value in full,value
    log=(HERE/'report_explained.log').read_text()
    for bad in ('Overfull \\hbox','Overfull \\vbox','Missing character:','undefined references'):
        assert bad not in log,bad
    rendered=[]
    if args.visual_reviewed:
        rendered=[build.ROOT/'build/remaining_report_20261109'/f'qa-{i}.png' for i in range(1,10)]
        for p in rendered:
            assert p.is_file() and p.stat().st_mtime>=(HERE/'report_explained.pdf').stat().st_mtime,p
    files=[HERE/n for n in ('README.md','report_explained.tex','report_explained.pdf','build_report.py',
        'validate_report.py','requirements.txt','report_constants.tex','report_sources.json')]
    files+=sorted((HERE/'tables').glob('*.tex'))
    receipt=dict(schema='remaining-scenarios-report-validation-v1',date='2026-11-09',
        page_count=9,contents_pages=1,main_pages=4,appendix_pages=4,generated_tables=11,
        main_release_pins_verified=108,tasks_v1=66,tasks_v2=24,all_six_arms_retained=True,
        source_tables_reproducible=True,source_outputs_regenerated=False,
        S4_proxy_and_S8_alignment_disclosed=True,post_v1_attacker_repair_disclosed=True,
        initial_verifier_false_gate_retained=True,scope_NA_not_scored_as_success=True,
        text_bounds_verified=True,no_overfull_boxes=True,visual_qa_completed=args.visual_reviewed,
        visual_qa_scope='all nine pages, linked contents, tables/CI and page boundaries',
        implementation_certified=False,
        files_sha256={str(p.relative_to(HERE)):sha(p) for p in files},
        renders_sha256={p.name:sha(p) for p in rendered})
    (HERE/'report_manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print('PASS:9 pages;11 source-derived tables;90 retained tasks;108 main pins; scope/layout')


if __name__=='__main__':main()
