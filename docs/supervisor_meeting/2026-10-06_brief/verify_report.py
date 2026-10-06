"""Check source pins, links and the reviewed meeting report exports."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

import pymupdf

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


class Reader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.sections = 0
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a' and 'href' in attrs: self.links.append(attrs['href'])
        if tag == 'section': self.sections += 1


def main():
    content = json.loads((OUT/'content.json').read_text())
    evidence = json.loads((OUT/'evidence.json').read_text())
    assert content['build_status'] == 'complete'
    assert content['meeting_date'] == 'not specified'
    assert content['source_sha256'] == evidence['next_meeting_iteration_sources']
    for path, expected in content['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path
    for path, expected in evidence['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path
    # Read the authored prose, avoiding evidence/raw method identifiers.
    prose = json.dumps([content['report'], content['preparation']], ensure_ascii=False)
    assert not re.search(r'PublicCover|S(?:1|2|3|9|10)\.[ABC]|A/B/C|TODO|TBD', prose, re.I)
    assert 'Shadow KNN' in prose and '**ExtraTrees' in prose
    assert len(content['report']['pages']) == 7
    assert len(content['preparation']['pages']) == 3
    checks = {}
    for name, key in [('report_explained','report'),('preparation_guide','preparation')]:
        reader = Reader()
        reader.feed((OUT/f'{name}.html').read_text())
        assert reader.sections == len(content[key]['pages'])
        for link in reader.links:
            if not link.startswith(('https://','http://','#')):
                assert (OUT/link).resolve().exists(), link
        doc = pymupdf.open(OUT/f'{name}.pdf')
        assert len(doc) == len(content[key]['pages']), (name,len(doc))
        chars = 0
        for index, pg in enumerate(doc):
            text = pg.get_text()
            assert '\ufffd' not in text and len(text) > 100, (name,index)
            chars += len(text)
            for block in pg.get_text('blocks'):
                assert block[0] >= 40 and block[2] <= pg.rect.width-40, (name,index,'horizontal clipping')
                assert block[1] >= 12 and block[3] <= pg.rect.height-12, (name,index,'vertical clipping')
        checks[name] = {'pages':len(doc),'unicode_chars':chars,'links':len(reader.links),
                        'html_sha256':hashlib.sha256((OUT/f'{name}.html').read_bytes()).hexdigest(),
                        'pdf_sha256':hashlib.sha256((OUT/f'{name}.pdf').read_bytes()).hexdigest()}
    review = {'status':'pass','checked_on':'2026-10-06',
              'content_sha256':hashlib.sha256((OUT/'content.json').read_bytes()).hexdigest(),
              'source_pins_checked':len(content['source_sha256']), 'exports':checks,
              'preceding_evidence_pins_checked':len(evidence['source_hashes']),
              'visual_review':'PDF pages and vector architecture rendered and inspected locally',
              'browser_preview':'Unavailable: enabled CUA surfaces contain no browser; HTML structure/links checked.',
              'scope':'Source/format checks, not independent confirmation of the research method.'}
    (OUT/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    print('Report checks passed:',len(content['source_sha256']),'source pins; PDF pages 7+3; local links/layout.')


if __name__ == '__main__': main()
