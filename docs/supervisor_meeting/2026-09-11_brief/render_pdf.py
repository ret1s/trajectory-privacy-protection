"""Print the canonical, compiled report; preserve its text, tables and SVG charts."""
import argparse
import json
from pathlib import Path
from hashlib import sha256
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:4183/')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless=True)
    page = browser.new_page(viewport={'width':1280,'height':900},device_scale_factor=1)
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(args.url, wait_until='networkidle')
    page.wait_for_selector('.brief-page-10')
    page.emulate_media(media='print', color_scheme='light', reduced_motion='reduce')
    page.evaluate('document.fonts.ready')
    page.wait_for_function("document.querySelectorAll('.brief-chart svg.recharts-surface').length === 2")
    # Wait two animation frames for print-size SVG geometry; no source changes.
    page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
    geometry = page.locator('.brief-page').evaluate_all('els => els.map(el => ({height:el.getBoundingClientRect().height, width:el.getBoundingClientRect().width, text:el.innerText.length}))')
    charts = page.locator('.brief-chart').evaluate_all('els => els.map(el => ({text:el.innerText, bars:el.querySelectorAll(".recharts-bar-rectangle").length, width:el.getBoundingClientRect().width}))')
    assert not errors, errors
    # Two exact zero differences have labels but no non-zero painted bar.
    assert [c['bars'] for c in charts] == [8,4], charts
    assert all(str(family) in charts[1]['text'] for family in range(307,313))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    page.pdf(path=str(args.output),format='A4',landscape=True,prefer_css_page_size=True,
             print_background=True,display_header_footer=False)
    result = dict(output=str(args.output),pdf_sha256=sha256(args.output.read_bytes()).hexdigest(),runtime_errors=errors,sections=geometry,charts=charts,
                  scope='Headless document conversion; no claim of interactive browser or mobile QA.')
    args.output.with_suffix('.render.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    browser.close()
