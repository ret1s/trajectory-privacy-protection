"""Optional browser QA for local replay (requires Playwright + Chromium).

Start web.benchmark_app --port 5051, then run this module (or pass --port).
Screenshots remain
under ignored tmp/, never replace the interactive app.
"""
from pathlib import Path
import argparse
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5051)
    args = parser.parse_args()
    origin = f'http://127.0.0.1:{args.port}'
    Path("tmp").mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        errors, urls = [], []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: urls.append(r.url))
        page.goto(origin + '/report-demo')
        page.wait_for_function("document.getElementById('step-info').textContent.includes('Bước')")
        assert not any(url.endswith("/evaluation") for url in urls)
        page.check("#evaluator")
        page.wait_for_function("document.querySelectorAll('#summary tr').length===120")
        page.select_option("#method", "br_selected")
        page.wait_for_function("document.querySelector('#metrics').textContent.includes('100.0%')")
        page.click("#play")
        page.wait_for_function("Number(document.querySelector('#step').value)>0")
        page.click("#play")
        page.screenshot(path="tmp/report_demo_desktop.png")
        page.select_option('#scenario', 'S9')
        page.wait_for_function("document.querySelector('#metrics').textContent.includes('Sai số điểm biên')")
        page.screenshot(path='tmp/paper_endpoint_desktop.png')
        page.select_option("#scenario", "S2")
        page.select_option("#method", "anotherme_adaptation")
        page.wait_for_function("document.querySelector('#run-status').textContent.includes('Không áp dụng')")
        assert page.locator("#play").is_disabled()
        assert "0.0%" not in page.locator("#metrics").inner_text()
        page.select_option("#method", "dls_graph_adaptation")
        page.wait_for_function("!document.querySelector('#play').disabled")
        page.uncheck("#evaluator")
        assert page.locator("#results").is_hidden()
        assert page.locator("#metrics").inner_text() == ""
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path="tmp/report_demo_mobile.png", full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert not errors, errors
        assert all(url.startswith(origin + '/') for url in urls), urls
        print({"page_errors": errors, "external_requests": 0, "checks": [
            "initial public-only", "evaluator opt-in", "120 summary rows", "BR selected", "playback", "hidden endpoint",
            "AnotherMe not-applicable", "DLS", "hide evaluator", "390px no overflow"]})
        browser.close()


if __name__ == "__main__":
    main()
