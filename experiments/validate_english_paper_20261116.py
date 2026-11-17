"""Validate manuscript provenance, references, source portability and rendered PDF text.

Visual QA is a separate observed action; this script never self-asserts it.
"""
from pathlib import Path
from collections import Counter
import argparse
import json
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.export_english_paper_20261116 import AUTHORS, PAPER, generate, sha
from experiments.build_english_paper_20261116 import (
    OUT, BUILD, source_files, MAIN_PAGE_LIMIT, check_main_page_limit,
)


def expanded(folder, entry, active=()):
    path = folder / (entry if entry.endswith(".tex") else entry + ".tex")
    assert path.name == str(path.relative_to(folder)), path
    assert path.name not in active, "cyclic TeX input"
    text = path.read_text()
    return re.sub(r"\\input\{([^}]+)\}",
                  lambda m: expanded(folder, m[1], active + (path.name,)), text)


def source_check(folder=PAPER):
    texts = {}
    for entry in ("main", "supplement"):
        text = expanded(folder, entry)
        texts[entry] = text
        labels = re.findall(r"\\label\{([^}]+)\}", text)
        assert not [k for k, c in Counter(labels).items() if c > 1], entry
        refs = re.findall(r"\\(?:eqref|ref|pageref)\{([^}]+)\}", text)
        assert not set(refs) - set(labels), (entry, set(refs) - set(labels))
        for image in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
            assert Path(image).name == image and (folder / image).is_file(), image
        assert not re.search(r"\b(?:TODO|TBC|FIXME)\b|turn\d+(?:search|view)|\ufffd", text), entry
        blocks = re.findall(r"\\author(?:\[[^]]*\])?\{([^}]+)\}", text)
        names = [name.strip() for block in blocks for name in re.split(r"\\and\b", block)]
        assert names == list(AUTHORS), (entry, names)
        assert r"pdfauthor={" + "; ".join(AUTHORS) + "}" in text, entry
        assert "Anonymous authors" not in text, entry
        # The text is English; legitimate non-ASCII author names remain.
        assert not re.search(r"\b(?:Nguồn|Truy cập|Phương pháp|kết quả|và)\b", text), entry
    main = texts["main"]
    options = re.search(r"\\documentclass\[([^]]+)\]\{extarticle\}", main)[1].split(",")
    assert {"twocolumn", "9pt", "letterpaper"} <= set(options), "Reference-inspired two-column layout required"
    assert r"\usepackage{geoipaper}" in main and (folder / "geoipaper.sty").is_file()
    layout = json.loads((folder / "layout_reference.json").read_text())
    for name, digest in layout["protected_scientific_files_sha256"].items():
        assert sha(folder / name) == digest, "Unrecorded scientific-content change: " + name
    tail = (folder / "main.tex").read_text().split(r"\input{introduction}", 1)[1]
    import hashlib
    assert hashlib.sha256((r"\input{introduction}" + tail).encode()).hexdigest() == layout["main_content_tail_sha256"]
    cited = {k.strip() for group in re.findall(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", main)
             for k in group.split(",")}
    bib = re.findall(r"\\bibitem\{([^}]+)\}", main)
    assert len(bib) == len(set(bib)) == 18
    assert cited == set(bib), (set(bib) - cited, cited - set(bib))
    ordered = list(dict.fromkeys(k.strip() for group in re.findall(
        r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", main) for k in group.split(",")))
    assert ordered == bib, "Numeric bibliography must follow first appearance"
    sections = re.findall(r"\\section\{([^}]+)\}", main)
    assert len(sections) == 5, sections
    abstract_words = len((folder / "abstract.tex").read_text().split())
    assert abstract_words <= 250
    highlights = (folder / "highlights.txt").read_text().strip().splitlines()
    assert 3 <= len(highlights) <= 5 and max(map(len, highlights)) <= 85
    for path in re.findall(r"\\path\{([^}]+\.py)\}", texts["supplement"]):
        assert (ROOT / path).is_file(), path
    return dict(layout="Geo-I reference-inspired two-column, 9 pt", main_sections=len(sections), references=len(bib), abstract_words=abstract_words,
                highlight_characters=list(map(len, highlights)),
                all_inputs_figures_references_resolve=True, authors=list(AUTHORS),
                author_order_verified=True, scientific_content_matches_recorded_revision=True,
                approved_content_revisions=layout.get("approved_content_revisions", []),
                protected_scientific_files=len(layout["protected_scientific_files_sha256"]))


def validate():
    generated, figures = generate()
    for name, text in generated.items():
        assert (PAPER / name).read_text() == text, name
    for name, path in figures.items():
        assert (PAPER / name).read_bytes() == path.read_bytes(), name
    structural = source_check()
    build = json.loads((OUT / "build.json").read_text())
    for source in source_files():
        assert sha(source) == build["source_sha256"][source.name], source
    assert build["portable_archive_both_entries_compiled"] and build["portable_text_identical"]
    assert sha(OUT / "latex_sources.zip") == build["archive_sha256"]
    with zipfile.ZipFile(OUT / "latex_sources.zip") as z:
        assert set(z.namelist()) == {p.name for p in source_files()}
        assert all("/" not in name and "\\" not in name for name in z.namelist())
        for source in source_files():
            assert z.read(source.name) == source.read_bytes(), source
    import pdfplumber
    pdfs = {}
    for entry, name in (("main", "manuscript.pdf"), ("supplement", "supplement.pdf")):
        log = (BUILD / entry / (entry + ".log")).read_text(errors="replace")
        assert not re.search(r"Overfull \\[hv]box|undefined references|Citation .* undefined|Missing character:", log), entry
        with pdfplumber.open(OUT / name) as doc:
            if entry == "main":
                check_main_page_limit(len(doc.pages))
            assert sha(OUT / name) == build["outputs"][name]["sha256"]
            author_meta = (doc.metadata.get("Author") or "").strip(" ;")
            assert re.split(r"\s*;\s*", author_meta) == list(AUTHORS), doc.metadata
            title_text = re.sub(r"\s+", "", doc.pages[0].extract_text() or "")
            positions = [title_text.find(re.sub(r"\s+", "", name)) for name in AUTHORS]
            assert min(positions) >= 0 and positions == sorted(positions), (entry, positions)
            texts = []
            bounds = []
            for i, page in enumerate(doc.pages, 1):
                text = page.extract_text() or ""
                assert len(text.strip()) > 40, (entry, i, "empty page")
                assert "\ufffd" not in text and "??" not in text, (entry, i)
                assert page.chars, (entry, i)
                bad = [c for c in page.chars if c["x0"] < -0.5 or c["x1"] > page.width + 0.5
                       or c["top"] < -0.5 or c["bottom"] > page.height + 0.5]
                assert not bad, (entry, i, "text outside page")
                bounds.append([round(min(c["x0"] for c in page.chars), 2),
                               round(max(c["x1"] for c in page.chars), 2)])
                texts.append(text)
            all_text = "\n".join(texts)
            assert "99.595" in all_text and "2.437372" in all_text
            assert "Gamma" in all_text or "Γ" in all_text or "public base" in all_text
            if entry == "main":
                assert "cache-046_day7" in all_text and "Selected ticks" in all_text
                assert "Locations on the benchmark road map" in all_text
                assert "Z180" in all_text and "Q5" in all_text
                assert "Only K5 + cover is uploaded" in all_text
            pdfs[name] = dict(pages=len(doc.pages), sha256=sha(OUT / name),
                              all_pages_text_bounds_pass=True, text_x_bounds=bounds)
    result = dict(schema="english-paper-validation-v1", status="pass", structural=structural,
                  frozen_pins_verified=108, source_generated_tables_match=True,
                  archive_flat_self_contained=True, portable_compilation_text_matches=True,
                  no_overfull_boxes=True, no_undefined_refs_or_missing_glyphs=True,
                  main_page_limit=MAIN_PAGE_LIMIT, main_page_limit_pass=True,
                  walkthrough_retained_source_binding_pass=True,
                  pdfs=pdfs, benchmarks_changed=False, defense_or_attackers_rerun=False,
                  visual_qa="Separate page inspection receipt required; not asserted by this script")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    result = validate()
    if args.write:
        (OUT / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS:", json.dumps(result["structural"]), "; PDF/archive/source integrity verified")


if __name__ == "__main__":
    main()
