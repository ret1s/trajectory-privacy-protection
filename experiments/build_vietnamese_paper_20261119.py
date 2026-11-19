"""Build the separate Vietnamese HK261 review paper without changing English inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.build_english_paper_20261116 import compile_one
from experiments.validate_english_paper_20261116 import expanded

EN = ROOT / "paper_hk261"
PAPER = ROOT / "paper_hk261_vi"
OUT = ROOT / "artifacts/papers/geoi_bq_hk261_vi_20261119_v1"
BUILD = ROOT / "build/paper/hk261_vi_20261119"
PROSE = ("abstract", "introduction", "related_work", "method", "analysis", "evaluation", "conclusion")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_files():
    return sorted(p for p in PAPER.iterdir() if p.is_file())


def original_hashes():
    folders = (EN, ROOT / "paper", ROOT / "artifacts/papers/geoi_bq_hk261_20261119_v1",
               ROOT / "artifacts/papers/geoi_bq_20261116_v1",
               ROOT / "artifacts/figures/paper_walkthrough_20261117_v1")
    return {str(p.relative_to(ROOT)): sha(p) for folder in folders for p in folder.iterdir() if p.is_file()}


def math_tokens(text):
    # Compare all inline and display math, including formulas embedded in prose.
    pattern = r"\$(?:[^$]|\\\$)*\$|\\begin\{(equation|align)\}.*?\\end\{\1\}"
    return [re.sub(r"\s+", "", m[0]) for m in re.finditer(pattern, text, re.S)]


def source_check(folder=PAPER):
    for name in ("llncs.cls", "splncs04.bst", "references.bib", "author_info.json", "template_origin.json"):
        assert (folder / name).read_bytes() == (EN / name).read_bytes(), name
    for stem in PROSE:
        original = (EN / (stem + ".tex")).read_text()
        translated = (folder / (stem + ".tex")).read_text()
        assert math_tokens(original) == math_tokens(translated), (stem, "Mathematics changed")
        for command in ("label", "cite", "ref"):
            pattern = r"\\" + command + r"\{([^}]+)\}"
            assert re.findall(pattern, original) == re.findall(pattern, translated), (stem, command)
        # Decimal-valued constants and results must survive translation verbatim.
        pattern = r"(?<![\w.])\d+\.\d+(?![\w.])"
        assert re.findall(pattern, original) == re.findall(pattern, translated), (stem, "Decimal values changed")
    for stem in ("table_privacy", "table_service"):
        original = (EN / (stem + ".tex")).read_text()
        translated = (folder / (stem + ".tex")).read_text()
        numeric = lambda t: re.findall(r"\d+\.\d+", t.split(r"\midrule")[1].split(r"\bottomrule")[0])
        assert numeric(original) == numeric(translated), stem
    text = expanded(folder, "main")
    assert r"\usepackage{fontspec}" in text and "texgyretermes" in text
    assert re.findall(r"\\section\{([^}]+)\}", text) == [
        "Giới thiệu", "Nghiên cứu liên quan", "Phương pháp đề xuất", "Thực nghiệm và phân tích", "Kết luận"]
    assert "Data and code availability" not in text
    assert "Declaration of generative AI assistance" not in text
    assert "Tác giả liên hệ:" in text and "nmhanh.sdh241@hcmut.edu.vn" in text
    assert "Hanh Ngo-Minh" in text and "Phan Trong Nhan" in text
    return {"math_identical_to_english": True, "result_cells_identical": True,
            "references_and_author_info_identical": True, "labels_and_citations_preserved": True,
            "scientific_prose_files_checked": len(PROSE)}


def make_archive():
    target = OUT / "latex_sources.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in source_files():
            entry = zipfile.ZipInfo(path.name, date_time=(2026, 11, 19, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, path.read_bytes())


def validate():
    result = source_check()
    record = json.loads((OUT / "build.json").read_text())
    assert record["original_files_sha256"] == original_hashes(), "Existing English versions changed"
    assert record["source_sha256"] == {p.name: sha(p) for p in source_files()}
    assert record["pdf_sha256"] == sha(OUT / "manuscript.pdf")
    assert record["archive_sha256"] == sha(OUT / "latex_sources.zip")
    with zipfile.ZipFile(OUT / "latex_sources.zip") as archive:
        assert set(archive.namelist()) == {p.name for p in source_files()}
        for path in source_files():
            assert archive.read(path.name) == path.read_bytes()
    log = (BUILD / "main/main.log").read_text(errors="replace")
    assert not re.search(r"Overfull \\[hv]box|undefined references|Citation .* undefined|Missing character:", log), "Layout/reference/glyph error"
    bbl = (BUILD / "main/main.bbl").read_text()
    assert len(re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", bbl)) == 18
    aux = (BUILD / "main/main.aux").read_text()
    for label, number in [("prop:rem", "3.1"), ("prop:history", "3.2"),
                          ("prop:kernel", "3.3"), ("prop:segments", "3.4"), ("thm:endtoend", "3.5")]:
        assert r"\newlabel{" + label + "}{{" + number + "}" in aux, label
    import pdfplumber
    with pdfplumber.open(OUT / "manuscript.pdf") as doc:
        assert 1 <= len(doc.pages) <= 6, "Keep the compact paper within six pages"
        texts = []
        for i, page in enumerate(doc.pages, 1):
            assert abs(page.width - 595.28) < 1 and abs(page.height - 841.89) < 1
            text = page.extract_text() or ""
            assert len(text.strip()) > 40 and "??" not in text and "\ufffd" not in text, i
            assert not [c for c in page.chars if c["x0"] < -.5 or c["x1"] > page.width + .5
                        or c["top"] < -.5 or c["bottom"] > page.height + .5], i
            texts.append(text)
        combined = "\n".join(texts)
        for required in ("Giới thiệu", "Phân tích privacy", "Kết luận", "99.595", "2.437372",
                         "cache-046_day7", "Tài liệu tham khảo", "cùng thời điểm"):
            assert re.sub(r"\s+", "", required) in re.sub(r"\s+", "", combined), required
        assert (doc.metadata.get("Author") or "").strip() == "Hanh Ngo-Minh; Phan Trong Nhan"
        result.update(pages=len(doc.pages), a4=True, no_missing_glyphs=True, no_overfull_boxes=True,
                      no_undefined_references=True, theorem_numbers_preserved=True, all_text_within_page=True)
    assert record["portable_text_identical"]
    result.update(status="pass", english_unchanged=True, portable_archive_compiled=True,
                  visual_qa="Separate observed page inspection receipt required")
    return result


def build(compiler):
    structural = source_check()
    originals = original_hashes()
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = compile_one(compiler, PAPER, "main", BUILD / "main")
    shutil.copyfile(pdf, OUT / "manuscript.pdf")
    make_archive()
    portable = BUILD / "portable"
    if portable.exists():
        shutil.rmtree(portable)
    portable.mkdir(parents=True)
    with zipfile.ZipFile(OUT / "latex_sources.zip") as archive:
        archive.extractall(portable)
    standalone = compile_one(compiler, portable, "main", BUILD / "portable_main")
    import pdfplumber
    with pdfplumber.open(pdf) as a, pdfplumber.open(standalone) as b:
        assert [p.extract_text() for p in a.pages] == [p.extract_text() for p in b.pages]
        pages = len(a.pages)
    assert originals == original_hashes()
    record = dict(schema="hk261-vietnamese-build-v1", date="2026-11-19", language="vi",
                  source_sha256={p.name: sha(p) for p in source_files()}, original_files_sha256=originals,
                  pages=pages, pdf_sha256=sha(pdf), archive_sha256=sha(OUT / "latex_sources.zip"),
                  portable_text_identical=True, method_or_benchmark_changed=False,
                  defenses_or_attackers_rerun=False, translation_checks=structural)
    (OUT / "build.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    result = validate()
    (OUT / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS: Vietnamese review PDF, {pages} pages; mathematics/results unchanged; portable ZIP compiled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tectonic", default=shutil.which("tectonic"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(validate(), ensure_ascii=False, indent=2))
    else:
        if not args.tectonic:
            parser.error("Pass --tectonic /path/to/tectonic")
        build(str(Path(args.tectonic).resolve()))
