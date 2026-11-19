"""Build a separate compact paper from the user's exact HK261 LLNCS template.

Scientific inputs are copied from the audited English paper; only headings,
front matter, typography and bibliography presentation are adapted.
"""
from pathlib import Path
import argparse
import json
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.export_english_paper_20261116 import AUTHORS, PAPER as ORIGINAL, generate, sha
from experiments.build_english_paper_20261116 import compile_one, check_main_page_limit
from experiments.validate_english_paper_20261116 import expanded

PAPER = ROOT / "paper_hk261"
OUT = ROOT / "artifacts/papers/geoi_bq_hk261_20261119_v1"
BUILD = ROOT / "build/paper/hk261_20261119"
INHERITED = (
    "abstract.tex", "introduction.tex", "related_work.tex", "method.tex", "analysis.tex",
    "evaluation.tex", "conclusion.tex", "figure_system.tex", "figure_model.tex",
    "figure_walkthrough.pdf", "table_related.tex", "table_privacy.tex", "table_service.tex",
    "highlights.txt",
)
PRESENTATION_CHANGES = {
    "introduction.tex": [
        (r"\begin{enumerate}", r"\begin{itemize}"),
        (r"\end{enumerate}", r"\end{itemize}"),
    ],
    "related_work.tex": [(r"\section{Related work}", r"\section{Related Work}")],
    "method.tex": [
        (r"\section{Proposed method and analysis}", r"\section{Proposed Method}"),
        (r"\subsection{System and assumptions}", r"\subsection{Framework}"),
        (r"\subsection{Protected reference and belief}",
         "\\subsection{Workflow}\n\\subsubsection{Protected reference and belief}"),
        (r"\subsection{Bounded segment selection and local answers}",
         r"\subsubsection{Query-segment selection and local answers}"),
        (r"\begin{figure}[t]", r"\begin{figure}[!ht]"),
    ],
    "analysis.tex": [
        (r"\subsection{Transcript privacy}", r"\subsection{Privacy analysis}"),
        (r"\subsection{Utility and scope}", r"\subsection{Utility analysis and scope}"),
    ],
    "evaluation.tex": [
        (r"\section{Experiments and analysis}", r"\section{Experiments and Analysis}"),
        (r"\subsection{Data and protocol}",
         "\\subsection{Environment Settings}\n\\subsubsection{Data and evaluation protocol}"),
        (r"\subsection{Comparators and attackers}",
         r"\subsubsection{Comparison methods and attackers}"),
        (r"\subsection{Main results}",
         "\\subsection{Empirical Experiments}\n\\subsubsection{Main results}"),
        (r"\subsection{Extended tasks and attribution}",
         r"\subsubsection{Extended tasks and attribution}"),
    ],
    "conclusion.tex": [(r"\section{Conclusion}", r"\section{Summary}")],
}


def protected_original_files():
    paths = [p for p in ORIGINAL.iterdir() if p.is_file()]
    paths += list((ROOT / "artifacts/papers/geoi_bq_20261116_v1").glob("*"))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths if p.is_file()}


def expected_sources():
    generated, figures = generate()  # Read-only authentication, no new experiments.
    for name, text in generated.items():
        assert (ORIGINAL / name).read_text() == text, name
    for name, source in figures.items():
        assert (ORIGINAL / name).read_bytes() == source.read_bytes(), name
    expected = {}
    for name in INHERITED:
        content = (ORIGINAL / name).read_bytes()
        for before, after in PRESENTATION_CHANGES.get(name, []):
            assert content.decode().count(before) == 1, name
            content = content.decode().replace(before, after, 1).encode()
        expected[name] = content
    return expected


def template_check(folder=PAPER):
    origin = json.loads((folder / "template_origin.json").read_text())
    for name, record in origin["members"].items():
        assert sha(folder / name) == record["sha256"], ("Supplied template changed", name)
    return origin


def source_check(folder=PAPER):
    origin = template_check(folder)
    expected = expected_sources()
    for name, content in expected.items():
        assert (folder / name).read_bytes() == content, ("Scientific input changed", name)
    text = expanded(folder, "main")
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    assert len(labels) == len(set(labels)), "Duplicate labels"
    refs = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", text))
    assert refs <= set(labels), refs - set(labels)
    for image in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
        assert Path(image).name == image and (folder / image).is_file(), image
    assert r"\documentclass[runningheads,twocolumn,envcountsame,envcountsect]{llncs}" in text
    assert r"\bibliographystyle{splncs04}" in text
    author_block = re.search(r"\\author\{(.*?)\}\s*\\authorrunning", text, re.S)[1]
    names = [re.sub(r"\\(?:inst|textsuperscript)\{[^}]*\}", "", n).strip()
             for n in author_block.split(r"\and")]
    assert names == list(AUTHORS)
    info = json.loads((folder / "author_info.json").read_text())
    assert [a["name"] for a in info["authors"]] == list(AUTHORS)
    corresponding = [a for a in info["authors"] if a["corresponding"]]
    assert len(corresponding) == 1 and corresponding[0]["name"] == AUTHORS[0]
    blocks = author_block.split(r"\and")
    for block, author in zip(blocks, info["authors"]):
        assert r"\inst{1,2}" in block and author["affiliation_ids"] == [1, 2]
        assert (r"\textsuperscript{\Letter}" in block) == author["corresponding"]
    normalized = re.sub(r"\s+", " ", text)
    for affiliation in info["affiliations"]:
        assert affiliation["text"] in normalized
    assert r"\email{\{nmhanh.sdh241, nhanpt\}@hcmut.edu.vn}" in text
    assert r"\Letter\ Corresponding author: \email{" + corresponding[0]["email"] + "}" in text
    assert [a["email"] for a in info["authors"]] == [
        "nmhanh.sdh241@hcmut.edu.vn", "nhanpt@hcmut.edu.vn"]
    for heading in ("Framework", "Workflow", "Environment Settings", "Empirical Experiments"):
        assert r"\subsection{" + heading + "}" in text
    assert r"\subsubsection{\ackname}" in text
    assert "for supporting this study." in normalized
    sections = re.findall(r"\\section\{([^}]+)\}", text)
    assert sections == ["Introduction", "Related Work", "Proposed Method", "Experiments and Analysis", "Summary"]
    assert not re.search(r"\b(?:TODO|TBC|FIXME)\b|turn\d+(?:search|view)", text)
    cited = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text) for k in group.split(",")}
    original_refs = (ORIGINAL / "references.tex").read_text()
    original_keys = set(re.findall(r"\\bibitem\{([^}]+)\}", original_refs))
    bib = (folder / "references.bib").read_text()
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    assert cited == bib_keys == original_keys and len(cited) == 18
    original_dois = set(re.findall(r"\b10\.\d{4,9}/[^}\s]+", original_refs))
    assert original_dois <= set(re.findall(r"\b10\.\d{4,9}/[^}\s,]+", bib)), "Bibliography lost source DOI"
    return dict(template_class_exact=True, template_bst_exact=True,
                class_version=origin["class_version"], main_sections=sections,
                inherited_inputs_verified=len(expected), references=18, authors=list(AUTHORS),
                scientific_change="Headings, contribution-list markers and float placement only; scientific paragraphs, equations, result values and figures unchanged",
                bibliography="Same 18 works and source DOIs; splncs04 presentation/order",
                author_metadata="Two supplied affiliations and user-provided emails; Hanh Ngo-Minh is the sole corresponding author",
                affiliations=info["affiliations"], corresponding_author=corresponding[0]["name"])


def prepare():
    template_check()
    expected = expected_sources()
    for name, content in expected.items():
        (PAPER / name).write_bytes(content)
    record = dict(schema="hk261-content-manifest-v1", date="2026-11-19",
                  inherited_source="paper/", source_sha256={name: sha(ORIGINAL / name) for name in INHERITED},
                  adapted_sha256={name: sha(PAPER / name) for name in INHERITED},
                  presentation_changes={name: [list(change) for change in changes]
                                        for name, changes in PRESENTATION_CHANGES.items()},
                  original_references_sha256=sha(ORIGINAL / "references.tex"),
                  original_evidence_manifest_sha256=sha(ORIGINAL / "evidence_manifest.json"),
                  bibliography_sha256=sha(PAPER / "references.bib"),
                  template_origin_sha256=sha(PAPER / "template_origin.json"),
                  author_info_sha256=sha(PAPER / "author_info.json"),
                  method_or_benchmark_changed=False, defense_or_attackers_rerun=False)
    (PAPER / "content_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    source_check()


def source_files():
    return sorted(p for p in PAPER.iterdir()
                  if p.suffix in (".tex", ".sty", ".cls", ".bst", ".bib", ".pdf", ".txt", ".md", ".json"))


def make_archive():
    with zipfile.ZipFile(OUT / "latex_sources.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in source_files():
            info = zipfile.ZipInfo(path.name, date_time=(2026, 11, 19, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, path.read_bytes())


def validate():
    structural = source_check()
    record = json.loads((OUT / "build.json").read_text())
    assert record["source_sha256"] == {p.name: sha(p) for p in source_files()}
    for name, digest in record["original_files_sha256"].items():
        assert sha(ROOT / name) == digest, ("Original version changed", name)
    assert sha(OUT / "latex_sources.zip") == record["archive_sha256"]
    with zipfile.ZipFile(OUT / "latex_sources.zip") as z:
        assert set(z.namelist()) == {p.name for p in source_files()}
        for p in source_files():
            assert z.read(p.name) == p.read_bytes()
    log = (BUILD / "main/main.log").read_text(errors="replace")
    assert not re.search(r"Overfull \\[hv]box|undefined references|Citation .* undefined|Missing character:", log), "TeX layout/reference/glyph problem"
    bbl = (BUILD / "main/main.bbl").read_text()
    assert len(re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", bbl)) == 18
    aux = (BUILD / "main/main.aux").read_text()
    for label, number in [("prop:rem", "3.1"), ("prop:history", "3.2"),
                          ("prop:kernel", "3.3"), ("prop:segments", "3.4"), ("thm:endtoend", "3.5")]:
        assert r"\newlabel{" + label + "}{{" + number + "}" in aux, (label, "Theorem numbering changed")
    import pdfplumber
    with pdfplumber.open(OUT / "manuscript.pdf") as doc:
        check_main_page_limit(len(doc.pages))
        assert sha(OUT / "manuscript.pdf") == record["pdf_sha256"]
        assert (doc.metadata.get("Author") or "").strip() == "; ".join(AUTHORS)
        texts = []
        for i, page in enumerate(doc.pages, 1):
            assert abs(page.width-595.28) < 1 and abs(page.height-841.89) < 1, "A4 required"
            text = page.extract_text() or ""
            assert len(text.strip()) > 40 and "??" not in text and "\ufffd" not in text, i
            assert not [c for c in page.chars if c["x0"] < -.5 or c["x1"] > page.width+.5
                        or c["top"] < -.5 or c["bottom"] > page.height+.5], i
            texts.append(text)
        combined = "\n".join(texts)
        for required in ("HCMUT", "VNU-HCM", "nmhanh.sdh241", "nhanpt", "Corresponding author:"):
            assert re.sub(r"\s+", "", required) in re.sub(r"\s+", "", texts[0]), (
                "Author block missing from page 1", required)
        assert "Acknowledgments." in combined
        for required in ("99.595", "2.437372", "cache-046_day7", "Summary", "same-time pair"):
            assert required in combined, required
        assert record["portable_archive_compiled"] and record["portable_text_identical"]
        result = dict(schema="hk261-validation-v1", status="pass", structural=structural,
                      pages=len(doc.pages), main_page_limit=6, page_limit_pass=True,
                      pdf_sha256=sha(OUT / "manuscript.pdf"), a4=True, two_columns=True,
                      no_overfull_boxes=True, no_undefined_references=True,
                      all_pdf_text_within_page=True, theorem_numbers_preserved=True,
                      portable_archive_compiled=True, original_version_unchanged=True,
                      visual_qa="Separate observed page inspection receipt required")
    return result


def build(compiler):
    original = protected_original_files()
    prepare()
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = compile_one(compiler, PAPER, "main", BUILD / "main")
    import pdfplumber
    with pdfplumber.open(pdf) as doc:
        check_main_page_limit(len(doc.pages))
        pages = len(doc.pages)
    shutil.copyfile(pdf, OUT / "manuscript.pdf")
    make_archive()
    portable = BUILD / "portable"
    if portable.exists():
        shutil.rmtree(portable)
    portable.mkdir(parents=True)
    with zipfile.ZipFile(OUT / "latex_sources.zip") as z:
        z.extractall(portable)
    standalone = compile_one(compiler, portable, "main", BUILD / "portable_main")
    with pdfplumber.open(pdf) as a, pdfplumber.open(standalone) as b:
        assert [p.extract_text() for p in a.pages] == [p.extract_text() for p in b.pages]
    assert original == protected_original_files(), "Do not change the existing paper/version"
    record = dict(schema="hk261-build-v1", date="2026-11-19",
                  source_sha256={p.name: sha(p) for p in source_files()},
                  original_files_sha256=original, pages=pages, pdf_sha256=sha(pdf),
                  archive_sha256=sha(OUT / "latex_sources.zip"),
                  portable_archive_compiled=True, portable_text_identical=True,
                  method_or_benchmark_changed=False)
    (OUT / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    result = validate()
    (OUT / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: HK261 template version, {pages} pages; original paper unchanged; portable source ZIP verified")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tectonic", default=shutil.which("tectonic"))
    p.add_argument("--check", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if args.check:
        result = validate()
        print("PASS:", json.dumps(result))
    elif args.prepare_only:
        prepare()
        print("PASS: template and inherited scientific sources verified")
    else:
        if not args.tectonic:
            p.error("Pass --tectonic /path/to/tectonic")
        build(str(Path(args.tectonic).resolve()))


if __name__ == "__main__":
    main()
