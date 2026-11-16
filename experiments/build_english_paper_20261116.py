"""Build the two author-review PDFs and a flat, independently compilable source ZIP."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.export_english_paper_20261116 import PAPER, generate, sha

OUT = ROOT / "artifacts/papers/geoi_bq_20261116_v1"
BUILD = ROOT / "build/paper/current"


def source_files():
    return sorted(p for p in PAPER.iterdir() if p.suffix in (".tex", ".sty", ".pdf", ".txt", ".md", ".json"))


def make_archive(path):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for source in source_files():
            info = zipfile.ZipInfo(source.name, date_time=(2026, 11, 16, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, source.read_bytes())


def compile_one(compiler, folder, entry, destination):
    destination.mkdir(parents=True, exist_ok=True)
    cmd = [compiler, "-X", "compile", str(folder / (entry + ".tex")), "--outdir",
           str(destination), "--keep-logs", "--keep-intermediates"]
    run = subprocess.run(cmd, cwd=folder, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (destination / "compiler_output.txt").write_text(run.stdout)
    if run.returncode:
        raise RuntimeError(run.stdout[-12000:])
    print(entry + ": compiled")
    return destination / (entry + ".pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tectonic", default=shutil.which("tectonic"))
    args = p.parse_args()
    if not args.tectonic:
        p.error("Tectonic required; pass --tectonic /path/to/tectonic")
    compiler = str(Path(args.tectonic).resolve())
    generated, figures = generate()
    for name, text in generated.items():
        (PAPER / name).write_text(text)
    for name, source in figures.items():
        (PAPER / name).write_bytes(source.read_bytes())
    OUT.mkdir(parents=True, exist_ok=True)
    import pdfplumber
    outputs = {}
    for entry, target in (("main", "manuscript.pdf"), ("supplement", "supplement.pdf")):
        pdf = compile_one(compiler, PAPER, entry, BUILD / entry)
        shutil.copyfile(pdf, OUT / target)
        with pdfplumber.open(pdf) as doc:
            outputs[target] = dict(pages=len(doc.pages), sha256=sha(pdf))
    make_archive(OUT / "latex_sources.zip")
    # Prove that the delivered archive has no hidden repository dependency.
    portable = BUILD / "portable"
    if portable.exists():
        shutil.rmtree(portable)
    portable.mkdir(parents=True)
    with zipfile.ZipFile(OUT / "latex_sources.zip") as z:
        z.extractall(portable)
    for entry, target in (("main", "manuscript.pdf"), ("supplement", "supplement.pdf")):
        pdf = compile_one(compiler, portable, entry, BUILD / ("portable_" + entry))
        with pdfplumber.open(pdf) as one, pdfplumber.open(OUT / target) as two:
            assert len(one.pages) == len(two.pages), entry
            assert [p.extract_text() for p in one.pages] == [p.extract_text() for p in two.pages], entry
    record = dict(schema="english-paper-build-v1", compiler=Path(compiler).name,
                  source_sha256={p.name: sha(p) for p in source_files()},
                  outputs=outputs, archive_sha256=sha(OUT / "latex_sources.zip"),
                  portable_archive_both_entries_compiled=True,
                  portable_text_identical=True, benchmark_or_method_changed=False)
    (OUT / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print("PASS: both PDFs and flat source archive; independent builds have identical text")


if __name__ == "__main__":
    main()
