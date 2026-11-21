"""Export an executed notebook to PDF without LaTeX: nbconvert -> HTML (+ print CSS) -> headless Edge/Chrome.

Usage:  py -3.14 -X utf8 scripts/export_notebook_pdf.py Lab2_Windowing.ipynb Lab2_Windowing.pdf
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]

PRINT_CSS = """
@page { size: A4; margin: 12mm 11mm; }
body { font-size: 10pt; }
.jp-InputPrompt, .jp-OutputPrompt { display: none !important; }
pre, .highlight pre, .jp-OutputArea-output pre {
  white-space: pre-wrap !important; word-break: break-word; font-size: 7.6pt !important; line-height: 1.3; }
img { max-width: 100% !important; height: auto !important; }
.jp-OutputArea-output img, .jp-RenderedImage, table.dataframe { break-inside: avoid; page-break-inside: avoid; }
.jp-RenderedHTMLCommon table, table.dataframe { font-size: 7.4pt !important; }
.jp-RenderedHTMLCommon td, .jp-RenderedHTMLCommon th { padding: 2px 5px !important; }
.jp-RenderedHTMLCommon h2 { break-after: avoid; margin-top: 1.4em; }
.jp-RenderedHTMLCommon h3 { break-after: avoid; }
"""


def main(nb_path: str, pdf_path: str) -> None:
    nb = nbformat.read(nb_path, as_version=4)
    html, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
    html = html.replace("</head>", f"<style>{PRINT_CSS}</style></head>", 1)
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        sys.exit("No Edge/Chrome executable found.")
    out = Path(pdf_path).resolve()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        page = Path(tmp) / "notebook.html"
        page.write_text(html, encoding="utf-8")
        # isolated profile -> never attaches to a running browser session of the user
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-pdf-header-footer",
                        f"--user-data-dir={Path(tmp) / 'profile'}", "--virtual-time-budget=15000",
                        f"--print-to-pdf={out}", page.as_uri()], check=True, timeout=300)
    print(f"written: {out.name} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
