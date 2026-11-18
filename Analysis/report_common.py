"""Shared helpers for the two PDF documents (HTML -> PDF with headless Chromium)."""
import json
from pathlib import Path
import pandas as pd
from common import ROOT, PROC, OUT, FIG

DEL = ROOT / "deliverables"
DEL.mkdir(exist_ok=True)
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)

CSS = r"""
@page { size: A4; margin: 17mm 17mm 18mm 17mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: 'Lora', Georgia, serif; font-size: 9.7pt; line-height: 1.52; color: #1E1E24; margin: 0; }
h1, h2, h3, h4, .sans, table, figcaption, .kicker, .callout, .stat, .toc, .tag { font-family: 'Inter', Arial, sans-serif; }
h1 { font-size: 19pt; color: #14295E; margin: 0 0 4mm 0; letter-spacing: -0.2px; font-weight: 700; }
h2 { font-size: 14pt; color: #14295E; margin: 7mm 0 2.5mm 0; font-weight: 700; break-after: avoid; }
h2 .num { color: #E5197F; margin-right: 6px; }
h3 { font-size: 10.8pt; color: #254AA5; margin: 5mm 0 1.5mm 0; font-weight: 650; break-after: avoid; }
h4 { font-size: 9.6pt; color: #1E1E24; margin: 3.5mm 0 1mm 0; font-weight: 650; break-after: avoid; }
p { margin: 0 0 2.6mm 0; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 2.6mm 0; padding-left: 5.5mm; }
li { margin-bottom: 1mm; }
a { color: #254AA5; text-decoration: none; }
.section { break-before: page; }
.kicker { font-size: 7.6pt; letter-spacing: 1.6px; text-transform: uppercase; color: #E5197F; font-weight: 700; margin-bottom: 1mm; }
figure { margin: 3mm 0 4mm 0; break-inside: avoid; }
figure img { width: 100%; display: block; }
figure.narrow img { width: 78%; margin: 0 auto; }
figure.mid img { width: 88%; margin: 0 auto; }
figcaption { font-size: 7.8pt; color: #5A5A66; margin-top: 1.5mm; line-height: 1.4; }
figcaption b { color: #1E1E24; }
table { border-collapse: collapse; width: 100%; font-size: 7.6pt; margin: 2mm 0 4mm 0; break-inside: avoid; }
th { background: #14295E; color: #fff; font-weight: 600; text-align: left; padding: 1.6mm 1.4mm; vertical-align: bottom; }
td { padding: 1.3mm 1.4mm; border-bottom: 0.4pt solid #DADCE5; vertical-align: top; }
tr:nth-child(even) td { background: #F6F7FB; }
td.r, th.r { text-align: right; font-variant-numeric: tabular-nums; }
td.rank { font-weight: 700; color: #E5197F; width: 6mm; }
td.name { font-weight: 600; white-space: nowrap; }
table.compact td, table.compact th { padding: 1mm 1.2mm; }
.callout { background: #F3F5FC; border-left: 3.5pt solid #254AA5; padding: 3mm 4mm; margin: 3mm 0 4mm 0; font-size: 8.7pt; line-height: 1.5; break-inside: avoid; }
.callout.pink { border-left-color: #E5197F; background: #FDF1F7; }
.callout.gold { border-left-color: #B8860B; background: #FBF7EC; }
.callout b { color: #14295E; }
.statrow { display: flex; gap: 3mm; margin: 3mm 0 4mm 0; break-inside: avoid; }
.stat { flex: 1; border-top: 2.5pt solid #254AA5; padding-top: 2mm; }
.stat .v { font-size: 17pt; font-weight: 700; color: #14295E; line-height: 1.1; }
.stat .l { font-size: 7.4pt; color: #5A5A66; line-height: 1.35; margin-top: 1mm; }
.stat.pink { border-top-color: #E5197F; } .stat.pink .v { color: #E5197F; }
.two { display: flex; gap: 6mm; } .two > div { flex: 1; }
.eq { font-family: 'Inter', sans-serif; font-size: 9pt; background: #FAFAFC; border: 0.5pt solid #E1E3EC; padding: 2.2mm 3.5mm; margin: 2mm 0 3mm 0; text-align: center; break-inside: avoid; }
.eq i { font-family: 'Lora', serif; }
.note { font-size: 7.6pt; color: #5A5A66; }
.tag { display: inline-block; font-size: 6.6pt; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; padding: 0.4mm 1.6mm; border-radius: 2mm; background: #E8ECF8; color: #254AA5; margin-right: 1mm; }
.tag.pink { background: #FBE3EF; color: #C0126A; }
.tag.gold { background: #F6EDD5; color: #8A6508; }
.profile { break-inside: avoid; border-bottom: 0.4pt solid #DADCE5; padding: 2.2mm 0 2.4mm 0; }
.profile h4 { margin: 0 0 0.8mm 0; font-size: 9.8pt; }
.profile h4 .rk { color: #E5197F; margin-right: 4px; }
.profile p { margin-bottom: 1.2mm; font-size: 9pt; }
.profile .ext { font-size: 8.1pt; color: #3C3C48; }
.toc { font-size: 9pt; columns: 2; column-gap: 10mm; }
.toc div { display: flex; justify-content: space-between; border-bottom: 0.4pt dotted #C7CAD6; padding: 1.1mm 0; }
.toc span.n { color: #E5197F; font-weight: 700; margin-right: 2mm; }
.refs { font-size: 7.6pt; line-height: 1.45; }
.refs li { margin-bottom: 1.2mm; }
.cover { height: 297mm; width: 210mm; position: relative; overflow: hidden; }
.cover .band { position: absolute; top: 0; left: 0; right: 0; height: 140mm;
  background: linear-gradient(135deg, #14295E 0%, #254AA5 58%, #E5197F 135%); }
.cover .band .stripe { position: absolute; bottom: 0; left: 0; right: 0; height: 3.5mm; background: #E5197F; }
.cover .band .gold { position: absolute; bottom: 3.5mm; left: 0; width: 42%; height: 1.2mm; background: #C9A227; }
.cover .inner { position: absolute; top: 34mm; left: 20mm; right: 22mm; color: #fff; font-family: 'Inter', sans-serif; }
.cover .pre { font-size: 8.5pt; letter-spacing: 2.5px; text-transform: uppercase; opacity: 0.85; font-weight: 600; }
.cover .title { font-size: 31pt; font-weight: 800; line-height: 1.03; margin: 7mm 0 5mm 0; letter-spacing: -0.5px; }
.cover .sub { font-size: 11.5pt; line-height: 1.45; opacity: 0.95; max-width: 150mm; font-weight: 400; }
.cover .below { position: absolute; top: 142mm; left: 20mm; right: 20mm; font-family: 'Inter', sans-serif; }
.cover .below .lead { font-family: 'Lora', serif; font-size: 11pt; line-height: 1.6; color: #1E1E24; margin-top: 10mm; }
.cover .meta { position: absolute; bottom: 14mm; left: 20mm; right: 20mm; font-size: 8pt; color: #5A5A66; font-family: 'Inter', sans-serif;
  border-top: 0.6pt solid #DADCE5; padding-top: 3mm; display: flex; justify-content: space-between; }
.listcard { display: flex; gap: 5mm; margin-top: 9mm; }
.listcard > div { flex: 1; border: 0.6pt solid #DADCE5; border-top: 3pt solid #254AA5; padding: 3mm 4mm; }
.listcard > div.b { border-top-color: #E5197F; }
.listcard h4 { margin: 0 0 2mm 0; font-size: 8.4pt; letter-spacing: 1.2px; text-transform: uppercase; color: #14295E; }
.listcard ol { margin: 0; padding-left: 5mm; font-size: 9.2pt; line-height: 1.55; font-family: 'Inter', sans-serif; }
"""

HEADER = """<div style="font-family:Inter,Arial;font-size:6.6pt;color:#8A8A96;width:100%;padding:0 17mm;display:flex;justify-content:space-between;">
<span>{left}</span><span style="color:#E5197F;font-weight:700;letter-spacing:1px;">SUPERR SELECTOR S2 · ROUND 1</span></div>"""
FOOTER = """<div style="font-family:Inter,Arial;font-size:6.6pt;color:#8A8A96;width:100%;padding:0 17mm;display:flex;justify-content:space-between;">
<span>{left}</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>"""


def img(name):
    return (FIG / f"{name}.png").as_uri()


def fig(name, caption, cls=""):
    return f'<figure class="{cls}"><img src="{img(name)}"><figcaption>{caption}</figcaption></figure>'


def j(name):
    return json.loads((OUT / name).read_text())


def f1(x, d=1):
    return f"{x:,.{d}f}"


def page(body, title):
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>"


def render_pdf(cover_html, body_html, out_pdf, title, header_left, footer_left):
    """Cover rendered full-bleed without header/footer; body with running header/footer; then merged."""
    from playwright.sync_api import sync_playwright
    from pypdf import PdfReader, PdfWriter
    stem = Path(out_pdf).stem
    cpath, bpath = BUILD / f"{stem}_cover.html", BUILD / f"{stem}_body.html"
    cpath.write_text(page(cover_html, title).replace("@page { size: A4; margin: 17mm 17mm 18mm 17mm; }", "@page { size: A4; margin: 0; }"), encoding="utf-8")
    bpath.write_text(page(body_html, title), encoding="utf-8")
    cpdf, bpdf = BUILD / f"{stem}_cover.pdf", BUILD / f"{stem}_body.pdf"
    with sync_playwright() as p:
        br = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
        pg = br.new_page()
        pg.goto(cpath.as_uri(), wait_until="networkidle")
        pg.pdf(path=str(cpdf), format="A4", print_background=True, margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        pg.goto(bpath.as_uri(), wait_until="networkidle")
        pg.pdf(path=str(bpdf), format="A4", print_background=True, display_header_footer=True,
               header_template=HEADER.format(left=header_left), footer_template=FOOTER.format(left=footer_left),
               margin={"top": "17mm", "bottom": "18mm", "left": "17mm", "right": "17mm"})
        br.close()
    w = PdfWriter()
    for f in (cpdf, bpdf):
        for pg_ in PdfReader(str(f)).pages:
            w.add_page(pg_)
    w.add_metadata({"/Title": title, "/Subject": "SupeRR Selector Season 2 - Round 1"})
    with open(out_pdf, "wb") as fh:
        w.write(fh)
    return out_pdf
