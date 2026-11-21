"""Build a lab report on the course template (DSP_TemplateReport.docx) and convert it to PDF with MS Word.

The page set-up and styles of the course template are kept. Its placeholder content is replaced by a university
title page (UBB logo, faculty, course, lab number, topic, date, student block pinned to the bottom of the page) and
by the report body given through the small API below. Used by scripts/make_report.py.
"""
import re
import subprocess
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Times New Roman"
TITLE_FONT = "Calibri"
MONO = "Consolas"
LOGO = Path(__file__).parent / "assets" / "ubb_logo.png"

FACULTY = "Wydział Budowy Maszyn i Informatyki"
TEACHER = "prof. dr hab. Vasyl Martsenyuk"
STUDENT = ("Radosław Skrzypczyński", "Nr indeksu: 29429", "Informatyka, studia II stopnia, niestacjonarne",
           "1 semestr, grupa 2")


class Report:
    def __init__(self, template, *, lab, date, topic, variant, course="Digital Signal Processing",
                 teacher=TEACHER, student=STUDENT, faculty=FACULTY, logo=LOGO):
        self.doc = docx.Document(template)
        self.fig_no = 0
        self.tab_no = 0
        body = self.doc.element.body
        for el in list(body):                       # keep only the section properties of the template
            if el.tag != qn("w:sectPr"):
                body.remove(el)
        self._title_page(lab, date, topic, variant, course, teacher, student, faculty, logo)

    def _tline(self, text, size, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=4):
        par = self.doc.add_paragraph()
        par.alignment = align
        pf = par.paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), 1.08
        for i, part in enumerate(text if isinstance(text, (list, tuple)) else [text]):
            r = par.add_run(part)
            r.font.name, r.font.size, r.bold = TITLE_FONT, Pt(size), bold
            r._element.rPr.rFonts.set(qn("w:cs"), TITLE_FONT)
        return par

    def _title_page(self, lab, date, topic, variant, course, teacher, student, faculty, logo):
        """University title page modelled on the UBB report layout."""
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_before = Pt(18)
        par.add_run().add_picture(str(logo), width=Cm(12.7))
        self._tline(faculty, 14, bold=True, before=14)
        self._tline(course, 20, bold=True, before=96)
        self._tline(f"(Laboratorium № {lab})", 14, bold=True, before=2)
        self._tline(f"Temat ćwiczenia: {topic}", 18, bold=True, before=44)
        left = WD_ALIGN_PARAGRAPH.LEFT
        self._tline(f"Data wykonania ćwiczenia: {date}", 16, align=left, before=28)
        self._tline(f"Prowadzący: {teacher}", 16, align=left)
        self._tline(f"Warianty: {variant}", 16, align=left)
        # student block: one frame anchored to the bottom margin, independent of the length of the topic
        for line in student:
            par = self._tline(line, 16, align=left, after=6)
            fr = OxmlElement("w:framePr")
            for k, v in (("w", str(int(16.0/2.54*1440))), ("wrap", "around"), ("vAnchor", "margin"),
                         ("hAnchor", "margin"), ("xAlign", "left"), ("yAlign", "bottom")):
                fr.set(qn(f"w:{k}"), v)
            par._p.get_or_add_pPr().insert(0, fr)
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ------------------------------------------------------------------------------------------------------------
    def _runs(self, par, text, size=12, italic=False):
        """Add text with **bold** and `code` inline markup."""
        for tok in re.split(r"(\*\*.+?\*\*|`.+?`)", text):
            if not tok:
                continue
            if tok.startswith("**"):
                r = par.add_run(tok[2:-2])
                r.bold = True
                r.font.name = FONT
            elif tok.startswith("`"):
                r = par.add_run(tok[1:-1])
                r.font.name = MONO
                size_r = size - 1.5
                r.font.size = Pt(size_r)
                continue
            else:
                r = par.add_run(tok)
                r.font.name = FONT
            r.font.size = Pt(size)
            r.italic = italic
        return par

    def heading(self, text):
        par = self.doc.add_paragraph()
        par.paragraph_format.space_before = Pt(14)
        par.paragraph_format.space_after = Pt(6)
        par.paragraph_format.keep_with_next = True
        r = par.add_run(text)
        r.bold, r.font.name, r.font.size = True, FONT, Pt(14)
        return par

    def subheading(self, text):
        par = self.doc.add_paragraph()
        par.paragraph_format.space_before = Pt(8)
        par.paragraph_format.space_after = Pt(3)
        par.paragraph_format.keep_with_next = True
        r = par.add_run(text)
        r.bold, r.font.name, r.font.size = True, FONT, Pt(12)
        return par

    def para(self, text, size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, keep_next=False):
        par = self.doc.add_paragraph()
        par.alignment = align
        par.paragraph_format.space_after = Pt(4)
        par.paragraph_format.keep_with_next = keep_next
        return self._runs(par, text, size)

    def bullets(self, items, size=12, numbered=False):
        for i, it in enumerate(items, 1):
            par = self.doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf = par.paragraph_format
            pf.left_indent, pf.first_line_indent, pf.space_after = Cm(0.9), Cm(-0.5), Pt(2)
            self._runs(par, (f"{i}. " if numbered else "• ") + it, size)

    def code(self, text, size=8.5):
        lines = text.strip("\n").splitlines()
        for i, line in enumerate(lines):
            par = self.doc.add_paragraph()
            pf = par.paragraph_format
            pf.space_before = Pt(4 if i == 0 else 0)
            pf.space_after = Pt(4 if i == len(lines) - 1 else 0)
            pf.line_spacing = 1.0
            pf.left_indent = Cm(0.3)
            pf.keep_together = True
            if i < len(lines) - 1:
                pf.keep_with_next = len(lines) < 40
            r = par.add_run(line if line else " ")
            r.font.name, r.font.size = MONO, Pt(size)
            r._element.rPr.rFonts.set(qn("w:eastAsia"), MONO)
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), "F3F2EE")
            par._p.get_or_add_pPr().append(shd)

    def figure(self, path, caption, width_cm=16.0):
        self.fig_no += 1
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.keep_with_next = True
        par.paragraph_format.space_before = Pt(6)
        par.add_run().add_picture(str(path), width=Cm(width_cm))
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(8)
        self._runs(cap, f"Figure {self.fig_no}: {caption}", size=10, italic=True)

    def table(self, header, rows, caption=None, size=9, col_widths=None, align_first_left=True, keep_together=True):
        if caption:
            self.tab_no += 1
            cap = self.doc.add_paragraph()
            cap.paragraph_format.keep_with_next = True
            cap.paragraph_format.space_before = Pt(6)
            self._runs(cap, f"Table {self.tab_no}: {caption}", size=10, italic=True)
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl_pr = t._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "4")
            el.set(qn("w:color"), "A6A59E")
            borders.append(el)
        tbl_pr.append(borders)
        for r_i, values in enumerate([header] + [list(r) for r in rows]):
            for c_i, val in enumerate(values):
                cell = t.cell(r_i, c_i)
                par = cell.paragraphs[0]
                par.alignment = (WD_ALIGN_PARAGRAPH.LEFT if (c_i == 0 and align_first_left)
                                 else WD_ALIGN_PARAGRAPH.CENTER)
                par.paragraph_format.space_after = Pt(0)
                if keep_together and r_i < len(rows):
                    par.paragraph_format.keep_with_next = True       # Word keeps the table on one page
                run = par.add_run(str(val))
                run.font.name, run.font.size = FONT, Pt(size)
                if r_i == 0:
                    run.bold = True
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear")
                    shd.set(qn("w:fill"), "ECEBE6")
                    cell._tc.get_or_add_tcPr().append(shd)
                if col_widths:
                    cell.width = Cm(col_widths[c_i])
        spacer = self.doc.add_paragraph()
        spacer.paragraph_format.space_after = Pt(2)
        return t

    def link(self, label, url):
        par = self.doc.add_paragraph()
        self._runs(par, label)
        # real hyperlink
        r_id = self.doc.part.relate_to(url, docx.opc.constants.RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
        h = OxmlElement("w:hyperlink")
        h.set(qn("r:id"), r_id)
        new_run = OxmlElement("w:r")
        rpr = OxmlElement("w:rPr")
        for tag, val in (("w:rFonts", None), ("w:color", "1F5FBF"), ("w:u", "single"), ("w:sz", "24")):
            el = OxmlElement(tag)
            if tag == "w:rFonts":
                el.set(qn("w:ascii"), FONT)
                el.set(qn("w:hAnsi"), FONT)
            else:
                el.set(qn("w:val"), val)
            rpr.append(el)
        new_run.append(rpr)
        txt = OxmlElement("w:t")
        txt.text = url
        new_run.append(txt)
        h.append(new_run)
        par._p.append(h)
        return par

    def save(self, docx_path, pdf_path=None):
        docx_path = Path(docx_path).resolve()
        self.doc.save(docx_path)
        print(f"written: {docx_path.name}")
        if pdf_path:
            docx_to_pdf(docx_path, pdf_path)


def docx_to_pdf(docx_path, pdf_path):
    """Convert with MS Word through COM (PowerShell); Word is started invisibly and closed again."""
    docx_path, pdf_path = Path(docx_path).resolve(), Path(pdf_path).resolve()
    ps = (
        "$ErrorActionPreference='Stop';"
        "$w = New-Object -ComObject Word.Application; $w.Visible = $false; $w.DisplayAlerts = 0;"
        f"try {{ $d = $w.Documents.Open('{docx_path}', $false, $true);"
        f"$d.SaveAs2('{pdf_path}', 17); $d.Close($false) }} finally {{ $w.Quit() }}"
    )
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps], check=True, timeout=300)
    print(f"written: {pdf_path.name} ({pdf_path.stat().st_size / 1024:.0f} KB)")


__all__ = ["Report", "docx_to_pdf", "RGBColor"]
