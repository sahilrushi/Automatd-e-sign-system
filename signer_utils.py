# signer_utils.py
import pdfplumber
import re
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PyPDF2 import PdfReader, PdfWriter
import io
from docx import Document

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')

def extract_emails_and_sigboxes_from_pdf(pdf_path):
    emails = []
    sig_boxes = []  # list of dicts {page, x0,y0,x1,y1}
    with pdfplumber.open(pdf_path) as pdf:
        for pageno, page in enumerate(pdf.pages):
            # extract words with bbox to detect signature placeholders
            words = page.extract_words()
            for w in words:
                txt = w.get("text","")
                # find emails
                m = EMAIL_RE.findall(txt)
                if m:
                    for em in m:
                        if em not in emails:
                            emails.append(em)
                # heuristics to find signature placeholder keywords
                low = txt.lower()
                if any(k in low for k in ("sign here", "signature", "signed by", "sign:")):
                    # expand box a bit
                    x0 = float(w['x0']) - 10
                    x1 = float(w['x1']) + 150
                    top = float(w['top']) - 5
                    bottom = float(w['bottom']) + 40
                    # pdfplumber coordinates: top is smaller? adapt if needed
                    sig_boxes.append({"page": pageno, "x0": x0, "top": top, "x1": x1, "bottom": bottom})
    return emails, sig_boxes

def extract_emails_and_sigpos_from_docx(docx_path):
    doc = Document(docx_path)
    emails = []
    sig_positions = []
    for i, p in enumerate(doc.paragraphs):
        text = p.text or ""
        for em in EMAIL_RE.findall(text):
            if em not in emails:
                emails.append(em)
        if "signature" in text.lower() or "[[SIGN_HERE]]" in text:
            sig_positions.append({"para_index": i, "text": text})
    return emails, sig_positions

def overlay_signature_on_pdf(pdf_path, signature_img_path):
    # Basic approach: find first page's placeholder using pdfplumber and overlay signature.png
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        words = page.extract_words()
        # find first signature placeholder
        target = None
        for w in words:
            low = w.get("text","").lower()
            if any(k in low for k in ("sign here","signature","sign:","signed by")):
                x0 = float(w['x0'])
                x1 = float(w['x1'])
                top = float(w['top'])
                bottom = float(w['bottom'])
                target = {"page": 0, "x0": x0, "x1": x1, "top": top, "bottom": bottom}
                break
        if not target:
            # fallback: place at bottom-right area
            target = {"page":0, "x0": page.width - 200, "x1": page.width - 20, "top": page.height - 200, "bottom": page.height - 100}

    # create a PDF overlay with reportlab matching page dimensions
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    for i, p in enumerate(reader.pages):
        packet = io.BytesIO()
        # page size in points — approximate using pdfplumber earlier or PyPDF2 mediaBox
        media = p.mediabox
        pw = float(media.width)
        ph = float(media.height)
        can = canvas.Canvas(packet, pagesize=(pw, ph))
        if i == target["page"]:
            # convert pdfplumber coords (top/bottom) to reportlab coordinates (origin=bottom-left)
            # pdfplumber top is distance from top; but we earlier used page.extract_words which gives top relative to top of page?
            # For a simple approach, try using bottom as y
            y = ph - target["bottom"]
            x = target["x0"]
            # width scale
            sig_width = target["x1"] - target["x0"]
            sig_height = sig_width * 0.4
            can.drawImage(signature_img_path, x, y, width=sig_width, height=sig_height, mask='auto')
        can.save()
        packet.seek(0)
        overlay_pdf = PdfReader(packet)
        # merge overlay onto original page
        base_page = p
        base_page.merge_page(overlay_pdf.pages[0])
        writer.add_page(base_page)

    out_stream = io.BytesIO()
    writer.write(out_stream)
    out_stream.seek(0)
    return out_stream
