# signer_utils.py
import os
import io
import re
import math
import logging
from typing import List, Dict, Tuple, Optional

from PIL import Image, UnidentifiedImageError
import pdfplumber
import pytesseract
from pytesseract import Output
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

# Optional: allow setting Tesseract path via environment variable
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", None)
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Configurable keywords and heuristics
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
SIG_KEYWORDS = ["signature", "sign here", "signed by", "sign:", "signature:"]
OCR_KEYWORDS = set([k.lower() for k in ["signature", "sign", "signed", "signatory", "sign here", "signature:"]])

# Utility types
Candidate = Dict[str, object]

# ---------------------------
# Helper: safe image open
# ---------------------------
def _open_image_safe(path: str) -> Image.Image:
    try:
        img = Image.open(path)
        img.load()
        return img
    except FileNotFoundError:
        logger.error("Signature image not found at %s", path)
        raise
    except UnidentifiedImageError:
        logger.error("Signature image at %s cannot be identified (corrupt/unsupported).", path)
        raise

# ---------------------------
# Email & signature keyword extraction from PDF (text layer)
# ---------------------------
def extract_emails_and_sigboxes_from_pdf(pdf_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Returns (emails, sig_boxes)
    - emails: de-duplicated list of email strings found in PDF text layer
    - sig_boxes: list of heuristic signature boxes detected from text layer,
      each box: {page, x0, x1, y0_top, y1_bottom}
    """
    emails = []
    sig_boxes = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pageno, page in enumerate(pdf.pages):
                # extract emails from text
                text = page.extract_text() or ""
                for em in EMAIL_RE.findall(text):
                    if em not in emails:
                        emails.append(em)

                # word-level detection for signature keywords
                try:
                    words = page.extract_words()
                except Exception:
                    words = []

                for w in words:
                    txt = (w.get("text") or "").strip().lower()
                    # simple match if keyword contained
                    for k in SIG_KEYWORDS:
                        if k in txt:
                            x0 = float(w.get("x0", 0)) - 5
                            x1 = float(w.get("x1", 0)) + 120
                            top = float(w.get("top", 0)) - 4
                            bottom = float(w.get("bottom", 0)) + 45
                            # clamp
                            x0 = max(0, x0)
                            x1 = min(page.width, x1)
                            top = max(0, top)
                            bottom = min(page.height, bottom)
                            sig_boxes.append({
                                "page": pageno,
                                "x0": x0,
                                "x1": x1,
                                "y0_top": top,
                                "y1_bottom": bottom
                            })
    except Exception as e:
        logger.exception("Error extracting emails/sigboxes from PDF: %s", e)
    # dedupe emails
    emails = list(dict.fromkeys(emails))
    logger.debug("extract_emails_and_sigboxes_from_pdf -> emails=%s sig_boxes=%d", emails, len(sig_boxes))
    return emails, sig_boxes

# ---------------------------
# Extract from DOCX (simple)
# ---------------------------
def extract_emails_and_sigpos_from_docx(docx_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Returns (emails, sig_positions)
    This is a lightweight DOCX extractor: DOCX doesn't have strict positional coordinates;
    returns paragraph-based placeholder positions as a heuristic.
    """
    emails = []
    sig_positions = []
    try:
        from docx import Document
        doc = Document(docx_path)
        for i, p in enumerate(doc.paragraphs):
            text = (p.text or "").strip()
            if not text:
                continue
            for em in EMAIL_RE.findall(text):
                if em not in emails:
                    emails.append(em)
            if any(k in text.lower() for k in SIG_KEYWORDS) or "[[SIGN_HERE]]" in text:
                # document-paragraph-level heuristic
                sig_positions.append({
                    "para_index": i,
                    "text": text
                })
    except Exception as e:
        logger.exception("Error reading DOCX: %s", e)
    emails = list(dict.fromkeys(emails))
    logger.debug("extract_emails_and_sigpos_from_docx -> emails=%s sig_positions=%d", emails, len(sig_positions))
    return emails, sig_positions

# ---------------------------
# OCR-based candidate finder (robust)
# ---------------------------
def find_signature_candidates_by_ocr(pdf_path: str, dpi: int = 200, min_candidate_score: float = 0.45) -> List[Candidate]:
    """
    Use pdfplumber to render pages -> pytesseract to extract word boxes.
    Returns list of candidate boxes in PDF points:
      {page, x, y, width, height, score, reason}
    Coordinates correspond to PDF points with origin bottom-left (suitable for reportlab).
    """
    candidates: List[Candidate] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pageno, page in enumerate(pdf.pages):
                try:
                    page_image = page.to_image(resolution=dpi).original  # PIL.Image
                except Exception:
                    logger.exception("Failed to render page %d as image", pageno)
                    continue

                img_w, img_h = page_image.size
                if img_w == 0 or img_h == 0:
                    continue

                # run OCR with word boxes
                try:
                    ocr = pytesseract.image_to_data(page_image, output_type=Output.DICT)
                except pytesseract.TesseractNotFoundError:
                    logger.error("Tesseract not found. Set TESSERACT_CMD or install Tesseract.")
                    raise
                except Exception:
                    logger.exception("pytesseract.image_to_data failed on page %d", pageno)
                    continue

                n = len(ocr.get("level", []))
                # collect words and check keywords
                for i in range(n):
                    text = (ocr.get("text", [""])[i] or "").strip()
                    if not text:
                        continue
                    txt_lower = text.lower()
                    # keyword match
                    if any(k in txt_lower for k in OCR_KEYWORDS):
                        left = int(ocr.get("left", [0])[i])
                        top = int(ocr.get("top", [0])[i])
                        width = int(ocr.get("width", [0])[i])
                        height = int(ocr.get("height", [0])[i])
                        # expand region to the right/down to get signing space
                        expand_w_px = int(width + img_w * 0.35)  # half page to the right heuristically
                        expand_h_px = int(height * 4.0)

                        # convert pixel -> pdf points
                        px_to_pt = page.width / float(img_w)
                        x_pt = left * px_to_pt
                        width_pt = min(page.width, expand_w_px * px_to_pt)
                        # Convert top-left image px -> bottom-left PDF pt
                        y_pt_top = (top + height) * px_to_pt
                        y_pt = page.height - y_pt_top  # bottom-left origin
                        height_pt = expand_h_px * px_to_pt

                        candidates.append({
                            "page": pageno,
                            "x": max(0, x_pt),
                            "y": max(0, y_pt),
                            "width": max(50, width_pt),
                            "height": max(30, height_pt),
                            "score": 0.95,
                            "reason": f"ocr_keyword:{text}"
                        })

                # whitespace heuristic: coarse occupancy scanning to find empty horizontal blocks
                # build occupancy grid on vertical slices
                try:
                    words_boxes = []
                    for i in range(n):
                        t = (ocr.get("text", [""])[i] or "").strip()
                        if not t:
                            continue
                        left = int(ocr.get("left", [0])[i])
                        top = int(ocr.get("top", [0])[i])
                        w = int(ocr.get("width", [0])[i])
                        h = int(ocr.get("height", [0])[i])
                        words_boxes.append((left, top, w, h))
                    # coarse rows
                    rows = 40
                    row_h = max(1, img_h // rows)
                    occupancy = [0] * rows
                    for (left, top, w, h) in words_boxes:
                        top_row = max(0, min(rows-1, top // row_h))
                        bottom_row = max(0, min(rows-1, (top + h) // row_h))
                        for r in range(top_row, bottom_row + 1):
                            occupancy[r] += 1
                    # find long empty ranges
                    threshold = 1
                    start = None
                    for r in range(rows):
                        if occupancy[r] <= threshold:
                            if start is None:
                                start = r
                        else:
                            if start is not None:
                                length = r - start
                                if length >= 2:
                                    y_pixel = start * row_h
                                    h_pixel = length * row_h
                                    # map to pdf pts
                                    px_to_pt = page.width / float(img_w)
                                    x_pt = 20 * px_to_pt
                                    width_pt = max(100, page.width - 40 * px_to_pt)
                                    y_pt = page.height - ((y_pixel + h_pixel) * px_to_pt)
                                    height_pt = h_pixel * px_to_pt
                                    candidates.append({
                                        "page": pageno,
                                        "x": x_pt,
                                        "y": max(0, y_pt),
                                        "width": width_pt,
                                        "height": max(30, height_pt),
                                        "score": 0.55,
                                        "reason": "whitespace_block"
                                    })
                                start = None
                    if start is not None:
                        r = rows
                        length = r - start
                        if length >= 2:
                            y_pixel = start * row_h
                            h_pixel = length * row_h
                            px_to_pt = page.width / float(img_w)
                            x_pt = 20 * px_to_pt
                            width_pt = max(100, page.width - 40 * px_to_pt)
                            y_pt = page.height - ((y_pixel + h_pixel) * px_to_pt)
                            height_pt = h_pixel * px_to_pt
                            candidates.append({
                                "page": pageno,
                                "x": x_pt,
                                "y": max(0, y_pt),
                                "width": width_pt,
                                "height": max(30, height_pt),
                                "score": 0.45,
                                "reason": "whitespace_tail"
                            })
                except Exception:
                    logger.exception("Whitespace scanning failed on page %d", pageno)
    except Exception:
        logger.exception("find_signature_candidates_by_ocr failed for %s", pdf_path)

    # sort by score desc
    candidates = sorted(candidates, key=lambda c: float(c.get("score", 0)), reverse=True)
    logger.debug("find_signature_candidates_by_ocr -> found %d candidates", len(candidates))
    # filter by min score
    filtered = [c for c in candidates if float(c.get("score", 0)) >= min_candidate_score]
    return filtered or candidates  # return filtered if any, else original list (so caller can fallback)

# ---------------------------
# Signature sizing
# ---------------------------
def compute_signature_dims(candidate: Candidate, signature_img_path: str,
                           max_fraction_width: float = 0.85,
                           min_fraction_width: float = 0.25) -> Tuple[float, float]:
    """
    Given a candidate box (in PDF points) and a signature image path,
    return (width_pt, height_pt) for drawing preserving aspect ratio and fitting inside candidate.
    """
    img = _open_image_safe(signature_img_path)
    img_w_px, img_h_px = img.size
    aspect = img_h_px / float(img_w_px)  # height / width

    avail_w = float(candidate.get("width", 200))
    avail_h = float(candidate.get("height", 100))

    # fraction heuristics
    if avail_w < 200:
        frac = max(min_fraction_width, 0.35)
    else:
        frac = min(max_fraction_width, 0.7 + min(avail_w / 1200.0, 0.2))

    target_w = avail_w * frac
    target_h = target_w * aspect

    # clamp by available height
    if target_h > (avail_h * 0.95):
        target_h = avail_h * 0.95
        target_w = target_h / aspect

    target_w = max(30.0, min(target_w, avail_w))
    target_h = max(10.0, min(target_h, avail_h))
    logger.debug("compute_signature_dims -> avail=(%0.1f,%0.1f) => target=(%0.1f,%0.1f)",
                 avail_w, avail_h, target_w, target_h)
    return target_w, target_h

# ---------------------------
# Overlay function
# ---------------------------
def overlay_signature_on_pdf_at_candidates(pdf_path: str,
                                           signature_img_path: str,
                                           candidates: List[Candidate],
                                           pick_first: bool = True,
                                           output_path: Optional[str] = None) -> io.BytesIO:
    """
    Overlay signature image on the PDF at the best candidate(s).
    - candidates coordinates must be in PDF points with origin bottom-left.
    - pick_first: place only on the top-scoring candidate if True; otherwise place on all.
    Returns BytesIO stream with final PDF.
    """
    out_stream = io.BytesIO()
    try:
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        # if no candidates, fallback to bottom-right
        if not candidates:
            logger.info("No candidates provided: using bottom-right fallback.")
            # build single fallback candidate per first page
            first_media = reader.pages[0].mediabox
            fallback = {
                "page": 0,
                "x": float(first_media.width) - 250,
                "y": 40.0,
                "width": 200.0,
                "height": 80.0,
                "score": 0.1,
                "reason": "fallback_bottom_right"
            }
            candidates = [fallback]

        # group candidates by page
        cand_by_page = {}
        for c in candidates:
            p = int(c.get("page", 0))
            cand_by_page.setdefault(p, []).append(c)

        for p_index, page in enumerate(reader.pages):
            media = page.mediabox
            pw = float(media.width)
            ph = float(media.height)

            # prepare overlay canvas same size as page (points)
            packet = io.BytesIO()
            c = canvas.Canvas(packet, pagesize=(pw, ph))

            page_cands = cand_by_page.get(p_index, [])
            if page_cands:
                # select candidates to draw
                to_draw = [page_cands[0]] if pick_first else page_cands
                for cand in to_draw:
                    try:
                        w_pt, h_pt = compute_signature_dims(cand, signature_img_path)
                        # anchor: left-align inside candidate x, center vertically inside candidate
                        x = float(cand.get("x", 0.0))
                        y = float(cand.get("y", 0.0))
                        # ensure inside page bounds
                        x = max(5.0, min(x, pw - w_pt - 5.0))
                        y = max(5.0, min(y, ph - h_pt - 5.0))

                        # draw image
                        c.drawImage(signature_img_path, x, y, width=w_pt, height=h_pt, mask='auto')
                        logger.info("Placed signature on page %d at x=%.1f y=%.1f w=%.1f h=%.1f reason=%s",
                                    p_index, x, y, w_pt, h_pt, cand.get("reason"))
                    except Exception:
                        logger.exception("Failed to draw signature for candidate %s", cand)
            c.save()
            packet.seek(0)

            try:
                overlay_pdf = PdfReader(packet)
                # Merge overlay page onto original
                base_page = page
                # merge_page is legacy; some PyPDF2 versions use merge_page, others use merge_page
                try:
                    base_page.merge_page(overlay_pdf.pages[0])
                except Exception:
                    # fallback to merging using /Contents
                    base_page.merge_page(overlay_pdf.pages[0])
                writer.add_page(base_page)
            except Exception:
                logger.exception("Failed to merge overlay on page %d", p_index)
                writer.add_page(page)

        writer.write(out_stream)
        out_stream.seek(0)

        # optional save
        if output_path:
            with open(output_path, "wb") as f:
                f.write(out_stream.getbuffer())

        return out_stream
    except Exception:
        logger.exception("overlay_signature_on_pdf_at_candidates failed for %s", pdf_path)
        raise
