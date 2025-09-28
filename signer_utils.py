import os
import io
import re
import difflib
import logging
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from PIL import Image
import pdfplumber

# Logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# Keywords + fuzzy helpers
KEYWORDS = [
    "signature", "sign here", "sign:", "sig.", "authorised signatory",
    "signed by", "please sign", "customer signature", "employee signature",
    "signatur", "signat", "siganture", "signature:"
]


def fuzzy_match(text: str, keywords=KEYWORDS, cutoff: float = 0.70) -> bool:
    """
    Quick fuzzy match using substring + difflib fallback.
    Substring test is fast and handles many OCR minor errors.
    """
    if not text:
        return False
    t = text.lower()
    for kw in keywords:
        if kw in t:
            return True
    # difflib fallback
    for kw in keywords:
        if difflib.SequenceMatcher(None, t, kw).ratio() >= cutoff:
            return True
    return False


# -------------------------
# Safe OCR wrapper
# -------------------------
def safe_ocr_image(pil_img: Image.Image, config: str = "--psm 6") -> dict:
    """
    Run pytesseract.image_to_data safely, returning a dict with keys present.
    If Tesseract isn't available or OCR fails, returns empty lists.
    """
    try:
        # PIL -> cv2-compatible array if needed
        if isinstance(pil_img, Image.Image):
            img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            img_cv = pil_img

        data = pytesseract.image_to_data(img_cv, output_type=Output.DICT, config=config)
        # ensure keys exist
        keys = ["level", "page_num", "block_num", "par_num", "line_num", "word_num",
                "left", "top", "width", "height", "conf", "text"]
        out = {}
        for k in keys:
            out[k] = data.get(k, []) if isinstance(data, dict) else []
        return out
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract not found. Please install or set TESSERACT_CMD.")
        return {"text": [], "left": [], "top": [], "width": [], "height": [], "conf": []}
    except Exception as e:
        logger.exception("safe_ocr_image failed: %s", e)
        return {"text": [], "left": [], "top": [], "width": [], "height": [], "conf": []}


# -------------------------
# Convert OCR coords -> PDF coords
# -------------------------
def convert_ocr_to_pdf_coords(ocr_x: float, ocr_y: float, ocr_w: float, ocr_h: float,
                              img_w: float, img_h: float,
                              pdf_w: float, pdf_h: float) -> Tuple[float, float, float, float]:
    """
    Convert OCR pixel coordinates (top-left origin) to PDF points (bottom-left origin),
    scaling properly based on image->pdf dimensions.
    """
    if img_w == 0 or img_h == 0:
        return 0, 0, ocr_w, ocr_h
    scale_x = pdf_w / img_w
    scale_y = pdf_h / img_h

    pdf_x = ocr_x * scale_x
    pdf_w_res = ocr_w * scale_x
    pdf_y = pdf_h - ((ocr_y + ocr_h) * scale_y)  # flip y
    pdf_h_res = ocr_h * scale_y
    return pdf_x, pdf_y, pdf_w_res, pdf_h_res


# -------------------------
# Preprocess image for OCR (cv2)
# -------------------------
def preprocess_image_for_ocr_cv(img_cv: np.ndarray, upscale_min_width: int = 1000) -> np.ndarray:
    """
    Input: cv2 BGR image. Output: processed grayscale image for OCR.
    """
    try:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # bilateral or median for denoising
        denoised = cv2.medianBlur(gray, 3)
        # adaptive threshold to handle uneven lighting
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        h, w = thresh.shape[:2]
        if w < upscale_min_width:
            scale = upscale_min_width / float(w)
            thresh = cv2.resize(thresh, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return thresh
    except Exception:
        logger.exception("preprocess_image_for_ocr_cv failed")
        # fallback: return original converted to grayscale
        try:
            return cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        except Exception:
            return img_cv


# -------------------------
# Remove duplicate/overlapping candidates
# -------------------------
def remove_duplicate_candidates(candidates: List[Dict], threshold_px: float = 30.0) -> List[Dict]:
    """
    Remove near-duplicate candidate boxes (simple non-max suppression by position).
    threshold_px is in PDF points (approx).
    """
    out = []
    for c in sorted(candidates, key=lambda x: -float(x.get("score", 0))):
        keep = True
        cx = c.get("x", 0)
        cy = c.get("y", 0)
        for u in out:
            ux = u.get("x", 0)
            uy = u.get("y", 0)
            if abs(cx - ux) < threshold_px and abs(cy - uy) < threshold_px:
                keep = False
                break
        if keep:
            out.append(c)
    return out


# -------------------------
# Convert image file -> single-page PDF (used by app)
# -------------------------
def convert_image_to_pdf(image_path: str) -> str:
    base_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_pdf = os.path.join(base_dir, f"{base_name}_conv.pdf")
    try:
        img = Image.open(image_path)
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
        img.save(out_pdf, "PDF", resolution=300.0)
        logger.info("Converted image %s -> %s", image_path, out_pdf)
        return out_pdf
    except Exception:
        logger.exception("convert_image_to_pdf failed for %s", image_path)
        raise


# -------------------------
# Extract emails + sig boxes from PDF text layer
# -------------------------
def extract_emails_and_sigboxes_from_pdf(pdf_path: str) -> Tuple[List[str], List[Dict]]:
    emails = []
    sig_boxes = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pageno, page in enumerate(pdf.pages):
                text = (page.extract_text() or "")
                for em in re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text):
                    if em not in emails:
                        emails.append(em)

                try:
                    words = page.extract_words()
                except Exception:
                    words = []

                for w in words:
                    txt = (w.get("text") or "").strip().lower()
                    if fuzzy_match(txt):
                        x0 = float(w.get("x0", 0)) - 5
                        x1 = float(w.get("x1", page.width))
                        top = float(w.get("top", 0)) - 4
                        bottom = float(w.get("bottom", 0)) + 45
                        sig_boxes.append({
                            "page": pageno,
                            "x0": max(0, x0),
                            "x1": min(page.width, x1),
                            "y0_top": max(0, top),
                            "y1_bottom": min(page.height, bottom)
                        })
    except Exception:
        logger.exception("extract_emails_and_sigboxes_from_pdf failed for %s", pdf_path)
    emails = list(dict.fromkeys(emails))
    return emails, sig_boxes


# -------------------------
# Enhanced OCR candidate finder (safe + confidence + scaling)
# -------------------------
def find_signature_candidates_by_ocr(pdf_path: str, dpi: int = 300, conf_threshold: int = 30) -> List[Dict]:
    """
    Return candidate boxes in PDF points with keys:
    {page, x, y, width, height, score, reason}

    conf_threshold: minimum OCR confidence to consider a word (0-100)
    """
    candidates: List[Dict] = []
    if not os.path.exists(pdf_path):
        logger.error("find_signature_candidates_by_ocr: file not found %s", pdf_path)
        return candidates

    try:
        # render pages as images
        pil_pages = convert_from_path(pdf_path, dpi=dpi)
        reader = PdfReader(pdf_path)
    except Exception:
        logger.exception("Failed to open/convert PDF: %s", pdf_path)
        return candidates

    for page_index, pil_img in enumerate(pil_pages):
        try:
            page = reader.pages[page_index]
        except Exception:
            logger.exception("PDF page access failed for index %d", page_index)
            continue

        pdf_w = float(page.mediabox.width)
        pdf_h = float(page.mediabox.height)

        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img_h, img_w = img_cv.shape[:2]

        proc = preprocess_image_for_ocr_cv(img_cv)

        # OCR passes (normal + inverted)
        for cfg_img in [proc, cv2.bitwise_not(proc)]:
            ocr = safe_ocr_image(cfg_img, config="--psm 6")
            texts = ocr.get("text", [])
            lefts = ocr.get("left", [])
            tops = ocr.get("top", [])
            widths = ocr.get("width", [])
            heights = ocr.get("height", [])
            confs = ocr.get("conf", [])

            n = len(texts)
            for i in range(n):
                try:
                    text = (texts[i] or "").strip()
                except Exception:
                    text = ""
                if not text:
                    continue

                # safe confidence parsing
                conf_raw = confs[i] if i < len(confs) else "-1"
                try:
                    conf_val = int(float(conf_raw))
                except Exception:
                    conf_val = -1

                if conf_val < conf_threshold:
                    # skip low-confidence OCR results
                    continue

                # bounding box (safe access)
                lx = int(lefts[i]) if i < len(lefts) else 0
                ty = int(tops[i]) if i < len(tops) else 0
                w_px = int(widths[i]) if i < len(widths) else 0
                h_px = int(heights[i]) if i < len(heights) else 0

                # Font-size like heuristic: skip tiny boxes
                if h_px < 8:
                    continue

                # fuzzy keyword match
                if fuzzy_match(text):
                    pdf_x, pdf_y, pdf_w_box, pdf_h_box = convert_ocr_to_pdf_coords(
                        lx, ty, w_px, h_px, img_w, img_h, pdf_w, pdf_h
                    )
                    # expand box a bit to give room for signature image
                    pdf_w_box = max(pdf_w_box, 120)
                    pdf_h_box = max(pdf_h_box, 30)
                    # keep inside page
                    pdf_x = max(0, min(pdf_x, pdf_w - pdf_w_box))
                    pdf_y = max(0, min(pdf_y, pdf_h - pdf_h_box))
                    candidates.append({
                        "page": page_index,
                        "x": pdf_x,
                        "y": pdf_y,
                        "width": pdf_w_box,
                        "height": pdf_h_box,
                        "score": max(0.5, conf_val / 100.0),
                        "reason": f"ocr:{text[:40]}"
                    })

        # Visual heuristics: detect horizontal lines and boxes (use proc image)
        try:
            edges = cv2.Canny(proc, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=100, maxLineGap=10)
            if lines is not None:
                for ln in lines:
                    x1, y1, x2, y2 = ln[0]
                    if abs(y2 - y1) < 10:
                        # convert coords to pdf units
                        pdf_x, pdf_y, pdf_w_box, pdf_h_box = convert_ocr_to_pdf_coords(
                            min(x1, x2), min(y1, y2)-10, abs(x2-x1), 30, img_w, img_h, pdf_w, pdf_h
                        )
                        pdf_x = max(0, min(pdf_x, pdf_w - pdf_w_box))
                        pdf_y = max(0, min(pdf_y, pdf_h - pdf_h_box))
                        candidates.append({
                            "page": page_index,
                            "x": pdf_x,
                            "y": pdf_y,
                            "width": pdf_w_box,
                            "height": pdf_h_box,
                            "score": 0.45,
                            "reason": "line_detect"
                        })
        except Exception:
            logger.exception("visual heuristics failed on page %d", page_index)

    # dedupe candidates (by proximity)
    candidates = remove_duplicate_candidates(candidates, threshold_px=20.0)
    # sort by score descending
    candidates = sorted(candidates, key=lambda c: float(c.get("score", 0)), reverse=True)
    logger.info("find_signature_candidates_by_ocr found %d candidates", len(candidates))
    return candidates


# -------------------------
# compute_signature_dims (unchanged logic but safe)
# -------------------------
def compute_signature_dims(candidate: dict, signature_img_path: str,
                           max_fraction_width: float = 0.85,
                           min_fraction_width: float = 0.25) -> Tuple[float, float]:
    try:
        img = Image.open(signature_img_path)
    except Exception:
        logger.exception("compute_signature_dims: failed opening signature image %s", signature_img_path)
        return candidate.get("width", 150), candidate.get("height", 50)

    img_w_px, img_h_px = img.size
    aspect = img_h_px / float(img_w_px) if img_w_px else 0.5

    avail_w = float(candidate.get("width", 200))
    avail_h = float(candidate.get("height", 100))

    if avail_w < 200:
        frac = max(min_fraction_width, 0.35)
    else:
        frac = min(max_fraction_width, 0.7 + min(avail_w / 1200.0, 0.2))

    target_w = avail_w * frac
    target_h = target_w * aspect

    if target_h > (avail_h * 0.95):
        target_h = avail_h * 0.95
        target_w = target_h / aspect

    target_w = max(30.0, min(target_w, avail_w))
    target_h = max(10.0, min(target_h, avail_h))
    logger.debug("compute_signature_dims -> avail=(%0.1f,%0.1f) => target=(%0.1f,%0.1f)",
                 avail_w, avail_h, target_w, target_h)
    return target_w, target_h


# -------------------------
# Overlay function: page-aware + bounds checking
# -------------------------
def overlay_signature_on_pdf_at_candidates(pdf_path: str,
                                           signature_img_path: str,
                                           candidates: List[dict],
                                           pick_first: bool = True,
                                           output_path: Optional[str] = None) -> io.BytesIO:
    """
    Overlay signature image on the PDF at the best candidate(s).
    Keeps interface same: single signature_img_path expected.
    """
    out_stream = io.BytesIO()
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not os.path.exists(signature_img_path):
        raise FileNotFoundError(f"Signature image not found: {signature_img_path}")

    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    # Group candidates by page
    by_page = {}
    for c in candidates:
        p = int(c.get("page", 0))
        by_page.setdefault(p, []).append(c)

    for p_index, page in enumerate(reader.pages):
        media = page.mediabox
        pw = float(media.width)
        ph = float(media.height)

        packet = io.BytesIO()
        c_canvas = canvas.Canvas(packet, pagesize=(pw, ph))

        page_cands = by_page.get(p_index, [])
        if page_cands:
            # choose which candidate(s) to draw
            to_draw = [page_cands[0]] if pick_first else page_cands
            for cand in to_draw:
                try:
                    # compute dims
                    w_pt, h_pt = compute_signature_dims(cand, signature_img_path)
                    x = float(cand.get("x", 0.0))
                    y = float(cand.get("y", 0.0))

                    # bounds checking: ensure inside page
                    x = max(5.0, min(x, pw - w_pt - 5.0))
                    y = max(5.0, min(y, ph - h_pt - 5.0))

                    c_canvas.drawImage(signature_img_path, x, y, width=w_pt, height=h_pt, mask='auto')
                    logger.info("Placed signature on page %d at x=%.1f y=%.1f w=%.1f h=%.1f reason=%s",
                                p_index, x, y, w_pt, h_pt, cand.get("reason"))
                except Exception:
                    logger.exception("Failed to draw signature for candidate %s", cand)

        c_canvas.save()
        packet.seek(0)

        try:
            overlay_pdf = PdfReader(packet)
            base_page = page
            base_page.merge_page(overlay_pdf.pages[0])
            writer.add_page(base_page)
        except Exception:
            logger.exception("Failed to merge overlay on page %d", p_index)
            writer.add_page(page)

    writer.write(out_stream)
    out_stream.seek(0)

    if output_path:
        with open(output_path, "wb") as f:
            f.write(out_stream.getbuffer())

    return out_stream
