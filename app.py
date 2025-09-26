import os
import hashlib
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, send_file, flash
from itsdangerous import URLSafeSerializer
from signer_utils import (
    extract_emails_and_sigboxes_from_pdf,
    extract_emails_and_sigpos_from_docx,
    overlay_signature_on_pdf_at_candidates,
    find_signature_candidates_by_ocr
)
from email.message import EmailMessage
import smtplib
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change_this_secret")

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SIGNED_FOLDER = os.path.join(BASE_DIR, "signed")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SIGNED_FOLDER, exist_ok=True)

# Email config
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
FROM_EMAIL = os.environ.get("FROM_EMAIL", SMTP_USER)
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:5000")

serializer = URLSafeSerializer(app.secret_key)


# Send signing email
def send_sign_email(to_email, token):
    link = f"{APP_BASE_URL}/place_preview/{token}"
    msg = EmailMessage()
    msg['Subject'] = 'Please review and sign the document'
    msg['From'] = FROM_EMAIL
    msg['To'] = to_email
    msg.set_content(f"You were requested to sign a document. Review it here: {link}")
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)


# Upload route
@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        if not f:
            flash("No file selected")
            return redirect(request.url)

        # Sanitize filename
        filename = secure_filename(f.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(save_path)

        # File hash
        with open(save_path, "rb") as fh:
            file_hash = hashlib.sha256(fh.read()).hexdigest()

        # Extract emails (from doc content)
        emails, sig_boxes = ([], None)
        if filename.lower().endswith('.pdf'):
            emails, sig_boxes = extract_emails_and_sigboxes_from_pdf(save_path)
        elif filename.lower().endswith(('.docx', '.doc')):
            emails, sig_boxes = extract_emails_and_sigpos_from_docx(save_path)

        # Token payload
        payload = {"filename": filename, "hash": file_hash, "email": (emails[0] if emails else "")}
        token = serializer.dumps(payload)

        # Redirect to preview page
        return redirect(url_for('place_signature_preview', token=token))

    return render_template("upload.html")


# Preview route
@app.route('/place_preview/<token>')
def place_signature_preview(token):
    try:
        data = serializer.loads(token)
    except Exception:
        return "Invalid link", 400

    filename = data['filename']
    file_url = url_for('uploaded_file', filename=filename, _external=True)
    return render_template("place_preview.html", file_url=file_url, token=token, filename=filename)


# Place signature route
@app.route('/place_signature/<token>', methods=['POST'])
def place_signature(token):
    try:
        data = serializer.loads(token)
    except Exception:
        return "Invalid link", 400

    filename = data['filename']
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    sig_image_path = os.path.join(STATIC_FOLDER, "signature.png")

    if not os.path.exists(sig_image_path):
        return "Signature image not found. Please upload one to /static/signature.png", 404

    # Step 1: Find candidates by OCR
    candidates = find_signature_candidates_by_ocr(pdf_path, dpi=200)

    # Step 2: Fallback using heuristic if no OCR candidates
    if not candidates:
        _, sig_boxes = extract_emails_and_sigboxes_from_pdf(pdf_path)
        if sig_boxes:
            candidates = []
            for s in sig_boxes:
                candidates.append({
                    "page": s.get('page', 0),
                    "x": s.get('x0', 50),
                    "y": s.get('y0', 50),
                    "width": max(150, s.get('x1', 200) - s.get('x0', 50)),
                    "height": max(40, s.get('y1', 100) - s.get('y0', 50)),
                    "score": 0.4,
                    "reason": "text_heuristic"
                })

    if not candidates:
        return "No suitable place for signature found", 400

    # Step 3: Overlay signature
    out_stream = overlay_signature_on_pdf_at_candidates(pdf_path, sig_image_path, candidates, pick_first=True)

    # Step 4: Save signed file
    signed_name = f"signed_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{filename}"
    out_path = os.path.join(SIGNED_FOLDER, signed_name)
    with open(out_path, "wb") as f:
        f.write(out_stream.read())

    # Return signed file
    return send_file(out_path, as_attachment=True)


# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))


# Serve signed files
@app.route('/signed/<filename>')
def signed_file(filename):
    return send_file(os.path.join(SIGNED_FOLDER, filename))


if __name__ == '__main__':
    app.run(debug=True)
