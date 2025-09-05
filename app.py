import os
import hashlib
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, send_file, flash
from itsdangerous import URLSafeSerializer
from signer_utils import extract_emails_and_sigboxes_from_pdf, extract_emails_and_sigpos_from_docx, overlay_signature_on_pdf
from email.message import EmailMessage
import smtplib
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

app = Flask(__name__)
app.secret_key = "change_this_to_a_secure_random_value"

UPLOAD_FOLDER = "uploads"
SIGNED_FOLDER = "signed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SIGNED_FOLDER, exist_ok=True)

# Email config from .env
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
FROM_EMAIL = os.environ.get("FROM_EMAIL", SMTP_USER)

# Token serializer
serializer = URLSafeSerializer(app.secret_key)


def send_sign_email(to_email, token):
    # link = url_for('sign_document', token=token, _external=True)
    link='https://www.google.com/webhp?hl=en&sa=X&ved=0ahUKEwjZvpPWhsGPAxXeyjgGHZGCHIgQPAgJ'
    msg = EmailMessage()
    msg['Subject'] = 'Please sign the document'
    msg['From'] = FROM_EMAIL
    msg['To'] = to_email
    msg.set_content(f'You were requested to sign a document. Click here to sign: {link}')

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        if not f:
            flash("No file selected")
            return redirect(request.url)

        filename = f.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(save_path)

        with open(save_path, "rb") as fh:
            file_hash = hashlib.sha256(fh.read()).hexdigest()

        emails, sig_boxes = [], None
        if filename.lower().endswith('.pdf'):
            emails, sig_boxes = extract_emails_and_sigboxes_from_pdf(save_path)
        elif filename.lower().endswith(('.docx', '.doc')):
            emails, sig_boxes = extract_emails_and_sigpos_from_docx(save_path)
        else:
            flash("Unsupported file type")
            return redirect(request.url)

        if not emails:
            flash("No signer email found in document")
            return redirect(request.url)

        candidate = emails[0]  # first email for demo
        payload = {"path": save_path, "filename": filename, "email": candidate, "hash": file_hash}
        token = serializer.dumps(payload)

        # Send sign email
        send_sign_email(candidate, token)

        flash(f"Email sent to {candidate}")
        return render_template("upload.html", message=f"Email sent to {candidate}")

    return render_template("upload.html")


@app.route('/sign/<token>', methods=['GET', 'POST'])
def sign_document(token):
    try:
        data = serializer.loads(token)
    except Exception:
        return "Invalid or expired link", 400

    filepath = data['path']
    filename = data['filename']
    recipient = data['email']

    if request.method == 'POST':
        sig_image = os.path.join("static", "signature.png")
        out_stream = overlay_signature_on_pdf(filepath, sig_image)
        signed_name = f"signed_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{filename}"
        out_path = os.path.join(SIGNED_FOLDER, signed_name)

        with open(out_path, "wb") as out_f:
            out_f.write(out_stream.read())

        return send_file(out_path, as_attachment=True)

    return render_template("sign_page.html", recipient=recipient, filename=filename, token=token)


if __name__ == '__main__':
    app.run(debug=True)
