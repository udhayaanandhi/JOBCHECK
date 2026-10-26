import os, re, html, json, uuid, sqlite3
from datetime import datetime
from urllib.parse import urlparse
import joblib, numpy as np, pandas as pd
from flask import (Flask, render_template, request, jsonify, Response,
                   redirect, url_for, abort, send_from_directory)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models.joblib")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "model_metrics.csv")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "jobcheck_history.sqlite3")
os.makedirs(UPLOAD_DIR, exist_ok=True)

bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Fields shown on the website (names unchanged)
FORM_FIELDS = ["title", "company_profile", "description", "requirements", "benefits", "location"]
# Same order used during training
MODEL_FIELDS = ["title", "company_profile", "description", "requirements", "benefits", "location",
                "department", "employment_type", "required_experience", "required_education",
                "industry", "function"]
# Employer details: used ONLY for the consistency checklist, never by the models
EMP_FIELDS = ["website", "recruiter_email", "contact"]
LABELS = {"title": "Job title", "company_profile": "About the company", "description": "Job description",
          "requirements": "Skills & requirements", "benefits": "Benefits & compensation", "location": "Location"}


def clean_text(value):
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


# ---------------- prediction (unchanged logic) ----------------
def predict_job(text):
    vectorizer = bundle["vectorizer"]
    models = bundle["models"]
    X = vectorizer.transform([text])
    X_knn = bundle["scaler"].transform(bundle["svd"].transform(X))
    results = []
    for name, model in models.items():
        if name == "K-Nearest Neighbors":
            prediction = int(model.predict(X_knn)[0])
        else:
            prediction = int(model.predict(X)[0])
        results.append({"model": name, "prediction": prediction,
                        "label": "Potentially Fake" if prediction == 1 else "Likely Genuine",
                        "class": "fake" if prediction == 1 else "genuine"})
    final_result = next((r for r in results if r["model"] == "Hard Voting Classifier"), None)
    if final_result is None:
        raise ValueError("Hard Voting Classifier not found.")
    fake_votes = sum(r["prediction"] == 1 for r in results)
    return results, final_result, fake_votes, len(results) - fake_votes


def predict_batch(texts):
    X = bundle["vectorizer"].transform(texts)
    Xk = bundle["scaler"].transform(bundle["svd"].transform(X))
    preds, scores = {}, {}
    for name, m in bundle["models"].items():
        xi = Xk if name == "K-Nearest Neighbors" else X
        preds[name] = np.asarray(m.predict(xi)).astype(int)
        if hasattr(m, "predict_proba"):
            scores[name] = m.predict_proba(xi)[:, 1]
        elif hasattr(m, "decision_function"):
            scores[name] = m.decision_function(xi)
    return preds, scores


# ---------------- explainable warning signs ----------------
def _r(p): return re.compile(p, re.I)
RULES = [
    ("Fee or payment request",
     _r(r"\b(registration|processing|security|training|joining|application|refundable)\s+(fee|fees|deposit|charges?)\b|\bpay\s+(a|an|the)?\s*[\w\s]{0,20}\b(fee|deposit)\b|\bsend\s+money\b"),
     "Genuine employers do not normally charge candidates to be hired."),
    ("Request for sensitive credentials",
     _r(r"\botp\b|\bpassword\b|bank\s+(account|details|login)|\bcvv\b|aadhaar|pan\s+card|credit\s+card"),
     "Never share OTPs, passwords or banking details as part of a job application."),
    ("Unrealistic earnings",
     _r(r"earn\s+(up\s+to\s+)?(?:₹|rs\.?|inr|\$)?\s*\d[\d,]*\s*(?:per|/|a)\s*(?:day|week)|guaranteed\s+income|easy\s+money|unlimited\s+earning|\blakhs?\b\s+(per|a)\s+(month|week)"),
     "Very high pay for little effort or skill is a common lure in job scams."),
    ("Pressure to apply immediately",
     _r(r"apply\s+(now|immediately|today)|limited\s+(slots|seats|vacancies)|urgent(ly)?\s+hiring|hurry|last\s+date\s+today"),
     "Artificial urgency discourages you from researching the employer."),
    ("Informal recruiter contact",
     _r(r"whats\s?app|telegram|text\s+hr|\bdm\s+(us|me)\b"),
     "Official recruitment normally uses company email or an official portal."),
    ("Vague selection claims",
     _r(r"no\s+(experience|interview|qualification)s?(\s+(needed|required))?|100%\s*(selection|placement|guarantee)|direct\s+joining|guaranteed\s+(job|selection)"),
     "Claims that skip screening entirely are unusual for genuine roles."),
]


def find_warnings(fields):
    out, seen = [], set()
    for f in FORM_FIELDS:
        txt = clean_text(fields.get(f, ""))
        for name, rx, why in RULES:
            for m in rx.finditer(txt):
                phrase = m.group(0).strip()
                key = (name, phrase.lower(), f)
                if key not in seen:
                    seen.add(key)
                    out.append({"rule": name, "phrase": phrase, "field": LABELS[f], "why": why})
    return out


# ---------------- employer consistency checklist ----------------
FREE_MAIL = {"gmail.com", "yahoo.com", "yahoo.in", "outlook.com", "hotmail.com", "rediffmail.com",
             "proton.me", "protonmail.com", "aol.com", "icloud.com"}


def _host(u):
    u = (u or "").strip().lower()
    if not u:
        return ""
    h = urlparse(u if "//" in u else "//" + u).hostname or ""
    return h[4:] if h.startswith("www.") else h


def employer_checks(website, email, contact):
    """Consistency checks only. Nothing here independently verifies a company."""
    if not (website or email or contact):
        return []
    c = []
    add = lambda s, l, d: c.append({"status": s, "label": l, "detail": d})
    wd = _host(website)
    ed = email.split("@")[-1].strip().lower() if "@" in email else ""
    if website:
        if website.strip().lower().startswith("http://"):
            add("warn", "Website does not use HTTPS", "Prefer sites that load securely over https.")
        else:
            add("info", "Website domain noted", f"{wd or website}: open it yourself and confirm it describes this role.")
    if email:
        if not ed or "." not in ed:
            add("warn", "Recruiter email looks incomplete", "Check the address for typing errors.")
        elif ed in FREE_MAIL:
            add("warn", "Free email provider used", f"{ed} is a public provider; companies usually recruit from their own domain.")
        elif wd:
            if ed == wd or ed.endswith("." + wd):
                add("ok", "Email and website domains are consistent", f"Both use {wd}. This is consistent, not proof of identity.")
            else:
                add("warn", "Email and website domains differ", f"Email uses {ed} but the website is {wd}.")
        else:
            add("info", "Email domain noted", f"{ed}. Add the website to compare domains.")
    if contact:
        if re.search(r"whats\s?app|telegram", contact, re.I):
            add("warn", "Contact is a chat app", "Prefer an official company phone number or email.")
        elif len(re.sub(r"\D", "", contact)) < 8:
            add("warn", "Contact does not look like a full phone number", "Ask for an official number.")
        else:
            add("info", "Contact detail provided", "Call the company's official number from its own website and confirm the role.")
    add("info", "This is not verification",
        "JobCheck only compares what you typed. It has not confirmed that this company or recruiter exists.")
    return c


def disagreement(fv, gv):
    n = fv + gv
    if fv == n:
        return {"level": "agree", "text": f"All {n} classifiers flag this posting as potentially fake."}
    if gv == n:
        return {"level": "agree", "text": f"All {n} classifiers classify this posting as likely genuine."}
    return {"level": "split", "text": f"The models disagree: {fv} of {n} flag this posting as fake while {gv} classify it as genuine. "
                                      "Treat the result as uncertain and review the evidence carefully."}


def recommendations(p):
    r = ["Verify the employer on its official website and professional networks before sharing documents.",
         "Never pay money or share OTPs, passwords or banking details to get a job.",
         "Contact the company using details you found yourself, not only those in the posting."]
    if p["warnings"]:
        r.insert(0, "Review each highlighted warning phrase. A warning is not proof of fraud, but each deserves a closer look.")
    if p["split"]["level"] == "split":
        r.insert(0, "The models disagree, so do not rely on the prediction alone.")
    return r


# ---------------- scan history & feedback (new tables; existing ones are untouched) ----------------
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS scans_v2(id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, title TEXT,
            label TEXT, fake_votes INTEGER, total INTEGER, payload TEXT);
        CREATE TABLE IF NOT EXISTS feedback_v2(id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id INTEGER,
            created TEXT, opinion TEXT, note TEXT);""")


init_db()


def load_metrics():
    if os.path.exists(METRICS_PATH):
        try:
            return pd.read_csv(METRICS_PATH).fillna("").to_dict("records")
        except Exception:
            pass
    return []


# ---------------- routes ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    ctx = dict(results=None, vote_result=None, error=None, submitted={}, fake_votes=0, genuine_votes=0,
               warnings=[], employer=[], split=None, scan_id=None)
    if request.method == "POST":
        sub = {f: request.form.get(f, "").strip() for f in FORM_FIELDS + EMP_FIELDS}
        ctx["submitted"] = sub
        text = " ".join(clean_text(sub.get(f, "")) for f in MODEL_FIELDS).strip()
        if len(text.split()) < 5:
            ctx["error"] = "Please enter sufficient job details."
        elif bundle is None:
            ctx["error"] = "Model not found. Please train the model first."
        else:
            try:
                res, final, fv, gv = predict_job(text)
                p = {"submitted": sub, "results": res, "final": final, "fv": fv, "gv": gv,
                     "warnings": find_warnings(sub),
                     "employer": employer_checks(sub["website"], sub["recruiter_email"], sub["contact"]),
                     "split": disagreement(fv, gv)}
                with db() as c:
                    cur = c.execute("INSERT INTO scans_v2(created,title,label,fake_votes,total,payload) VALUES(?,?,?,?,?,?)",
                                    (datetime.now().strftime("%Y-%m-%d %H:%M"), sub["title"][:120], final["label"], fv, len(res), json.dumps(p)))
                    ctx["scan_id"] = cur.lastrowid
                ctx.update(results=res, vote_result=final, fake_votes=fv, genuine_votes=gv,
                           warnings=p["warnings"], employer=p["employer"], split=p["split"])
            except Exception as e:
                ctx["error"] = f"Prediction failed: {e}"
    return render_template("index.html", metrics=load_metrics(), ready=bundle is not None, **ctx)


@app.route("/report/<int:sid>")
def report(sid):
    with db() as c:
        row = c.execute("SELECT * FROM scans_v2 WHERE id=?", (sid,)).fetchone()
    if not row:
        abort(404)
    p = json.loads(row["payload"])
    body = render_template("report.html", scan=row, p=p, recs=recommendations(p), labels=LABELS)
    return Response(body, mimetype="text/html",
                    headers={"Content-Disposition": f"attachment; filename=jobcheck_report_{sid}.html"})


@app.route("/history")
def history():
    with db() as c:
        rows = c.execute("SELECT s.*, (SELECT COUNT(*) FROM feedback_v2 f WHERE f.scan_id=s.id) AS fb "
                         "FROM scans_v2 s ORDER BY s.id DESC LIMIT 100").fetchall()
    return render_template("history.html", rows=rows, thanks=request.args.get("thanks"))


@app.route("/feedback/<int:sid>", methods=["POST"])
def feedback(sid):
    op = request.form.get("opinion", "other")[:30]
    note = request.form.get("note", "").strip()[:500]
    with db() as c:
        if not c.execute("SELECT 1 FROM scans_v2 WHERE id=?", (sid,)).fetchone():
            abort(404)
        c.execute("INSERT INTO feedback_v2(scan_id,created,opinion,note) VALUES(?,?,?,?)",
                  (sid, datetime.now().strftime("%Y-%m-%d %H:%M"), op, note))
    return redirect(url_for("history", thanks=1))


@app.route("/performance")
def performance():
    imgs = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith("_confusion_matrix.png")) if os.path.isdir(RESULTS_DIR) else []
    return render_template("performance.html", metrics=load_metrics(), imgs=imgs)


@app.route("/results/<path:fn>")
def results_file(fn):
    if not fn.endswith(".png"):
        abort(404)
    return send_from_directory(RESULTS_DIR, fn)


@app.route("/dataset", methods=["GET", "POST"])
def dataset():
    def page(**kw):
        return render_template("dataset.html", ready=bundle is not None, data=kw.get("data"), error=kw.get("error"))
    if request.method == "GET":
        return page()
    f = request.files.get("file")
    if bundle is None:
        return page(error="Model not found. Please train the model first.")
    if not f or not f.filename:
        return page(error="Please choose a CSV or Excel file.")
    ext = f.filename.lower().rsplit(".", 1)[-1]
    if ext not in ("csv", "xlsx", "xls"):
        return page(error="Unsupported file type. Upload .csv, .xlsx or .xls.")
    try:
        df = pd.read_csv(f) if ext == "csv" else pd.read_excel(f)
    except Exception as e:
        return page(error=f"Could not read the file: {e}")
    low = {str(c).lower(): c for c in df.columns}
    use = [low[c] for c in MODEL_FIELDS if c in low]
    if not use:
        return page(error="No supported job text columns found (e.g. title, description, requirements).")
    texts = [" ".join("" if pd.isna(v) else clean_text(v) for v in row).strip()
             for row in df[use].itertuples(index=False)]
    ok = [i for i, t in enumerate(texts) if len(t.split()) >= 5]
    if not ok:
        return page(error="No rows contained enough text to analyze.")
    failed = len(df) - len(ok)
    preds, scores = predict_batch([texts[i] for i in ok])
    names = list(preds)
    final = preds["Hard Voting Classifier"]
    votes = sum(preds.values())

    fc = {k: low[k] for k in FORM_FIELDS if k in low}
    recs = df[list(fc.values())].fillna("").astype(str).to_dict("records") if fc else [{}] * len(df)
    wc = [len(find_warnings({k: recs[i].get(c, "") for k, c in fc.items()})) for i in ok]
    tcol = low.get("title")
    out = pd.DataFrame({"row": [i + 2 for i in ok],
                        "title": df[tcol].iloc[ok].astype(str).str[:80].values if tcol else "",
                        "prediction": np.where(final == 1, "Potentially Fake", "Likely Genuine"),
                        "fake_votes_of_6": votes, "warning_count": wc})
    for n in names:
        out[n] = np.where(preds[n] == 1, "Fake", "Genuine")

    metrics = cms = None
    labelled = 0
    lc = low.get("fraudulent")
    if lc is not None:
        y = pd.to_numeric(df[lc].iloc[ok], errors="coerce").values
        m = np.isin(y, [0, 1])
        if m.sum() > 0:
            yt = y[m].astype(int)
            labelled, metrics, cms = int(m.sum()), [], []
            for n in names:
                p = preds[n][m]
                row = {"model": n, "accuracy": accuracy_score(yt, p),
                       "precision": precision_score(yt, p, zero_division=0),
                       "recall": recall_score(yt, p, zero_division=0),
                       "f1": f1_score(yt, p, zero_division=0), "auc": None}
                if n in scores and len(set(yt)) == 2:
                    try: row["auc"] = roc_auc_score(yt, scores[n][m])
                    except Exception: pass
                metrics.append(row)
                tn, fp, fn, tp = confusion_matrix(yt, p, labels=[0, 1]).ravel()
                cms.append({"model": n, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
            metrics.sort(key=lambda r: -r["f1"])
    token = uuid.uuid4().hex[:12]
    out.to_csv(os.path.join(UPLOAD_DIR, f"predictions_{token}.csv"), index=False)
    fake_n = int((final == 1).sum())
    data = dict(total=len(df), analyzed=len(ok), failed=failed, fake=fake_n, genuine=len(ok) - fake_n,
                pct=round(fake_n / len(ok) * 100, 1), warned=int(sum(1 for w in wc if w)),
                metrics=metrics, cms=cms, labelled=labelled, token=token,
                preview=out.head(25).to_dict("records"), filename=f.filename)
    return page(data=data)


@app.route("/dataset/download/<token>")
def dataset_download(token):
    if not re.fullmatch(r"[a-f0-9]{12}", token):
        abort(404)
    return send_from_directory(UPLOAD_DIR, f"predictions_{token}.csv", as_attachment=True)


@app.errorhandler(413)
def too_large(e):
    return "File too large (limit 25 MB).", 413


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": bundle is not None})


if __name__ == "__main__":
    app.run(debug=True)
