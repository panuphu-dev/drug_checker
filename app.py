"""
Flask Web Application - Online Drug Sales Legality Checker
Faculty of Pharmaceutical Sciences, Khon Kaen University, 2025
"""

from flask import Flask, render_template_string, request, jsonify
import joblib, re, json, os, numpy as np
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_output')
model = joblib.load(os.path.join(MODEL_DIR, 'best_model.joblib'))
tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
with open(os.path.join(MODEL_DIR, 'evaluation_results.json'), 'r', encoding='utf-8') as f:
    eval_results = json.load(f)
with open(os.path.join(MODEL_DIR, 'model_info.json'), 'r', encoding='utf-8') as f:
    model_info = json.load(f)

stop_words = set(thai_stopwords())
stop_words.update({'ที่','ใน','ของ','และ','เป็น','มี','ได้','ไม่','จะ','กับ',
    'ให้','แต่','ก็','หรือ','ว่า','ถ้า','จาก','โดย','เมื่อ','แล้ว','ยัง',
    'คือ','ด้วย','ไป','มา','ทำ','อยู่','เรา','นี้','ซึ่ง','กัน','ๆ',
    'nan','NaN','None',''})

LEGAL_CRITERIA = {
    'L1':{'name':'Advertising Permit','desc':'The product listing does not display a valid drug advertising license number (Kor Thor.) as required by law.'},
    'L2':{'name':'Drug Registration','desc':'The product listing does not display a valid drug registration number issued by the Thai FDA.'},
    'L3':{'name':'Exaggerated Claims','desc':'The advertisement contains exaggerated, miraculous, or scientifically unsubstantiated efficacy claims.'},
    'L4':{'name':'False Indications','desc':'The advertisement displays false or misleading therapeutic indications not supported by approved labeling.'},
    'L5':{'name':'Misrepresented Composition','desc':'The advertisement misrepresents the drug composition, ingredients, or active substances.'},
    'L6':{'name':'Prohibited Disease Claims','desc':'The advertisement claims efficacy against diseases prohibited from advertising, including cancer, diabetes, cardiovascular disease, renal disease, hepatic disease, AIDS, and tuberculosis.'},
    'L7':{'name':'Abortifacient or Aphrodisiac','desc':'The advertisement promotes drugs intended as abortifacients or aphrodisiacs.'},
    'L8':{'name':'Dangerous Drug Sales','desc':'The listing offers dangerous or specially controlled drugs for sale without a valid pharmaceutical license.'},
    'L9':{'name':'Personal Endorsement','desc':'The advertisement uses personal testimonials or celebrity endorsements to promote drug efficacy.'},
    'L10':{'name':'Inappropriate Methods','desc':'The advertisement employs inappropriate, misleading, or deceptive promotional methods.'},
}
DISEASE_KW = ['มะเร็ง','เบาหวาน','ความดัน','หัวใจ','ไต','ตับ','เอดส์','วัณโรค','โรคผิวหนัง','โรคจิต','อัมพาต','อัมพฤกษ์','ลมชัก']
MIRACLE_KW = ['หาย','หายขาด','ปาฏิหาริย์','มหัศจรรย์','วิเศษ','100%','การันตี','รับรองผล','เห็นผลทันที','ชัวร์']
DANGER_KW  = ['สเตียรอยด์','steroid','ยาปฏิชีวนะ','antibiotic','ยาอันตราย','ยาควบคุมพิเศษ','วัตถุออกฤทธิ์']

def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', '', str(text))
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize_thai(text):
    tokens = word_tokenize(str(text), engine='newmm')
    return ' '.join(t.strip() for t in tokens if t.strip() and len(t.strip()) > 1 and t not in stop_words)

def rule_check(name, ad, has_reg, has_lic):
    c = str(name) + ' ' + str(ad)
    v = []
    if not has_lic: v.append('L1')
    if not has_reg: v.append('L2')
    if any(k in c for k in MIRACLE_KW): v.append('L3')
    if any(k in c for k in DISEASE_KW): v.append('L6')
    if any(k in c.lower() for k in DANGER_KW): v.append('L8')
    return v

def predict(product_name, ad_text, has_reg=False, has_lic=False):
    combined = str(product_name) + ' ' + str(ad_text)
    X = tfidf.transform([tokenize_thai(clean_text(combined))])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else [1-pred, pred]
    violations = rule_check(product_name, ad_text, has_reg, has_lic)
    nv = len(violations)
    if nv >= 2 or (pred == 1 and nv >= 1):
        result, risk = 'ILLEGAL', 'High'
    elif pred == 1:
        result, risk = 'SUSPECTED ILLEGAL', 'Moderate'
    elif nv >= 1:
        result, risk = 'REQUIRES FURTHER REVIEW', 'Moderate'
    else:
        result, risk = 'LAWFUL', 'Low'
    return {
        'result': result, 'ml_prediction': 'Illegal' if pred == 1 else 'Lawful',
        'illegal_probability': round(float(proba[1]) * 100, 1),
        'confidence': round(float(max(proba)) * 100, 1),
        'risk_level': risk, 'violations': violations,
        'violation_details': [{'code': v, **LEGAL_CRITERIA[v]} for v in violations],
        'n_violations': nv,
    }

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Online Drug Sales Legality Checker</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --navy:#1b2a4a;--dark:#0f1c2e;--mid:#2c3e5a;--steel:#4a5c78;--slate:#64748b;
  --light:#8896ab;--bdr:#d1d9e6;--bg:#f4f6f9;--white:#fff;
  --green:#1a7a42;--green-bg:#eaf5ef;--red:#a32a2a;--red-bg:#fbf0f0;--amber:#8a6d1b;--amber-bg:#fdf8eb;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter','Segoe UI','Helvetica Neue',Arial,sans-serif;background:var(--bg);color:var(--navy);line-height:1.65;font-size:15px}
.hdr{background:var(--dark);color:#fff;padding:36px 24px;border-bottom:4px solid var(--navy)}
.hdr-inner{max-width:1000px;margin:0 auto}
.hdr-sup{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.2em;color:var(--light);margin-bottom:8px}
.hdr h1{font-size:1.55rem;font-weight:800;letter-spacing:-.02em;line-height:1.3}
.hdr p{font-size:.85rem;color:var(--light);margin-top:8px}
.wrap{max-width:1000px;margin:0 auto;padding:0 24px}
.tabs{display:flex;border-bottom:1px solid var(--bdr);background:var(--white)}
.tab{padding:13px 24px;border:none;background:transparent;cursor:pointer;font-size:.82rem;font-weight:500;color:var(--slate);font-family:inherit;border-bottom:3px solid transparent;margin-bottom:-1px;text-transform:uppercase;letter-spacing:.05em;transition:all .15s}
.tab:hover{color:var(--navy)}.tab.on{color:var(--navy);font-weight:700;border-bottom-color:var(--navy)}
.pane{display:none;padding:28px 0 60px}.pane.on{display:block;animation:fi .3s ease}
@keyframes fi{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.card{background:var(--white);border:1px solid var(--bdr);padding:32px 28px;margin-bottom:20px}
.card-h{font-size:1rem;font-weight:700;color:var(--navy);margin-bottom:16px;padding-bottom:10px;border-bottom:2px solid var(--navy);text-transform:uppercase;letter-spacing:.06em}
.card-sh{font-size:.92rem;font-weight:600;color:var(--navy);margin-bottom:12px}
.body{font-size:.9rem;color:var(--steel);line-height:1.75}
.fg{margin-bottom:20px}.fg label{display:block;font-weight:600;margin-bottom:6px;font-size:.84rem;color:var(--mid);letter-spacing:.02em}
.fg input,.fg textarea,.fg select{width:100%;padding:10px 14px;border:1px solid var(--bdr);font-size:.9rem;outline:none;font-family:inherit;transition:border .2s;background:var(--white)}
.fg input:focus,.fg textarea:focus{border-color:var(--navy)}
.fg textarea{resize:vertical;min-height:130px;line-height:1.7}
.chk{display:flex;gap:32px;flex-wrap:wrap}.chk label{display:flex;align-items:center;gap:8px;cursor:pointer;font-size:.88rem;color:var(--mid);font-weight:400}
.chk input[type=checkbox]{width:16px;height:16px;accent-color:var(--navy)}
.btn{padding:11px 32px;background:var(--navy);color:#fff;border:none;font-size:.88rem;font-weight:700;cursor:pointer;font-family:inherit;text-transform:uppercase;letter-spacing:.06em;transition:all .2s}
.btn:hover{background:var(--dark)}.btn:disabled{opacity:.45;cursor:not-allowed}
.res{border:1px solid var(--bdr);border-left:5px solid;padding:28px;margin-top:16px;animation:fi .35s ease}
.res-lawful{border-left-color:var(--green);background:var(--green-bg)}
.res-illegal{border-left-color:var(--red);background:var(--red-bg)}
.res-review{border-left-color:var(--amber);background:var(--amber-bg)}
.res-sup{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--slate);margin-bottom:6px}
.res-title{font-size:1.4rem;font-weight:800;letter-spacing:.02em}
.res-sub{font-size:.85rem;color:var(--steel);margin-top:4px}
.met-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:20px 0}
.met{padding:16px;background:rgba(255,255,255,.7);border:1px solid var(--bdr);text-align:center}
.met-v{font-size:1.4rem;font-weight:800;color:var(--navy)}.met-l{font-size:.72rem;color:var(--slate);font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-top:4px}
.viol{padding:12px 16px;margin-bottom:8px;background:rgba(255,255,255,.6);border:1px solid rgba(163,42,42,.12);font-size:.88rem;line-height:1.6}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;margin-bottom:24px;border:1px solid var(--bdr)}
.stat{background:var(--white);padding:22px 20px;text-align:center}
.stat-v{font-size:1.6rem;font-weight:800;color:var(--navy);letter-spacing:-.02em}
.stat-l{font-size:.75rem;color:var(--slate);font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-top:4px}
.stat-s{font-size:.72rem;color:var(--light);margin-top:2px}
.tbl{width:100%;border-collapse:collapse;font-size:.84rem}
.tbl th{background:var(--navy);color:#fff;padding:11px 10px;font-weight:600;font-size:.78rem;letter-spacing:.04em;text-transform:uppercase;border-bottom:2px solid var(--dark)}
.tbl td{padding:10px 12px;border-bottom:1px solid var(--bdr);font-variant-numeric:tabular-nums}
.tbl .best{background:#eef2f9;font-weight:600}.tbl .best td:first-child{color:var(--navy);font-weight:700}
.tbl tr:nth-child(even){background:var(--bg)}
.bar-row{display:flex;align-items:center;margin-bottom:12px;gap:12px}
.bar-lbl{width:150px;text-align:right;font-size:.82rem}
.bar-bg{flex:1;background:#e8ecf1;height:24px;overflow:hidden}
.bar-fill{height:100%;display:flex;align-items:center;justify-content:flex-end;padding-right:8px;font-size:.72rem;font-weight:700;color:#fff;transition:width .8s ease-out}
.cm{display:grid;grid-template-columns:100px 1fr 1fr;gap:1px;max-width:380px;margin:20px auto;font-size:.82rem;background:var(--bdr)}
.cm div{padding:12px;text-align:center;font-weight:600}
.cm-h{background:var(--navy);color:#fff}
.cm-tn{background:#dce6f0;color:var(--navy);font-weight:700;font-size:.92rem}
.cm-fp{background:#f5dede;color:var(--red);font-weight:700;font-size:.92rem}
.cm-fn{background:#f5edd8;color:var(--amber);font-weight:700;font-size:.92rem}
.cm-tp{background:#d8eede;color:var(--green);font-weight:700;font-size:.92rem}
.ftr{border-top:1px solid var(--bdr);background:var(--white);padding:24px;text-align:center}
.ftr p{font-size:.8rem;color:var(--slate)}.ftr p+p{margin-top:4px;font-size:.76rem;color:var(--light)}
@media(max-width:768px){.hdr h1{font-size:1.3rem}.tab{padding:10px 14px;font-size:.76rem}.stats{grid-template-columns:repeat(2,1fr)}.bar-lbl{width:100px;font-size:.72rem}.card{padding:24px 18px}.cm{max-width:300px}}
</style>
</head>
<body>

<div class="hdr"><div class="hdr-inner">
  <div class="hdr-sup">Faculty of Pharmaceutical Sciences &mdash; Khon Kaen University</div>
  <h1>Online Drug Sales Legality Checker</h1>
  <p>Machine Learning System for Detection of Illegal Pharmaceutical Advertisements on E-Commerce Platforms</p>
</div></div>

<div class="wrap">
<div class="tabs">
  <button class="tab on" onclick="go('check',this)">Legality Check</button>
  <button class="tab" onclick="go('dash',this)">Model Performance</button>
  <button class="tab" onclick="go('crit',this)">Legal Criteria</button>
  <button class="tab" onclick="go('about',this)">About</button>
</div>

<div id="check" class="pane on">
<div class="card">
  <h2 class="card-h">Product Verification Form</h2>
  <p class="body" style="margin-top:-6px;margin-bottom:24px">Enter the product name and full advertisement text from a Shopee or Lazada listing below. The system will evaluate legal compliance based on the Thai Drug Act B.E. 2510.</p>
  <form id="frm" onsubmit="return send(event)">
    <div class="fg"><label>Platform</label><select name="platform"><option>Shopee</option><option>Lazada</option><option>Other</option></select></div>
    <div class="fg"><label>Product Name <span style="color:var(--red)">*</span></label><input name="product_name" required placeholder="Enter the product name as displayed on the listing"></div>
    <div class="fg"><label>Advertisement Text / Product Description <span style="color:var(--red)">*</span></label><textarea name="ad_text" required placeholder="Paste the complete advertisement text or product description from the e-commerce listing"></textarea></div>
    <div class="fg"><label>Regulatory Information</label><div class="chk"><label><input type="checkbox" name="has_registration"> Has FDA Thailand registration number</label><label><input type="checkbox" name="has_ad_license"> Has advertising permit number (Kor Thor.)</label></div></div>
    <button class="btn" id="sbtn" type="submit">Submit for Verification</button>
  </form>
</div>
<div id="resArea"></div>
</div>

<div id="dash" class="pane">
<div class="stats">
  <div class="stat"><div class="stat-v">{{ mi.n_samples }}</div><div class="stat-l">Total Samples</div><div class="stat-s">Shopee and Lazada combined</div></div>
  <div class="stat"><div class="stat-v">{{ mi.n_legal }}</div><div class="stat-l">Classified Lawful</div><div class="stat-s">{{ "%.1f"|format(mi.n_legal/mi.n_samples*100) }}% of dataset</div></div>
  <div class="stat"><div class="stat-v">{{ mi.n_illegal }}</div><div class="stat-l">Classified Illegal</div><div class="stat-s">{{ "%.1f"|format(mi.n_illegal/mi.n_samples*100) }}% of dataset</div></div>
  <div class="stat"><div class="stat-v">1 : 12</div><div class="stat-l">Class Imbalance Ratio</div><div class="stat-s">Minority to majority</div></div>
</div>

<div class="card">
  <h2 class="card-h">Comparative Model Performance</h2>
  <p class="body" style="margin-top:-6px;margin-bottom:20px">All models were evaluated using 5-Fold Stratified Cross-Validation. SMOTE oversampling was applied within each training fold to prevent data leakage. The best-performing model is indicated with an asterisk.</p>
  <div style="overflow-x:auto">
  <table class="tbl"><thead><tr>
    <th style="text-align:left">Model</th><th>Accuracy</th><th>Precision</th><th>Sensitivity</th><th>Specificity</th><th>F1-Score</th><th>AUC-ROC</th>
  </tr></thead><tbody>
  {% for n,r in res.items() %}<tr class="{{ 'best' if n==mi.best_model }}">
    <td style="text-align:left">{{ n }}{{ ' *' if n==mi.best_model }}</td>
    <td>{{ "%.2f"|format(r.accuracy*100) }}%</td><td>{{ "%.2f"|format(r.precision*100) }}%</td>
    <td>{{ "%.2f"|format(r.recall_sensitivity*100) }}%</td><td>{{ "%.2f"|format(r.specificity*100) }}%</td>
    <td style="font-weight:700">{{ "%.4f"|format(r.f1_score) }}</td><td>{{ "%.4f"|format(r.auc_roc) }}</td>
  </tr>{% endfor %}
  </tbody></table></div>
  <p style="font-size:.78rem;color:var(--light);margin-top:12px;font-style:italic">* Denotes the selected model based on highest F1-Score.</p>
</div>

<div class="card">
  <h2 class="card-h">F1-Score Comparison</h2>
  {% for n,r in res.items() %}<div class="bar-row">
    <div class="bar-lbl" style="font-weight:{{ '700' if n==mi.best_model else '400' }};color:{{ 'var(--navy)' if n==mi.best_model else 'var(--steel)' }}">{{ n }}{{ ' *' if n==mi.best_model }}</div>
    <div class="bar-bg"><div class="bar-fill" style="width:{{ (r.f1_score/0.5*100)|round }}%;background:{{ 'var(--navy)' if n==mi.best_model else 'var(--light)' }}">{{ "%.4f"|format(r.f1_score) }}</div></div>
  </div>{% endfor %}
</div>

<div class="card">
  <h2 class="card-h">AUC-ROC Comparison</h2>
  {% for n,r in res.items() %}<div class="bar-row">
    <div class="bar-lbl" style="font-weight:{{ '700' if n==mi.best_model else '400' }};color:{{ 'var(--navy)' if n==mi.best_model else 'var(--steel)' }}">{{ n }}{{ ' *' if n==mi.best_model }}</div>
    <div class="bar-bg"><div class="bar-fill" style="width:{{ (r.auc_roc*100)|round }}%;background:{{ 'var(--navy)' if n==mi.best_model else 'var(--light)' }}">{{ "%.4f"|format(r.auc_roc) }}</div></div>
  </div>{% endfor %}
</div>

{% set cm = res[mi.best_model].confusion_matrix %}
<div class="card">
  <h2 class="card-h">Confusion Matrix &mdash; {{ mi.best_model }}</h2>
  <div class="cm">
    <div style="background:var(--bg)"></div><div class="cm-h">Predicted Lawful</div><div class="cm-h">Predicted Illegal</div>
    <div class="cm-h">Actual Lawful</div><div class="cm-tn">TN = {{ cm[0][0] }}</div><div class="cm-fp">FP = {{ cm[0][1] }}</div>
    <div class="cm-h">Actual Illegal</div><div class="cm-fn">FN = {{ cm[1][0] }}</div><div class="cm-tp">TP = {{ cm[1][1] }}</div>
  </div>
  <p style="text-align:center;color:var(--light);font-size:.76rem;margin-top:8px">TN = True Negative, FP = False Positive, FN = False Negative, TP = True Positive</p>
</div>

<div class="card">
  <h2 class="card-h">Summary of Findings</h2>
  <p class="body" style="margin-bottom:14px">Five machine learning models were evaluated using 5-Fold Stratified Cross-Validation with SMOTE oversampling applied within each training fold to prevent data leakage. <strong>{{ mi.best_model }}</strong> achieved the highest F1-Score of {{ "%.4f"|format(res[mi.best_model].f1_score) }}, with a Sensitivity of {{ "%.2f"|format(res[mi.best_model].recall_sensitivity*100) }}% and Specificity of {{ "%.2f"|format(res[mi.best_model].specificity*100) }}%.</p>
  <p class="body">The primary challenge is severe class imbalance (ratio 1:12), with only {{ mi.n_illegal }} illegal samples out of {{ mi.n_samples }} total. To address this limitation, the system employs a hybrid approach combining ML predictions with rule-based legal criteria derived from the Thai Drug Act B.E. 2510, which significantly improves detection reliability in operational use.</p>
</div>
</div>

<div id="crit" class="pane">
<div class="card">
  <h2 class="card-h">Legal Assessment Criteria</h2>
  <p class="body" style="margin-top:-6px;margin-bottom:20px">The following ten criteria are derived from the Thai Drug Act B.E. 2510 (1967) and related ministerial regulations. They define the legal framework used to evaluate whether online drug sales and advertisements comply with Thai pharmaceutical law.</p>
  <table class="tbl"><thead><tr>
    <th style="text-align:left;width:60px">Code</th><th style="text-align:left;width:180px">Criterion</th><th style="text-align:left">Description</th>
  </tr></thead><tbody>
  {% for k,c in criteria.items() %}<tr>
    <td style="text-align:left;font-weight:700;color:var(--navy)">{{ k }}</td>
    <td style="text-align:left;font-weight:600;color:var(--mid)">{{ c.name }}</td>
    <td style="text-align:left;color:var(--steel);line-height:1.6">{{ c.desc }}</td>
  </tr>{% endfor %}
  </tbody></table>
</div>
<div class="card">
  <h2 class="card-h">Classification Logic</h2>
  <p class="body" style="margin-top:-6px;margin-bottom:20px">The system produces a final determination by combining the ML model prediction with the rule-based criteria assessment. The decision matrix is as follows:</p>
  <table class="tbl"><thead><tr>
    <th style="text-align:left">Determination</th><th style="text-align:left">Conditions</th>
  </tr></thead><tbody>
    <tr><td style="text-align:left;font-weight:700;color:var(--navy)">Lawful</td><td style="text-align:left;color:var(--steel);line-height:1.6">No violations detected by either the ML model or rule-based criteria.</td></tr>
    <tr><td style="text-align:left;font-weight:700;color:var(--navy);white-space:nowrap">Requires Further Review</td><td style="text-align:left;color:var(--steel);line-height:1.6">Exactly one rule-based violation detected, or the ML model flags the listing for review.</td></tr>
    <tr><td style="text-align:left;font-weight:700;color:var(--navy);white-space:nowrap">Suspected Illegal</td><td style="text-align:left;color:var(--steel);line-height:1.6">The ML model predicts illegal status, but no rule-based violations are identified.</td></tr>
    <tr><td style="text-align:left;font-weight:700;color:var(--navy)">Illegal</td><td style="text-align:left;color:var(--steel);line-height:1.6">Two or more rule-based violations are detected, or both the ML model and rule-based criteria indicate non-compliance.</td></tr>
  </tbody></table>
</div>
</div>

<div id="about" class="pane">
<div class="card">
  <h2 class="card-h">Project Information</h2>
  <div style="margin-bottom:28px">
    <div class="card-sh">Title</div>
    <p class="body" style="font-weight:500;color:var(--navy)">Guidelines for Addressing the Problem of Online Drug Sales and the Development of Machine Learning for Verifying the Accuracy of Online Drug Advertisements</p>
    <p style="font-size:.82rem;color:var(--light);margin-top:4px">Special Project in Community Pharmacy (PS 125 776)</p>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:28px;margin-bottom:28px">
    <div>
      <div class="card-sh">Researchers</div>
      <table style="font-size:.88rem;color:var(--steel);line-height:2;border-collapse:collapse">
        <tr><td style="padding-right:16px;font-weight:500;color:var(--mid)">Norapat Khawsee</td><td style="color:var(--light)">643150079-4</td></tr>
        <tr><td style="padding-right:16px;font-weight:500;color:var(--mid)">Sirimongkol Phudphong</td><td style="color:var(--light)">643150091-4</td></tr>
        <tr><td style="padding-right:16px;font-weight:500;color:var(--mid)">Phatharadanai Layom</td><td style="color:var(--light)">643150188-9</td></tr>
      </table>
    </div>
    <div>
      <div class="card-sh">Advisor</div>
      <p style="font-size:.9rem;color:var(--steel)">Dr. Panumart Phumart</p>
      <div class="card-sh" style="margin-top:20px">Program and Institution</div>
      <p style="font-size:.9rem;color:var(--steel);line-height:1.8">Doctor of Pharmacy Program<br>Faculty of Pharmaceutical Sciences<br>Khon Kaen University, 2025</p>
    </div>
  </div>
  <hr style="border:none;border-top:1px solid var(--bdr);margin:28px 0">
  <div class="card-sh">Methodology</div>
  <p class="body" style="margin-bottom:14px">This study employs a quantitative research design across two phases. Phase 1 involves developing the AI tool by establishing binary legal criteria (lawful/unlawful) based on the Thai Drug Act B.E. 2510, creating a ground truth dataset from approximately 385 product listings sampled from Shopee and Lazada via systematic sampling, and training machine learning models using natural language processing for Thai text analysis. Phase 2 validates the tool's performance against the established ground truth using standard classification metrics.</p>
  <p class="body">The NLP pipeline consists of Thai word segmentation via PyThaiNLP (newmm engine), TF-IDF vectorization with unigram and bigram features (5,000 dimensions), and class imbalance handling with SMOTE. Five classifiers were evaluated: Logistic Regression, Support Vector Machine (RBF kernel), Random Forest, Gradient Boosting, and Multinomial Naive Bayes.</p>
  <hr style="border:none;border-top:1px solid var(--bdr);margin:28px 0">
  <div class="card-sh">Technology Stack</div>
  <p class="body">Python, scikit-learn, PyThaiNLP, Flask, TF-IDF, SMOTE (imbalanced-learn), Gradient Boosting, pandas, NumPy</p>
</div>
</div>

</div>

<div class="ftr">
  <p style="font-weight:600">Faculty of Pharmaceutical Sciences, Khon Kaen University</p>
  <p>Special Project in Community Pharmacy (PS 125 776) &mdash; Academic Year 2025</p>
  <p style="margin-top:8px;font-size:.72rem">This system is intended as a preliminary screening tool for research purposes and does not constitute official legal advice.</p>
</div>

<script>
function go(id,el){
  document.querySelectorAll('.pane').forEach(function(p){p.classList.remove('on')});
  document.querySelectorAll('.tab').forEach(function(t){t.classList.remove('on')});
  document.getElementById(id).classList.add('on');el.classList.add('on');
}
async function send(e){
  e.preventDefault();var b=document.getElementById('sbtn');
  b.disabled=true;b.textContent='Processing...';var f=document.getElementById('frm');
  try{
    var r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({product_name:f.product_name.value,ad_text:f.ad_text.value,
        platform:f.platform.value,has_registration:f.has_registration.checked,
        has_ad_license:f.has_ad_license.checked})});
    show(await r.json());
  }catch(err){document.getElementById('resArea').innerHTML='<div class="res res-review" style="border-left-color:var(--red)"><p>An error occurred: '+err.message+'</p></div>';}
  b.disabled=false;b.textContent='Submit for Verification';
}
function show(r){
  var cls='res-lawful',clr='var(--green)';
  if(r.result==='ILLEGAL'){cls='res-illegal';clr='var(--red)';}
  else if(r.result!=='LAWFUL'){cls='res-review';clr='var(--amber)';}
  var h='<div class="res '+cls+'"><div class="res-sup">Verification Result</div>';
  h+='<div class="res-title" style="color:'+clr+'">'+r.result+'</div>';
  h+='<div class="res-sub">Risk Level: '+r.risk_level+'</div>';
  h+='<div class="met-grid">';
  h+='<div class="met"><div class="met-v">'+r.illegal_probability+'%</div><div class="met-l">Illegal Probability</div></div>';
  h+='<div class="met"><div class="met-v">'+r.confidence+'%</div><div class="met-l">Confidence</div></div>';
  h+='<div class="met"><div class="met-v">'+r.ml_prediction+'</div><div class="met-l">ML Prediction</div></div>';
  h+='<div class="met"><div class="met-v" style="color:'+(r.n_violations>0?'var(--red)':'var(--green)')+'">'+r.n_violations+'</div><div class="met-l">Violations Identified</div></div>';
  h+='</div>';
  if(r.violation_details&&r.violation_details.length){
    h+='<div class="card-sh" style="color:var(--red);border-bottom:1px solid rgba(163,42,42,.2);padding-bottom:8px;margin-bottom:12px;margin-top:20px">Criteria Violations Identified</div>';
    r.violation_details.forEach(function(v){
      h+='<div class="viol"><strong style="color:var(--navy)">'+v.code+' &mdash; '+v.name+':</strong> <span style="color:var(--steel)">'+v.desc+'</span></div>';
    });
  }
  h+='</div>';
  document.getElementById('resArea').innerHTML=h;
}
</script>
</body></html>"""

@app.route('/')
def index():
    return render_template_string(HTML, res=eval_results, mi=model_info, criteria=LEGAL_CRITERIA)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    d = request.get_json()
    return jsonify(predict(d.get('product_name',''), d.get('ad_text',''),
                           d.get('has_registration',False), d.get('has_ad_license',False)))

@app.route('/api/model-info')
def api_model_info():
    return jsonify(model_info)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Online Drug Sales Legality Checker")
    print("  Faculty of Pharmaceutical Sciences")
    print("  Khon Kaen University - 2025")
    print("="*60)
    print(f"\n  Server running at http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
