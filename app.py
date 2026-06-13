from flask import Flask, render_template, request, jsonify
import joblib
import re
import os
import traceback
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

print("=" * 50)
print("Loading model files from:", MODEL_DIR)
try:
    classifier = joblib.load(os.path.join(MODEL_DIR, 'priority_classifier.pkl'))
    tfidf      = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    le         = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    print("Model loaded OK")
    print("Classes:", list(le.classes_))
except Exception as e:
    print("ERROR loading model:", e)
    raise
print("=" * 50)

# ── Department detection ──────────────────────────────────────────
CATEGORY_RULES = {
    'Hardware':       ['printer','laptop','computer','keyboard','mouse',
                       'monitor','screen','cable','ups','power','device',
                       'hardware','lan cable','voltage','kharab'],
    'Network':        ['wifi','internet','network','lan','connection',
                       'bandwidth','router','switch','connectivity',
                       'load shedding','bijli','offline','band'],
    'Software':       ['windows','excel','office','software','app',
                       'application','install','update','license',
                       'activation','error','crash','slow','freeze'],
    'Database':       ['database','db','backup','sql','data','server',
                       'restore','query','record'],
    'Security':       ['password','login','access','account','permission',
                       'bhool','reset','hack','breach','auth'],
    'Infrastructure': ['server','ups','restart','reboot','hosting',
                       'vm','virtual','deploy','uptime'],
}

STAFF = {
    'Hardware':       {'name':'Ahmed',   'role':'Hardware Engineer'},
    'Network':        {'name':'Bilal',   'role':'Network Administrator'},
    'Software':       {'name':'Sara',    'role':'Software Specialist'},
    'Database':       {'name':'Usman',   'role':'Database Administrator'},
    'Security':       {'name':'Fatima',  'role':'Security Analyst'},
    'Infrastructure': {'name':'Hamza',   'role':'Infrastructure Lead'},
    'General':        {'name':'Support', 'role':'Support Team'},
}

SLA_MAP = {
    'critical': '30 minutes',
    'high':     '2 hours',
    'medium':   '8 hours',
    'low':      '24 hours',
}

# ── Feature helpers — MUST match train.py exactly (8 features) ───
urdu_words      = ['kharab','nahi','chal','band','jaldi','bhool',
                   'bijli','gaya','gayi','hai','tha','thi','karo']
high_keywords   = ['urgent','crash','down','outage','critical','asap',
                   'immediately','emergency','not working','failed',
                   'kharab','band','jaldi','server down','network down']
medium_keywords = ['slow','error','issue','problem','broken','fail',
                   'intermittent','sometimes','occasionally','degraded']
low_keywords    = ['update','request','change','password','reset',
                   'please','when possible','minor','info','question',
                   'how to','bhool','inquiry','feedback']

BOILERPLATE = [
    r'dear customer support team\w*', r'dear customer support\w*',
    r'i hope this message finds you well',
    r'i am writing to (report|inform|request|let)',
    r'thank you for your (assistance|help|support|time)',
    r'best regards\w*', r'kind regards\w*', r'sincerely\w*',
]

URDU_VOCAB = {
    'ho gaya':'failed','ho gia':'failed','ho gai':'failed',
    'aa raha':'appearing','kharab':'broken','nahi':'not',
    'chal':'working','band':'offline','jaldi':'urgent',
    'zaroori':'important','bhool':'forgot','bijli':'electricity',
    'baad':'after','raha':'running','chali':'gone',
    'karo':'restart','hai':'is','tha':'was','thi':'was',
    'gaya':'gone','gayi':'gone','ke':'','se':'','ka':'','pe':'on',
}

def extract_core_text(text):
    text = str(text).lower()
    for pat in BOILERPLATE:
        text = re.sub(pat, ' ', text)
    for urdu, eng in sorted(URDU_VOCAB.items(), key=lambda x: -len(x[0])):
        text = re.sub(r'\b' + re.escape(urdu) + r'\b', eng, text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    if len(words) > 60:
        text = ' '.join(words[10:90])
    return text

def count_kw(text, kws):
    t = str(text).lower()
    return sum(1 for k in kws if k in t)

def caps_ratio(text):
    text = str(text)
    return round(sum(1 for c in text if c.isupper()) / max(len(text), 1), 3)

def detect_language(text):
    words    = str(text).lower().split()
    total    = max(len(words), 1)
    urdu_cnt = sum(1 for w in words if w in urdu_words)
    if urdu_cnt == 0:           return 'English'
    elif urdu_cnt > total / 2:  return 'Roman Urdu'
    else:                       return 'Code-Mixed'

def assign_department(text):
    tl     = str(text).lower()
    scores = {dept: sum(1 for kw in kws if kw in tl)
              for dept, kws in CATEGORY_RULES.items()}
    best   = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'General'

# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data   = request.get_json()
        ticket = (data.get('ticket') or '').strip()
        if not ticket:
            return jsonify({'error': 'Empty ticket text'}), 400

        core = extract_core_text(ticket)

        # ── 8 features — exactly matching train.py ──
        tfidf_feat = tfidf.transform([core])
        extra_feat = csr_matrix([[
            sum(1 for w in core.split() if w in urdu_words),  # urdu_word_count
            count_kw(core, high_keywords),                     # high_kw_count
            count_kw(core, medium_keywords),                   # medium_kw_count
            count_kw(core, low_keywords),                      # low_kw_count
            caps_ratio(ticket),                                # caps_ratio
            ticket.count('!'),                                 # exclamation_count
            len(core),                                         # text_length
            len(core.split()),                                 # word_count
        ]])

        features = hstack([tfidf_feat, extra_feat])
        pred_enc = classifier.predict(features)[0]
        priority = le.inverse_transform([pred_enc])[0]

        confidence = None
        if hasattr(classifier, 'predict_proba'):
            proba      = classifier.predict_proba(features)[0]
            confidence = round(float(max(proba)) * 100, 1)

        dept  = assign_department(ticket)
        staff = STAFF.get(dept, STAFF['General'])
        lang  = detect_language(ticket)
        sla   = SLA_MAP.get(priority, '8 hours')

        print(f"  [{priority.upper():8s}] {confidence}% | {dept} | {ticket[:60]}")

        return jsonify({
            'priority':   priority,
            'confidence': confidence,
            'department': dept,
            'staff_name': staff['name'],
            'staff_role': staff['role'],
            'language':   lang,
            'sla':        sla,
        })

    except Exception as e:
        print("CLASSIFY ERROR:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
