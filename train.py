import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# ==================================================================
# STEP 1: LOAD CLEAN DATA
# ==================================================================
print("=" * 60)
print("STEP 1: Loading Clean Data")
print("=" * 60)

df = pd.read_csv('clean_tickets.csv')

if 'ticket_text_clean' in df.columns:
    df['text'] = df['ticket_text_clean'].fillna(df['ticket_text'].fillna(''))
else:
    df['text'] = df['ticket_text'].fillna('')

df['label'] = df['priority'].str.lower().str.strip()
df = df[df['text'].str.strip() != '']
df = df[df['label'].isin(['low', 'medium', 'high', 'critical'])]

print(f"Usable rows: {df.shape[0]}")
print(f"\nPriority distribution:\n{df['label'].value_counts()}")

# ==================================================================
# STEP 2: SMART TEXT EXTRACTION
# Strip the generic "dear customer support team" boilerplate
# and focus on the actual problem description
# ==================================================================
print("\n" + "=" * 60)
print("STEP 2: Smart Text Extraction")
print("=" * 60)

# Boilerplate phrases to remove
BOILERPLATE = [
    r'dear customer support team\w*',
    r'dear customer support\w*',
    r'i hope this message finds you well',
    r'i hope this email finds you well',
    r'i am writing to (report|inform|request|let)',
    r'i am reaching out to',
    r'i am contacting you (regarding|about|to)',
    r'thank you for your (assistance|help|support|time)',
    r'please (let me know|feel free|do not hesitate)',
    r'best regards\w*',
    r'kind regards\w*',
    r'sincerely\w*',
]

def extract_core_text(text):
    """Remove boilerplate and keep the actual problem description."""
    text = str(text).lower()

    # Remove boilerplate
    for pattern in BOILERPLATE:
        text = re.sub(pattern, ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # If text is long, focus on the most informative middle section
    # (first 50 chars are usually still generic, meat is in middle)
    words = text.split()
    if len(words) > 60:
        # Skip first 10 words (still generic after boilerplate removal)
        # Take next 80 words where problem is described
        text = ' '.join(words[10:90])

    return text

df['text_core'] = df['text'].apply(extract_core_text)

print("Sample extracted core text:")
print(df[['label', 'text_core']].head(3).to_string())

# ==================================================================
# STEP 3: FEATURE ENGINEERING
# ==================================================================
print("\n" + "=" * 60)
print("STEP 3: Feature Engineering")
print("=" * 60)

urdu_words = ['kharab', 'nahi', 'chal', 'band', 'jaldi', 'bhool',
              'bijli', 'gaya', 'gayi', 'hai', 'tha', 'thi', 'karo']

high_keywords = [
    'urgent', 'crash', 'down', 'outage', 'critical', 'asap',
    'immediately', 'emergency', 'not working', 'failed', 'offline',
    'blocking', 'cannot access', 'unable to', 'stopped working',
    'kharab', 'band', 'jaldi', 'server down', 'network down',
    'persistent', 'significant', 'disruption', 'inaccessible'
]

medium_keywords = [
    'slow', 'error', 'issue', 'problem', 'broken', 'fail',
    'intermittent', 'sometimes', 'occasionally', 'degraded',
    'not responding', 'freezing', 'crashing', 'unexpected'
]

low_keywords = [
    'update', 'request', 'change', 'password', 'reset', 'please',
    'when possible', 'minor', 'info', 'question', 'how to', 'bhool',
    'inquiry', 'feedback', 'clarification', 'billing', 'understand',
    'curious', 'wondering', 'would like to know', 'general'
]

def count_urdu_words(text):
    words = str(text).lower().split()
    return sum(1 for w in words if w in urdu_words)

def count_keywords(text, keywords):
    text_lower = str(text).lower()
    return sum(1 for kw in keywords if kw in text_lower)

def caps_ratio(text):
    text = str(text)
    if len(text) == 0: return 0.0
    return round(sum(1 for c in text if c.isupper()) / len(text), 3)

def exclamation_count(text):
    return str(text).count('!')

def question_count(text):
    return str(text).count('?')

# Apply on CORE text (boilerplate removed)
df['urdu_word_count']   = df['text_core'].apply(count_urdu_words)
df['high_kw_count']     = df['text_core'].apply(lambda x: count_keywords(x, high_keywords))
df['medium_kw_count']   = df['text_core'].apply(lambda x: count_keywords(x, medium_keywords))
df['low_kw_count']      = df['text_core'].apply(lambda x: count_keywords(x, low_keywords))
df['caps_ratio']        = df['text'].apply(caps_ratio)
df['exclamation_count'] = df['text'].apply(exclamation_count)
df['question_count']    = df['text'].apply(question_count)
df['text_length']       = df['text_core'].str.len()
df['word_count']        = df['text_core'].str.split().str.len()

# Priority keyword score = high_kw - low_kw (net urgency signal)
df['urgency_net'] = df['high_kw_count'] - df['low_kw_count']

extra_cols = ['urdu_word_count', 'high_kw_count', 'medium_kw_count',
              'low_kw_count', 'caps_ratio', 'exclamation_count',
              'question_count', 'text_length', 'word_count', 'urgency_net']

print(f"Features: {extra_cols}")
print(f"\nAvg high_kw by priority:")
print(df.groupby('label')[['high_kw_count', 'low_kw_count', 'urgency_net']].mean().round(2))

# ==================================================================
# STEP 4: ENCODE LABELS
# ==================================================================
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])
print(f"\nClasses: {list(le.classes_)}")

# ==================================================================
# STEP 5: TRAIN / TEST SPLIT
# ==================================================================
print("\n" + "=" * 60)
print("STEP 5: Train/Test Split")
print("=" * 60)

X_text  = df['text_core']   # use core text for TF-IDF
X_extra = df[extra_cols].values
y       = df['label_enc']

X_text_train, X_text_test, X_extra_train, X_extra_test, y_train, y_test = \
    train_test_split(X_text, X_extra, y, test_size=0.2,
                     random_state=42, stratify=y)

print(f"Train: {len(X_text_train)} | Test: {len(X_text_test)}")

# ==================================================================
# STEP 6: TF-IDF ON CORE TEXT
# ==================================================================
print("\n" + "=" * 60)
print("STEP 6: TF-IDF Vectorization")
print("=" * 60)

tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=2,
    strip_accents='unicode'
)

X_tfidf_train = tfidf.fit_transform(X_text_train)
X_tfidf_test  = tfidf.transform(X_text_test)

X_train = hstack([X_tfidf_train, csr_matrix(X_extra_train)])
X_test  = hstack([X_tfidf_test,  csr_matrix(X_extra_test)])

print(f"Feature matrix: {X_train.shape}")

# ==================================================================
# STEP 7: SMOTE BALANCING
# ==================================================================
print("\n" + "=" * 60)
print("STEP 7: Balancing Classes")
print("=" * 60)

try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_train_bal.shape[0]} rows")
    unique, counts = np.unique(y_train_bal, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {le.classes_[u]}: {c}")
except ImportError:
    print("imbalanced-learn not found — using class_weight='balanced'")
    X_train_bal, y_train_bal = X_train, y_train

# ==================================================================
# STEP 8: TRAIN MODELS
# ==================================================================
print("\n" + "=" * 60)
print("STEP 8: Training & Comparing Models")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2,
        class_weight='balanced_subsample', random_state=42, n_jobs=-1),
    'Linear SVM': LinearSVC(
        class_weight='balanced', max_iter=3000, C=0.5, random_state=42),
}

best_f1    = 0
best_name  = ''
best_model = None

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    print(f"{name:25s} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
    if f1 > best_f1:
        best_f1    = f1
        best_name  = name
        best_model = model

print(f"\nBest model: {best_name} (F1: {best_f1:.4f})")

# ==================================================================
# STEP 9: DETAILED REPORT
# ==================================================================
print("\n" + "=" * 60)
print(f"STEP 9: Detailed Report - {best_name}")
print("=" * 60)

y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

print("Confusion Matrix:")
cm    = confusion_matrix(y_test, y_pred_best)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print(cm_df)

# ==================================================================
# STEP 10: SAVE MODEL
# ==================================================================
print("\n" + "=" * 60)
print("STEP 10: Saving Model")
print("=" * 60)

os.makedirs('model', exist_ok=True)
joblib.dump(best_model, 'model/priority_classifier.pkl')
joblib.dump(tfidf,      'model/tfidf_vectorizer.pkl')
joblib.dump(le,         'model/label_encoder.pkl')

print("[DONE] Saved to model/")

# ==================================================================
# STEP 11: PREDICTION FUNCTION
# ==================================================================
print("\n" + "=" * 60)
print("STEP 11: Testing Predictions")
print("=" * 60)

def predict_priority(ticket_text):
    core = extract_core_text(ticket_text)
    tfidf_feat = tfidf.transform([core])
    extra_feat = csr_matrix([[
        count_urdu_words(core),
        count_keywords(core, high_keywords),
        count_keywords(core, medium_keywords),
        count_keywords(core, low_keywords),
        caps_ratio(ticket_text),
        exclamation_count(ticket_text),
        question_count(ticket_text),
        len(core),
        len(core.split()),
        count_keywords(core, high_keywords) - count_keywords(core, low_keywords)
    ]])
    features = hstack([tfidf_feat, extra_feat])
    pred_enc = best_model.predict(features)[0]
    return le.inverse_transform([pred_enc])[0]

test_tickets = [
    ("UPS kharab ho gaya server down",                        "high"),
    ("password bhool gaya reset please",                      "low"),
    ("My laptop is very slow and keeps freezing",             "medium"),
    ("URGENT: entire network is down office cannot work!",    "high"),
    ("load shedding se wifi crash urgent",                    "high"),
    ("Please update my email signature when possible",        "low"),
    ("database backup fail ho gia",                           "high"),
    ("Can you clarify the billing cycle for my account?",     "low"),
    ("centralized portal is offline blocking all access",     "high"),
    ("screen flickering intermittently sometimes",            "medium"),
]

print(f"\n{'Ticket':<50} | Expected | Predicted")
print("-" * 75)
correct = 0
for ticket, expected in test_tickets:
    predicted = predict_priority(ticket)
    match = "[OK]" if predicted == expected else "[!!]"
    if predicted == expected:
        correct += 1
    print(f"{ticket:<50} | {expected:<8} | {predicted.upper()} {match}")

print(f"\nSample accuracy: {correct}/{len(test_tickets)}")
print("\n" + "=" * 60)
print("[DONE] Training complete!")
print("=" * 60)