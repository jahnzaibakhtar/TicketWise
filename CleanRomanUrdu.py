import pandas as pd
import numpy as np
import re

# ==================================================================
# CUSTOM REPLACEMENTS (no urduhack needed)
# ==================================================================

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', str(text))

def normalize(text):
    return re.sub(r'\s+', ' ', str(text)).strip()

# ==================================================================
# STEP 1: LOAD BOTH DATASETS
# ==================================================================
print("=" * 60)
print("STEP 1: Loading Datasets")
print("=" * 60)

# --- Kaggle Dataset ---
df_kaggle = pd.read_csv('IT Support Ticket Data.csv')
print(f"Kaggle raw: {df_kaggle.shape}")

# Keep only relevant columns and rename to unified names
df_kaggle = df_kaggle[['Body', 'Department', 'Priority']].copy()
df_kaggle.columns = ['ticket_text', 'category', 'priority']
df_kaggle['source'] = 'kaggle'

# --- Roman Urdu Dataset ---
df_pak = pd.read_csv('PakRomanUrdu.csv')
print(f"Roman Urdu raw: {df_pak.shape}")
df_pak = df_pak[['ticket_text', 'category', 'priority']].copy()
df_pak['source'] = 'roman_urdu'

# --- Combine ---
df = pd.concat([df_kaggle, df_pak], ignore_index=True)
print(f"Combined total: {df.shape}")

# ==================================================================
# STEP 2: CLEAN MISSING VALUES
# ==================================================================
print("\n" + "=" * 60)
print("STEP 2: Cleaning Missing Values")
print("=" * 60)

print(f"Missing before:\n{df.isnull().sum()}")

df['ticket_text'] = df['ticket_text'].fillna('')
df['category']    = df['category'].fillna('Unknown')
df['priority']    = df['priority'].fillna('low')

# Remove empty tickets
df = df[df['ticket_text'].str.strip() != '']

# Normalize priority to lowercase
df['priority'] = df['priority'].str.lower().str.strip()

# Keep only valid priority labels
df = df[df['priority'].isin(['low', 'medium', 'high', 'critical'])]

print(f"\nMissing after:\n{df.isnull().sum()}")
print(f"\nUsable rows: {df.shape[0]}")
print(f"\nPriority distribution:\n{df['priority'].value_counts()}")
print(f"\nCategory distribution:\n{df['category'].value_counts().head(10)}")

# ==================================================================
# STEP 3: REMOVE DUPLICATES
# ==================================================================
print("\n" + "=" * 60)
print("STEP 3: Removing Duplicates")
print("=" * 60)

print(f"Duplicates before: {df.duplicated(subset=['ticket_text']).sum()}")
df = df.drop_duplicates(subset=['ticket_text'])
print(f"Duplicates after:  {df.duplicated(subset=['ticket_text']).sum()}")
print(f"Rows after dedup:  {df.shape[0]}")

# ==================================================================
# STEP 4: TEXT NORMALIZATION
# ==================================================================
print("\n" + "=" * 60)
print("STEP 4: Text Normalization")
print("=" * 60)

df['ticket_text'] = df['ticket_text'].str.lower().str.strip()

# ==================================================================
# STEP 5: ROMAN URDU NLP PREPROCESSING
# ==================================================================
print("\n" + "=" * 60)
print("STEP 5: Roman Urdu NLP Preprocessing")
print("=" * 60)

URDU_VOCAB = {
    'ho gaya':  'failed',
    'ho gia':   'failed',
    'ho gai':   'failed',
    'aa raha':  'appearing',
    'kharab':   'broken',
    'nahi':     'not',
    'chal':     'working',
    'band':     'offline',
    'jaldi':    'urgent',
    'zaroori':  'important',
    'bhool':    'forgot',
    'bijli':    'electricity',
    'baad':     'after',
    'raha':     'running',
    'chali':    'gone',
    'karo':     'restart',
    'hai':      'is',
    'tha':      'was',
    'thi':      'was',
    'gaya':     'gone',
    'gayi':     'gone',
    'ke':       '',
    'se':       '',
    'ka':       '',
    'pe':       'on',
}

def preprocess_roman_urdu(text):
    text = str(text).lower().strip()
    for urdu, eng in sorted(URDU_VOCAB.items(), key=lambda x: -len(x[0])):
        text = re.sub(r'\b' + re.escape(urdu) + r'\b', eng, text)
    text = remove_punctuation(text)
    text = normalize(text)
    return text

df['ticket_text_clean'] = df['ticket_text'].apply(preprocess_roman_urdu)

print("\nSample cleaned (Roman Urdu):")
urdu_sample = df[df['source'] == 'roman_urdu'][['ticket_text', 'ticket_text_clean']].head(5)
print(urdu_sample)

# ==================================================================
# STEP 6: LANGUAGE DETECTION
# ==================================================================
print("\n" + "=" * 60)
print("STEP 6: Language Detection")
print("=" * 60)

urdu_words = ['kharab', 'nahi', 'chal', 'band', 'jaldi', 'bhool',
              'bijli', 'gaya', 'gayi', 'hai', 'tha', 'thi', 'karo',
              'baad', 'raha', 'chali', 'ho', 'se', 'ka', 'ke', 'pe']

def detect_language(text):
    words    = str(text).lower().split()
    total    = max(len(words), 1)
    urdu_cnt = sum(1 for w in words if w in urdu_words)
    if urdu_cnt == 0:
        return 'English'
    elif urdu_cnt > total / 2:
        return 'Roman Urdu'
    else:
        return 'Code-Mixed'

df['language_type'] = df['ticket_text'].apply(detect_language)
print(f"\nLanguage distribution:\n{df['language_type'].value_counts()}")

# ==================================================================
# STEP 7: URGENCY SCORE
# ==================================================================
print("\n" + "=" * 60)
print("STEP 7: Urgency Scoring")
print("=" * 60)

urdu_urgent = ['jaldi', 'urgent', 'zaroori', 'crash', 'down', 'failed',
               'kharab', 'band', 'fail', 'ho gaya', 'gaya', 'ho gai',
               'not working', 'error', 'broken', 'offline', 'critical']

def extract_urgency(text):
    urgency    = 0.0
    text_lower = str(text).lower()
    for word in urdu_urgent:
        if word in text_lower:
            urgency += 0.3
    if len(text) > 0:
        urgency += sum(1 for c in text if c.isupper()) / len(text) * 0.2
    return round(urgency, 3)

df['urgency_score'] = df['ticket_text'].apply(extract_urgency)

# ==================================================================
# STEP 8: FEATURE ENGINEERING
# ==================================================================
print("\n" + "=" * 60)
print("STEP 8: Feature Engineering")
print("=" * 60)

df['text_length'] = df['ticket_text_clean'].str.len()
df['word_count']  = df['ticket_text_clean'].str.split().str.len()

print(f"\nFinal dataset shape: {df.shape}")
print(f"\nFeature summary:")
print(df[['text_length', 'word_count', 'urgency_score']].describe())

# ==================================================================
# STEP 9: SAVE
# ==================================================================
print("\n" + "=" * 60)
print("STEP 9: Saving Clean Data")
print("=" * 60)

df.to_csv('clean_tickets.csv', index=False)
print(f"[DONE] Saved: clean_tickets.csv ({df.shape})")

print(f"\nSource breakdown:\n{df['source'].value_counts()}")
print(f"\nPriority distribution:\n{df['priority'].value_counts()}")
print(f"\nCategory distribution:\n{df['category'].value_counts().head(10)}")

print("\n" + "=" * 60)
print("[DONE] Data cleaning complete!")
print("=" * 60)