# TicketWise 🎫
### AI-powered IT Support Ticket Classifier for Pakistani Teams

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Flask](https://img.shields.io/badge/Flask-3.0-green) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange) ![NLP](https://img.shields.io/badge/NLP-Roman%20Urdu-purple)

---

## The Problem

In Pakistani offices, employees don't write support tickets in perfect English. They write things like:

> *"server band ho gaya koi kaam nahi"*  
> *"UPS kharab ho gaya urgent hai"*  
> *"bijli gai network down ho gaya"*

Every existing ticket triage system is built for English only — completely blind to Roman Urdu. IT teams waste hours manually sorting through hundreds of mixed-language tickets every day.

**TicketWise solves this.**

---

## What It Does

TicketWise automatically classifies IT support tickets written in **English, Roman Urdu, or Code-Mixed** text into four priority levels:

| Priority | Response Time | Example |
|---|---|---|
| 🚨 Critical | 30 minutes | "Entire network down, 20 employees idle" |
| ⚠️ High | 2 hours | "UPS kharab ho gaya server down" |
| 🔵 Medium | 8 hours | "Laptop slow hai kabhi kabhi freeze hota" |
| ✅ Low | 24 hours | "Password bhool gaya reset please" |

It also:
- Detects the **department** (Hardware, Network, Software, Database, Security)
- **Auto-assigns** the right IT agent
- Recommends **SLA timelines**
- Detects **language** (English / Roman Urdu / Code-Mixed)

---

## Demo

![TicketWise Demo](demo.png)

The Gmail-style interface lets IT teams triage an entire inbox with one click.

---

## Tech Stack

- **Backend:** Python, Flask
- **ML:** Scikit-learn (Random Forest, Logistic Regression, Linear SVM)
- **NLP:** TF-IDF (unigrams to trigrams), Custom Roman Urdu feature engineering
- **Balancing:** SMOTE (Synthetic Minority Over-sampling)
- **Frontend:** HTML, CSS, JavaScript

---

## Model Performance

| Model | Accuracy | F1 Score |
|---|---|---|
| Random Forest ⭐ | 0.67 | 0.67 |
| Logistic Regression | 0.65 | 0.64 |
| Linear SVM | 0.63 | 0.62 |

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/jahnzaibakhtar/TicketWise.git
cd TicketWise
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Train the model**
```bash
python train.py
```

**5. Run the app**
```bash
python app.py
```

**6. Open browser**
```
http://127.0.0.1:5000
```

---

## Project Structure

```
TicketWise/
├── model/                  # Trained ML model files
├── templates/
│   └── index.html          # Gmail-style web interface
├── app.py                  # Flask backend
├── train.py                # Model training pipeline
├── clean_tickets.csv       # Cleaned dataset
├── requirements.txt
└── README.md
```

---

## Key Innovation

Most ticket classifiers are trained on English corporate datasets from Western companies. TicketWise is specifically designed for the **Pakistani IT environment** with:

- Custom **Roman Urdu lexicon** for urgency detection
- **Code-mixed NLP** handling English + Roman Urdu in the same ticket
- Localized dataset with Pakistani IT scenarios (load shedding, UPS failures, bijli outages)

---

## Built By

**Jahanzaib Akhtar**  
Computer Science Student — Khawaja Fareed University of Engineering & IT  
[LinkedIn](https://linkedin.com/in/jahnzaibakhtar)

---

*Semester Project — 2026*
