# SchemeXpress — Project Documentation

**Document Version:** 1.0 (Phase 1 — Project Setup)  
**Last Updated:** Phase 1  
**Status:** Living document — updated after every phase

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Target Users](#4-target-users)
5. [Existing Problem Analysis](#5-existing-problem-analysis)
6. [Proposed Solution](#6-proposed-solution)
7. [Features](#7-features)
8. [Dataset](#8-dataset)
9. [Data Preprocessing](#9-data-preprocessing)
10. [Exploratory Data Analysis](#10-exploratory-data-analysis)
11. [Feature Engineering](#11-feature-engineering)
12. [NLP Pipeline](#12-nlp-pipeline)
13. [Recommendation Methodology](#13-recommendation-methodology)
14. [Recommendation Evaluation](#14-recommendation-evaluation)
15. [RAG Architecture](#15-rag-architecture)
16. [Software Architecture](#16-software-architecture)
17. [Database Design](#17-database-design)
18. [Testing](#18-testing)
19. [Limitations](#19-limitations)
20. [Future Improvements](#20-future-improvements)

---

## 1. Project Overview

**Project Name:** SchemeXpress  
**Subtitle:** Smart Government Scheme Recommendation Platform  
**Type:** Full-stack data science web application  
**Domain:** GovTech / Public Service / Information Retrieval

SchemeXpress is a data-driven platform that helps Indian citizens discover government welfare schemes relevant to their personal profile. It combines classical NLP techniques, a rule-based eligibility engine, and a hybrid recommendation system to surface the most relevant schemes from a dataset of approximately 3,400 Indian government schemes.

The platform presents results in a clear, explainable format — showing not just which schemes match, but why they match and which eligibility criteria were satisfied.

A secondary feature, built after the core recommendation engine, is a RAG-based (Retrieval-Augmented Generation) Q&A assistant that allows users to ask natural-language questions about government schemes and receive grounded, source-cited answers.

---

## 2. Problem Statement

India operates thousands of Central and State government schemes across sectors including education, healthcare, agriculture, housing, employment, skill development, and social welfare. The primary challenge is not the absence of schemes — it is the gap between scheme availability and citizen awareness.

**Specific problems:**

1. **Fragmentation:** Scheme information is spread across dozens of ministry portals, state government websites, and PDF circulars with no unified discovery interface.

2. **No personalization:** Existing portals display all schemes in bulk with no filtering based on a citizen's profile (age, income, occupation, state, social category).

3. **Language and literacy barriers:** Scheme descriptions are often written in bureaucratic language that is difficult for general citizens to parse.

4. **Information decay:** Scheme eligibility criteria, application deadlines, and benefit amounts change over time with no consistent update mechanism on citizen-facing portals.

5. **No guidance on why:** Even when citizens find a scheme, they are not told whether they are likely to qualify — leaving the burden of eligibility interpretation entirely on the citizen.

**The consequence:** A large proportion of eligible citizens — particularly those in rural areas, elderly citizens, and those with limited digital literacy — are unaware of schemes they are legally entitled to benefit from.

---

## 3. Objectives

**Primary Objectives:**

1. Build a working hybrid recommendation system that returns relevant government schemes based on a user's demographic and socioeconomic profile.

2. Provide clear, criterion-level explanations for every recommendation so users understand why a scheme was surfaced.

3. Present a professional, accessible, responsive web interface that non-technical users can navigate confidently.

**Data Science Objectives:**

4. Perform rigorous data understanding, cleaning, and preprocessing on the Kaggle dataset before any modeling.

5. Build a complete NLP pipeline using TF-IDF and cosine similarity as the classical baseline.

6. Engineer structured features from unstructured text fields to enable rule-based filtering.

7. Evaluate the recommendation system using appropriate metrics and document the methodology and limitations honestly.

8. Build a RAG-based Q&A assistant grounded in the scheme dataset.

**Software Engineering Objectives:**

9. Maintain a clean, modular codebase with clear separation between data science logic and web application logic.

10. Use Git with meaningful commits to demonstrate professional development practice.

11. Deploy the application to a publicly accessible URL with MongoDB Atlas as the cloud database.

---

## 4. Target Users

| User Type | Description | Primary Need |
|---|---|---|
| General Citizens | Rural and urban citizens seeking welfare benefits | Discover schemes they qualify for |
| Students | School/college students seeking scholarships | Find education-related schemes |
| Farmers | Agricultural workers | Find agriculture and crop insurance schemes |
| Women | Female citizens seeking gender-specific schemes | Find women empowerment schemes |
| Senior Citizens | Elderly users | Find pension and healthcare schemes |
| Differently-abled | Citizens with disabilities | Find accessibility and support schemes |
| Small Business Owners | Entrepreneurs and MSMEs | Find startup and business loan schemes |
| Researchers / Journalists | Those studying welfare policy | Quickly explore the scheme landscape |

**Secondary users:** Government outreach workers, NGOs, and social workers who assist citizens with scheme applications.

---

## 5. Existing Problem Analysis

### Current State of Government Scheme Discovery

**myScheme (myscheme.gov.in):** India's official scheme discovery portal.
- Provides a questionnaire-based filter
- Coverage is good but relies on citizens already knowing how to navigate it
- No NLP-based relevance ranking — results are binary filter matches
- No explanation of why a scheme was shown

**State Government Portals:** Highly fragmented. Each state has its own portal format with no standardized structure.

**Direct Ministry Websites:** Written for policy administrators, not citizens. Navigation is complex and search is poor.

**What is missing across all existing solutions:**
- Personalized relevance ranking (not just binary filter)
- Natural language requirement input ("I need help paying my child's college fees")
- Explanation of which criteria matched
- A conversational Q&A layer for follow-up questions

SchemeXpress addresses each of these gaps directly.

---

## 6. Proposed Solution

SchemeXpress provides a three-layer solution:

### Layer 1 — Structured Eligibility Matching
The user fills a short profile questionnaire. A rule-based engine evaluates their profile against extracted scheme eligibility criteria (age, gender, state, income bracket, occupation, education level, social category, rural/urban status). This produces a set of candidate schemes with structured match scores.

### Layer 2 — NLP Relevance Ranking
The user also provides a free-text description of their requirement (e.g., "looking for financial assistance for higher education"). A TF-IDF + cosine similarity engine compares this against the text representation of every candidate scheme and produces a relevance score.

### Layer 3 — Hybrid Scoring and Explanation
The eligibility score and NLP score are combined using a weighted formula to produce a final recommendation score. The results are ranked and presented with per-criterion explanations.

### Layer 4 (RAG) — Conversational Q&A
A separate RAG pipeline allows users to ask free-form questions. The question is embedded, matched against indexed scheme chunks via vector search, and the relevant context is passed to an LLM to generate a grounded answer with source citations.

---

## 7. Features

### Core Features

| Feature | Description | Phase |
|---|---|---|
| User profile questionnaire | Clean, sectioned form collecting demographic info | Phase 12 |
| Eligibility engine | Rule-based filter against structured scheme criteria | Phase 7 |
| NLP similarity matching | TF-IDF + cosine similarity on text fields | Phase 6 |
| Hybrid recommender | Weighted combination of eligibility + NLP scores | Phase 8 |
| Recommendation explanation | Per-criterion match breakdown for every result | Phase 8 |
| Scheme detail page | Full scheme info: description, benefits, eligibility, documents, application | Phase 12 |
| No-results guidance | Helpful suggestions when no match is found | Phase 12 |
| Responsive UI | Desktop and mobile support via CSS | Phase 12 |

### Advanced Features

| Feature | Description | Phase |
|---|---|---|
| Analytics dashboard | Dataset-level visualizations and statistics | Phase 13 |
| RAG Q&A assistant | Natural language scheme question answering | Phase 14 |

---

## 8. Dataset

### Source
- **Platform:** Kaggle
- **Content:** Indian government schemes (Central and State)
- **Approximate size:** ~3,400 records

### Known Fields

| Field | Type | Description |
|---|---|---|
| `scheme_name` | Text | Official name of the scheme |
| `slug` | Text | URL-friendly identifier |
| `details` | Long text | Scheme overview and description |
| `benefits` | Long text | What the scheme provides |
| `eligibility` | Long text | Who can apply |
| `application` | Long text | How to apply |
| `documents` | Long text | Required documents |
| `level` | Categorical | Central / State / District |
| `schemeCategory` | Categorical | Sector/domain of the scheme |
| `tags` | Text / List | Keywords associated with the scheme |

### Known Issues (to be confirmed in Phase 2)
- One column appears to be unnamed or empty
- Possible duplicate scheme records
- Missing values in text fields
- Inconsistent category naming
- Text fields may contain HTML entities or encoded characters from web scraping

### Data Handling Rules
- Raw data is stored in `data/raw/` and **never modified**
- All cleaning is done programmatically and outputs go to `data/processed/`
- Intermediate outputs go to `data/interim/`
- Every preprocessing decision is documented in Section 9 below

---

## 9. Data Preprocessing

> **Status:** [TO BE COMPLETED — Phase 3]

This section will be populated after the preprocessing notebook (`notebooks/03_preprocessing.ipynb`) is completed. The following structure defines what will be documented.

### 9.1 Missing Value Analysis

For each column, we will document:
- Count and percentage of missing values
- Whether missingness is structural (the scheme genuinely lacks that info) or a data quality issue
- Imputation or handling strategy with justification

**Decision template:**
> **Problem:** [describe the missing value situation]  
> **Why it matters:** [explain impact on recommendation quality]  
> **Solution:** [describe what we did]  
> **Justification:** [explain why this was the right choice]

### 9.2 Duplicate Detection

- Method: exact match on `scheme_name` and `slug`, fuzzy match on `details`
- Decision criteria for removal
- Count of duplicates found and removed

### 9.3 Unnecessary Column Removal

- Identify columns with no analytical value
- Document each removed column and the reason

### 9.4 Text Normalization

- Whitespace normalization
- Case normalization
- HTML entity removal
- Special character handling

### 9.5 Category Standardization

- Inconsistent category values (e.g., "Women & Child" vs "Women and Child")
- Standardization mapping documented here

### 9.6 Data Validation

- Post-cleaning shape: [rows × columns]
- Null counts after cleaning
- Schema validation

---

## 10. Exploratory Data Analysis

> **Status:** [TO BE COMPLETED — Phase 4]

This section will be populated after `notebooks/04_eda.ipynb` is completed.

### Questions to Answer

Each visualization will be tied to a specific question:

1. How many schemes are in the cleaned dataset?
2. What is the Central vs State vs District distribution?
3. Which scheme categories have the most schemes?
4. Which states have the most schemes?
5. What is the completeness of each text field?
6. What are the most common tags?
7. What types of benefits are most frequently mentioned?
8. What target groups appear most often in eligibility text?
9. What is the distribution of scheme text lengths?
10. Are there any patterns in missing data?

### Visualization Standards

Every chart will include:
- A clear title stating the question being answered
- Labelled axes
- A 2-3 sentence interpretation below the chart

---

## 11. Feature Engineering

> **Status:** [TO BE COMPLETED — Phase 5]

This section will document all features extracted from text fields.

### Target Features

| Feature | Source Field | Extraction Method | Possible Values |
|---|---|---|---|
| `target_gender` | eligibility | Keyword matching | Male / Female / All |
| `min_age` | eligibility | Regex | Integer / Unknown |
| `max_age` | eligibility | Regex | Integer / Unknown |
| `income_category` | eligibility | Keyword matching | BPL / Low / Any / Unknown |
| `education_level` | eligibility | Keyword matching | None / Primary / Secondary / Graduate / Any |
| `occupation` | eligibility | Keyword matching | Farmer / Student / Self-employed / Any / Unknown |
| `social_category` | eligibility | Keyword matching | SC / ST / OBC / General / Any / Unknown |
| `state_specific` | level + details | Field check | State name / Central / Unknown |
| `rural_urban` | eligibility | Keyword matching | Rural / Urban / Both / Unknown |
| `benefit_type` | benefits | Keyword matching | Financial / Educational / Healthcare / Other |

### Important Constraint
> Features are extracted on a best-effort basis. If reliable extraction is not possible for a field, the value is set to **Unknown** and the eligibility engine treats **Unknown** as a **non-blocking match** (i.e., we do not disqualify the user on the basis of information we cannot confirm).

---

## 12. NLP Pipeline

> **Status:** [TO BE COMPLETED — Phase 6]

### 12.1 Text Representation Strategy

Each scheme is represented by a combined text field built from:
```
combined_text = scheme_name + " " + details + " " + benefits + " " + eligibility + " " + tags
```

This combined field is what gets vectorized by TF-IDF.

### 12.2 Text Preprocessing Steps

| Step | Library | Purpose |
|---|---|---|
| Lowercase | Python built-in | Normalize case |
| Remove special chars | `re` (regex) | Remove punctuation, HTML artifacts |
| Tokenization | `nltk.tokenize` | Split text into individual tokens |
| Stop-word removal | `nltk.corpus.stopwords` | Remove words like "the", "is", "for" |
| Lemmatization | `nltk.stem.WordNetLemmatizer` | Reduce words to base form |

### 12.3 TF-IDF Vectorization

**Library:** `sklearn.feature_extraction.text.TfidfVectorizer`

**Why TF-IDF over bag-of-words:**
- Bag-of-words gives equal weight to common and rare words
- TF-IDF downweights words that appear in many documents (e.g., "scheme", "government") and upweights rare, distinctive words
- This produces better similarity scores for information retrieval tasks

**Key parameters to tune:** [TO BE DOCUMENTED — Phase 6]

### 12.4 Cosine Similarity

**Why cosine similarity:**
- Two TF-IDF vectors can be compared by the angle between them
- Cosine similarity is robust to document length differences (a short eligibility description and a long one can still score similarly if they share the right terms)
- Range: 0 (no similarity) to 1 (identical)

---

## 13. Recommendation Methodology

> **Status:** [TO BE COMPLETED — Phase 8]

### 13.1 Approach Comparison

| Approach | Method | Strengths | Weaknesses |
|---|---|---|---|
| Approach 1 | Rule-based only | Precise for structured criteria | Misses schemes with relevant but textually rich eligibility |
| Approach 2 | TF-IDF similarity only | Captures semantic relevance | Ignores explicit eligibility constraints |
| Approach 3 | Hybrid (eligibility + NLP) | Combines both signals | Requires weight tuning |

### 13.2 Hybrid Score Formula

```
hybrid_score = α × eligibility_score + β × nlp_score
```

Where:
- `eligibility_score` ∈ [0, 1] — fraction of extractable criteria that match
- `nlp_score` ∈ [0, 1] — cosine similarity between user requirement and scheme text
- α + β = 1 (weights are constrained to sum to 1)

**Weight tuning:** [TO BE DOCUMENTED after Phase 9 evaluation]

### 13.3 Explainability Output

For each recommendation, the system generates a structured explanation:

```
Why this scheme was recommended:
✓ State requirement matched (Tamil Nadu)
✓ Gender requirement matched (Female)
✓ Education level matched (Graduate)
✗ Income limit could not be verified (information unavailable)
~ Age requirement: scheme specifies 18-35, you are 24 ✓

Match Score: 87%
NLP Relevance Score: 0.73
Eligibility Score: 0.80
```

---

## 14. Recommendation Evaluation

> **Status:** [TO BE COMPLETED — Phase 9]

### 14.1 Challenge: No Ground Truth Labels

This dataset does not contain user preference labels (i.e., "user X found scheme Y relevant"). This is a common challenge in recommendation system evaluation.

**Approaches we will use:**

1. **Top-K precision/recall using manually created test cases** — We will define synthetic user profiles with known expected schemes and measure whether the system retrieves them.

2. **Comparative evaluation** — Compare Approach 1, 2, and 3 on the same test cases and measure which retrieves more relevant schemes at rank K.

3. **Ranking quality** — Measure whether the most relevant schemes appear at the top of the ranked list.

### 14.2 Metrics

| Metric | Formula | What it measures |
|---|---|---|
| Precision@K | Relevant in top K / K | Of the top K results, how many are actually relevant? |
| Recall@K | Relevant in top K / Total relevant | Of all relevant schemes, how many did we find in top K? |

### 14.3 Honest Limitations

- Ground truth labels are manually created and limited
- Evaluation reflects our test cases, not real-world user satisfaction
- Results will be reported exactly as measured — no cherry-picking

---

## 15. RAG Architecture

> **Status:** [TO BE COMPLETED — Phase 14]

### 15.1 Overview

The RAG (Retrieval-Augmented Generation) assistant answers natural language questions about government schemes. It grounds every answer in the actual scheme data, preventing the LLM from hallucinating scheme details.

### 15.2 Pipeline

```
User Question
      │
      ▼
Text Embedding
(sentence-transformers or OpenAI embeddings)
      │
      ▼
Vector Similarity Search
(MongoDB Atlas Vector Search)
      │
      ▼
Top-K Relevant Scheme Chunks Retrieved
      │
      ▼
Context + Question → LLM Prompt
      │
      ▼
Grounded Answer with Source Citations
```

### 15.3 Design Decisions

[TO BE COMPLETED — Phase 14]

- Choice of embedding model
- Chunk size strategy
- Number of retrieved chunks (K)
- LLM choice and prompt design
- Hallucination prevention strategy

---

## 16. Software Architecture

### 16.1 Directory Structure

```
SchemeXpress/
│
├── data/
│   ├── raw/              ← Original dataset. NEVER modified.
│   ├── processed/        ← Output of preprocessing pipeline
│   └── interim/          ← Intermediate transformation artifacts
│
├── notebooks/            ← Exploratory Jupyter notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_nlp.ipynb
│   ├── 07_eligibility_engine.ipynb
│   ├── 08_recommendation.ipynb
│   └── 09_evaluation.ipynb
│
├── src/                  ← Production Python modules
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py         ← Data cleaning functions
│   │   └── feature_extractor.py  ← Feature engineering
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── text_processor.py  ← Text cleaning pipeline
│   │   └── tfidf_engine.py    ← TF-IDF vectorizer wrapper
│   ├── recommendation/
│   │   ├── __init__.py
│   │   ├── eligibility_engine.py  ← Rule-based scoring
│   │   └── recommender.py         ← Hybrid recommendation logic
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py       ← Precision@K, Recall@K
│   ├── database/
│   │   ├── __init__.py
│   │   └── mongo_client.py    ← PyMongo connection and queries
│   └── rag/
│       ├── __init__.py
│       └── rag_pipeline.py    ← RAG retrieval and generation
│
├── backend/
│   ├── app.py                 ← Flask app factory
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── main.py            ← Home, about routes
│   │   ├── recommendation.py  ← Profile form, results routes
│   │   └── schemes.py         ← Scheme detail routes
│   └── services/
│       ├── __init__.py
│       └── recommendation_service.py  ← Orchestrates the pipeline
│
├── templates/             ← Jinja2 HTML templates
│   ├── base.html
│   ├── home.html
│   ├── profile.html
│   ├── results.html
│   ├── scheme_detail.html
│   └── no_results.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/               ← Only if absolutely necessary
│   └── images/
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_eligibility.py
│   ├── test_recommender.py
│   └── test_routes.py
│
└── docs/
    └── architecture_diagram.png
```

### 16.2 Separation of Concerns

| Layer | Responsibility |
|---|---|
| `notebooks/` | Exploration, experimentation, visualization |
| `src/` | Tested, reusable production modules |
| `backend/routes/` | HTTP request handling only — no business logic |
| `backend/services/` | Orchestrating the recommendation pipeline |
| `templates/` | Presentation only — no business logic |

**Key principle:** The `notebooks/` are where we learn and experiment. The `src/` modules are where that learning crystallizes into clean, reusable code. Flask routes never contain data science logic directly.

---

## 17. Database Design

### 17.1 Connection

- Database: MongoDB Atlas (cloud hosted)
- Driver: PyMongo
- Connection string stored in `.env` as `MONGO_URI`
- Connection managed in `src/database/mongo_client.py`

### 17.2 Collections

#### `schemes` Collection

The primary collection. Each document represents one government scheme.

```json
{
  "_id": "ObjectId (auto-generated by MongoDB)",
  "scheme_name": "PM Kisan Samman Nidhi",
  "slug": "pm-kisan-samman-nidhi",
  "level": "Central",
  "schemeCategory": "Agriculture,Rural & Environment",
  "tags": ["farmer", "agriculture", "income support"],
  "details": "...",
  "benefits": "...",
  "eligibility": "...",
  "application": "...",
  "documents": "...",
  "extracted_features": {
    "target_gender": "All",
    "min_age": null,
    "max_age": null,
    "income_category": "BPL",
    "education_level": "Any",
    "occupation": "Farmer",
    "social_category": "Any",
    "state_specific": "Central",
    "rural_urban": "Rural",
    "benefit_type": "Financial"
  },
  "combined_text": "...",
  "tfidf_index": 42,
  "created_at": "ISODate",
  "updated_at": "ISODate"
}
```

#### `recommendation_sessions` Collection (Phase 11+)

```json
{
  "_id": "ObjectId",
  "session_id": "UUID string",
  "user_profile": {
    "age": 24,
    "gender": "Female",
    "state": "Tamil Nadu",
    "annual_income": 120000,
    "occupation": "Student",
    "education_level": "Graduate",
    "social_category": "OBC",
    "rural_urban": "Urban",
    "requirement": "Financial assistance for higher education"
  },
  "recommendations": [
    {
      "scheme_id": "ObjectId ref",
      "scheme_name": "...",
      "hybrid_score": 0.87,
      "eligibility_score": 0.80,
      "nlp_score": 0.73,
      "matched_criteria": ["gender", "education_level", "state"],
      "unmatched_criteria": []
    }
  ],
  "created_at": "ISODate"
}
```

### 17.3 Indexes

| Collection | Field | Index Type | Reason |
|---|---|---|---|
| `schemes` | `scheme_name` | Text index | Full-text search fallback |
| `schemes` | `level` | Single field | Filter by Central/State |
| `schemes` | `schemeCategory` | Single field | Category browse |
| `schemes` | `extracted_features.state_specific` | Single field | State filtering |
| `recommendation_sessions` | `session_id` | Unique | Session lookup |
| `recommendation_sessions` | `created_at` | Single field | TTL expiry |

---

## 18. Testing

> **Status:** [TO BE COMPLETED — Phase 15]

### Test Strategy

| Test Type | Tool | Coverage |
|---|---|---|
| Unit tests | pytest | Individual functions in `src/` |
| Integration tests | pytest | Pipeline end-to-end |
| Route tests | Flask test client | HTTP responses |
| Data validation tests | pytest | Schema checks on processed data |

---

## 19. Limitations

1. **Dataset currency:** The Kaggle dataset is a snapshot in time. Scheme eligibility criteria and benefits change. The platform cannot reflect real-time government updates.

2. **Feature extraction accuracy:** Eligibility criteria in the dataset are written in natural language with significant variation. Structured feature extraction is best-effort. Some eligibility information will be missed or misclassified.

3. **NLP model scope:** TF-IDF is a bag-of-words model. It does not understand semantic meaning (e.g., "financial support" and "monetary assistance" are treated as different terms). This is a known limitation of the baseline that transformer-based embeddings can address.

4. **No official eligibility determination:** The system provides preliminary recommendations only. It is not connected to any government eligibility verification system.

5. **English-centric:** The current NLP pipeline assumes English text. Many scheme descriptions may contain transliterated Hindi or regional language terms that the current pipeline handles suboptimally.

6. **Evaluation without ground truth:** Without real user preference data, evaluation is limited to manually constructed test cases.

---

## 20. Future Improvements

| Improvement | Description | Priority |
|---|---|---|
| Multilingual NLP | Support Hindi, Tamil, Telugu text in NLP pipeline | High |
| Transformer embeddings | Replace TF-IDF with SBERT or similar for better semantic matching | Medium |
| Real-time data updates | API integration with myScheme or government data APIs | High |
| User accounts | Save profiles and recommendation history | Medium |
| Collaborative filtering | Learn from user interactions to improve recommendations | Low |
| Mobile application | Flutter or React Native companion app | Low |
| Scheme deadline alerts | Notify users of upcoming application deadlines | Medium |
| Offline mode | PWA for users with poor internet connectivity | Medium |

---

*This document is maintained as a living record of all design decisions, experimental results, and implementation details throughout the development of SchemeXpress.*
