# SchemeXpress
### Smart Government Scheme Recommendation Platform

> A full-stack, data-driven platform that helps Indian citizens discover government schemes relevant to their profile using NLP, hybrid recommendation systems, and an AI-powered Q&A assistant.

---

## Problem Statement

India has thousands of Central and State government schemes spanning education, healthcare, agriculture, housing, employment, and social welfare. Most citizens — especially those in rural areas or with limited digital literacy — are unaware of schemes they are eligible for. Existing government portals are fragmented, difficult to navigate, and provide no personalized guidance.

**The result:** Eligible citizens miss out on benefits they are legally entitled to.

---

## Proposed Solution

SchemeXpress solves this by:

1. **Profiling the user** — collecting basic demographic and socioeconomic information
2. **Filtering schemes** — using a rule-based eligibility engine against structured criteria
3. **Ranking schemes** — using NLP (TF-IDF + cosine similarity) against the user's stated requirement
4. **Explaining recommendations** — showing exactly why each scheme was matched
5. **Answering questions** — using a RAG (Retrieval-Augmented Generation) assistant for natural language Q&A about schemes

---

## Key Features

| Feature | Description |
|---|---|
| Personalized Recommendations | Hybrid eligibility + NLP matching |
| Explainable Results | Every recommendation shows matched criteria |
| Scheme Details | Full description, benefits, eligibility, documents, application process |
| Q&A Assistant | RAG-based chatbot for scheme-related questions |
| Analytics Dashboard | Dataset-level statistics and visualizations |
| No-match Guidance | Helpful suggestions when no schemes match |
| Responsive UI | Works on desktop and mobile |

---

## Architecture Overview

```
User Profile Input
        │
        ▼
┌─────────────────────┐
│  Eligibility Engine │  ← Rule-based structured matching
│  (Age, Gender,      │    (State, Income, Occupation,
│   Income, State...) │     Education, Category)
└────────┬────────────┘
         │  Filtered Candidate Schemes
         ▼
┌─────────────────────┐
│   NLP Engine        │  ← TF-IDF vectorization
│   TF-IDF +          │    Cosine similarity between
│   Cosine Similarity │    user requirement and scheme text
└────────┬────────────┘
         │  Similarity Scores
         ▼
┌─────────────────────┐
│  Hybrid Scorer      │  ← Weighted combination of
│  & Ranker           │    eligibility score + NLP score
└────────┬────────────┘
         │  Ranked Recommendations
         ▼
┌─────────────────────┐
│  Explainability     │  ← Per-criterion match explanation
│  Layer              │    for every recommendation
└────────┬────────────┘
         │
         ▼
    Flask + Jinja2
    MongoDB Atlas
    HTML/CSS Frontend
```

**RAG Assistant (separate pipeline):**
```
User Question → Text Embedding → Vector Search (MongoDB Atlas)
→ Retrieve Relevant Scheme Chunks → LLM → Grounded Answer
```

---

## Tech Stack

### Backend
| Tool | Purpose |
|---|---|
| Python 3.11+ | Core language |
| Flask 3.0 | Web framework |
| PyMongo 4.7 | MongoDB driver |
| python-dotenv | Environment variable management |

### Data Science
| Tool | Purpose |
|---|---|
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical computation |
| Matplotlib / Seaborn | Data visualization |
| scikit-learn | TF-IDF, cosine similarity, evaluation metrics |
| NLTK | Text preprocessing (tokenization, lemmatization) |

### Frontend
| Tool | Purpose |
|---|---|
| HTML5 | Structure |
| CSS3 | Styling and layout |
| Jinja2 | Server-side templating |

### Database
| Tool | Purpose |
|---|---|
| MongoDB Atlas | Cloud NoSQL database |
| MongoDB Atlas Vector Search | Vector similarity for RAG (Phase 14) |

### Infrastructure
| Tool | Purpose |
|---|---|
| Git / GitHub | Version control |
| python-dotenv | Secret management |
| Gunicorn / Waitress | Production WSGI server (Phase 16) |

---

## Dataset

- **Source:** [Kaggle — Indian Government Schemes Dataset](https://www.kaggle.com/)
- **Size:** ~3,400 schemes
- **Format:** CSV
- **Key Fields:** `scheme_name`, `slug`, `details`, `benefits`, `eligibility`, `application`, `documents`, `level`, `schemeCategory`, `tags`
- **Nature:** Primarily text-based; requires NLP preprocessing

> The raw dataset is **not committed** to this repository due to size and licensing.
> Download the dataset from Kaggle and place it at `data/raw/schemes.csv`.

---

## Data Science Pipeline

```
Raw Kaggle Dataset (data/raw/)
        │
        ▼
01. Data Understanding       → notebooks/01_data_understanding.ipynb
02. Data Quality Analysis    → notebooks/02_data_quality.ipynb
03. Data Preprocessing       → notebooks/03_preprocessing.ipynb
                               src/preprocessing/cleaner.py
        │
        ▼  data/processed/schemes_clean.csv
04. EDA                      → notebooks/04_eda.ipynb
05. Feature Engineering      → notebooks/05_feature_engineering.ipynb
                               src/preprocessing/feature_extractor.py
        │
        ▼  data/processed/schemes_features.csv
06. NLP Processing           → notebooks/06_nlp.ipynb
                               src/nlp/text_processor.py
                               src/nlp/tfidf_engine.py
        │
        ▼  data/processed/tfidf_matrix.pkl
07. Eligibility Engine       → src/recommendation/eligibility_engine.py
08. Recommendation Engine    → src/recommendation/recommender.py
09. Evaluation               → notebooks/09_evaluation.ipynb
                               src/evaluation/evaluator.py
10. RAG Assistant            → src/rag/rag_pipeline.py
```

---

## NLP Pipeline

```
Raw Text (details + eligibility + benefits + tags)
        │
        ▼
1. Lowercase normalization
2. Remove special characters and extra whitespace
3. Tokenization (split into words)
4. Stop-word removal (common words like "the", "is", "for")
5. Lemmatization (reduce words to base form: "farming" → "farm")
6. Build combined text representation per scheme
        │
        ▼
TF-IDF Vectorization
(Term Frequency - Inverse Document Frequency)
        │
        ▼
Cosine Similarity Matrix
(compare user requirement vector against all scheme vectors)
```

**Why TF-IDF first, not transformers?**
TF-IDF is interpretable, fast, and works well on this type of structured government text. We build and evaluate the TF-IDF baseline first, then decide whether transformer-based embeddings add meaningful improvement.

---

## Recommendation Methodology

SchemeXpress uses a **hybrid approach**:

1. **Eligibility Score** — rule-based matching of user profile against extracted scheme criteria (age, gender, state, income, occupation, education, social category)
2. **NLP Similarity Score** — TF-IDF cosine similarity between the user's stated requirement and each scheme's text representation
3. **Hybrid Score** — weighted combination: `score = α × eligibility_score + β × nlp_score`

The weights α and β are experimentally tuned and documented in the evaluation phase.

---

## Project Structure

```
SchemeXpress/
│
├── data/
│   ├── raw/              ← Original Kaggle dataset (not committed)
│   ├── processed/        ← Cleaned and engineered datasets
│   └── interim/          ← Intermediate transformation outputs
│
├── notebooks/            ← Jupyter notebooks for each phase
│
├── src/
│   ├── preprocessing/    ← Data cleaning and feature extraction modules
│   ├── nlp/              ← Text processing and TF-IDF engine
│   ├── recommendation/   ← Eligibility engine and recommender
│   ├── evaluation/       ← Recommendation evaluation metrics
│   ├── database/         ← MongoDB connection and queries
│   └── rag/              ← RAG pipeline (Phase 14)
│
├── backend/
│   ├── app.py            ← Flask application entry point
│   ├── routes/           ← URL route handlers
│   └── services/         ← Business logic layer
│
├── templates/            ← Jinja2 HTML templates
├── static/               ← CSS, JS, images
├── tests/                ← Unit and integration tests
├── docs/                 ← Additional documentation and diagrams
│
├── .env                  ← Secret config (NOT committed — see .env.example)
├── .env.example          ← Template showing required env variables
├── .gitignore
├── requirements.txt
├── README.md
└── PROJECT_DOCUMENTATION.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.11 or higher
- pip
- Git
- A MongoDB Atlas account (free tier is sufficient)

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/SchemeXpress.git
cd SchemeXpress
```

### Step 2 — Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Configure environment variables
```bash
# Copy the example file
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux

# Edit .env and fill in your values
```

### Step 5 — Add the dataset
Download the dataset from Kaggle and place it at:
```
data/raw/schemes.csv
```

### Step 6 — Run the data pipeline
```bash
# Run notebooks in order, or run the pipeline script (Phase 11+)
jupyter notebook
```

### Step 7 — Run the Flask application
```bash
python backend/app.py
```
Then open `http://localhost:5000` in your browser.

---

## Environment Variables

Create a `.env` file in the project root. Required variables:

```
# MongoDB
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/schemexpress

# Flask
FLASK_SECRET_KEY=your-random-secret-key-here
FLASK_ENV=development
FLASK_DEBUG=1

# RAG (Phase 14 — leave blank until needed)
OPENAI_API_KEY=
```

**Never commit your `.env` file. It is listed in `.gitignore`.**

---

## MongoDB Setup

1. Create a free account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a new cluster (M0 free tier)
3. Create a database user with read/write permissions
4. Whitelist your IP address (or use `0.0.0.0/0` for development)
5. Copy the connection string and set it as `MONGO_URI` in your `.env`

**Collections used:**

| Collection | Purpose |
|---|---|
| `schemes` | All government schemes from the processed dataset |
| `users` | User profiles (future phase) |
| `recommendation_history` | Past recommendation sessions (future phase) |

---

## Screenshots

> Screenshots will be added after the frontend is built in Phase 12.

---

## Evaluation Results

> Evaluation metrics (Precision@K, Recall@K) will be added after Phase 9.

---

## Limitations

- The dataset contains government schemes data that may be outdated. Always verify eligibility and application details on official government portals.
- The recommendation system provides **preliminary matches only** — not official government eligibility determinations.
- Text-based NLP matching may miss schemes with relevant eligibility but sparse or inconsistent text descriptions.
- The current system is optimized for Indian government schemes and the Hindi/English mixed text present in the dataset.

---

## Future Improvements

- [ ] User authentication and saved profiles
- [ ] Multilingual support (Hindi, Tamil, Telugu, Bengali)
- [ ] Real-time scheme data updates via government APIs
- [ ] Collaborative filtering based on user interaction history
- [ ] Mobile application
- [ ] SMS/WhatsApp notification for scheme deadlines
- [ ] Transformer-based embeddings (SBERT) for improved NLP similarity
- [ ] Fine-tuned LLM for the RAG assistant

---

## Government Data Disclaimer

> SchemeXpress provides **preliminary recommendations** based on the information available in its dataset. Eligibility rules, benefits, application procedures, and scheme availability are subject to change by the concerned government authority. Users must verify the latest information on the official government source before applying. **SchemeXpress does not guarantee government approval or actual eligibility.**

---

## Data Attribution

Dataset sourced from [Kaggle](https://www.kaggle.com/). All scheme information is derived from publicly available Indian government data. SchemeXpress does not claim ownership of any scheme data.

---

## License

This project is developed for educational purposes as part of a Data Science academic portfolio.

---

*Built with Python, Flask, MongoDB, and a genuine interest in making government information accessible.*
