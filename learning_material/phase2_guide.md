# Phase 2 Learning Guide — Data Collection & Preprocessing

This guide explains every concept you encounter in Phase 2 in plain language.
Read it alongside the notebooks — one section per notebook.

---

## Part 1 — What is a DataFrame?

A **DataFrame** is the core data structure in pandas. Think of it as a spreadsheet in Python.

```
     scheme_name          level    schemeCategory
0    PM Kisan...         Central   Agriculture
1    Beti Bachao...      Central   Women Welfare
2    AICTE STTP...       Central   Education
```

- Each **row** is one record (one government scheme).
- Each **column** is one attribute (name, level, category, etc.).
- The numbers on the left (0, 1, 2) are the **index** — the row identifier.

### Creating and loading a DataFrame

```python
import pandas as pd

# Load from a CSV file
df = pd.read_csv('data/raw/updated_data.csv')

# Check dimensions: (rows, columns)
print(df.shape)          # e.g. (3406, 11)

# See column names
print(df.columns.tolist())

# See first 5 rows
print(df.head())

# See last 5 rows
print(df.tail())
```

---

## Part 2 — Missing Values (NaN)

**NaN** stands for "Not a Number". In pandas, it represents an empty cell — a missing value.

### Why missing values are a problem

If the `eligibility` field of a scheme is NaN, our recommendation engine has no text to
match against a user's profile. The scheme might be perfectly relevant but gets ignored.

### Detecting missing values

```python
# Count missing values per column
df.isnull().sum()

# As a percentage
(df.isnull().sum() / len(df) * 100).round(2)

# Check if a specific cell is NaN
pd.isna(df['eligibility'].iloc[0])   # True if missing, False if not
```

### Fixing missing values — fillna()

```python
# Fill all NaN in 'details' with an empty string
df['details'] = df['details'].fillna('')

# Fill NaN in 'schemeCategory' with a placeholder
df['schemeCategory'] = df['schemeCategory'].fillna('Uncategorized')
```

**Why empty string, not 'Unknown'?**
For text fields used in NLP, an empty string contributes zero signal to TF-IDF.
That's correct behaviour — we don't want fake text influencing the similarity score.
For category fields used in filtering, a labeled placeholder like 'Uncategorized'
is honest and filterable.

---

## Part 3 — Duplicate Rows

A **duplicate row** is a record that appears more than once in the dataset.
This happens in web-scraped data when the scraper visits the same page twice.

### Why duplicates are a problem

If PM Kisan Samman Nidhi appears twice, the recommendation engine will
suggest it twice at the top of every farmer's results — making the system
look broken.

### Detecting duplicates

```python
# Count exact full-row duplicates
df.duplicated().sum()

# Count rows where scheme_name appears more than once
df.duplicated(subset=['scheme_name']).sum()

# See the actual duplicated rows
df[df.duplicated(subset=['scheme_name'], keep=False)]
```

### Removing duplicates

```python
# Keep the first occurrence, drop all later copies
df = df.drop_duplicates(keep='first')

# Drop duplicates based on one specific column
df = df.drop_duplicates(subset=['slug'], keep='first')
```

---

## Part 4 — Text Cleaning

Raw text from web scraping is dirty. Here's what you'll find and how to fix it.

### 1. BOM Characters (\\ufeff)

A **BOM** (Byte Order Mark) is an invisible character that some text editors and
scrapers place at the beginning of a text field. It looks like `` (nothing) but
it corrupts the first token in TF-IDF.

```python
text = '\ufeffgovernment scheme'
cleaned = text.replace('\ufeff', '')
print(cleaned)  # 'government scheme'
```

### 2. HTML Entities

When websites store text, special characters like `&`, `<`, `>` are encoded:
- `&amp;`  → `&`
- `&lt;`   → `<`
- `&gt;`   → `>`
- `&nbsp;` → (non-breaking space)

```python
import html

text = 'Women &amp; Child Development'
cleaned = html.unescape(text)
print(cleaned)  # 'Women & Child Development'
```

### 3. Newlines and Tabs

Text scraped from HTML often contains `\n` (newline) and `\t` (tab) characters
from the original HTML formatting. For TF-IDF, these should just be spaces.

```python
import re

text = 'Step 1: Apply online\nStep 2: Upload documents\n'
cleaned = re.sub(r'[\n\r\t]+', ' ', text)
print(cleaned)  # 'Step 1: Apply online Step 2: Upload documents '
```

### 4. Regular Expressions (re module)

`re.sub(pattern, replacement, string)` — find all matches of `pattern` and replace them.

| Pattern | Matches | Example |
|---|---|---|
| `\n` | newline character | `re.sub(r'\n', ' ', text)` |
| `[\n\r\t]+` | one or more newlines/returns/tabs | Same as above, combined |
| ` {2,}` | 2 or more consecutive spaces | `re.sub(r' {2,}', ' ', text)` |
| `&[a-zA-Z]+;` | HTML entity like `&amp;` | |
| `[^a-z0-9\s]` | anything NOT a letter, digit, or space | |

---

## Part 5 — What is TF-IDF? (Preview for Phase 6)

You'll build this in Phase 6, but it helps to understand *why* we're building `combined_text` now.

**TF-IDF** = Term Frequency - Inverse Document Frequency.

It converts a text document into a list of numbers (a vector).
Words that are rare across all documents get high scores; common words get low scores.

Example with 3 schemes:
```
Scheme A: "financial assistance farmer agriculture"
Scheme B: "scholarship student education university"
Scheme C: "financial health medical insurance"
```

The word "financial" appears in schemes A and C — medium score.
The word "farmer" appears only in scheme A — high score.
The word "the" would appear everywhere — score ≈ 0 (filtered out as a stopword).

**Why we build combined_text:**
TF-IDF works on one document per scheme. We combine all relevant fields
so the single vector captures scheme_name + category + description + eligibility + tags.

```python
df['combined_text'] = (
    df['scheme_name']    + ' ' +
    df['schemeCategory'] + ' ' +
    df['details']        + ' ' +
    df['benefits']       + ' ' +
    df['eligibility']    + ' ' +
    df['tags']
)
```

---

## Part 6 — The src/ vs notebooks/ Separation

### Why two places?

| Location | Purpose | Who uses it |
|---|---|---|
| `notebooks/` | Experiment, learn, visualize | You, during development |
| `src/` | Clean, tested, reusable code | Flask app, other modules |

Think of the notebook as your rough work on paper.
Think of `src/` as the clean answer you write in the exam.

### How they relate

The notebook `03_preprocessing.ipynb` and the module `src/preprocessing/cleaner.py`
contain the **exact same logic**. The difference:

- In the notebook, every step prints output so you can see what's happening.
- In `cleaner.py`, the logic is wrapped in functions that can be imported anywhere
  in the project with `from src.preprocessing.cleaner import run_full_pipeline`.

### The pattern in every phase

```
notebook/  →  experiment + learn  →  extract logic  →  src/ module
```

---

## Part 7 — File Organization for Data

```
data/
  raw/          ← Original file. NEVER touch this.
  interim/      ← Intermediate steps (optional, for large pipelines)
  processed/    ← Final clean file. This is what all later phases use.
```

**Why never touch raw/?**
If your cleaning code has a bug, you can always rerun it on the original.
If you overwrite the original, the mistake is permanent.

This is the same principle as version control: you don't edit your git history.

---

## Quick Reference — Most Used pandas Commands in Phase 2

```python
# Load
df = pd.read_csv('file.csv')

# Inspect
df.shape                    # (rows, cols)
df.head(n)                  # first n rows
df.dtypes                   # data type of each column
df.describe()               # statistics for numeric columns

# Missing values
df.isnull().sum()           # count NaN per column
df['col'].fillna('')        # fill NaN with empty string
df.dropna(subset=['col'])   # drop rows where 'col' is NaN

# Duplicates
df.duplicated().sum()
df.drop_duplicates(keep='first')

# Column operations
df.drop(columns=['col'])                 # remove a column
df['col'].str.strip()                    # remove whitespace
df['col'].str.replace(old, new)          # simple string replace
df['col'].apply(my_function)             # apply custom function to every cell

# Save
df.to_csv('output.csv', index=False)
```

---

## Self-Check Questions

After completing Phase 2, you should be able to answer:

1. What is a DataFrame? How is it different from a Python list?
2. What does `df.isnull().sum()` return?
3. Why do we fill missing text fields with `''` instead of dropping the rows?
4. What is a duplicate row and why is it a problem for recommendation?
5. What does `html.unescape()` do? Give one example.
6. What is the purpose of `combined_text`?
7. What is the difference between `notebooks/03_preprocessing.ipynb` and `src/preprocessing/cleaner.py`?
8. Why do we never modify files in `data/raw/`?

Answers to all of these are in the notebooks and in this guide.
