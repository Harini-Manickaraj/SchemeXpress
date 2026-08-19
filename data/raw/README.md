# data/raw/

This directory holds the original, unmodified source dataset.

## Important Rule

**Never modify files in this directory.**

Raw data is sacred. All transformations happen in code and output to `data/processed/` or `data/interim/`. If you need to re-run the pipeline from scratch, the raw file must be available exactly as it was downloaded.

## Dataset

**File expected:** `schemes.csv`  
**Source:** Kaggle — Indian Government Schemes Dataset  
**Approximate size:** ~3,400 records

## How to Get the Dataset

1. Go to [Kaggle](https://www.kaggle.com/) and search for "Indian government schemes"
2. Download the dataset CSV file
3. Rename it to `schemes.csv` if needed
4. Place it at: `data/raw/schemes.csv`

## Why This File is Not in the Repository

- The CSV file is large and Kaggle datasets have their own licensing terms
- GitHub has a 100MB file size limit; large CSVs should not be committed
- Raw data files are listed in `.gitignore`

## Known Contents (to be confirmed in Phase 2)

| Column | Description |
|---|---|
| `scheme_name` | Official scheme name |
| `slug` | URL-friendly identifier |
| `details` | Scheme overview |
| `benefits` | What the scheme provides |
| `eligibility` | Who can apply |
| `application` | How to apply |
| `documents` | Required documents |
| `level` | Central / State |
| `schemeCategory` | Scheme sector |
| `tags` | Associated keywords |

One column appears to be unnamed/empty — this will be confirmed and handled in Phase 2.
