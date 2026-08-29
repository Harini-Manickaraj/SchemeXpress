"""
src/preprocessing/cleaner.py
============================
Production data cleaning module for SchemeXpress.

This module contains the same logic developed in notebooks/03_preprocessing.ipynb,
but packaged as clean, importable functions that the rest of the application can use.

KEY PRINCIPLE
-------------
Notebooks are for learning and experimenting.
This module is the result — tested, reusable, no Jupyter dependency.

USAGE
-----
    from src.preprocessing.cleaner import run_full_pipeline

    df_clean = run_full_pipeline(
        raw_path='data/raw/updated_data.csv',
        save_path='data/processed/schemes_clean.csv'
    )
"""

import re
import html
import logging
import os

import pandas as pd

# ── Logger ────────────────────────────────────────────────────────────────────
# logging lets us print informative messages from inside a module
# without using bare print() statements (which can't be turned off).
logger = logging.getLogger(__name__)


# =============================================================================
# INDIVIDUAL CLEANING FUNCTIONS
# Each function does exactly one thing. Small functions are easy to test and
# easy to understand. This is the Single Responsibility Principle.
# =============================================================================

def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are completely empty or have unnamed/blank headers.

    The raw CSV has an extra comma in the header row which creates a column
    with an empty string name. It contains no data and must be removed.

    Args:
        df: Raw DataFrame loaded from CSV.

    Returns:
        DataFrame with empty-header columns removed.
    """
    cols_to_drop = [
        col for col in df.columns
        if str(col).strip() == '' or str(col).startswith('Unnamed:')
    ]
    if cols_to_drop:
        logger.info(f'Dropping empty columns: {cols_to_drop}')
        df = df.drop(columns=cols_to_drop)
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Two passes:
      1. Exact full-row duplicates (every column matches).
      2. Rows with duplicate 'slug' values (same scheme URL = same scheme).

    Args:
        df: DataFrame after empty columns have been dropped.

    Returns:
        DataFrame with duplicates removed.
    """
    before = len(df)

    # Pass 1: exact full-row duplicates
    df = df.drop_duplicates(keep='first')
    after_full = len(df)
    logger.info(f'Exact duplicate rows removed: {before - after_full}')

    # Pass 2: duplicate slugs
    df = df.drop_duplicates(subset=['slug'], keep='first')
    after_slug = len(df)
    logger.info(f'Duplicate-slug rows removed:  {after_full - after_slug}')
    logger.info(f'Total rows after deduplication: {len(df)}')

    return df


def _make_slug(name: str) -> str:
    """
    Convert a scheme name into a URL-friendly slug.

    Example: 'PM Kisan Samman Nidhi' → 'pm-kisan-samman-nidhi'

    Args:
        name: The scheme name string.

    Returns:
        A lowercase, hyphen-separated slug string.
    """
    s = str(name).lower().strip()
    s = re.sub(r'[^a-z0-9\s-]', '', s)   # keep only alphanumeric, spaces, hyphens
    s = re.sub(r'\s+', '-', s)            # spaces → hyphens
    s = re.sub(r'-+', '-', s)             # collapse multiple hyphens
    return s.strip('-')


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values with safe, meaningful defaults.

    Strategy:
      - Text fields used in NLP → empty string ''
        (missing text = no signal, not an error)
      - Category fields used in filtering → labeled placeholder
      - slug → derived from scheme_name if missing

    Args:
        df: DataFrame after duplicate removal.

    Returns:
        DataFrame with no NaN values.
    """
    # Text fields → empty string (missing text = no NLP signal, not an error)
    text_cols = ['details', 'benefits', 'eligibility',
                 'application', 'documents', 'tags']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # Category fields → meaningful placeholder
    if 'schemeCategory' in df.columns:
        df['schemeCategory'] = df['schemeCategory'].fillna('Uncategorized')

    if 'level' in df.columns:
        df['level'] = df['level'].fillna('Unknown')

    # slug → derive from scheme_name if missing
    if 'slug' in df.columns and 'scheme_name' in df.columns:
        df['slug'] = df['slug'].fillna(df['scheme_name'].apply(_make_slug))

    remaining_nulls = df.isnull().sum().sum()
    logger.info(f'Remaining null values after fill: {remaining_nulls}')

    return df


def clean_text(text: str) -> str:
    """
    Clean a single text string.

    Cleaning steps (in order):
      1. Return '' if input is not a string (handles NaN that slipped through)
      2. Remove BOM and zero-width Unicode characters
      3. Decode HTML entities  (&amp; → &,  &nbsp; → space,  &lt; → <)
      4. Replace newlines and tab characters with a space
      5. Collapse multiple consecutive spaces into one
      6. Strip leading and trailing whitespace

    Args:
        text: A raw text string from the dataset.

    Returns:
        A cleaned, normalized text string.
    """
    if not isinstance(text, str):
        return ''

    # Step 1: Remove BOM and invisible Unicode characters
    text = text.replace('\ufeff', '')   # BOM character
    text = text.replace('\u200b', '')   # zero-width space
    text = text.replace('\u00a0', ' ')  # non-breaking space → regular space

    # Step 2: Decode HTML entities
    text = html.unescape(text)

    # Step 3: Replace newlines and tabs with a space
    text = re.sub(r'[\n\r\t]+', ' ', text)

    # Step 4: Collapse multiple spaces → one
    text = re.sub(r' {2,}', ' ', text)

    # Step 5: Strip edges
    text = text.strip()

    return text


def clean_all_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clean_text() to every text column in the DataFrame.

    Args:
        df: DataFrame after missing value fill.

    Returns:
        DataFrame with all text columns cleaned.
    """
    text_cols = ['scheme_name', 'details', 'benefits', 'eligibility',
                 'application', 'documents', 'tags']

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Extra step: strip outer quote characters from scheme_name
    if 'scheme_name' in df.columns:
        df['scheme_name'] = (
            df['scheme_name']
            .str.strip()
            .str.strip('"')
            .str.strip("'")
            .str.strip()
        )

    logger.info('Text cleaning applied to all columns.')
    return df


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize schemeCategory and level column values.

    Problems fixed:
      - Inconsistent casing ('education & learning' vs 'Education & Learning')
      - Extra whitespace
      - Unknown level values outside the expected set

    Args:
        df: DataFrame after text cleaning.

    Returns:
        DataFrame with standardized categorical columns.
    """
    valid_levels = {'Central', 'State', 'District', 'Unknown'}

    if 'schemeCategory' in df.columns:
        df['schemeCategory'] = df['schemeCategory'].str.strip().str.title()

    if 'level' in df.columns:
        df['level'] = df['level'].str.strip().str.title()
        df['level'] = df['level'].apply(
            lambda x: x if x in valid_levels else 'Unknown'
        )

    logger.info('Category standardization complete.')
    return df


def build_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a 'combined_text' column for TF-IDF vectorization.

    The TF-IDF engine needs one text string per scheme. This function
    concatenates the most content-rich fields into a single combined field.

    Formula:
        combined_text = scheme_name + schemeCategory + details
                        + benefits + eligibility + tags

    Note: 'application' and 'documents' are excluded because they describe
    how to apply — irrelevant to relevance matching.

    Args:
        df: DataFrame after category standardization.

    Returns:
        DataFrame with new 'combined_text' column.
    """
    df['combined_text'] = (
        df.get('scheme_name', pd.Series([''] * len(df))).fillna('')    + ' ' +
        df.get('schemeCategory', pd.Series([''] * len(df))).fillna('') + ' ' +
        df.get('details', pd.Series([''] * len(df))).fillna('')        + ' ' +
        df.get('benefits', pd.Series([''] * len(df))).fillna('')       + ' ' +
        df.get('eligibility', pd.Series([''] * len(df))).fillna('')    + ' ' +
        df.get('tags', pd.Series([''] * len(df))).fillna('')
    )

    # Normalize whitespace in combined_text
    df['combined_text'] = df['combined_text'].apply(
        lambda t: re.sub(r' {2,}', ' ', str(t).strip())
    )

    avg_len = df['combined_text'].str.len().mean()
    logger.info(f'combined_text built. Average length: {avg_len:.0f} chars')

    return df


# =============================================================================
# MAIN PIPELINE FUNCTION
# Chains all individual steps in the correct order.
# =============================================================================

def run_full_pipeline(
    raw_path: str,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline from raw CSV to clean DataFrame.

    Steps (in order):
        1. Load raw CSV
        2. Drop empty/unnamed columns
        3. Drop duplicate rows
        4. Fill missing values
        5. Clean all text columns
        6. Standardize categories
        7. Build combined_text
        8. Save to CSV (optional)

    Args:
        raw_path:  Path to the raw input CSV file.
        save_path: Path to save the cleaned CSV. If None, the result
                   is only returned as a DataFrame (not saved).

    Returns:
        The cleaned pandas DataFrame.

    Raises:
        FileNotFoundError: If raw_path does not exist.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f'Raw data file not found: {raw_path}')

    logger.info(f'Loading raw data from: {raw_path}')
    df = pd.read_csv(raw_path)
    logger.info(f'Raw shape: {df.shape}')

    # Pipeline steps
    df = drop_empty_columns(df)
    df = drop_duplicates(df)
    df = fill_missing_values(df)
    df = clean_all_text_columns(df)
    df = standardize_categories(df)
    df = build_combined_text(df)

    logger.info(f'Clean shape: {df.shape}')

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, encoding='utf-8')
        logger.info(f'Clean dataset saved to: {save_path}')

    return df
