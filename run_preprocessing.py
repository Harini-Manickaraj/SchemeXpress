"""
run_preprocessing.py
====================
Command-line script to run the full SchemeXpress preprocessing pipeline.

Run this from the project root:
    python run_preprocessing.py

What it does:
    1. Reads  data/raw/updated_data.csv
    2. Runs the full cleaning pipeline  (src/preprocessing/cleaner.py)
    3. Saves data/processed/schemes_clean.csv

After running this script successfully, you are ready for Phase 3 (EDA).
"""

import logging
import sys
import os

# ── Set up logging so we can see what each step is doing ──────────────────────
# basicConfig sets the default log level to INFO, which means all logger.info()
# calls inside cleaner.py will print to the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def main():
    # ── Paths ─────────────────────────────────────────────────────────────────
    # os.path.dirname(__file__) = the directory containing this script
    # This makes the script work from any working directory.
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    raw_path   = os.path.join(base_dir, 'data', 'raw',       'updated_data.csv')
    clean_path = os.path.join(base_dir, 'data', 'processed', 'schemes_clean.csv')

    logger.info('=' * 55)
    logger.info('SchemeXpress — Preprocessing Pipeline')
    logger.info('=' * 55)
    logger.info(f'Input:  {raw_path}')
    logger.info(f'Output: {clean_path}')

    # ── Import and run the pipeline ───────────────────────────────────────────
    # We import here (not at the top) so a clear error is shown if the
    # venv isn't activated and pandas is missing.
    try:
        from src.preprocessing.cleaner import run_full_pipeline
    except ImportError as e:
        logger.error(f'Import failed: {e}')
        logger.error('Make sure you have activated the virtual environment:')
        logger.error('  .venv\\Scripts\\Activate.ps1   (Windows PowerShell)')
        sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        df = run_full_pipeline(raw_path=raw_path, save_path=clean_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error('Place the dataset at data/raw/updated_data.csv and try again.')
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info('─' * 55)
    logger.info('PIPELINE COMPLETE')
    logger.info(f'  Rows in clean dataset:    {len(df)}')
    logger.info(f'  Columns:                  {len(df.columns)}')
    logger.info(f'  Null values remaining:    {df.isnull().sum().sum()}')
    logger.info(f'  Output file:              {clean_path}')
    logger.info('─' * 55)
    logger.info('Next step: open notebooks/03_preprocessing.ipynb')
    logger.info('           to verify the output interactively.')


if __name__ == '__main__':
    main()
