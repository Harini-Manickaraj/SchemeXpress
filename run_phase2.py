"""
run_phase2.py  —  SchemeXpress Phase 2 pipeline
================================================
Runs end-to-end:
  1. Data understanding  (prints stats, saves to docs/_phase2_stats.json)
  2. Data quality analysis
  3. Preprocessing       (saves data/processed/cleaned_schemes.csv)
  4. EDA charts          (saves PNGs to docs/_eda_charts/)

Usage (from project root with venv active):
    python run_phase2.py

Raw data is NEVER modified.
"""

import os, sys, re, html, json, warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
RAW_PATH   = os.path.join(ROOT, "data", "raw", "updated_data.csv")
OUT_PATH   = os.path.join(ROOT, "data", "processed", "cleaned_schemes.csv")
CHARTS_DIR = os.path.join(ROOT, "docs", "_eda_charts")
STATS_PATH = os.path.join(ROOT, "docs", "_phase2_stats.json")

os.makedirs(os.path.join(ROOT, "data", "processed"), exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── imports ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed; saves to files
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 5)})

def sep(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)

# ══════════════════════════════════════════════════════════════
# SECTION 1 — LOAD & UNDERSTAND RAW DATA
# ══════════════════════════════════════════════════════════════
sep("1. LOAD RAW DATA")

if not os.path.exists(RAW_PATH):
    sys.exit(f"ERROR: dataset not found at {RAW_PATH}")

raw = pd.read_csv(RAW_PATH, low_memory=False)
print(f"Raw shape  : {raw.shape[0]:,} rows × {raw.shape[1]} columns")
print(f"\nColumns ({raw.shape[1]}):")
for i, col in enumerate(raw.columns):
    print(f"  {i:>2}. {col!r}")

sep("1a. DATA TYPES")
print(raw.dtypes.to_string())

sep("1b. MISSING VALUES (raw)")
null_counts = raw.isnull().sum()
null_pct    = (null_counts / len(raw) * 100).round(2)
missing_df  = pd.DataFrame({"missing_count": null_counts, "missing_pct": null_pct})
print(missing_df[missing_df.missing_count > 0].to_string()
      if missing_df.missing_count.sum() > 0 else "  No missing values.")

sep("1c. DUPLICATES (raw)")
exact_dupes = int(raw.duplicated().sum())
print(f"Exact full-row duplicates : {exact_dupes}")
if "slug" in raw.columns:
    slug_dupes = int(raw.duplicated(subset=["slug"]).sum())
    print(f"Duplicate slug values     : {slug_dupes}")
else:
    slug_dupes = 0
    print("No 'slug' column found.")

sep("1d. UNIQUE VALUES PER COLUMN")
for col in raw.columns:
    print(f"  {col!r:<30}  {raw[col].nunique():>5} unique  |  dtype={raw[col].dtype}")

sep("1e. TEXT FIELD LENGTH STATS")
text_cols = [c for c in ["details","benefits","eligibility","application","documents","tags"]
             if c in raw.columns]
for col in text_cols:
    series = raw[col].dropna().astype(str)
    series = series[series.str.strip() != ""]
    if len(series):
        lens = series.str.len()
        print(f"  {col:<15}  n={len(series):>5}  "
              f"min={int(lens.min()):>5}  "
              f"median={int(lens.median()):>6}  "
              f"max={int(lens.max()):>7}")

sep("1f. CATEGORICAL VALUE COUNTS")
cat_cols = [c for c in ["level", "schemeCategory"] if c in raw.columns]
for col in cat_cols:
    vc = raw[col].value_counts(dropna=False)
    print(f"\n  {col} ({vc.shape[0]} unique values):")
    print(vc.head(25).to_string())

sep("1g. SAMPLE RECORDS")
print(raw.head(2).T.to_string())

# ══════════════════════════════════════════════════════════════
# SECTION 2 — DATA QUALITY ANALYSIS
# ══════════════════════════════════════════════════════════════
sep("2. DATA QUALITY ANALYSIS")

# Empty / whitespace-only strings (not null but blank)
print("\nEmpty / whitespace-only strings per text column:")
for col in text_cols:
    blank_count = raw[col].fillna("").astype(str).str.strip().eq("").sum()
    print(f"  {col:<15}  {blank_count:>5} blank entries "
          f"(includes {int(raw[col].isnull().sum())} NaN)")

# Scheme name quality
if "scheme_name" in raw.columns:
    quoted = raw["scheme_name"].astype(str).str.startswith('"').sum()
    print(f"\nScheme names starting with quote char : {quoted}")
    print(f"Scheme name avg length                : "
          f"{raw['scheme_name'].astype(str).str.len().mean():.0f} chars")

# Check for BOM / HTML entities in text
bom_total = 0
html_ent_total = 0
for col in text_cols:
    bom_count = raw[col].dropna().astype(str).str.contains("\ufeff", regex=False).sum()
    ent_count = raw[col].dropna().astype(str).str.contains(r"&[a-zA-Z]+;|&#\d+;", regex=True).sum()
    bom_total += bom_count
    html_ent_total += ent_count
    if bom_count or ent_count:
        print(f"  {col:<15}  BOM chars in {bom_count} rows  |  HTML entities in {ent_count} rows")
print(f"\nTotal rows with BOM chars    : {bom_total}")
print(f"Total rows with HTML entities: {html_ent_total}")

# ══════════════════════════════════════════════════════════════
# SECTION 3 — PREPROCESSING
# ══════════════════════════════════════════════════════════════
sep("3. PREPROCESSING")

df = raw.copy()   # raw is NEVER touched after this point

# 3a — Drop unnamed / empty-header columns
unnamed = [c for c in df.columns
           if str(c).strip() == "" or str(c).startswith("Unnamed:")]
if unnamed:
    print(f"Dropping unnamed/empty columns: {unnamed}")
    df = df.drop(columns=unnamed)
else:
    print("No unnamed columns to drop.")
print(f"Shape after column drop: {df.shape}")

# 3b — Remove exact full-row duplicates
before = len(df)
df = df.drop_duplicates(keep="first")
rows_removed_exact = before - len(df)
print(f"Exact duplicate rows removed : {rows_removed_exact}")

# 3c — Remove duplicate slugs
rows_removed_slug = 0
if "slug" in df.columns:
    before = len(df)
    df = df.drop_duplicates(subset=["slug"], keep="first")
    rows_removed_slug = before - len(df)
    print(f"Duplicate-slug rows removed  : {rows_removed_slug}")
print(f"Rows after deduplication     : {len(df):,}")

# 3d — Fill missing values
#   Text NLP fields → empty string  (missing = no signal, not an error)
#   Category fields → labeled placeholder
TEXT_FILL = [c for c in ["details","benefits","eligibility",
                          "application","documents","tags"] if c in df.columns]
for col in TEXT_FILL:
    df[col] = df[col].fillna("")

if "schemeCategory" in df.columns:
    df["schemeCategory"] = df["schemeCategory"].fillna("Uncategorized")
if "level" in df.columns:
    df["level"] = df["level"].fillna("Unknown")

# Derive missing slugs from scheme_name
if "slug" in df.columns and "scheme_name" in df.columns:
    def _slugify(name):
        s = str(name).lower().strip()
        s = re.sub(r"[^a-z0-9\s-]", "", s)
        s = re.sub(r"\s+", "-", s)
        s = re.sub(r"-+", "-", s)
        return s.strip("-")
    df["slug"] = df["slug"].fillna(df["scheme_name"].apply(_slugify))

nulls_after_fill = int(df.isnull().sum().sum())
print(f"Null values after fill       : {nulls_after_fill}")

# 3e — Clean text  (light NLP-safe cleaning — does NOT lowercase / stem)
def clean_text(text):
    """
    NLP-safe cleaning:
    • Remove BOM, zero-width chars, non-breaking spaces
    • Decode HTML entities (&amp; → &, etc.)
    • Collapse newlines/tabs → single space
    • Collapse multiple spaces → single space
    • Strip leading/trailing whitespace
    Does NOT lowercase or stem — those happen inside the NLP pipeline (Phase 6).
    """
    if not isinstance(text, str) or text == "":
        return ""
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u00a0", " ")
    text = html.unescape(text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

ALL_TEXT = [c for c in ["scheme_name","details","benefits","eligibility",
                         "application","documents","tags"] if c in df.columns]
for col in ALL_TEXT:
    df[col] = df[col].apply(clean_text)

# Strip outer quotes from scheme_name (CSV scraping artifact)
if "scheme_name" in df.columns:
    df["scheme_name"] = (df["scheme_name"]
                         .str.strip()
                         .str.strip('"')
                         .str.strip("'")
                         .str.strip())

# 3f — Standardize categoricals
VALID_LEVELS = {"Central", "State", "District", "Unknown"}
if "schemeCategory" in df.columns:
    df["schemeCategory"] = df["schemeCategory"].str.strip().str.title()
if "level" in df.columns:
    df["level"] = df["level"].str.strip().str.title()
    df["level"] = df["level"].apply(lambda x: x if x in VALID_LEVELS else "Unknown")

# 3g — Build combined_text for TF-IDF (used in Phase 6+, built here for completeness)
ct_parts = [c for c in ["scheme_name","schemeCategory","details",
                          "benefits","eligibility","tags"] if c in df.columns]
df["combined_text"] = df[ct_parts].fillna("").agg(" ".join, axis=1)
df["combined_text"] = df["combined_text"].apply(
    lambda t: re.sub(r" {2,}", " ", t.strip()))

print(f"\nFinal clean shape            : {df.shape}")
print(f"Columns                      : {list(df.columns)}")
print(f"Null values in clean dataset : {df.isnull().sum().sum()}")

# 3h — Save clean dataset
# Ensure all text fill columns are pure string (no NaN) before writing
for _col in TEXT_FILL:
    if _col in df.columns:
        df[_col] = df[_col].astype(str)
        df[_col] = df[_col].replace("nan", "").replace("<NA>", "")
df.to_csv(OUT_PATH, index=False, encoding="utf-8")
verify = pd.read_csv(OUT_PATH, keep_default_na=False)
print(f"\nSaved → {OUT_PATH}")
print(f"Reload verification          : {verify.shape[0]:,} rows × {verify.shape[1]} columns  ✓")
print(f"Null values in saved file    : {verify.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════
# SECTION 4 — EDA
# ══════════════════════════════════════════════════════════════
sep("4. EDA — GENERATING CHARTS")

def save_fig(name):
    path = os.path.join(CHARTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ── Chart 1: Level distribution ──────────────────────────────
if "level" in df.columns:
    fig, ax = plt.subplots(figsize=(7, 4))
    vc = df["level"].value_counts()
    colors = sns.color_palette("muted", len(vc))
    bars = ax.bar(vc.index, vc.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Distribution of Schemes by Level (Central / State / District)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Level", fontsize=11)
    ax.set_ylabel("Number of Schemes", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top","right"]].set_visible(False)
    save_fig("01_level_distribution.png")

# ── Chart 2: Top 20 scheme categories ────────────────────────
if "schemeCategory" in df.columns:
    fig, ax = plt.subplots(figsize=(12, 7))
    top_cats = df["schemeCategory"].value_counts().head(20)
    sns.barplot(x=top_cats.values, y=top_cats.index, ax=ax, palette="Blues_d")
    for i, val in enumerate(top_cats.values):
        ax.text(val + 3, i, f"{val:,}", va="center", fontsize=9)
    ax.set_title("Top 20 Scheme Categories by Count", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Number of Schemes", fontsize=11)
    ax.set_ylabel("Scheme Category", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    save_fig("02_top_categories.png")

# ── Chart 3: Missing-value heatmap (raw) ─────────────────────
raw_no_unnamed = raw.drop(columns=[c for c in raw.columns
                                   if str(c).strip() == "" or str(c).startswith("Unnamed:")],
                          errors="ignore")
null_pct_series = (raw_no_unnamed.isnull().sum() / len(raw_no_unnamed) * 100).round(1)
null_pct_nonzero = null_pct_series[null_pct_series > 0]

if len(null_pct_nonzero) > 0:
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(null_pct_nonzero.index, null_pct_nonzero.values,
                   color=sns.color_palette("Reds_d", len(null_pct_nonzero)))
    for bar, val in zip(bars, null_pct_nonzero.values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_title("Missing Values (%) per Column — Raw Dataset",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("% Missing", fontsize=11)
    ax.set_ylabel("Column", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    save_fig("03_missing_values.png")
else:
    print("  No missing values — skipping chart 03.")

# ── Chart 4: Text field length distributions ─────────────────
plot_text_cols = [c for c in ["details","eligibility","benefits"] if c in df.columns]
if plot_text_cols:
    fig, axes = plt.subplots(1, len(plot_text_cols),
                             figsize=(6 * len(plot_text_cols), 5), sharey=False)
    if len(plot_text_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, plot_text_cols):
        series = df[col].astype(str)
        series = series[series.str.strip() != ""].str.len()
        if len(series) == 0:
            continue
        series_clipped = series.clip(upper=series.quantile(0.99))
        ax.hist(series_clipped, bins=50, color=sns.color_palette("muted")[2],
                edgecolor="white", alpha=0.85)
        ax.set_title(f"'{col}' text length distribution\n"
                     f"(clipped at 99th pct = {int(series.quantile(0.99)):,} chars)",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Characters", fontsize=10)
        ax.set_ylabel("Number of Schemes", fontsize=10)
        ax.spines[["top","right"]].set_visible(False)
    plt.suptitle("Text Field Length Distributions", fontsize=13, fontweight="bold", y=1.02)
    save_fig("04_text_lengths.png")

# ── Chart 5: Top 20 tags ──────────────────────────────────────
if "tags" in df.columns:
    all_tags = (
        df["tags"]
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .str.title()
    )
    all_tags = all_tags[all_tags.str.len() > 1]
    top_tags = all_tags.value_counts().head(20)
    if len(top_tags) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(x=top_tags.values, y=top_tags.index, ax=ax, palette="Greens_d")
        for i, val in enumerate(top_tags.values):
            ax.text(val + 1, i, f"{val:,}", va="center", fontsize=9)
        ax.set_title("Top 20 Most Frequent Tags", fontsize=13,
                     fontweight="bold", pad=12)
        ax.set_xlabel("Frequency", fontsize=11)
        ax.set_ylabel("Tag", fontsize=11)
        ax.spines[["top","right"]].set_visible(False)
        save_fig("05_top_tags.png")

# ── Chart 6: Schemes per level and category (stacked bar, top 10 cats) ───
if "level" in df.columns and "schemeCategory" in df.columns:
    top10_cats = df["schemeCategory"].value_counts().head(10).index.tolist()
    sub = df[df["schemeCategory"].isin(top10_cats)]
    pivot = sub.groupby(["schemeCategory","level"]).size().unstack(fill_value=0)
    # Reorder columns
    for lv in ["Central","State","District","Unknown"]:
        if lv not in pivot.columns:
            pivot[lv] = 0
    pivot = pivot[["Central","State","District","Unknown"]]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(13, 6))
    pivot.plot(kind="barh", stacked=True, ax=ax,
               color=["#4878CF","#6ACC65","#D65F5F","#B47CC7"],
               edgecolor="white")
    ax.set_title("Top 10 Categories: Scheme Count by Level",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Schemes", fontsize=11)
    ax.set_ylabel("Scheme Category", fontsize=11)
    ax.legend(title="Level", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines[["top","right"]].set_visible(False)
    save_fig("06_category_by_level.png")

# ── Chart 7: combined_text length distribution ───────────────
if "combined_text" in df.columns:
    ct_lens = df["combined_text"].str.len()
    fig, ax = plt.subplots(figsize=(9, 4))
    ct_clipped = ct_lens.clip(upper=ct_lens.quantile(0.99))
    ax.hist(ct_clipped, bins=60, color=sns.color_palette("muted")[0],
            edgecolor="white", alpha=0.85)
    ax.axvline(ct_lens.median(), color="crimson", linestyle="--", linewidth=1.5,
               label=f"Median = {int(ct_lens.median()):,}")
    ax.set_title("Combined Text Length Distribution (clipped at 99th pct)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Characters", fontsize=11)
    ax.set_ylabel("Number of Schemes", fontsize=11)
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    save_fig("07_combined_text_length.png")

# ══════════════════════════════════════════════════════════════
# SECTION 5 — COLLECT STATS  (for docs and notebook)
# ══════════════════════════════════════════════════════════════
sep("5. COLLECTING STATS")

stats = {
    "raw_rows":             int(raw.shape[0]),
    "raw_cols":             int(raw.shape[1]),
    "raw_columns":          list(raw.columns),
    "clean_rows":           int(df.shape[0]),
    "clean_cols":           int(df.shape[1]),
    "clean_columns":        list(df.columns),
    "exact_dupes_removed":  int(rows_removed_exact),
    "slug_dupes_removed":   int(rows_removed_slug),
    "nulls_after_clean":    int(df.isnull().sum().sum()),
    "raw_missing": {str(k): int(v)
                    for k, v in raw.isnull().sum().items() if v > 0},
}

if "level" in df.columns:
    stats["level_dist"] = {k: int(v) for k, v in df["level"].value_counts().items()}

if "schemeCategory" in df.columns:
    stats["top20_categories"] = {
        k: int(v) for k, v in df["schemeCategory"].value_counts().head(20).items()
    }

# text length summaries
stats["text_length_stats"] = {}
for col in text_cols:
    if col in df.columns:
        s = df[col].astype(str)
        s = s[s.str.strip() != ""].str.len()
        if len(s):
            stats["text_length_stats"][col] = {
                "count":  int(len(s)),
                "min":    int(s.min()),
                "median": int(s.median()),
                "mean":   int(s.mean()),
                "max":    int(s.max()),
            }

if "combined_text" in df.columns:
    ct = df["combined_text"].str.len()
    stats["combined_text_stats"] = {
        "min": int(ct.min()), "median": int(ct.median()),
        "mean": int(ct.mean()), "max": int(ct.max()),
    }

with open(STATS_PATH, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
print(f"Stats saved → {STATS_PATH}")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
sep("PHASE 2 COMPLETE — SUMMARY")
print(f"Raw dataset        : {stats['raw_rows']:,} rows × {stats['raw_cols']} cols")
print(f"Clean dataset      : {stats['clean_rows']:,} rows × {stats['clean_cols']} cols")
print(f"Rows removed       : {stats['exact_dupes_removed'] + stats['slug_dupes_removed']} "
      f"({stats['exact_dupes_removed']} exact + {stats['slug_dupes_removed']} slug dupes)")
print(f"Nulls remaining    : {stats['nulls_after_clean']}")
print(f"Output CSV         : {OUT_PATH}")
print(f"Charts             : {CHARTS_DIR}")
print(f"Stats JSON         : {STATS_PATH}")
print(f"\nAll charts generated:")
for f in sorted(os.listdir(CHARTS_DIR)):
    if f.endswith(".png"):
        print(f"  {f}")
