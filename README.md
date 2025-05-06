# Data Processing Pipeline Documentation

This document explains the complete data processing pipeline divided into two main phases: **Data Cleaning** and **Feature Engineering**.

## Phase 1: Data Cleaning (data_cleaning.ipynb)

The data cleaning notebook transforms raw datasets into a clean, structured format ready for analysis.

### Setup and Initialization
```python
import pandas as pd, numpy as np, re, ast, logging
from typing import List
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("clean")
RAW_PATH   = "dataset.csv"
CLEAN_PATH = "cleaned-dataset.csv"
```
This section imports necessary libraries and configures logging to track progress. It defines input/output file paths.

### Step 1: Load Raw Dataset
```python
df = pd.read_csv(RAW_PATH)
display(df.head(3)); display(df.info())
```
Loads the raw dataset from CSV format and displays the first 3 rows plus information about the dataframe structure, allowing for initial inspection.

### Step 2: Column Hygiene
```python
df.columns = (df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_"))
df = df.rename(columns={"plot_kyeword": "plot_keyword",
                        "generes": "genres"})
```
Standardizes column names by:
- Stripping whitespace
- Converting to lowercase
- Converting spaces to underscores
- Fixing specific typos in column names ("plot_kyeword" → "plot_keyword" and "generes" → "genres")

### Step 3: Numeric Parsing
```python
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
def parse_votes(v):
    if pd.isna(v): return None
    if isinstance(v, (int, float)): return int(v)
    v = (str(v).strip().upper()
                    .replace(",", "")
                    .replace("M", "000000")
                    .replace("K", "000"))
    try: return int(float(v))
    except ValueError: return None
df["votes"] = df["user_rating"].apply(parse_votes)
```
Handles numerical conversions for ratings and votes:
- Converts ratings to proper numeric format using pandas
- Creates a custom function to parse vote counts that handles special formats:
  - Converts "K" to thousands (e.g., "10K" → 10,000)
  - Converts "M" to millions (e.g., "2.5M" → 2,500,000)
  - Handles comma-separated numbers

### Step 4: Runtime & Budget Extraction
```python
run_re = re.compile(r"(?:(\d+)\s*hours?)?\s*(?:(\d+)\s*minutes?)?", re.I)
def runtime_min(txt):
    if pd.isna(txt): return None
    m = run_re.search(str(txt))
    if m and (m.group(1) or m.group(2)):
        h = int(m.group(1) or 0); m_ = int(m.group(2) or 0)
        return h*60 + m_
    try: return int(float(str(txt).replace(",", "")))
    except ValueError: return None
def budget_usd(txt):
    if pd.isna(txt): return None
    m = re.search(r"\$([\d,]+)", str(txt))
    return int(m.group(1).replace(",", "")) if m else None
df["runtime_min"] = df["run_time"].apply(runtime_min)
df["budget_usd"]  = df["run_time"].apply(budget_usd)
```
Extracts structured data from text fields:
- Creates regex patterns to find runtime information in "hour/minute" format
- Converts runtime to total minutes (e.g., "2 hours 30 minutes" → 150 minutes)
- Extracts budget amounts from text by finding "$" followed by numbers
- Creates standardized numeric columns for both runtime and budget

### Step 5: Fix Year and Convert Stringified Lists
```python
df["year"] = (pd.to_numeric(df["year"].astype(str)
                                       .str.replace(r"[^0-9]", "", regex=True),
                          errors="coerce")
             .astype("Int64"))
def safe_eval(x): 
    try: return ast.literal_eval(x) if pd.notna(x) else []
    except Exception: return []
for col in ["genres", "plot_keyword", "top_5_casts"]:
    df[col] = df[col].apply(safe_eval)
```
Handles data type conversions for complex fields:
- Cleans and standardizes the year column, removing non-numeric characters
- Safely converts string representations of lists (like "[action, drama]") into actual Python lists
- Applies this conversion to genres, plot keywords, and cast lists
- Uses a safe evaluation approach that handles errors gracefully

### Step 6: Drop Dirty Columns & De-duplicate
```python
df = df.drop(columns={"run_time", "overview", "path"} & set(df.columns))
df = df.drop_duplicates(subset=["movie_title", "year"], keep="first")
log.info("Rows after cleaning: %d", len(df))
```
Finalizes the dataset by:
- Removing unnecessary columns (but only if they exist in the dataset)
- Removing duplicate movies based on title and year
- Keeping the first occurrence of each movie
- Logging the final row count after deduplication

### Step 7: Save Cleaned Dataset
```python
df.to_csv(CLEAN_PATH, index=False)
log.info("Saved %s", CLEAN_PATH)
```
Exports the cleaned dataset to a CSV file without including row indices, creating a file ready for the feature engineering phase.

## Phase 2: Feature Engineering (feature_engineering.ipynb)

The feature engineering notebook transforms clean data into enriched features for analysis and modeling.

### Setup and Initialization
```python
import pandas as pd, numpy as np, itertools, logging, math, ast
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.stem import WordNetLemmatizer
import nltk; nltk.download("wordnet", quiet=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log  = logging.getLogger("feature")
lemm = WordNetLemmatizer()
CLEAN_PATH   = "cleaned-dataset.csv"
FEATURE_PATH = "featured-dataset.csv"
TOP_RATING   = 7.5
MIN_VOTES    = 10_000
RARE_KW_MAX  = 3
TOP_ACTORS_N = 100
TOP_DIRS_N   = 100
```
Imports specialized libraries for feature engineering and sets constants for filtering high-quality content and managing feature generation.

### Step 1: Load Cleaned Dataset
```python
df = pd.read_csv(CLEAN_PATH)
log.info("Loaded %d clean rows", len(df))
```
Loads the output from the previous cleaning phase and logs the row count.

### Step 2: High-Quality Subset Creation
```python
hq = df[(df["rating"] >= TOP_RATING) & (df["votes"] >= MIN_VOTES)].copy()
hq.reset_index(drop=True, inplace=True)
log.info("High‑quality subset: %d rows", len(hq))
```
Creates a subset with only high-quality entries:
- Filters for movies with ratings ≥ 7.5
- Requires at least 10,000 votes to ensure statistical significance
- Resets the index for clean sequencing
- Logs the size of the high-quality subset

### Step 3: One-Hot Encode Genres
```python
mlb = MultiLabelBinarizer()
ohe = mlb.fit_transform(hq["genres"])
genre_cols = [f"genre_{g.replace(' ','_').lower()}" for g in mlb.classes_]
hq[genre_cols] = ohe
hq["genre_count"] = hq["genres"].str.len()
```
Transforms genre lists into machine-learning friendly format:
- Uses MultiLabelBinarizer to convert genre lists to one-hot encoded columns
- Creates standardized column names for each genre (e.g., "genre_action")
- Adds a count of genres per movie for quick filtering by genre diversity

### Step 4: Clean & Prune Plot Keywords
```python
hq["plot_keyword"] = hq["plot_keyword"].apply(
    lambda lst: sorted({lemm.lemmatize(str(k).lower().strip()) for k in lst})
)
kw_freq = Counter(itertools.chain.from_iterable(hq["plot_keyword"]))
rare_kw = {k for k,c in kw_freq.items() if c <= RARE_KW_MAX}
hq["plot_keyword"] = hq["plot_keyword"].apply(
    lambda ks: [k for k in ks if k not in rare_kw]
)
hq["kw_count"] = hq["plot_keyword"].str.len()
```
Optimizes keyword features:
- Standardizes keywords through lemmatization (converting words to their root form)
- Counts keyword frequency across the entire dataset
- Identifies and removes rare keywords (appearing ≤ 3 times)
- Adds a count of keywords per movie after filtering

### Step 5: Actor & Director Indicator Columns
```python

norm_ascii  = lambda s: "".join(
    c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
)                       # strip accents → “René” → “Rene”
clean_token = lambda s: re.sub(r"[^a-z0-9_]", "", s.lower())  # drop punctuation

# Track used column names to avoid collisions
used_cols = set(hq.columns)

# ---------------------------  Actors  -----------------------------
actor_freq = Counter(itertools.chain.from_iterable(hq["top_5_casts"]))
top_actors = [a for a, _ in actor_freq.most_common(TOP_ACTORS_N)]

for actor in top_actors:
    if not isinstance(actor, str) or not actor.strip():
        continue  # skip NaN / empty
    
    # Use last token of name → “Tom Cruise” → “cruise”
    last = clean_token(norm_ascii(actor).split()[-1])
    if not last:
        continue
    col = f"actor_{last}"

    # Ensure uniqueness if two actors share a last name
    suffix = 1
    while col in used_cols:
        col = f"{col}_{suffix}"
        suffix += 1
    used_cols.add(col)
    
    # Vectorised membership mask, stored as int8 to save memory
    hq[col] = hq["top_5_casts"].apply(lambda lst, a=actor: int(a in lst)).astype("int8")

# --------------------------  Directors ----------------------------
top_dirs = hq["director"].value_counts().head(TOP_DIRS_N).index

for director in top_dirs:
    if not isinstance(director, str) or not director.strip():
        continue
    
    last = clean_token(norm_ascii(director).split()[-1])
    if not last:
        continue
    col = f"director_{last}"

    suffix = 1
    while col in used_cols:
        col = f"{col}_{suffix}"
        suffix += 1
    used_cols.add(col)
    
    # Binary flag: 1 if this row’s director matches
    hq[col] = (hq["director"] == director).astype("int8")

print("Added",
      sum(c.startswith("actor_")    for c in used_cols),    "actor columns &",
      sum(c.startswith("director_") for c in used_cols), "director columns")
```
Creates robust binary flags for top on‑screen and behind‑camera talent:

- Finds the 100 most‑frequent actors (top_5_casts) across the high‑quality subset.
- Generates a binary column for each actor (1 = actor appears in that movie, otherwise 0).
- Finds the 100 most‑frequent directors and creates matching binary columns.
Column‑naming rules:
- Uses the cleaned last token of the name for readability (e.g. actor_cruise, director_nolan).
- Strips accents and punctuation (René Zellweger → actor_zellweger).
- If two people share the same last name, appends _1, _2, … to keep names unique (actor_smith, actor_smith_1).
- All indicator columns are stored as int8 to minimise memory.

### Step 6: Popularity & Temporal Features
```python
C = hq["rating"].mean(); m = MIN_VOTES
hq["weighted_rating"] = ((hq["votes"]/(hq["votes"]+m))*hq["rating"] +
                         (m/(hq["votes"]+m))*C)
hq["log_votes"] = np.log10(hq["votes"]+1)
hq["decade"]    = (hq["year"]//10)*10
hq["runtime_bucket"] = pd.cut(
    hq["runtime_min"],
    bins=[0,90,110,140,1e9],
    labels=["<90","90‑110","110‑140",">140"]
)
```
Adds derived features useful for analysis:
- Creates a "weighted rating" that balances user ratings with popularity (similar to IMDB's Bayesian average)
- Converts vote counts to logarithmic scale to handle the wide distribution
- Groups movies by decade for temporal analysis
- Categorizes movies into runtime buckets for easy filtering (<90 min, 90-110 min, etc.)

### Step 7: Export Feature-Rich Dataset
```python
hq.to_csv(FEATURE_PATH, index=False)
log.info("Saved %s  (rows=%d, cols=%d)", FEATURE_PATH, *hq.shape)
hq.head()
```
Finalizes and exports the feature-engineered dataset:
- Saves the high-quality, feature-rich dataset to CSV
- Logs the dimensions of the final dataset (rows and columns)
- Displays the first few rows to inspect the results

## Summary

This two-phase pipeline transforms raw movie data into a highly structured format with rich features suitable for:

1. **Recommendation systems**: Using genre, actor, director, and keyword indicators
2. **Trend analysis**: Using temporal features like decade
3. **Quality filtering**: Using rating, weighted rating, and vote count
4. **Content categorization**: Using runtime buckets and genre counts

The resulting dataset enables sophisticated queries and analytics beyond what would be possible with the raw data.
