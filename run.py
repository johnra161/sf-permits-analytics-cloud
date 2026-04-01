"""
run.py — SF Building Permits: full pipeline + dashboard launcher
================================================================
What this script does:
  1. Checks if the BigQuery tables already exist and have data
     → If YES: skips everything and opens the Streamlit dashboard
     → If NO:  runs the full pipeline first, then opens the dashboard

Pipeline steps (only when data is missing):
  - Downloads the SF Building Permits dataset from Kaggle (in-memory)
  - Cleans both CSVs  (permits + contacts)
  - Uploads cleaned data to BigQuery
  - Opens the Streamlit Community Cloud dashboard in your browser

Requirements:
  pip install pandas requests google-cloud-bigquery google-cloud-bigquery-storage pyarrow db-dtypes

Environment variables (set these before running):
  KAGGLE_USERNAME      your Kaggle username
  KAGGLE_KEY           your Kaggle API key
  GCP_PROJECT          GCP project ID          (e.g. logical-carver-489015-h1)
  BQ_DATASET           BigQuery dataset name   (e.g. sf_permits)
  GOOGLE_APPLICATION_CREDENTIALS   path to your GCP service account JSON key
  STREAMLIT_APP_URL    your Streamlit Community Cloud URL

  Optional — only needed if you want to re-run even when data exists:
  FORCE_RELOAD         set to "true" to always re-run the full pipeline
"""

import ast
import io
import os
import re
import sys
import webbrowser
import zipfile

import pandas as pd
import requests
from google.cloud import bigquery


# ─────────────────────────────────────────────
# CONFIG  (all from environment variables)
# ─────────────────────────────────────────────

KAGGLE_USERNAME   = os.environ.get("KAGGLE_USERNAME")
KAGGLE_KEY        = os.environ.get("KAGGLE_KEY")
GCP_PROJECT       = os.environ.get("GCP_PROJECT")
BQ_DATASET        = os.environ.get("BQ_DATASET")
STREAMLIT_APP_URL = os.environ.get("STREAMLIT_APP_URL")
FORCE_RELOAD      = os.environ.get("FORCE_RELOAD", "false").lower() == "true"

KAGGLE_DATASET    = "san-francisco/sf-building-permits-and-contacts"
PERMITS_CSV       = "building-permits.csv"
CONTACTS_CSV      = "building-permits-contacts.csv"
BQ_PERMITS_TABLE  = "building_permits"
BQ_CONTACTS_TABLE = "building_permits_contacts"


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_env():
    """Fail fast with a clear message if any required variable is missing."""
    required = {
        "KAGGLE_USERNAME":              KAGGLE_USERNAME,
        "KAGGLE_KEY":                   KAGGLE_KEY,
        "GCP_PROJECT":                  GCP_PROJECT,
        "BQ_DATASET":                   BQ_DATASET,
        "STREAMLIT_APP_URL":            STREAMLIT_APP_URL,
        "GOOGLE_APPLICATION_CREDENTIALS": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        print("\n❌  Missing required environment variables:")
        for var in missing:
            print(f"    {var}")
        print("\nSet them in your shell before running:")
        print("  export KAGGLE_USERNAME=your_username")
        print("  export KAGGLE_KEY=your_api_key")
        print("  export GCP_PROJECT=logical-carver-489015-h1")
        print("  export BQ_DATASET=sf_permits")
        print("  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json")
        print("  export STREAMLIT_APP_URL=https://your-app.streamlit.app")
        sys.exit(1)


# ─────────────────────────────────────────────
# BIGQUERY — CHECK IF DATA EXISTS
# ─────────────────────────────────────────────

def tables_exist(bq: bigquery.Client) -> bool:
    """Return True only if both tables exist AND contain at least one row."""
    for table_id in (BQ_PERMITS_TABLE, BQ_CONTACTS_TABLE):
        full = f"{GCP_PROJECT}.{BQ_DATASET}.{table_id}"
        try:
            result = bq.query(f"SELECT COUNT(*) AS n FROM `{full}`").result()
            count = next(iter(result))["n"]
            if count == 0:
                print(f"  Table {full} exists but is empty — will reload.")
                return False
        except Exception:
            print(f"  Table {full} not found — will run pipeline.")
            return False
    return True


# ─────────────────────────────────────────────
# KAGGLE DOWNLOAD  (fully in-memory)
# ─────────────────────────────────────────────

def download_kaggle_zip() -> zipfile.ZipFile:
    print(f"  Downloading dataset from Kaggle: {KAGGLE_DATASET}")
    url  = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
    resp = requests.get(url, auth=(KAGGLE_USERNAME, KAGGLE_KEY), timeout=300)
    resp.raise_for_status()
    mb = len(resp.content) / 1_048_576
    print(f"  Downloaded {mb:.1f} MB")
    return zipfile.ZipFile(io.BytesIO(resp.content))


def read_csv_from_zip(zf: zipfile.ZipFile, filename: str) -> pd.DataFrame:
    matches = [n for n in zf.namelist() if n.endswith(filename)]
    if not matches:
        raise FileNotFoundError(
            f"'{filename}' not found in zip. Contents: {zf.namelist()}"
        )
    with zf.open(matches[0]) as f:
        return pd.read_csv(f, dtype=str, low_memory=False)


# ─────────────────────────────────────────────
# PII SCRUBBING
# ─────────────────────────────────────────────

_EMAIL_RE = re.compile(r'\S+@\S+\.\S+')
_PHONE_RE = re.compile(r'\+?\d[\d\s\-\(\)]{7,}\d')
_NAME_RE  = re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b')


def scrub_text(text):
    if pd.isna(text):
        return text
    text = str(text)
    text = _EMAIL_RE.sub("[REDACTED]", text)
    text = _PHONE_RE.sub("[REDACTED]", text)
    text = _NAME_RE.sub("[REDACTED]",  text)
    return text


# ─────────────────────────────────────────────
# LOCATION PARSING
# ─────────────────────────────────────────────

def extract_coordinates(loc):
    if pd.isna(loc):
        return pd.Series([None, None])
    try:
        d = ast.literal_eval(str(loc))
        return pd.Series([d.get("latitude"), d.get("longitude")])
    except Exception:
        return pd.Series([None, None])


# ─────────────────────────────────────────────
# CLEAN PERMITS
# ─────────────────────────────────────────────

PERMITS_KEEP = [
    "Permit Number", "Permit Type", "Permit Type Definition",
    "Permit Creation Date", "Block", "Lot", "Street Number",
    "Street Name", "Street Suffix", "Description", "Current Status",
    "Current Status Date", "Filed Date", "Issued Date", "Completed Date",
    "Permit Expiration Date", "Estimated Cost", "Revised Cost",
    "Existing Use", "Proposed Use", "Existing Units", "Proposed Units",
    "Number of Existing Stories", "Number of Proposed Stories",
    "Existing Construction Type", "Existing Construction Type Description",
    "Proposed Construction Type", "Proposed Construction Type Description",
    "Plansets", "Supervisor District", "Neighborhoods - Analysis Boundaries",
    "Zipcode", "Latitude", "Longitude",
]

PERMITS_NUMERIC = [
    "Estimated Cost", "Revised Cost", "Existing Units", "Proposed Units",
    "Number of Existing Stories", "Number of Proposed Stories",
]


def clean_permits(df: pd.DataFrame) -> pd.DataFrame:
    if "Location" in df.columns:
        coords = df["Location"].apply(extract_coordinates)
        df = df.copy()
        df["Latitude"]  = coords[0]
        df["Longitude"] = coords[1]
        df = df.drop(columns=["Location"])

    df = df[[c for c in PERMITS_KEEP if c in df.columns]].copy()

    if "Description" in df.columns:
        df["Description"] = df["Description"].apply(scrub_text)

    before = len(df)
    df = df.drop_duplicates()
    print(f"    Removed {before - len(df):,} duplicate rows")

    for col in PERMITS_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("Latitude", "Longitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# CLEAN CONTACTS
# ─────────────────────────────────────────────

CONTACTS_DROP = [
    "ID", "Agent Address", "Agent Address2", "City", "State",
    "Agent Zipcode", "To Date", "License2",
    "First Name", "Last Name",
]

CONTACTS_FILL = [
    "Firm Name", "Firm Address", "Firm City",
    "Firm State", "Firm Zipcode", "License1",
]


def normalize_zip(z):
    return "Unknown" if pd.isna(z) else str(z).split("-")[0].strip()


def clean_contacts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in CONTACTS_DROP if c in df.columns])

    for col in CONTACTS_FILL:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    if "Firm Zipcode" in df.columns:
        df["Firm Zipcode"] = df["Firm Zipcode"].apply(normalize_zip)

    before = len(df)
    df = df.drop_duplicates()
    print(f"    Removed {before - len(df):,} duplicate rows")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# BIGQUERY UPLOAD
# ─────────────────────────────────────────────

def upload_to_bigquery(bq: bigquery.Client, df: pd.DataFrame, table_id: str):
    full = f"{GCP_PROJECT}.{BQ_DATASET}.{table_id}"
    print(f"  Uploading {len(df):,} rows → {full}")
    job = bq.load_table_from_dataframe(
        df, full,
        job_config=bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            autodetect=True,
        ),
    )
    job.result()
    print(f"  ✓ {full}")


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(bq: bigquery.Client):
    # Ensure BQ dataset exists
    ds = bigquery.Dataset(f"{GCP_PROJECT}.{BQ_DATASET}")
    ds.location = "US"
    bq.create_dataset(ds, exists_ok=True)

    # Download
    print("\n── Step 1 of 3: Downloading from Kaggle ───────────────────")
    zf = download_kaggle_zip()

    print("\n── Step 2 of 3: Cleaning data ──────────────────────────────")
    print("  Permits:")
    permits_raw  = read_csv_from_zip(zf, PERMITS_CSV)
    print(f"    Raw rows: {len(permits_raw):,}")
    permits_clean = clean_permits(permits_raw)
    print(f"    Clean rows: {len(permits_clean):,}  |  {len(permits_clean.columns)} columns")

    print("  Contacts:")
    contacts_raw  = read_csv_from_zip(zf, CONTACTS_CSV)
    print(f"    Raw rows: {len(contacts_raw):,}")
    contacts_clean = clean_contacts(contacts_raw)
    print(f"    Clean rows: {len(contacts_clean):,}  |  {len(contacts_clean.columns)} columns")

    print("\n── Step 3 of 3: Uploading to BigQuery ──────────────────────")
    upload_to_bigquery(bq, permits_clean,  BQ_PERMITS_TABLE)
    upload_to_bigquery(bq, contacts_clean, BQ_CONTACTS_TABLE)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SF Building Permits — Pipeline & Dashboard")
    print("=" * 60)

    validate_env()

    bq = bigquery.Client(project=GCP_PROJECT)

    if not FORCE_RELOAD and tables_exist(bq):
        print("\n✅  BigQuery tables already exist with data.")
        print("    Skipping pipeline — opening dashboard directly.\n")
    else:
        if FORCE_RELOAD:
            print("\n🔄  FORCE_RELOAD=true — re-running full pipeline.\n")
        else:
            print("\n⚙️   Data not found — running full pipeline.\n")

        run_pipeline(bq)

        print("\n✅  Pipeline complete. Data is in BigQuery.")

    print(f"\n🚀  Opening dashboard: {STREAMLIT_APP_URL}")
    webbrowser.open(STREAMLIT_APP_URL)
    print("\nDone.")


if __name__ == "__main__":
    main()
