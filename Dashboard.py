import json
import re
import warnings

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pydeck as pdk
from google.cloud import bigquery
from google.oauth2 import service_account
from google import genai
from wordcloud import WordCloud, STOPWORDS

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="SF Permit Analytics",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ SF Permit Analytics Dashboard")

# ─────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────

GCP_PROJECT = st.secrets["GCP_PROJECT"]
BQ_DATASET = st.secrets["BQ_DATASET"]
MODEL = "gemini-2.5-flash"


@st.cache_resource
def get_bq_client():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return bigquery.Client(project=GCP_PROJECT, credentials=creds)


@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


bq = get_bq_client()
gemini = get_gemini_client()


# ─────────────────────────────────────────────
# VISUALIZATION DATA  (BigQuery does all work)
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading map data…")
def load_map_data():
    df = bq.query(f"""
        SELECT
            ROUND(Latitude,  3) AS lat,
            ROUND(Longitude, 3) AS lon,
            COUNT(*)            AS count
        FROM `{GCP_PROJECT}.{BQ_DATASET}.building_permits`
        WHERE Latitude  IS NOT NULL
          AND Longitude IS NOT NULL
          AND Latitude  BETWEEN 37.6 AND 37.9
          AND Longitude BETWEEN -122.6 AND -122.3
        GROUP BY lat, lon
    """).to_dataframe()
    return df


@st.cache_data(ttl=3600, show_spinner="Loading timeline data…")
def load_timeline_data():
    df = bq.query(f"""
        SELECT
            EXTRACT(YEAR FROM PARSE_DATETIME('%Y-%m-%dT%H:%M:%E3S', `Issued Date`)) AS Year,
            COUNT(*) AS permits_issued
        FROM `{GCP_PROJECT}.{BQ_DATASET}.building_permits`
        WHERE `Issued Date` IS NOT NULL
        GROUP BY Year
        ORDER BY Year
    """).to_dataframe()
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)
    return df


@st.cache_data(ttl=3600, show_spinner="Loading word frequencies…")
def load_word_frequencies():
    df = bq.query(f"""
        WITH words AS (
            SELECT LOWER(word) AS word
            FROM `{GCP_PROJECT}.{BQ_DATASET}.building_permits`
            CROSS JOIN UNNEST(SPLIT(REGEXP_REPLACE(Description, r'[^a-zA-Z ]', ' '), ' ')) AS word
            WHERE Description IS NOT NULL
            AND LENGTH(word) >= 3
        )
        SELECT word, COUNT(*) AS frequency
        FROM words
        WHERE word NOT IN (
            'the','and','for','that','this','with','from','have','not',
            'are','was','were','but','its','had','has','been','will',
            'redacted','nan','none','unknown','san','francisco','permit',
            'work','install','existing','new','replace','per','plan',
            'app','ref','use','non','type','ft','sq','fl','blvd','bldg',
            'flr','extg','one','two','single','back','full','side','cti',
            'str','int','via','any','all','our','can','may','also','into',
            'they','them','their','than','then','when','what','which','who'
        )
        AND LENGTH(word) >= 3
        GROUP BY word
        ORDER BY frequency DESC
        LIMIT 100
    """).to_dataframe()
    return df


# ─────────────────────────────────────────────
# GEMINI — NATURAL LANGUAGE TO SQL
# ─────────────────────────────────────────────

_SQL_SYSTEM = f"""
You are a data analyst assistant for the SF Building Permits dataset stored in BigQuery.

Your job is to convert a user's natural language question into a valid BigQuery SQL query.
Return ONLY the SQL query — no explanation, no markdown fences, no preamble.
If the question is off-topic or unanswerable, return exactly: INVALID

════════════════════════════════════════════════════════
TABLES & SCHEMA
════════════════════════════════════════════════════════

`{GCP_PROJECT}.{BQ_DATASET}.building_permits`
  STRING:  `Permit Number`, `Permit Type`, `Permit Type Definition`, `Permit Creation Date`,
           `Block`, `Lot`, `Street Number`, `Street Name`, `Street Suffix`, `Description`,
           `Current Status`, `Current Status Date`, `Filed Date`, `Issued Date`,
           `Completed Date`, `Permit Expiration Date`, `Existing Use`, `Proposed Use`,
           `Existing Construction Type`, `Existing Construction Type Description`,
           `Proposed Construction Type`, `Proposed Construction Type Description`,
           `Plansets`, `Supervisor District`, `Neighborhoods - Analysis Boundaries`, `Zipcode`
  FLOAT64: `Estimated Cost`, `Revised Cost`, `Existing Units`, `Proposed Units`,
           `Number of Existing Stories`, `Number of Proposed Stories`, `Latitude`, `Longitude`

`{GCP_PROJECT}.{BQ_DATASET}.building_permits_contacts`
  STRING:  `Permit Number`, `Role`, `Firm Name`, `Firm Address`, `Firm City`,
           `Firm State`, `Firm Zipcode`, `PTS Agent ID`, `From Date`, `License1`

════════════════════════════════════════════════════════
COLUMN DISAMBIGUATION  (map vague user language → exact column)
════════════════════════════════════════════════════════

── COST / VALUE ───────────────────────────────────────
"cost", "price", "value", "budget", "how much"        → `Estimated Cost`  (default)
"revised cost", "updated cost", "final cost"           → `Revised Cost`
There is NO lot size, parcel area, or square footage column.
If the user asks for area/size → return INVALID

── DATES ──────────────────────────────────────────────
"issued", "approved", "granted", "permit date"         → `Issued Date`  (DEFAULT for unqualified "date")
"filed", "submitted", "applied", "application date"    → `Filed Date`
"completed", "finished", "closed", "done date"         → `Completed Date`
"created", "opened", "started"                         → `Permit Creation Date`
"expired", "expiration"                                → `Permit Expiration Date`
"status date", "last updated"                          → `Current Status Date`
"from date", "contact date"                            → `From Date`  (contacts table only)

── STATUS ─────────────────────────────────────────────
"status", "stage", "state"  → `Current Status`
Exact values (use lowercase verbatim in WHERE):
  appeal, approved, cancelled, complete, denied, disapproved, expired,
  filed, filing, granted, incomplete, inspection, issued, issuing,
  plancheck, reinstated, revoked, suspend, unknown, upheld, withdrawn
Common phrasing → exact value:
  "active" / "in progress"   → issued
  "done" / "finished"        → complete
  "pending" / "under review" → plancheck
  "rejected"                 → denied
  "lapsed"                   → expired

── LOCATION / GEOGRAPHY ───────────────────────────────
"neighbourhood", "neighborhood", "area", "district", "part of the city"
  → `Neighborhoods - Analysis Boundaries`
"supervisor district", "supervisorial district"
  → `Supervisor District`  (values are numeric strings: "1" through "11")
"zip", "zipcode", "postal code"
  → `Zipcode`
"address", "street"
  → `Street Number`, `Street Name`, `Street Suffix`

── PERMIT TYPE ────────────────────────────────────────
"type of permit", "permit category", "kind of permit"
  → `Permit Type Definition`  (ALWAYS prefer over `Permit Type`)
Exact values (lowercase, use verbatim in WHERE):
  additions alterations or repairs
  demolitions
  grade or quarry or fill or excavate
  new construction
  new construction wood frame
  otc alterations permit
  sign - erect
  wall or painted sign
`Permit Type` is a numeric code string — use only if user explicitly asks for the code.
`Permit Number` is a unique ID — use only for lookup queries, never aggregate on it.

── CONSTRUCTION ───────────────────────────────────────
"construction type", "building material"
  → `Existing Construction Type Description` or `Proposed Construction Type Description`
Exact values: constr type 1, constr type 2, constr type 3, constr type 4, wood frame (5)
"use", "purpose", "occupancy", "what it's used for"
  → `Existing Use` or `Proposed Use`  (free-text; use LIKE for filtering)
"stories", "floors", "height"
  → `Number of Existing Stories` or `Number of Proposed Stories`
"units", "apartments", "dwellings", "homes"
  → `Existing Units` or `Proposed Units`

── PARCEL IDENTITY ────────────────────────────────────
"lot", "parcel"  → `Lot`   (string ID, NOT a size — no area data exists)
"block"          → `Block` (string ID)
Do not compute AVG or SUM on these — they are identifiers.

── CONTACTS TABLE ─────────────────────────────────────
"firm", "company", "contractor", "agency"  → `Firm Name`
"role", "capacity", "who filed"            → `Role`
Exact role values (lowercase):
  architect, attorney, authorized agent-others, contractor, designer,
  engineer, lessee, pmt consultant/expediter, project contact
"license", "credential"   → `License1`
"agent ID"                → `PTS Agent ID`

PRIVACY: The following columns do NOT exist — removed before upload:
  First Name, Last Name, Contact Name, Agent Address, ID
If the user asks for individual names → return INVALID

════════════════════════════════════════════════════════
DATE HANDLING
════════════════════════════════════════════════════════

All date columns in building_permits are STRING stored as: "2005-10-04T00:00:00.000"
All date columns in building_permits_contacts are STRING stored as: "2008-03-19T00:00:00.000Z"

Always use the SAFE prefix to handle NULL or malformed date strings without crashing.

Parsing permits dates:
  SAFE.PARSE_DATETIME('%Y-%m-%dT%H:%M:%E3S', `Issued Date`)

Parsing contacts `From Date`:
  SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E3SZ', `From Date`)

Extracting year from a permits date:
  EXTRACT(YEAR FROM SAFE.PARSE_DATETIME('%Y-%m-%dT%H:%M:%E3S', `Issued Date`))

Filtering by year:
  EXTRACT(YEAR FROM SAFE.PARSE_DATETIME('%Y-%m-%dT%H:%M:%E3S', `Issued Date`)) = 2023

Filtering by date range:
  SAFE.PARSE_DATETIME('%Y-%m-%dT%H:%M:%E3S', `Issued Date`) BETWEEN
    DATETIME '2023-01-01' AND DATETIME '2023-12-31'

Relative date language → resolve to concrete filters on `Issued Date`:
  "this year"        → year = 2025
  "last year"        → year = 2024
  "recent"/"lately"  → year >= 2023
  "last N years"     → year >= (2025 - N)
  "last decade"      → year >= 2015
  Never leave a date filter empty when the user implies recency.

════════════════════════════════════════════════════════
NUMERIC COLUMN RULES
════════════════════════════════════════════════════════

`Estimated Cost`, `Revised Cost`, `Existing Units`, `Proposed Units`,
`Number of Existing Stories`, `Number of Proposed Stories`, `Latitude`, `Longitude`
are all stored as FLOAT64 — never cast them, they are already numeric.

When computing AVG or SUM on cost columns, always exclude zeros and NULLs:
  WHERE `Estimated Cost` > 0
  (add as an additional AND condition if other filters already exist)

For COUNT queries, use COUNT(`Permit Number`) not COUNT(*) to avoid counting null rows.

════════════════════════════════════════════════════════
SQL CONSTRUCTION RULES
════════════════════════════════════════════════════════

1.  Always backtick-quote every column name — many contain spaces and special characters.
2.  Always backtick-quote table names.
3.  String comparisons: always wrap both sides in LOWER() for case-insensitivity.
    Example: LOWER(`Current Status`) = 'complete'
4.  Substring matches: use LOWER(`Description`) LIKE LOWER('%keyword%')
5.  Aggregation queries (COUNT, AVG, SUM, MIN, MAX): no LIMIT clause.
6.  Row-returning queries (SELECT without aggregation): default LIMIT 10.
7.  "Top N" / "bottom N" questions: use ORDER BY + LIMIT N.
8.  Never GROUP BY high-cardinality free-text columns: `Description`, `Firm Name`,
    `Street Name`, `Permit Number`. These produce useless results.
9.  Cross-dataset questions (e.g. "which firms filed the most permits in Mission"):
    These require a JOIN. Write the JOIN using `Permit Number` as the key:
    JOIN `{GCP_PROJECT}.{BQ_DATASET}.building_permits_contacts` USING (`Permit Number`)
10. Scalar aggregation (single number, no GROUP BY): no ORDER BY, no LIMIT.
11. Never fabricate column names, table names, or values not listed in this prompt.
12. Empty string check: some STRING columns may contain "" instead of NULL.
    Use: col IS NOT NULL AND col != '' when filtering for present values.
13. For "how many per year" questions, always ORDER BY year ASC.
14. For "top/most/highest" questions, always ORDER BY metric DESC.
15. For "least/lowest/fewest" questions, always ORDER BY metric ASC.

════════════════════════════════════════════════════════
EDGE CASES
════════════════════════════════════════════════════════

"show me some permits" / "what's in the data" / vague browse requests
  → SELECT * FROM `{GCP_PROJECT}.{BQ_DATASET}.building_permits` LIMIT 10

"how many permits total"
  → SELECT COUNT(`Permit Number`) FROM `{GCP_PROJECT}.{BQ_DATASET}.building_permits`

Questions about individual people's names
  → return INVALID  (names were removed for privacy)

Questions about lot size, parcel area, square footage
  → return INVALID  (column does not exist; suggest asking about cost or units instead)

Questions requiring joining permits + contacts
  → write the JOIN using `Permit Number` as the key (rule 9 above)

Questions about "average cost" with no other filters
  → always add WHERE `Estimated Cost` > 0

Questions about construction activity by neighbourhood
  → GROUP BY `Neighborhoods - Analysis Boundaries`

Questions about trends over time
  → GROUP BY EXTRACT(YEAR FROM SAFE.PARSE_DATETIME('%Y-%m-%dT%H:%M:%E3S', `Issued Date`))
     ORDER BY year ASC
"""


def generate_sql(question: str) -> str:
    prompt = _SQL_SYSTEM + f"\n\nUser question: {question}"
    try:
        resp = gemini.models.generate_content(model=MODEL, contents=prompt)
        sql = resp.text.strip()
        sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"^```\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql)
        return sql.strip()
    except Exception as e:
        return f"ERROR: {e}"


def run_sql(sql: str) -> pd.DataFrame:
    return bq.query(sql).to_dataframe()


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2 = st.tabs(["💬 AI Chat", "📊 Visualizations"])

# ─────────────────────────────────────────────
# TAB 1 — AI CHAT
# ─────────────────────────────────────────────

with tab1:
    st.subheader("Ask Questions About the Data")
    st.caption(
        "Examples: *What is the average permit cost in the Mission district?* · "
        "*How many permits were issued per year?* · "
        "*Which neighbourhood has the most completed permits?*"
    )

    question = st.text_input(
        "Your question:",
        placeholder="Ask anything about SF building permits…"
    )

    if question:
        with st.spinner("Generating SQL…"):
            sql = generate_sql(question)

        if sql.startswith("INVALID"):
            st.warning(
                "The AI couldn't map that question to the dataset.\n\n"
                + (sql.replace("INVALID", "").strip() or
                   "Try rephrasing — e.g. *average Estimated Cost by neighbourhood*.")
            )
        elif sql.startswith("ERROR"):
            st.error(sql)
        else:
            col_q, col_r = st.columns([1, 2])

            with col_q:
                st.subheader("Generated SQL")
                st.code(sql, language="sql")

            with col_r:
                try:
                    result = run_sql(sql)
                    st.subheader("Result")
                    st.dataframe(result, use_container_width=True)

                    # Auto-chart for two-column label+value results
                    if (
                            isinstance(result, pd.DataFrame)
                            and result.shape[1] == 2
                            and result.shape[0] > 1
                            and pd.api.types.is_numeric_dtype(result.iloc[:, 1])
                    ):
                        st.subheader("Chart")
                        st.bar_chart(result.set_index(result.columns[0])[result.columns[1]])

                except Exception as e:
                    st.error(f"Query error: {e}")

# ─────────────────────────────────────────────
# TAB 2 — VISUALIZATIONS
# ─────────────────────────────────────────────

with tab2:
    # ── 1. HEATMAP ──────────────────────────────

    st.subheader("📍 Permit Location Heatmap")
    st.markdown(
        """
        This heatmap shows where building permit activity is concentrated across San Francisco.
        Brighter, more saturated areas indicate higher permit density — useful for identifying
        construction hotspots by neighbourhood or corridor. Zoom and pan to explore.
        """
    )

    map_df = load_map_data()

    if map_df.empty:
        st.warning("No valid coordinates found in the dataset.")
    else:
        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=map_df,
            get_position=["lon", "lat"],
            get_weight="count",
            aggregation="SUM",
            color_range=[
                [0, 0, 255, 0],
                [0, 128, 255, 100],
                [0, 255, 200, 160],
                [255, 255, 0, 210],
                [255, 128, 0, 235],
                [255, 0, 0, 255],
            ],
            opacity=0.85,
            radius_pixels=60,
            intensity=2,
            threshold=0.01,
        )

        view = pdk.ViewState(
            latitude=map_df["lat"].median(),
            longitude=map_df["lon"].median(),
            zoom=11,
            pitch=0,
        )

        st.pydeck_chart(pdk.Deck(
            layers=[heatmap_layer],
            initial_view_state=view,
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        ))

    st.divider()

    # ── 2. PERMITS OVER TIME ─────────────────────

    st.subheader("📈 Permits Issued Over Time")
    st.markdown(
        """
        Tracks the volume of building permits issued each year across San Francisco.
        Use the year filter below to zoom into a specific period — the default shows
        the most recent two years for a quick snapshot of recent activity.
        """
    )

    timeline_df = load_timeline_data()

    if timeline_df.empty:
        st.warning("No valid dates found in the dataset.")
    else:
        min_year = int(timeline_df["Year"].min())
        max_year = int(timeline_df["Year"].max())
        default_start = max(min_year, max_year - 1)

        year_range = st.slider(
            "Select year range:",
            min_value=min_year,
            max_value=max_year,
            value=(default_start, max_year),
            step=1,
        )

        filtered = timeline_df[timeline_df["Year"].between(year_range[0], year_range[1])]
        st.line_chart(filtered.set_index("Year")["permits_issued"])
        st.caption(
            f"Showing {year_range[0]}–{year_range[1]}. "
            f"Total permits in this range: {filtered['permits_issued'].sum():,}"
        )

    st.divider()

    # ── 3. DESCRIPTION WORD CLOUD ────────────────

    st.subheader("☁️ Description Word Cloud")
    st.markdown(
        """
        The most frequently used words in permit Description fields.
        Larger words appear more often across all descriptions — helpful for spotting
        common project types (e.g., *plumbing*, *electrical*, *renovation*).
        Common English words and privacy-redaction markers are excluded.
        """
    )

    word_df = load_word_frequencies()

    if word_df.empty:
        st.warning("No descriptions found in the dataset.")
    else:
        freq_dict = dict(zip(word_df["word"], word_df["frequency"]))


        def multicolor_func(word, font_size, position, orientation, random_state=None, **kwargs):
            import random
            colors = [
                "hsl(210, 80%, 35%)",
                "hsl(200, 75%, 45%)",
                "hsl(185, 70%, 42%)",
                "hsl(160, 60%, 40%)",
                "hsl(25,  80%, 50%)",
                "hsl(15,  85%, 48%)",
                "hsl(45,  80%, 48%)",
            ]
            rng = random_state or random.Random()
            return colors[rng.randint(0, len(colors) - 1)]


        wc = WordCloud(
            width=1000,
            height=450,
            background_color="white",
            max_words=100,
            collocations=False,
            min_word_length=3,
            color_func=multicolor_func,
            prefer_horizontal=0.85,
        ).generate_from_frequencies(freq_dict)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)