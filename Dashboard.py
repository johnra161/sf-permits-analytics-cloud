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
BQ_DATASET  = st.secrets["BQ_DATASET"]
MODEL       = "gemini-2.5-flash"
 
 
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
 
 
bq     = get_bq_client()
gemini = get_gemini_client()
 
# ─────────────────────────────────────────────
# VISUALIZATION DATA  (only columns needed)
# ─────────────────────────────────────────────
 
@st.cache_data(ttl=3600, show_spinner="Loading map data…")
def load_map_data():
    df = bq.query(f"""
        SELECT Latitude, Longitude
        FROM `{GCP_PROJECT}.{BQ_DATASET}.building_permits`
        WHERE Latitude IS NOT NULL AND Longitude IS NOT NULL
    """).to_dataframe()
    df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df.dropna().rename(columns={"Latitude": "lat", "Longitude": "lon"})
 
 
@st.cache_data(ttl=3600, show_spinner="Loading timeline data…")
def load_timeline_data():
    df = bq.query(f"""
        SELECT `Issued Date`
        FROM `{GCP_PROJECT}.{BQ_DATASET}.building_permits`
        WHERE `Issued Date` IS NOT NULL
    """).to_dataframe()
    df["Issued Date"] = pd.to_datetime(df["Issued Date"], errors="coerce")
    df = df.dropna()
    df["Year"] = df["Issued Date"].dt.year.astype(int)
    return df
 
 
@st.cache_data(ttl=3600, show_spinner="Loading description data…")
def load_description_data():
    df = bq.query(f"""
        SELECT Description
        FROM `{GCP_PROJECT}.{BQ_DATASET}.building_permits`
        WHERE Description IS NOT NULL
    """).to_dataframe()
    return df
 
 
# ─────────────────────────────────────────────
# GEMINI — NATURAL LANGUAGE TO SQL
# ─────────────────────────────────────────────
 
_SQL_SYSTEM = f"""
You are a data analyst assistant for the SF Building Permits dataset stored in BigQuery.
 
Your job is to convert a user's natural language question into a valid BigQuery SQL query.
Return ONLY the SQL query — no explanation, no markdown fences, no preamble.
 
════════════════════════════════════════════════════════
TABLES AVAILABLE
════════════════════════════════════════════════════════
 
`{GCP_PROJECT}.{BQ_DATASET}.building_permits`
Columns: `Permit Number`, `Permit Type`, `Permit Type Definition`, `Permit Creation Date`,
`Block`, `Lot`, `Street Number`, `Street Name`, `Street Suffix`, `Description`,
`Current Status`, `Current Status Date`, `Filed Date`, `Issued Date`, `Completed Date`,
`Permit Expiration Date`, `Estimated Cost`, `Revised Cost`, `Existing Use`, `Proposed Use`,
`Existing Units`, `Proposed Units`, `Number of Existing Stories`, `Number of Proposed Stories`,
`Existing Construction Type`, `Existing Construction Type Description`,
`Proposed Construction Type`, `Proposed Construction Type Description`,
`Plansets`, `Supervisor District`, `Neighborhoods - Analysis Boundaries`,
`Zipcode`, `Latitude`, `Longitude`
 
`{GCP_PROJECT}.{BQ_DATASET}.building_permits_contacts`
Columns: `Permit Number`, `Role`, `Firm Name`, `Firm Address`, `Firm City`,
`Firm State`, `Firm Zipcode`, `License1`, `From Date`, `PTS Agent ID`
 
════════════════════════════════════════════════════════
COLUMN DISAMBIGUATION
════════════════════════════════════════════════════════
"cost/price/value"        → `Estimated Cost`
"revised/final cost"      → `Revised Cost`
"issued/approved date"    → `Issued Date`
"filed/submitted date"    → `Filed Date`
"completed/finished date" → `Completed Date`
"neighbourhood/area"      → `Neighborhoods - Analysis Boundaries`
"zip/postal code"         → `Zipcode`
"permit type/category"    → `Permit Type Definition`
"stories/floors"          → `Number of Existing Stories` or `Number of Proposed Stories`
"units/apartments"        → `Existing Units` or `Proposed Units`
"firm/company/contractor" → `Firm Name`
"role/capacity"           → `Role`
 
Status values: appeal, approved, cancelled, complete, denied, disapproved, expired,
filed, filing, granted, incomplete, inspection, issued, issuing, plancheck,
reinstated, revoked, suspend, unknown, upheld, withdrawn
"active/in progress" → issued | "done/finished" → complete | "pending" → plancheck
 
Relative dates on `Issued Date`:
"this year" → >= '2025-01-01'
"last year" → BETWEEN '2024-01-01' AND '2024-12-31'
"recent/lately" → >= '2023-01-01'
 
════════════════════════════════════════════════════════
RULES
════════════════════════════════════════════════════════
1. Always use backtick-quoted column names since many have spaces.
2. Default LIMIT 10 for row-returning queries; no LIMIT for aggregations.
3. For off-topic questions return exactly: INVALID
4. For questions about individual names return exactly: INVALID (names were removed for privacy)
5. String comparisons use LOWER() for case-insensitivity.
6. Use SAFE_CAST for numeric columns to avoid errors.
"""
 
 
def generate_sql(question: str) -> str:
    prompt = _SQL_SYSTEM + f"\n\nUser question: {question}"
    try:
        resp = gemini.models.generate_content(model=MODEL, contents=prompt)
        sql = resp.text.strip()
        # Strip markdown fences if model adds them anyway
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
            aggregation="SUM",
            color_range=[
                [0,   0,   255, 0],
                [0,   128, 255, 100],
                [0,   255, 200, 160],
                [255, 255, 0,   210],
                [255, 128, 0,   235],
                [255, 0,   0,   255],
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
        min_year      = int(timeline_df["Year"].min())
        max_year      = int(timeline_df["Year"].max())
        default_start = max(min_year, max_year - 1)
 
        year_range = st.slider(
            "Select year range:",
            min_value=min_year,
            max_value=max_year,
            value=(default_start, max_year),
            step=1,
        )
 
        filtered = timeline_df[timeline_df["Year"].between(year_range[0], year_range[1])]
        timeline = filtered.groupby("Year").size().rename("Permits Issued")
        st.line_chart(timeline)
        st.caption(
            f"Showing {year_range[0]}–{year_range[1]}. "
            f"Total permits in this range: {len(filtered):,}"
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
 
    desc_df = load_description_data()
 
    if desc_df.empty:
        st.warning("No descriptions found in the dataset.")
    else:
        raw_text = " ".join(desc_df["Description"].dropna().astype(str).tolist())
 
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update([
            "redacted", "REDACTED",
            "na", "nan", "none", "unknown",
            "sf", "san", "francisco",
            "permit", "work", "install", "existing", "new", "replace",
            "per", "plan", "app", "ref", "use", "non", "type",
            "ft", "sq", "fl", "st", "av", "blvd", "dr", "bldg",
            "t", "e", "f", "n", "s", "w", "el", "pa", "re", "ot",
            "ap", "oti", "appl",
            "flr", "extg", "finishe", "kindg", "maher",
            "1st", "2nd", "3rd", "nov",
            "one", "two", "single",
            "back", "full", "side",
            "cti", "will",
        ])
 
        tokens = [
            word for word in raw_text.split()
            if len(word) >= 3 and word.lower() not in custom_stopwords
        ]
        filtered_text = " ".join(tokens)
 
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
            stopwords=custom_stopwords,
            max_words=100,
            collocations=False,
            min_word_length=3,
            color_func=multicolor_func,
            prefer_horizontal=0.85,
        ).generate(filtered_text)
 
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)