"""
Dashboard.py — SF Permit Analytics (Streamlit Community Cloud)
Reads directly from BigQuery. Deployed separately on share.streamlit.io.
"""

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

st.set_page_config(page_title="SF Permit Analytics", page_icon="🏗️", layout="wide")
st.title("🏗️ SF Permit Analytics Dashboard")

# ── BigQuery client ──────────────────────────────────────────

GCP_PROJECT = st.secrets["GCP_PROJECT"]
BQ_DATASET  = st.secrets["BQ_DATASET"]


@st.cache_resource
def get_bq_client():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return bigquery.Client(project=GCP_PROJECT, credentials=creds)


bq = get_bq_client()

# ── Gemini client ────────────────────────────────────────────

MODEL  = "gemini-2.5-flash"
gemini = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ── Data loading ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading permits…")
def load_permits():
    df = bq.query(
        f"SELECT * FROM `{GCP_PROJECT}.{BQ_DATASET}.building_permits`"
    ).to_dataframe()
    for col in ["Estimated Cost", "Revised Cost", "Existing Units", "Proposed Units",
                "Number of Existing Stories", "Number of Proposed Stories",
                "Latitude", "Longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner="Loading contacts…")
def load_contacts():
    return bq.query(
        f"SELECT * FROM `{GCP_PROJECT}.{BQ_DATASET}.building_permits_contacts`"
    ).to_dataframe()


@st.cache_data(ttl=3600, show_spinner="Building schema…")
def build_schema(_permits, _contacts):
    def infer_type(series):
        sample = series.dropna().head(30)
        if sample.empty:
            return "string"
        try:
            pd.to_numeric(sample); return "number"
        except (ValueError, TypeError): pass
        try:
            if pd.to_datetime(sample, errors="coerce").notna().sum() > len(sample) * 0.7:
                return "date"
        except Exception: pass
        return "string"

    def schema_for(df):
        out = {}
        for col in df.columns:
            t = infer_type(df[col])
            entry = {"type": t}
            if t == "string":
                uniq = df[col].dropna().unique()
                if len(uniq) <= 30:
                    entry["values"] = sorted(str(v) for v in uniq)
            out[col] = entry
        return out

    return {"permits": schema_for(_permits), "contacts": schema_for(_contacts)}


permits_df  = load_permits()
contacts_df = load_contacts()
SCHEMA      = build_schema(permits_df, contacts_df)

st.success(f"✅  {len(permits_df):,} permit rows · {len(contacts_df):,} contact rows")

# ── Gemini query generation ──────────────────────────────────

_QUERY_SYSTEM = """
You are a data analyst assistant for the SF Building Permits dataset.
Convert the user's natural language question into a structured JSON query object.
Return ONLY valid JSON — no explanation, no markdown fences, no preamble.

════════════════════════════════════════════════════════
RESPONSE TYPES
════════════════════════════════════════════════════════
A) Valid query → return the query object.
B) Concept exists but no matching column →
   {"invalid_query": true, "reason": "...", "suggestion": "..."}
C) Completely off-topic →
   {"invalid_query": true, "reason": "not related to SF building permits"}

════════════════════════════════════════════════════════
DATASET OVERVIEW
════════════════════════════════════════════════════════
"permits"  — one row per permit: identity, dates, location, cost, construction, status.
"contacts" — one row per person/firm on a permit. Joined via "Permit Number".
Cross-dataset questions → invalid_query (engine is single-dataset only).

════════════════════════════════════════════════════════
COLUMN DISAMBIGUATION
════════════════════════════════════════════════════════
COST: "cost/price/value" → "Estimated Cost"; "revised/final cost" → "Revised Cost"
DATES: "issued/approved" → "Issued Date" (default); "filed/submitted" → "Filed Date";
       "completed/finished" → "Completed Date"; "expired" → "Permit Expiration Date"
Relative dates → filter on "Issued Date":
  "this year"→>="2025-01-01"; "last year"→2024; "recent/lately"→>="2023-01-01"
STATUS → "Current Status". Values: appeal, approved, cancelled, complete, denied,
  disapproved, expired, filed, filing, granted, incomplete, inspection, issued, issuing,
  plancheck, reinstated, revoked, suspend, unknown, upheld, withdrawn
  "active"→"issued"; "done"→"complete"; "pending"→"plancheck" or "filed"
LOCATION: "neighbourhood/area" → "Neighborhoods - Analysis Boundaries";
  "zip" → "Zipcode"; "supervisor district" → "Supervisor District"
PERMIT TYPE → "Permit Type Definition". Values: additions alterations or repairs,
  demolitions, new construction, new construction wood frame, otc alterations permit,
  sign - erect, wall or painted sign
CONSTRUCTION: "stories/floors" → "Number of Existing Stories"/"Number of Proposed Stories";
  "units/apartments" → "Existing Units"/"Proposed Units"
CONTACTS: "firm/company/contractor" → "Firm Name"; "role" → "Role"
  Roles: architect, attorney, authorized agent-others, contractor, designer,
  engineer, lessee, pmt consultant/expediter, project contact
PRIVACY: First Name, Last Name, Contact Name do NOT exist → invalid_query if asked.

════════════════════════════════════════════════════════
QUERY RULES
════════════════════════════════════════════════════════
1. Only use columns that exist in the schema.
2. Operators: "=" exact; "contains" substring; ">","<",">=","<=" numeric/date.
3. group_by: pair with metric; sort desc by default; never group Description, Firm Name,
   Street Name, Permit Number.
4. "count" needs no column. "average/sum/min/max" need a numeric column.
5. limit: default 10; omit for scalar queries.
6. Vague questions → no filters, no metric, limit 10.
7. Default dataset when ambiguous → "permits".

════════════════════════════════════════════════════════
QUERY SCHEMA
════════════════════════════════════════════════════════
{
  "dataset": "permits" | "contacts",
  "filters": [{"column": "...", "operator": "="|">"|"<"|">="|"<="|"contains", "value": "..."}],
  "metric":  {"operation": "count"|"average"|"sum"|"min"|"max", "column": "..."},
  "group_by": ["..."],
  "sort":    {"column": "...", "order": "asc"|"desc"},
  "limit":   <integer>
}
All fields except "dataset" are optional.

════════════════════════════════════════════════════════
DATASET SCHEMA
════════════════════════════════════════════════════════
"""


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"invalid_query": True}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {"invalid_query": True}


def generate_query(question):
    prompt = _QUERY_SYSTEM + json.dumps(SCHEMA, indent=2) + f"\n\nUser question: {question}"
    try:
        resp = gemini.models.generate_content(model=MODEL, contents=prompt)
        return extract_json(resp.text)
    except Exception as e:
        return {"invalid_query": True, "reason": f"Gemini error: {e}"}


# ── Query engine ─────────────────────────────────────────────

def execute_query(query):
    df = permits_df.copy() if query.get("dataset", "permits") == "permits" else contacts_df.copy()

    for f in query.get("filters", []):
        col, op, val = f.get("column"), f.get("operator", "="), f.get("value")
        if col not in df.columns:
            continue
        try:
            nval = float(val)
        except (TypeError, ValueError):
            nval = None

        if op in ("=", "=="):
            if nval is not None and pd.api.types.is_numeric_dtype(df[col]):
                df = df[df[col] == nval]
            else:
                df = df[df[col].astype(str).str.lower() == str(val).lower()]
        elif op == "contains":
            df = df[df[col].astype(str).str.lower().str.contains(str(val).lower(), na=False)]
        elif nval is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            ops = {">": df[col].__gt__, ">=": df[col].__ge__,
                   "<": df[col].__lt__, "<=": df[col].__le__}
            if op in ops:
                df = df[ops[op](nval)]

    if "metric" in query:
        op_name  = query["metric"].get("operation", "count")
        col      = query["metric"].get("column")
        group_by = [g for g in query.get("group_by", []) if g in df.columns]
        agg      = {"average": "mean", "sum": "sum", "count": "count",
                    "min": "min", "max": "max"}.get(op_name, "count")

        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if group_by and col:
            result = df.groupby(group_by)[col].agg(agg).reset_index()
            result.columns = group_by + [f"{op_name}_{col}"]
        elif group_by:
            result = df.groupby(group_by).size().reset_index(name="count")
        else:
            value = getattr(df[col], agg)() if col and col in df.columns else len(df)
            result = pd.DataFrame({"Result": [value]})
    else:
        result = df

    if "sort" in query:
        sc = query["sort"].get("column")
        if sc and sc in result.columns:
            result = result.sort_values(sc, ascending=(query["sort"].get("order") == "asc"))

    if "limit" in query:
        result = result.head(int(query["limit"]))

    return result


# ── Tabs ─────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["💬 AI Chat", "📊 Visualizations"])

with tab1:
    st.subheader("Ask Questions About the Data")
    st.caption(
        "Examples: *Average permit cost in the Mission?* · "
        "*Permits issued per year?* · *Neighbourhood with most completed permits?*"
    )

    question = st.text_input("Your question:", placeholder="Ask anything about SF building permits…")

    if question:
        with st.spinner("Thinking…"):
            query = generate_query(question)

        if query.get("invalid_query"):
            msg = "The AI couldn't map that question to the dataset."
            if query.get("reason"):
                msg += f"\n\n**Why:** {query['reason']}"
            if query.get("suggestion"):
                msg += f"\n\n**Try instead:** {query['suggestion']}"
            st.warning(msg)
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Generated Query")
                st.json(query)
            with c2:
                try:
                    result = execute_query(query)
                    st.subheader("Result")
                    st.dataframe(result, use_container_width=True)
                    if (isinstance(result, pd.DataFrame) and result.shape[1] == 2
                            and result.shape[0] > 1
                            and pd.api.types.is_numeric_dtype(result.iloc[:, 1])):
                        st.subheader("Chart")
                        st.bar_chart(result.set_index(result.columns[0])[result.columns[1]])
                except Exception as e:
                    st.error(f"Query error: {e}")

with tab2:

    # Heatmap
    st.subheader("📍 Permit Location Heatmap")
    st.markdown("Where permit activity is concentrated across SF. Brighter = higher density.")

    map_df = (permits_df[["Latitude", "Longitude"]].copy()
              .pipe(lambda d: d.assign(
                  Latitude=pd.to_numeric(d["Latitude"], errors="coerce"),
                  Longitude=pd.to_numeric(d["Longitude"], errors="coerce")))
              .dropna().rename(columns={"Latitude": "lat", "Longitude": "lon"}))

    if map_df.empty:
        st.warning("No valid coordinates found.")
    else:
        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer(
                "HeatmapLayer", data=map_df, get_position=["lon", "lat"],
                aggregation="SUM", opacity=0.85, radius_pixels=60,
                intensity=2, threshold=0.01,
                color_range=[[0,0,255,0],[0,128,255,100],[0,255,200,160],
                             [255,255,0,210],[255,128,0,235],[255,0,0,255]],
            )],
            initial_view_state=pdk.ViewState(
                latitude=map_df["lat"].median(), longitude=map_df["lon"].median(),
                zoom=11, pitch=0),
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        ))

    st.divider()

    # Timeline
    st.subheader("📈 Permits Issued Over Time")
    st.markdown("Volume of permits issued per year. Use the slider to zoom in.")

    if "Issued Date" in permits_df.columns:
        tdf = permits_df[["Issued Date"]].copy()
        tdf["Issued Date"] = pd.to_datetime(tdf["Issued Date"], errors="coerce")
        tdf = tdf.dropna()
        tdf["Year"] = tdf["Issued Date"].dt.year.astype(int)
        min_y, max_y = int(tdf["Year"].min()), int(tdf["Year"].max())
        yr = st.slider("Year range:", min_value=min_y, max_value=max_y,
                       value=(max(min_y, max_y - 1), max_y), step=1)
        filt = tdf[tdf["Year"].between(yr[0], yr[1])]
        st.line_chart(filt.groupby("Year").size().rename("Permits Issued"))
        st.caption(f"Showing {yr[0]}–{yr[1]} · {len(filt):,} permits")
    else:
        st.warning("'Issued Date' column not found.")

    st.divider()

    # Word cloud
    st.subheader("☁️ Description Word Cloud")
    st.markdown("Most frequent words in permit descriptions. Boilerplate and PII markers removed.")

    if "Description" in permits_df.columns:
        stops = set(STOPWORDS) | {
            "redacted", "REDACTED", "na", "nan", "none", "unknown",
            "sf", "san", "francisco", "permit", "work", "install", "existing",
            "new", "replace", "per", "plan", "app", "ref", "use", "non", "type",
            "ft", "sq", "fl", "st", "av", "blvd", "dr", "bldg",
            "t", "e", "f", "n", "s", "w", "el", "pa", "re", "ot", "ap", "oti", "appl",
            "flr", "extg", "finishe", "kindg", "maher",
            "1st", "2nd", "3rd", "nov", "one", "two", "single",
            "back", "full", "side", "cti", "will",
        }
        tokens = [w for w in " ".join(permits_df["Description"].dropna().astype(str)).split()
                  if len(w) >= 3 and w.lower() not in stops]

        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            import random
            colors = ["hsl(210,80%,35%)", "hsl(200,75%,45%)", "hsl(185,70%,42%)",
                      "hsl(160,60%,40%)", "hsl(25,80%,50%)", "hsl(15,85%,48%)", "hsl(45,80%,48%)"]
            return (random_state or random.Random()).choice(colors)

        wc = WordCloud(width=1000, height=450, background_color="white", stopwords=stops,
                       max_words=100, collocations=False, min_word_length=3,
                       color_func=color_func, prefer_horizontal=0.85).generate(" ".join(tokens))
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("'Description' column not found.")
