# 🏗️ SF Building Permits Analytics Dashboard

An end-to-end data pipeline and analytics dashboard for exploring San Francisco building permits using **BigQuery + Streamlit + Gemini AI**.

This project:

* Downloads and cleans real-world data from Kaggle
* Uploads it to Google BigQuery
* Provides interactive visualizations
* Allows natural language queries via AI → SQL

---

# 🚀 Features

* 📥 **Automated Data Pipeline**

  * Fetches dataset from Kaggle
  * Cleans and removes duplicates
  * Scrubs sensitive data (PII)
  * Uploads to BigQuery

* 📊 **Interactive Dashboard (Streamlit)**

  * Heatmap of permit locations
  * Permits over time
  * Word cloud from descriptions

* 🤖 **AI-Powered Query System**

  * Ask questions in plain English
  * Gemini converts them into SQL
  * Executes queries on BigQuery

---

# 📁 Project Structure

```
.
├── Dashboard.py              # Streamlit dashboard
├── run.py                    # Pipeline + launcher
├── requirements.txt          # Dependencies
├── secrets.template.toml     # Template for credentials
├── .gitignore
└── .streamlit/
    └── secrets.toml          # (NOT included — you create this)
```

---

# ⚙️ Setup Instructions

## 1️⃣ Clone the repository

```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

---

## 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## 3️⃣ Create your secrets file

This project uses **Streamlit secrets** for all credentials.

### Step 1:

Create a folder:

```
.streamlit/
```

### Step 2:

Copy the template:

```
secrets.template.toml → .streamlit/secrets.toml
```

### Step 3:

Fill in your credentials based on this template:

👉 See template here: 

---

# 🔑 Required Credentials

You MUST provide the following:

---

## 🟡 Kaggle API

Used to download the dataset.

Get from:
https://www.kaggle.com/settings/account

Fill:

```
KAGGLE_USERNAME = "your_username"
KAGGLE_KEY = "your_api_key"
```

---

## 🔵 Google Cloud (BigQuery)

You need:

* A GCP project
* A BigQuery dataset
* A Service Account JSON key

### Steps:

1. Create project in Google Cloud
2. Enable BigQuery API
3. Create a service account
4. Download JSON key
5. Paste JSON values into `[gcp_service_account]`

Fill:

```
GCP_PROJECT = "your-project-id"
BQ_DATASET = "sf_permits"
```

---

## 🤖 Gemini API

Used for AI → SQL conversion.

Get API key from:
https://ai.google.dev/

Fill:

```
GEMINI_API_KEY = "your_api_key"
```

---

## 🌐 Streamlit App URL

Used to open the dashboard automatically after pipeline runs.

```
STREAMLIT_APP_URL = "https://your-app.streamlit.app"
```

(If running locally, you can leave this empty or set to localhost)

---

# ▶️ Running the Project

## Option 1 — Full pipeline + dashboard

```
python run.py
```

This will:

1. Check if data exists in BigQuery
2. If not → download + clean + upload
3. Open the dashboard automatically

---

## Option 2 — Running the Dashboard (after setup)

If the data has already been uploaded to BigQuery, you can run only the dashboard:

streamlit run Dashboard.py
```

---

# 🔄 Forcing a Data Reload

If you want to rerun the pipeline even if data exists:

In `secrets.toml`:

```
FORCE_RELOAD = "true"
```

---

# ⚠️ Important Notes

## 🔐 Secrets Safety

The following files are **NOT tracked by Git**:

* `.streamlit/secrets.toml`
* credentials JSON
* API keys

Never commit them.

---

## 📦 Dataset

* Source: Kaggle SF Building Permits dataset
* Size: ~700–800 MB
* Processed automatically in memory

---

## 💡 AI Limitations

* Gemini free tier has request limits
* Complex or vague questions may return `INVALID`
* Queries are constrained to dataset schema

---

# 🛠️ Tech Stack

* Python
* Streamlit
* Google BigQuery
* Gemini API
* Pandas
* PyDeck (maps)
* Matplotlib / WordCloud

---

# 📈 Example Questions

Try asking:

* "What is the average permit cost in Mission district?"
* "How many permits were issued per year?"
* "Which neighborhood has the most completed permits?"


