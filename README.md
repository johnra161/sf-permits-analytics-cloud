# 🏗️ SF Building Permits — Pipeline & Dashboard

One script (`run.py`) that handles everything:
- First run: downloads Kaggle data → cleans it → uploads to BigQuery → opens dashboard
- Repeat runs: detects data already exists → opens dashboard immediately

---

## Repo structure

```
sf-permits/
├── run.py                              # The one script you run
├── requirements.txt                    # Dependencies for run.py
├── dashboard/
│   ├── Dashboard.py                    # Streamlit app (deployed on Community Cloud)
│   ├── requirements.txt                # Streamlit dependencies
│   └── .streamlit/
│       └── secrets.toml.example        # Template for Streamlit secrets
└── .gitignore
```

---

## One-time setup

### 1. Clone the repo and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/sf-permits.git
cd sf-permits
pip install -r requirements.txt
```

### 2. Get your credentials

**Kaggle API key**
Go to kaggle.com → Account → API → Create New Token. This downloads `kaggle.json`.
Your username and key are inside that file.

**GCP service account key**
In GCP Console → IAM & Admin → Service Accounts → create one (or use an existing one) with:
- BigQuery Data Editor
- BigQuery Job User

Go to its Keys tab → Add Key → JSON → download the file. Keep the path handy.

**Gemini API key**
Get a free key at aistudio.google.com.

### 3. Set environment variables

On Mac/Linux:
```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"
export GCP_PROJECT="logical-carver-489015-h1"
export BQ_DATASET="sf_permits"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/sa-key.json"
export STREAMLIT_APP_URL="https://your-app.streamlit.app"
```

On Windows (Command Prompt):
```cmd
set KAGGLE_USERNAME=your_kaggle_username
set KAGGLE_KEY=your_kaggle_api_key
set GCP_PROJECT=logical-carver-489015-h1
set BQ_DATASET=sf_permits
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\sa-key.json
set STREAMLIT_APP_URL=https://your-app.streamlit.app
```

On Windows (PowerShell):
```powershell
$env:KAGGLE_USERNAME="your_kaggle_username"
$env:KAGGLE_KEY="your_kaggle_api_key"
$env:GCP_PROJECT="logical-carver-489015-h1"
$env:BQ_DATASET="sf_permits"
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\sa-key.json"
$env:STREAMLIT_APP_URL="https://your-app.streamlit.app"
```

### 4. Run

```bash
python run.py
```

First run takes ~5–10 minutes (download + clean + upload).
Every run after that opens the dashboard instantly.

To force a full re-run even if data exists:
```bash
export FORCE_RELOAD=true
python run.py
```

---

## Deploying the dashboard on Streamlit Community Cloud

This only needs to be done once (by the repo owner). Users who pull and run `run.py` just need the `STREAMLIT_APP_URL` pointing to your already-deployed app.

1. Push the repo to GitHub
2. Go to share.streamlit.io → sign in with GitHub → **Create app**
3. Set:
   - Repository: `YOUR_USERNAME/sf-permits`
   - Branch: `main`
   - Main file path: `dashboard/Dashboard.py`
4. Click **Advanced settings → Secrets** and paste the contents of `dashboard/.streamlit/secrets.toml.example`, filled with your real values (the `[gcp_service_account]` block is the fields from your GCP JSON key file, reformatted as TOML)
5. Click **Deploy**

Copy the URL Streamlit gives you — that's your `STREAMLIT_APP_URL`.
