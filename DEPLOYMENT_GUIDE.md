# 🚀 Deployment Guide — Online Drug Sales Legality Checker

This guide walks you through deploying the Flask web application so that **anyone with the link can access and use it**. Three free hosting options are provided below, ranked from easiest to most flexible.

---

## 📁 Project Structure

```
drug-checker/
├── app.py                  ← Main Flask application (English)
├── wsgi.py                 ← WSGI entry point (for PythonAnywhere)
├── Procfile                ← Process file (for Render)
├── requirements.txt        ← Python dependencies
├── train_model.py          ← ML training script (reference)
└── model_output/
    ├── best_model.joblib       ← Trained Gradient Boosting model
    ├── tfidf_vectorizer.joblib ← TF-IDF vectorizer
    ├── evaluation_results.json ← All model metrics
    └── model_info.json         ← Dataset statistics
```

---

## Option A: PythonAnywhere (Recommended — Easiest)

**PythonAnywhere** is a free cloud platform specifically designed for Python web apps. No credit card required. Your site will be live at `https://yourusername.pythonanywhere.com`.

### Step 1 — Create an Account

Go to [www.pythonanywhere.com](https://www.pythonanywhere.com) and click **"Create a Beginner account"** (free).

### Step 2 — Upload Project Files

1. After logging in, click the **"Files"** tab in the top menu.
2. Navigate to `/home/yourusername/` (this is your home directory).
3. Click **"Upload a file"** and upload the `drug-checker.zip` file.
4. Open a **Bash console** (from the "Consoles" tab → "Other: Bash") and run:

```bash
cd ~
unzip drug-checker.zip
cd drug-checker
```

### Step 3 — Install Dependencies

In the same Bash console, run:

```bash
pip3 install --user -r requirements.txt
```

> ⚠️ If `pythainlp` fails, try: `pip3 install --user pythainlp==4.0.2`

### Step 4 — Create the Web App

1. Go to the **"Web"** tab in the top menu.
2. Click **"Add a new web app"**.
3. Choose **"Manual configuration"** (not Flask — we'll configure manually).
4. Select **Python 3.10** (or the latest available).

### Step 5 — Configure WSGI

1. On the Web tab, find **"WSGI configuration file"** and click the link (it looks like `/var/www/yourusername_pythonanywhere_com_wsgi.py`).
2. **Delete ALL existing content** and replace with:

```python
import sys
import os

# Add your project directory to the path
project_home = '/home/yourusername/drug-checker'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

os.chdir(project_home)

from app import app as application
```

3. **Replace `yourusername`** with your actual PythonAnywhere username.
4. Click **"Save"**.

### Step 6 — Set Source Code Directory

On the Web tab, set these fields:
- **Source code:** `/home/yourusername/drug-checker`
- **Working directory:** `/home/yourusername/drug-checker`

### Step 7 — Reload & Visit

1. Click the green **"Reload"** button on the Web tab.
2. Visit: `https://yourusername.pythonanywhere.com`

🎉 **Done! Share this URL with anyone.**

### Troubleshooting PythonAnywhere

If you see an error page, check the **error log** (link is on the Web tab). Common issues:

| Problem | Solution |
|---------|----------|
| ModuleNotFoundError: pythainlp | Run `pip3 install --user pythainlp` in Bash |
| FileNotFoundError: model files | Verify files are in `/home/yourusername/drug-checker/model_output/` |
| 502 Bad Gateway | Click "Reload" on the Web tab |

---

## Option B: Render (Free — Auto-Deploy from GitHub)

**Render** provides free hosting with automatic deployment from GitHub.

### Step 1 — Push to GitHub

1. Create a GitHub account at [github.com](https://github.com) if you don't have one.
2. Create a **new repository** (e.g., `drug-checker`).
3. Upload all project files to the repository.

Using Git from the command line:

```bash
cd drug-checker
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/drug-checker.git
git push -u origin main
```

### Step 2 — Deploy on Render

1. Go to [render.com](https://render.com) and sign up (free).
2. Click **"New +"** → **"Web Service"**.
3. Connect your GitHub account and select the `drug-checker` repository.
4. Configure:
   - **Name:** `drug-checker`
   - **Runtime:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
5. Select the **Free** plan.
6. Click **"Create Web Service"**.

Render will build and deploy automatically. Your site will be at:
`https://drug-checker.onrender.com`

> ⚠️ Free tier on Render sleeps after 15 minutes of inactivity. The first visit after sleep takes ~30 seconds to wake up.

---

## Option C: Run Locally & Share via ngrok

If you want to quickly share from your own computer without any cloud hosting:

### Step 1 — Install Python

Download Python 3.10+ from [python.org/downloads](https://python.org/downloads). During installation, **check "Add Python to PATH"**.

### Step 2 — Install Dependencies

Open a terminal/command prompt and run:

```bash
cd drug-checker
pip install -r requirements.txt
```

### Step 3 — Run the App

```bash
python app.py
```

The app starts at `http://localhost:5000`. This works on your computer only.

### Step 4 — Share via ngrok (Optional)

To let others access your local server through a public URL:

1. Sign up at [ngrok.com](https://ngrok.com) (free).
2. Download and install ngrok.
3. Run in a **separate terminal**:

```bash
ngrok http 5000
```

4. ngrok gives you a public URL like `https://abc123.ngrok-free.app` — share this with anyone.

> ⚠️ This only works while your computer is running and ngrok is active.

---

## 📊 Quick Comparison

| Feature | PythonAnywhere | Render | Local + ngrok |
|---------|:---:|:---:|:---:|
| Difficulty | ⭐ Easy | ⭐⭐ Medium | ⭐ Easy |
| Always Online | ✅ Yes | ⚠️ Sleeps | ❌ No |
| Free | ✅ Yes | ✅ Yes | ✅ Yes |
| Custom Domain | ❌ | ✅ Paid | ❌ |
| Auto-Deploy | ❌ | ✅ Yes | ❌ |
| Speed | Fast | Fast (after wake) | Fastest |

**Recommendation:** Use **PythonAnywhere** for the simplest, always-online deployment.

---

## 🔗 After Deployment

Once your site is live, you can share the URL with anyone. They can:

1. Open the link in any web browser (desktop or mobile).
2. Go to the **"Check"** tab to verify product listings.
3. View model performance in the **"Results"** tab.
4. Review legal criteria in the **"Legal Criteria"** tab.

No installation or login is required for end users.

---

*Faculty of Pharmaceutical Sciences, Khon Kaen University — 2025*
