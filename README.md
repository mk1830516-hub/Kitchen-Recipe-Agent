# 🍳 Kitchen Recipe AI

A Streamlit chat agent that gives recipes, formats ingredients into clean
tables, adapts output for **Normal** vs **Diabetic / Health-conscious**
users (with product swaps + healthier cooking methods), remembers chat
history, and falls back across three LLM providers in this exact order:
**Groq → Cohere → Gemini**.

## Project files

| File | Purpose |
|---|---|
| `agent.py` | The app (run this one) |
| `requirements.txt` | Python dependencies |
| `chat_memory.json` | Auto-created/updated chat history — not pushed to GitHub |
| `.gitignore` | Keeps secrets and chat history out of the public repo |
| `.streamlit/secrets.toml` | Your real API keys — **you create this yourself, never commit it** |

---

## Part 1 — Run it in Google Colab

**Step 1 — Upload files**
Upload `agent.py`, `requirements.txt`, and `.gitignore` into `/content/` via the Colab Files panel.

**Step 2 — Install everything**
```python
!pip install -r requirements.txt
!npm install -g localtunnel
```

**Step 3 — Add your keys as Colab Secrets**
Click the 🔑 icon (left sidebar) → add `GROQ_API_KEY`, `COHERE_API_KEY`, `GEMINI_API_KEY` → turn on "Notebook access" for each.

**Step 4 — Write them into `.streamlit/secrets.toml`**
This is what `agent.py` reads via `st.secrets` — no keys ever appear in your code:
```python
import os
from google.colab import userdata

os.makedirs(".streamlit", exist_ok=True)
with open(".streamlit/secrets.toml", "w") as f:
    f.write(f'GROQ_API_KEY = "{userdata.get("GROQ_API_KEY")}"\n')
    f.write(f'COHERE_API_KEY = "{userdata.get("COHERE_API_KEY")}"\n')
    f.write(f'GEMINI_API_KEY = "{userdata.get("GEMINI_API_KEY")}"\n')

print("Secrets file created.")
```

**Step 5 — Run the app and open a public tunnel**
```python
!streamlit run agent.py & npx localtunnel --port 8501
```
This prints a `.loca.lt` link. Open it — it will ask for a "Tunnel Password", which is your Colab machine's public IP. Get it with:
```python
import urllib
print(urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip())
```
Paste that IP into the password field on the tunnel page.

---

## Part 2 — Push to GitHub from Colab

Only `agent.py`, `requirements.txt`, and `.gitignore` get pushed. `chat_memory.json` and `.streamlit/secrets.toml` are excluded by `.gitignore` on purpose — they hold your chat data and real keys.

```python
!git init
!git config --global user.name "Ammara"
!git config --global user.email "your_email@example.com"

!git add agent.py requirements.txt .gitignore

!git commit -m "Kitchen Recipe Agent commit"

!git branch -M main

!git remote add origin https://github.com/<your-username>/kitchen-recipe-ai.git

!git push -u origin main
```

**What each command does:**
- `git init` — turns the current Colab folder into a Git repository
- `git config --global user.name / user.email` — sets your identity so GitHub knows who made each commit
- `git add agent.py requirements.txt .gitignore` — stages only these 3 safe files, leaving out `chat_memory.json` and secrets
- `git commit -m "..."` — saves the staged files as a snapshot with a message
- `git branch -M main` — renames the default branch to `main` (GitHub's standard)
- `git remote add origin <URL>` — links this local repo to your GitHub repo
- `git push -u origin main` — uploads your commit to GitHub for the first time

If `git push` asks for a password, GitHub requires a **Personal Access Token** (Settings → Developer settings → Personal access tokens → generate one with `repo` scope) instead of your account password.

If you'd rather skip typing a token every time, install GitHub CLI (`gh`) in Colab and run `gh auth login` once — it authenticates via a browser link instead.

---

## Part 3 — Connect to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
2. **Create app** → pick your repo → branch `main` → main file `agent.py` → **Deploy**
3. Once deployed, open **⋮ → Settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "your_real_groq_key"
   COHERE_API_KEY = "your_real_cohere_key"
   GEMINI_API_KEY = "your_real_gemini_key"
   ```
4. **Save** — the app restarts and reads these automatically via `st.secrets`, same as it did from `.streamlit/secrets.toml` in Colab. No code changes needed.

From now on, any time you edit `agent.py` in Colab and re-run the Part 2 push commands, Streamlit Cloud auto-detects the new commit and redeploys within a minute or two.
