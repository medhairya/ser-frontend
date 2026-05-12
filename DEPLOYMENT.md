# Deploying the SER web app (Vercel + Modal + Hugging Face)

This guide matches the layout under `new/`:

- **`frontend/`** — Vite + TypeScript UI (static hosting on **Vercel**).
- **`backend/`** — FastAPI inference packaged for **Modal** (GPU: **T4** by default).

**Why Modal (not Render)?** Render’s practical tier for web services is mostly CPU; GPU instances are a separate, heavier product. Modal gives on-demand **CUDA** for PyTorch with a simple `modal deploy` flow and scales to zero between requests.

---

## 1. Put `best_model.pt` on Hugging Face (recommended)

Do **not** commit large weight files to GitHub. Use a **Model** repository (not a Dataset repo), unless you prefer Dataset for a single file (Model repos are the usual place for `.pt` checkpoints).

1. Create a Hugging Face account: [https://huggingface.co/join](https://huggingface.co/join)
2. **New model** → name it (e.g. `your-username/avtca-ravdess-weights`).
3. Upload the checkpoint:
   - Training saves **`best_model.pt`** (see `deploy/train.py`). You can rename for clarity, but if you rename, set `HF_CHECKPOINT_FILENAME` accordingly.
   - Optional: add a small `README.md` in the repo describing RAVDESS, license, and citation.
4. **Public repo:** no token needed for download.  
   **Private repo:** create an access token (Settings → Access Tokens, read scope) and set it as **`HF_TOKEN`** in Modal (below).

CLI alternative (after `pip install huggingface_hub` and `huggingface-cli login`):

```bash
huggingface-cli upload your-username/avtca-ravdess-weights best_model.pt --repo-type model
```

---

## 2. Deploy the API on Modal

1. Install CLI: `pip install modal`
2. One-time login: `modal setup`
3. In the [Modal dashboard](https://modal.com/), open your app → **Secrets / Environment** (or per-function env) and set:
   - **`HF_REPO_ID`** — e.g. `your-username/avtca-ravdess-weights`
   - **`HF_CHECKPOINT_FILENAME`** — default `best_model.pt` if you use that name
   - **`HF_TOKEN`** — only if the model repo is private
   - **`ALLOWED_ORIGINS`** — e.g. `https://your-app.vercel.app` (comma-separated for multiple). Use `*` only for quick tests.

4. From the `backend` folder:

```bash
cd backend
modal deploy modal_app.py
```

5. Copy the **HTTPS URL** Modal shows for the ASGI app (ends with `.modal.run`) and use it as the frontend API base.

**Local CPU test (optional):**

```bash
cd backend
pip install -r requirements.txt
set HF_REPO_ID=your-username/avtca-ravdess-weights
uvicorn local_app:app --reload --port 8000
```

---

## 3. Deploy the frontend on Vercel

1. Push this repo (or the `new/` folder as the repo root) to **GitHub**.
2. In Vercel: **New Project** → import the repo.
3. Set **Root Directory** to `frontend` (if the repo contains more than the frontend).
4. **API URL:** The production Space URL is **fixed** in `frontend/src/main.ts` (`API_BASE`). You do **not** need `VITE_API_URL` on Vercel unless you fork and change the code.
5. Deploy.

---

## 4. GitHub layout

Suggested new repo root (what you have under `new/`):

- `frontend/` — production UI
- `backend/` — Modal + local FastAPI + copied `src/models` for inference
- `deploy/` — training code (existing)
- `DEPLOYMENT.md` — this file
- `.gitignore` — excludes `*.pt`, `node_modules`, etc.

Training stays reproducible from `deploy/`; inference weights live on Hugging Face.

---

## API contract

- `GET /health` → `{ "status": "ok", ... }`
- `POST /predict` — `multipart/form-data` field **`file`** (video with audio recommended)  
  → `{ "status": "success", "emotion": "...", "confidence": 0.xx, "probabilities": { ... } }`

The model is **multimodal** (audio mel + video frames), matching RAVDESS-style clips.
