# EmotionAI

EmotionAI is a multimodal emotion monitoring project with:
- A Python/Flask backend for video+audio emotion analysis
- A React UI (inside Electron) for dashboard, history, chatbot, and settings
- Background Auto Mode that records at intervals, runs analysis, stores results, and triggers native notifications on negative emotional shifts

## Current Repository Scope

This repository contains:
- Runtime app: `app.py` (Flask API + model inference + Groq integrations)
- Desktop client: `frontend/` (React + Vite + Electron shell)
- Pretrained models: `models/*.h5`
- Data/training/testing utilities: `scripts/` and `tests/`
- Local DB: `emotionai.db` (SQLite)

## Core Features (As Implemented)

1. Multimodal processing
- Accepts uploaded/recorded video (`/process`)
- Extracts audio via FFmpeg
- Runs video model + audio model and computes timeline-based result

2. Temporal emotion summary
- Produces per-segment audio/video timelines
- Computes fused dominant emotion, confidence, stability, transition rate, and distribution

3. AI content generation
- Uses Groq (`llama-3.3-70b-versatile`) for:
  - Story, quote, suggested video, books, songs
  - Chatbot replies (`/chat`)
  - History trend analysis (`/analyze_history`)
- Falls back to internal templates for content generation if LLM response fails

4. Background Auto Mode (Electron frontend)
- Interval-based recording (`intervalMinutes`)
- Recording duration control (`recordDurationMinutes`)
- Background processing + persistent save
- Negative-shift notification with optional auto-play intervention music

5. Persistence
- Backend SQLite (`emotionai.db`): history + emotion->music mappings
- Electron user data:
  - `settings.json` (persistent app settings)
  - `results.json` (analysis history used by calendar/history view)
  - `analyses/*.json` (cached range analysis text)

6. Dashboard modules
- Recording/Upload panel
- Processing progress polling (`/status`)
- Emotion cards + temporal chart + cognitive insights
- Calendar/history with range filters (today/week/month/all)
- AI assistant chat panel
- Settings for monitoring schedule and music mapping

## Tech Stack

Backend:
- Flask, Flask-CORS
- TensorFlow/Keras
- OpenCV
- Librosa
- NumPy/Pandas/scikit-learn
- ffmpeg-python

Frontend/Desktop:
- React 19 + Vite
- Tailwind CSS
- Chart.js + react-chartjs-2
- Electron

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- emotionai.db
|-- models/
|-- scripts/
|   |-- data_processing/
|   |-- model_training/
|   `-- model_testing/
|-- tests/
`-- frontend/
    |-- src/
    |-- main.cjs
    |-- vite.config.js
    `-- package.json
```

## Setup

### 1) Backend

```bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# Linux/macOS
# source myenv/bin/activate

pip install -r requirements.txt
```

Set Groq key (optional but recommended):

```bash
# Windows (cmd)
set GROQ_API_KEY=your_key_here

# PowerShell
$env:GROQ_API_KEY="your_key_here"

# Linux/macOS
export GROQ_API_KEY=your_key_here
```

Run backend:

```bash
python app.py
```

Backend listens on `http://127.0.0.1:5000`.

### 2) Frontend / Electron

```bash
cd frontend
npm install
```

Build UI bundle:

```bash
npm run build
```

Run desktop app:

```bash
npm run electron:dist
```

Notes:
- `electron:start` currently references `npm run dev`, but `dev` script is not defined in `frontend/package.json`.
- If you need Vite dev mode, run Vite manually and then launch Electron.

## API Endpoints

Main Flask routes in `app.py`:
- `GET /status` -> processing progress state
- `POST /process` -> upload video and run full pipeline
- `GET /music/search?q=...` -> search/cache track and return preview URL
- `GET /downloaded_music/<filename>` -> serve cached music file
- `GET /history` / `DELETE /history`
- `GET /mappings` / `POST /mappings`
- `POST /analyze_history`
- `POST /chat`
- `GET /stream_local?path=...`

## Behavioral Notes

- Backend process is started by Electron via child process (`python app.py`) and can be started on-demand through IPC.
- Current Electron window behavior: app opens visibly on launch and minimizes to tray on close.
- Upload size limit is set to 100MB in Flask.

## Scripts and Utilities

- Model/data pipelines are in `scripts/model_training/` and `scripts/data_processing/`
- Test/inspection helpers are in `tests/`
- Plot artifacts are stored under `plots/`

## Known Gaps / Cleanup Candidates

- `frontend/package.json` is missing a `dev` script used by `electron:start`
- Some backend code includes legacy paths/unused helper functions that can be simplified
- Root README and `frontend/README.md` were previously inconsistent (this root README is updated to match current repo)
