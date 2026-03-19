# EmotionAI Frontend (React + Electron)

This folder contains the desktop client UI and Electron runtime shell used by EmotionAI.

## What Is Here

- React renderer app (`src/`) for:
  - Dashboard (manual recording/upload + result cards/charts)
  - History/Calendar analytics
  - Settings (auto mode, interval, recording duration, intervention mapping)
  - Chat assistant panel
- Electron main process (`main.cjs`) for:
  - Tray lifecycle
  - Backend child-process control (`python app.py`)
  - Native notifications
  - Persistent local storage via userData files

## Scripts

```bash
npm run build          # Build Vite bundle to dist/
npm run preview        # Preview built frontend
npm run lint           # Lint React code
npm run electron:dist  # Launch Electron app
```

Note: `electron:start` exists but expects a `dev` script that is currently not defined.

## Runtime Integration

- Renderer sets API base URL to Flask: `http://localhost:5000`
- Vite proxy is configured for backend routes in `vite.config.js`
- Electron IPC is used for settings/history/cache and backend lifecycle

## Persistent Files (Electron userData)

- `settings.json`
- `results.json`
- `analyses/*.json`

## Main Files

- `main.cjs`: Electron main process and IPC handlers
- `src/App.jsx`: top-level app shell and tabs
- `src/hooks/useDaemon.js`: auto-mode background monitor
- `src/hooks/useProcessing.js`: `/process` + `/status` workflow
- `src/components/CalendarView.jsx`: historical analytics + AI trend analysis

For complete backend architecture and API contracts, see the root README.
