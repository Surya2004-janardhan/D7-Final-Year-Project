# Application Flow & Workflow Explanation

This file explains, in plain language and with concrete details, how the app's core flow works: how data moves between components, what runs in foreground vs background, the timing and delay mechanics, and the exact behavior of each button/control.

Use this as a reference for developers and product reviewers who want a clear mental model of system behavior.

---

## 1. High-level overview

- The system continuously monitors user audio and video to detect emotion changes. When a significant emotion transition is detected (subject to thresholds, hysteresis, and cooldowns), the app can pause the automatic recording/daemon, ask the user via a native notification whether to play a mapped song, and if allowed, start playback and show a meme follow-up notification after a delay.

- Main runtime pieces:
  - Frontend Renderer (Electron + React): UI, audio playback, scheduling meme notifications, user controls.
  - Main Process (Electron main.cjs): native integrations, OS notifications, window lifecycle and IPC endpoints.
  - Background Daemon (renderer hook `useDaemon.js`): continuous timed capture and process loop that records short clips, sends them to the backend, and triggers notifications when shifts are detected.
  - Backend (Flask `app.py`): analysis service that runs models, returns emotion probabilities and optional meme content.

All communication between renderer and main process uses Electron IPC; communication between renderer/daemon and backend uses HTTP endpoints (e.g., `/process`).

## 2. Component responsibilities (simple)

- Renderer (`frontend/src/App.jsx`)
  - Show UI and controls (Play, Stop Music, Mirror toggle, RecordingPanel).
  - Manage audio playback and playlist queue.
  - Pause/resume daemon when music plays or stops (controlled via `stopDaemon()`/`startDaemon()`).
  - Schedule the meme-only follow-up notification (e.g., 20s after playback starts) and cancel it if playback stops early.
  - Listen for `notification-action` IPC messages (Play/No button clicks from native OS notifications) and act accordingly.

- Main Process (`frontend/main.cjs`)
  - Show native OS notifications (ask with action buttons; meme-only notifications with icon/title).
  - Translate native notification button clicks into `notification-action` IPC events sent to renderer.
  - Set AppUserModelID and attempt shortcut setup on Windows to improve toast reliability.

- Daemon (`frontend/src/hooks/useDaemon.js`)
  - Operates a timed loop: wait -> capture short recording (audio+video) -> send to backend -> receive analysis -> update internal state and trigger initial notification (no meme) via IPC.
  - Maintains state flags: `isDaemonActive`, `daemonStatus`, `pausedForMusic`, `lastShiftTime`, etc.
  - Uses smoothing, deltas and slope checks (EMA, windows) to detect trends and transitions (see FORMALE.md for math).

- Backend (`app.py`)
  - Runs emotion detection models for audio/visual inputs, returns probabilities and optional memes (caption + optional image path/url).
  - Exposes REST endpoints used by the daemon/renderer.

## 3. Step-by-step runtime flow (what happens over time)

1. App starts: renderer boots, main process initializes, `useDaemon` registers and begins monitoring (if auto mode is enabled).
2. Daemon loop sleeps for configured `idle_interval` (seconds) then starts a short capture for `record_duration` seconds (example: capture 1.5s to 3.0s). Capture is performed with MediaRecorder in the renderer context.
3. Daemon sends captured clip to backend `/process` endpoint. Backend returns fused probabilities for K emotions and optionally a `memes` payload.
4. Daemon updates smoothed probabilities, computes delta and slope, and checks transition rules (per FORMALE.md). If a candidate transition is detected and sustained, the daemon triggers a notification request to main process with a payload EXCLUDING the meme (initial ask).
5. Main process shows a native notification with two action buttons: "Play" and "No". The notification contains an emoji-prefixed title, short reason text, and a tag identifying the shift event.
6. If the user taps "Play" in the native notification, the main process emits `notification-action` IPC to the renderer with action='play' and the payload identifying the shift. The renderer immediately:
   - Calls `stopDaemon()` (sets `pausedForMusic=true`).
   - Clears any running daemon timers and pending captures.
   - Starts or queues the music track mapped to the detected emotion.
   - Starts a meme follow-up timer (e.g., 20s). After the delay the renderer asks main process to show a meme-only notification (title+caption and optional image).
7. If the user taps "No" or dismisses the notification, the renderer leaves the daemon paused only if previously paused for music, otherwise continues monitoring; no playback occurs.
8. When playback ends (naturally or via Stop Music button), the renderer clears the meme timer and, if `pausedForMusic` is true, calls `startDaemon()` to resume automatic monitoring.

## 4. Buttons and controls (exact behavior)

- Play (in notification)
  - Action: sent from OS notification to renderer via `notification-action` IPC with `{action: 'play', payload}`.
  - Renderer response: immediately `stopDaemon()`, set `pausedForMusic=true`, start playback of emotion-mapped song, schedule meme follow-up timer.

- No (in notification)
  - Action: send `{action:'no'}` back to renderer.
  - Renderer response: do not play. If daemon was not paused before the ask, keep it running. If already paused for some other reason, maintain prior paused state.

- Stop Music (UI button / dashboard)
  - Action: stops current audio playback immediately.
  - Renderer response: clears meme timer, sets `pausedForMusic=false`, and if auto mode was enabled before, restarts the daemon after a short guard delay (100–500ms) to avoid immediate re-triggering.

- Mirror Preview toggle (RecordingPanel)
  - Action: toggles whether a small mirrored video preview is shown while recording.
  - Renderer response: when `isRecording && mirrorPreview` -> show preview element streaming `getUserMedia()`; when not recording or toggle off -> stop preview stream.

- Auto Mode Toggle (daemon control)
  - Action: controls whether the daemon runs automatically in the background.
  - Renderer response: toggles `isDaemonActive`. If disabled, `stopDaemon()` is invoked and periodic captures stop until user re-enables.

## 5. Foreground vs Background operations

- Foreground (UI/main interaction):
  - User clicks, responds to notifications, toggles settings, and interacts with playback controls. These actions are handled in the renderer and via IPC to main.

- Background (daemon & processing):
  - The daemon loop runs in renderer background context (not blocking UI threads) and schedules captures asynchronously. Network I/O to backend is non-blocking.
  - Heavy model inference runs in backend; the renderer only waits for results and updates internal probabilities.

- Important note about threading & event loops:
  - MediaRecorder and capture APIs run in renderer/browser threads; do not perform heavy synchronous CPU work there.
  - All IPC handlers are event-driven; ensure handlers are idempotent and tolerant of repeated messages.

## 6. Timings, delays, and scheduling (concrete)

- Capture cycle
  - `idle_interval`: time between capture windows when nothing interesting detected (example 5–10s).
  - `record_duration`: how long each recorded clip lasts (example 1.5–3.0s). Shorter clips reduce latency; longer clips provide more signal.

- Detection rules
  - Use EMA smoothing and a 2s rolling window for delta/slope calculations (see FORMALE.md defaults: `alpha=0.30`, `N = 2s * fs`).
  - Require `T_sustain` (2s) of consistent signal before confirming a transition to avoid false positives.

- Notification and playback
  - Initial ask notification appears immediately when a transition is confirmed.
  - `T_meme` (meme delay): schedule meme-only follow-up 20s after playback starts (cancel if playback stops early).
  - `T_cooldown` (play cooldown): default 30s between automatic plays to avoid rapid successive songs.

## 7. State flags and guards (what to check before actions)

- `isDaemonActive` — whether the auto-monitoring loop should run.
- `pausedForMusic` — set when playback is started by user action (notification Play or UI Play). Used to resume daemon after playback ends only if this flag is true.
- `isPlaying` — whether audio playback is currently active.
- `lastPlayTimestamp` — used to enforce `T_cooldown`.

Guard examples:
- Before starting playback: verify `!isPlaying && (now - lastPlayTimestamp) > T_cooldown && userAllowsAutoPlay`.
- Before resuming daemon after playback: only call `startDaemon()` if `pausedForMusic === true` (so we don't accidentally restart if user manually paused the daemon earlier).

## 8. Error handling and edge cases (simple rules)

- Network failures when calling `/process`: retry once quickly (e.g., 500ms), then drop that frame and continue loop to avoid blocking.
- If MediaRecorder fails to start, escalate to UI: show an inline error and stop the daemon until user re-enables capture.
- If backend returns low-confidence results (all probabilities below `p_min`), treat the frame as "no decision": do not trigger notifications.

## 9. Where to change behavior (code pointers)

- Daemon loop & detection: `frontend/src/hooks/useDaemon.js` — change intervals, smoothing alpha, decision thresholds, sustain times.
- Playback control, notification handling, meme scheduling: `frontend/src/App.jsx` — change how Play/No are handled and where `stopDaemon()`/`startDaemon()` are called.
- Native notification formatting and action wiring: `frontend/main.cjs` — change titles, actions, and Windows toast setup.
- Backend models and meme generation: `app.py` — change model calls, meme content generation, and response shape.

## 10. Quick interaction example (end-to-end)

1. Daemon captures 2s clip and sends to backend.
2. Backend returns fused probs: `happy:0.67`, `neutral:0.42`.
3. Daemon smoothed `happy` crosses threshold and sustained 2s → triggers ask notification.
4. User taps "Play" in native notification → renderer stops daemon, plays happy song, sets `pausedForMusic=true`, schedules meme in 20s.
5. After 20s of playback, renderer asks main process to show meme-only notification.
6. Playback ends; renderer clears timers and resumes daemon if `pausedForMusic`.

---

If you'd like, I can now:
- Add a small JS snippet implementing the exact scheduling/flags used in the app to paste into `useDaemon.js` or `App.jsx`.
- Add a diagram (Mermaid) showing message flow between daemon → backend → main → renderer.

End of explanation.
