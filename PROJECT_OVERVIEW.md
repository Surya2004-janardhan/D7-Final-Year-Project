# EmotionAI — Project Overview & Presentation Guide

> **For Faculty Presentation · Final Year Project · B.Tech 4th Year**
> Simple explanations. Strong points. Real-world impact.

---

## What Is This Project? (One Line)

> **EmotionAI is a desktop application that silently watches how you feel while working, and steps in with music or a joke when you are getting too stressed — automatically, without you having to do anything.**

---

## The Problem (Why Does This Matter?)

Software engineers and IT professionals spend **8–12 hours a day** staring at screens.

Over time, this creates:
- 😤 **Stress** — from debugging, deadlines, or repeated failures
- 😶 **Mental fatigue** — the brain slows down, focus drops
- 😞 **Emotional burnout** — which is invisible, hard to measure, and often ignored

The critical issue: **no one notices it in real-time.**

By the time a person realises they are burned out, it has already impacted their health and output. There is no tool today that:
1. Silently monitors your emotional state *while* you work
2. Detects when you are shifting toward stress or sadness
3. Automatically intervenes with something to help — music, a joke — at the *right moment*

---

## Existing Solutions — Where They Fall Short

| Existing Tool | What It Does | Problem |
|---|---|---|
| **Headspace / Calm** | Meditation apps | Requires the user to *already know* they need help and open the app |
| **Mood tracking apps** | Manual check-ins | You have to stop and type how you feel — people don't do this |
| **Wearables (Fitbit, etc.)** | Tracks heart rate | Expensive hardware, no contextual understanding of *why* you're stressed |
| **HR Wellness Programs** | Weekly check-ins | Too infrequent, one-size-fits-all, no real-time awareness |
| **Productivity apps** | Time tracking | Track *output*, not *emotion* or *well-being* |

**The gap:** None of them work **silently, in real-time, in the background, while you work, without interrupting you** unless something is actually wrong.

---

## Our Solution — What EmotionAI Does Differently

EmotionAI runs **passively in the background** on your computer like an antivirus — you don't see it unless it needs to tell you something.

Here is what happens from the user's perspective:

```
You sit down at your computer and start working.
EmotionAI starts watching — quietly.

Every 15 minutes, it takes a quick look at:
   → Your face (via webcam) — are you frowning? tense?
   → Your voice (via microphone) — is your tone flat? sharp?

It analyses both together and builds a picture of your emotional state.

If it sees you went from "fine" to "stressed" or "angry" — it sends a
Windows notification (like a Teams ping) that says:

   "Hey — looks like you're getting stressed. Want me to play your focus music?"

You click YES → your mapped music plays automatically.
You click NO  → it leaves you alone and keeps watching.
```

That's it. No app to open. No forms to fill. No action needed unless you want it.

---

## How It Works — Simple Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EMOTIONAI SYSTEM                             │
│                                                                     │
│  ┌────────────┐     Every X minutes (set by user)                  │
│  │   WEBCAM   │────►                                                │
│  └────────────┘     ┌──────────────────────┐                       │
│                     │  CAPTURE ENGINE      │  Records short clip   │
│  ┌────────────┐     │  (silent, bg)        │  of face + voice      │
│  │ MICROPHONE │────►└──────────────────────┘                       │
│  └────────────┘              │                                      │
│                              ▼                                      │
│                   ┌──────────────────────┐                         │
│                   │   AI ANALYSIS        │  Facial expression      │
│                   │   (Face + Voice)     │  + Voice tone combined  │
│                   └──────────────────────┘                         │
│                              │                                      │
│               ┌──────────────┴──────────────┐                      │
│               ▼                             ▼                       │
│     [No change detected]         [BIG SHIFT detected]              │
│      → Keep watching              e.g. Happy → Angry               │
│                                             │                       │
│                                             ▼                       │
│                              ┌──────────────────────────┐          │
│                              │  WINDOWS NOTIFICATION    │          │
│                              │  "Feeling stressed?      │          │
│                              │   Want to play music?"   │          │
│                              └──────────────────────────┘          │
│                                      │           │                  │
│                                   [YES]         [NO]               │
│                                      │           │                  │
│                                      ▼           ▼                  │
│                              Plays music    Keeps watching          │
│                              (user-mapped)                          │
│                                      │                              │
│                   ┌──────────────────┘                             │
│                   ▼                                                 │
│         ┌─────────────────┐                                        │
│         │ HISTORY SAVED   │  → Stored locally, privately           │
│         │ (SQLite DB)     │  → Viewable in Calendar tab            │
│         └─────────────────┘                                        │
│                   │                                                 │
│                   ▼                                                 │
│         ┌─────────────────┐                                        │
│         │  AI REPORT      │  "You get stressed most on             │
│         │  (Weekly/Daily) │   Wednesdays around 3 PM"              │
│         └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Three Tabs — Three Functions

### 1. 📊 Dashboard (Live View)
- Shows your **live camera feed** and **microphone waveform** (to confirm it's active)
- **Auto Mode toggle** — one button to turn background monitoring ON or OFF
- When the daemon detects something, results appear here
- Manual "Analyse Now" button if you want an instant check

### 2. 📅 History (Calendar View)
- Shows your **emotional pattern over time** — hour by hour, day by day
- Colour-coded grid: green = calm/happy, red = stressed/angry
- **Time filters**: Today / This Week / This Month / All Time
- "Run AI Analysis" button — sends your history to an AI which writes:
  > *"You show signs of emotional fatigue most on Mondays. 3 PM appears to be your highest-stress hour. Consider scheduling breaks during this window."*
- Once generated, this report is **saved locally** — no need to regenerate

### 3. ⚙️ Settings (Full Control)
| Setting | Options | What It Does |
|---|---|---|
| Check interval | 2 / 10 / 15 / 30 / 60 min | How often background check runs |
| Recording duration | 1 / 2 / 5 / 10 min | How long each sample is |
| Notification style | Ask me / Auto-play | Do you want to approve, or just let it play? |
| Music mapping | Per emotion (Happy, Sad, Angry…) | Your own playlist for each mood |

All settings **persist across restarts** — set once, works forever.

---

## Technology Stack (Simple Labels)

| Layer | Technology | Role |
|---|---|---|
| **Desktop Wrapper** | Electron (Node.js) | Makes it a real Windows desktop app |
| **UI** | React (JavaScript) | What you see on screen |
| **Face Analysis** | TensorFlow / MobileNetV2 | Reads emotions from your face |
| **Voice Analysis** | Librosa + CNN | Reads tone/emotion from your voice |
| **AI Fusion** | Custom weighted model | Combines face + voice into one answer |
| **Local Database** | SQLite | Stores all history — privately, offline |
| **AI Report** | Groq API (LLaMA model) | Writes your weekly/daily trend report |
| **OS Notifications** | Electron Notification API | Real Windows toast notifications |

> **Key point for faculty:** All face and voice data is processed **locally on the device**. Nothing is uploaded to the internet. Privacy is built in by design.

---

## How We Detect an Emotional Shift

The system does not just look at one moment. It uses a pattern across time:

```
Step 1 → Record short clip (e.g. 2 minutes of your face + voice)
Step 2 → Analyse: face model gives emotion probabilities
                  voice model gives emotion probabilities
Step 3 → Combine both (weighted fusion: face 60%, voice 40%)
Step 4 → Compare to the PREVIOUS reading
          Did you go from Happy → Angry? (shift)
          Did Stress probability jump by more than 12%? (threshold)
Step 5 → If yes: send notification
          If no: save reading and wait for next interval
```

This means **one bad moment doesn't trigger a false alarm** — the shift has to be real and sustained.

---

## Real-World Use Cases

### 1. 🏢 IT Company / Software Team
A company installs EmotionAI on developer machines. HR can see (with consent) aggregated trend data:
- *"Team stress spikes on sprint deadlines (Fridays)"*
- *"Developer 3 shows consistent fatigue indicators after 6 PM"*
- Result: Better scheduling, fewer burnouts, lower attrition

### 2. 🎓 Student in Hostel
A student studying for competitive exams has the app running:
- It detects growing frustration during a tough subject
- Plays their mapped "focus playlist" automatically
- After 30 minutes, sends a meme to lighten the mood
- Over time, shows them what hours they study best

### 3. 🏥 Remote Work Health Monitoring
With remote work becoming permanent post-COVID:
- Companies can monitor workforce well-being without intrusive surveillance
- Employees get *personal* support, not generic wellness emails
- Mental health becomes measurable, trackable, and actionable

### 4. 🔬 Research / Academic Use
Universities can use the system to study:
- How cognitive fatigue develops over a study session
- Correlation between emotional state and task performance
- Personalized intervention timing (when is music most effective?)

---

## Why This Is Different — Our 5 Strongest Points

### ✅ 1. It Works Without You Doing Anything
Every other solution asks you to open an app, fill a form, or remember to check in. EmotionAI just runs. You do your work. It does its job.

### ✅ 2. Multimodal = More Accurate
Using **both face and voice** together is significantly more reliable than either alone. A person can fake a calm expression but their voice trembles. The system catches both.

### ✅ 3. Privacy First
Your face, your voice, your emotional history — **nothing leaves your computer**. The SQLite database is on your machine. The only external call is for the AI text report (which sends summarised data, not raw recordings).

### ✅ 4. Personalised Interventions
You choose your music. You choose your intervals. You choose whether it asks or acts automatically. The system adapts to *your* preferences, not a generic template.

### ✅ 5. Actionable Insights, Not Just Data
Most wellness tools show you charts. EmotionAI tells you what they mean:
> *"You experience the most emotional fluctuation on Tuesdays between 2–5 PM. This pattern has appeared for 3 consecutive weeks."*

---

## How We Present This to Faculty

### Opening Line
> *"Every developer has had a day where they were clearly burning out — but nobody noticed. We built a system that notices."*

### Core Argument
> This is not an emotion tracker. It is a **real-time workplace wellness assistant** that intervenes at the right moment, using AI that is already proven (face recognition, speech emotion recognition), packaged in a way that actually works in the real world — silently, privately, and personally.

### What Makes It a Real System (Not Just a Demo)
- ✅ Runs as a **real desktop app** (Electron — installable on Windows)
- ✅ Has a **real database** (SQLite — data persists across sessions)
- ✅ Uses **real OS notifications** (Windows Action Center — not just browser popups)
- ✅ Has **real AI analysis** (Groq/LLaMA for trend reports)
- ✅ Settings **persist across restarts** (not reset every time)
- ✅ Works **completely offline** for all core functionality

### The Research Underneath
- Face emotion recognition: **MobileNetV2 + CNN** (transfer learning on FER-2013 dataset)
- Voice emotion recognition: **Librosa MFCC + CNN** (trained on RAVDESS / TESS dataset)
- Fusion: **Weighted probability average** (mathematical formula in `docs/FORMALE.md`)
- Trend detection: **Exponential Moving Average + slope detection** (prevents false positives)

---

## Summary Table — Our System vs Alternatives

| Feature | Existing Apps | EmotionAI |
|---|---|---|
| Requires user action to activate | ✅ Yes (every time) | ❌ No (runs automatically) |
| Real-time detection | ❌ No | ✅ Yes (every X minutes) |
| Personalised music/intervention | ❌ No | ✅ Yes (user-mapped) |
| Works offline / privately | ❌ Usually cloud-based | ✅ Fully local |
| Persistent historical data | ❌ Rarely | ✅ SQLite database |
| AI-generated insight report | ❌ No | ✅ Yes (LLM-powered) |
| Desktop native notifications | ❌ No | ✅ Yes (Windows Action Center) |
| Free / no subscription | ❌ Paid | ✅ Open, self-hosted |

---

## One Sentence for Each Section When Presenting

- **Problem:** "Software workers burn out silently, and no tool detects it in real-time."
- **Solution:** "We built an app that watches your face and voice every 15 minutes and steps in when it detects stress — with music or encouragement."
- **How it works:** "Face + voice analysis, fused with AI, stored locally, and surfaced as native OS notifications."
- **Real world:** "An IT company can use this to detect team-wide burnout trends before they become attrition."
- **Why us:** "Everything is offline, personalised, and requires zero effort from the user."

---

*All technical formulas: see `docs/FORMALE.md`*
*Full application flow: see `docs/EXPLANATION.MD`*
*Full API and code documentation: see `docs/documentation.md`*
