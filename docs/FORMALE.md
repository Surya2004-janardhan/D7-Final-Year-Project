# Core Detection Phase — Formal Specification

This document records the full, formal logic used during the core detection phase: how we compute percentages for emotions, how we detect increases/decreases in flow, how we determine transitions between emotions, and the exact decision rules (thresholds, formulas, and pseudocode) used to trigger actions such as playing a song.

This is written to be implementable directly from the math and pseudocode below.

---

## 1. Inputs and preprocessing

- Modalities considered (each produces a probability distribution over the same K emotion classes):
  - Audio model outputs: vector $\mathbf{p}^{(a)}_t \in \mathbb{R}^K$ (per time step/frame t)
  - Visual model outputs: vector $\mathbf{p}^{(v)}_t \in \mathbb{R}^K$
  - (Optional) Text/contextual outputs: $\mathbf{p}^{(x)}_t \in \mathbb{R}^K$

- Every model output is expected to be probabilities (softmax) over classes; if raw logits are provided, the pipeline applies softmax first:
  $$\mathbf{p}_t = \mathrm{softmax}(\mathbf{z}_t)\quad\text{where }\mathrm{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}.$$ 

- Frame rate / time units: pipeline processes frames at a fixed cadence (frames per second) or event timestamps. All time-based thresholds below are expressed in seconds and must be converted to frame counts: $N = \text{round}(f_{s}\cdot T)$ where $f_s$ is frames/sec and $T$ is seconds.

## 2. Fusion across modalities

- Weighted fusion of probability vectors (recommended):
  $$\mathbf{s}_t = w_a\,\mathbf{p}^{(a)}_t + w_v\,\mathbf{p}^{(v)}_t + w_x\,\mathbf{p}^{(x)}_t$$
  where weights are non-negative and sum to 1: $w_a+w_v+w_x=1$.

- Renormalize fused scores to probabilities (optional but recommended to maintain probabilistic interpretation):
  $$\mathbf{p}^{(f)}_t = \mathrm{softmax}(\mathbf{s}_t)$$

- Alternative (logit fusion): fuse logits and then softmax; useful when model confidences vary in scale.

## 3. Percentages and confidence

- For a given class $c$, the per-frame percentage is
  $$P_{c,t} = 100\times p^{(f)}_{c,t}$$
  where $p^{(f)}_{c,t}$ is the fused probability for class $c$ at time $t$.

- Use these percentages for human-readable displays and thresholds; internal comparisons and math use the probability in $[0,1]$.

## 4. Temporal smoothing and stability

- Exponential moving average (EMA) smoothing per-class (recommended to reduce noise):
  $$\hat p_{c,t} = \alpha\,p^{(f)}_{c,t} + (1-\alpha)\,\hat p_{c,t-1},\qquad 0<\alpha\le 1.$$ 
  - Typical value: $\alpha=0.25\text{--}0.5$ (lower = more smoothing). Use $\alpha$ corresponding to desired time constant: $\tau \approx \frac{1}{\alpha}$ frames.

- Rolling-window mean and std (for anomaly/variance checks): for window length $N$ frames,
  $$\mu_{c,t}^{(N)} = \frac{1}{N}\sum_{i=t-N+1}^{t} p^{(f)}_{c,i}$$
  $$\sigma_{c,t}^{(N)} = \sqrt{\frac{1}{N}\sum_{i=t-N+1}^{t}\left(p^{(f)}_{c,i}-\mu_{c,t}^{(N)}\right)^2}.$$ 

## 5. Trend (flow) detection — increasing vs decreasing

We consider two complementary methods for detecting increasing/decreasing trends in an emotion's probability.

- (A) Finite difference (simple):
  $$\Delta_{c,t}^{(T)} = p^{(f)}_{c,t} - p^{(f)}_{c,t-T}.$$ 
  If $\Delta_{c,t}^{(T)} > \delta_{up}$ → increasing; if $\Delta_{c,t}^{(T)} < -\delta_{down}$ → decreasing.
  - Typical window $T$: 1–3 seconds (converted to frames). Typical thresholds: $\delta_{up}=0.05\text{--}0.12$ (5%–12%), $\delta_{down}=0.05$.

- (B) Linear regression slope over a window (more robust to jitter): compute slope $m_{c,t}$ via least-squares on the past $N$ frames (time index normalized). For sample times $x_i=i$ and values $y_i=p^{(f)}_{c,t-N+i}$,
  $$m_{c,t}=\frac{\sum_i (x_i-\bar x)(y_i-\bar y)}{\sum_i (x_i-\bar x)^2}.$$ 
  Interpret $m_{c,t}$ as probability change per frame; convert to per-second by multiplying by $f_s$ if needed.
  - Thresholds: slope threshold $m_{thresh}$ corresponds to a fractional change per second (e.g., 0.02 per second = 2%/s). Recommended: $m_{thresh}=0.02\text{--}0.05$.

- Combine both: require both $\Delta$ and $m$ conditions to hold for robust detection.

## 6. Hysteresis and state stability (avoid flapping)

- Use separate enter/exit thresholds (hysteresis):
  - Enter (start considering emotion dominant): $p > p_{enter}$
  - Exit (stop considering it dominant): $p < p_{exit}$
  with $p_{enter} > p_{exit}$ (example: 0.60 vs 0.50).

- Minimum sustained duration: once a candidate event is detected, require it to be sustained for $T_{sustain}$ seconds before taking action (e.g., $T_{sustain}=2$–4s).

## 7. Emotion transition detection (A → B)

- Define primary emotion at time $t$ as the argmax of smoothed probabilities: $e_t = \arg\max_c \hat p_{c,t}$.

- A transition from emotion $A$ to emotion $B$ is confirmed if all the following hold:
  1. $\hat p_{B,t} - \hat p_{A,t} > \Delta_{margin}$ (absolute margin threshold).
  2. $\hat p_{B,t} > p_{min}$ (absolute confidence floor, e.g., 0.45–0.60).
  3. The increase trend for $B$ is positive: $\Delta_{B,t}^{(T)}>\delta_{up}$ and/or $m_{B,t}>m_{thresh}$.
  4. Sustained for at least $T_{sustain}$ seconds.

- Recommended defaults:
  - $\Delta_{margin}=0.12$ (12%)
  - $p_{min}=0.55$ (55%)
  - $T_{sustain}=2.0$ seconds

## 8. Decision rule to play a song (mapping transitions to actions)

- Preconditions to allow playing music:
  - No song currently playing OR `allow_replacement` is true.
  - Last-play cooldown elapsed: ensure at least $T_{cooldown}$ seconds since last play (recommended $T_{cooldown}=30$ s).
  - User/global preference allows auto-play.

- Trigger rule (pseudocode):

```
If transition_detected(A->B):
    if (not is_playing) and (time_since_last_play > T_cooldown) and (p_B > play_confidence):
        pause_daemon()
        play_song(mapped_song_for(B))
        schedule_meme_notification_after( T_meme_delay )
```

- Concrete boolean rule expressed mathematically:
  Play if
  $$\hat p_{B,t} > p_{play} \quad\text{and}\quad (\hat p_{B,t}-\hat p_{A,t})>\Delta_{margin} \quad\text{and}\quad cooldown\_passed$$
  where $p_{play}$ is a play-confidence threshold (e.g., 0.60).

## 9. Cooldowns, priority and tie-breaking

- Cooldown: do not start another song until $T_{cooldown}$ elapsed since last play.

- Priority: if multiple transitions happen simultaneously, use one of:
  - Highest absolute confidence $\hat p$ wins;
  - Highest recent momentum (slope $m$) wins;
  - Weighted score combining both: $score_c = \lambda\,\hat p_{c,t} + (1-\lambda)\,\text{normalize}(m_{c,t})$.

- Tie-breaker: small random jitter or prefer the previously dominant emotion to avoid oscillation.

## 10. Meme notification scheduling (follow-up notification)

- Separate the initial Play/No notification (ask) and a later meme-only notification. The renderer schedules the meme-only notification after a fixed delay $T_{meme}$ (e.g., 20 s) measured from the moment playback starts.

- Cancel the meme notification if playback stopped early.

## 11. Probability fusion detail (for implementers)

- If combining logits from models with different calibration, either:
  - Calibrate each model (temperature scaling) before averaging probabilities, or
  - Convert to log-odds and average: $\ell_{c,t}=\log\frac{p_{c,t}}{1-p_{c,t}}$, then weighted sum and inverse logistic.

## 12. Numerical thresholds (suggested starting defaults)

- EMA alpha: $\alpha=0.30$
- Rolling window: $N = f_s \times 2.0$ seconds (2s window)
- Delta threshold: $\delta_{up}=0.08$ (8%)
- Slope threshold: $m_{thresh}=0.02$ (2% per second)
- Enter/Exit thresholds: $p_{enter}=0.60$, $p_{exit}=0.50$
- Transition margin: $\Delta_{margin}=0.12$
- Play confidence: $p_{play}=0.60$
- Minimum confidence: $p_{min}=0.45$
- Sustain duration: $T_{sustain}=2.0$ s
- Cooldown: $T_{cooldown}=30$ s
- Meme delay: $T_{meme}=20$ s

These are starting points; tune per dataset and user preference via experiments.

## 13. Edge cases and safety checks

- If no emotion exceeds $p_{enter}$, do not trigger transitions.
- If top two emotions are within $\epsilon$ (e.g., 0.05), consider the state ambiguous—do not auto-play.
- If microphone/camera model confidence is low (high $\sigma$), fall back to more conservative thresholds.

## 14. Example scenario

- Given $f_s=10$ fps, $N=20$ (2s). At time $t_0$ the primary emotion is `neutral` with $\hat p_{neutral}=0.42$. Over the next 3s the fused smoothed `happy` probability grows to $\hat p_{happy}=0.67` and $\hat p_{happy}-\hat p_{neutral}=0.25>0.12$ and slope $m_{happy}=0.04>0.02$ and sustained $>T_{sustain}$. Then the transition `neutral->happy` is confirmed and if cooldown passed and user allows, a `happy`-mapped song will play.

## 15. Pseudocode (full)

```
for each incoming frame t:
    compute per-modality probs p^{(a)}, p^{(v)}, p^{(x)}
    fuse -> p^{(f)}_t
    update EMA: hat_p_t = alpha * p^{(f)}_t + (1-alpha) * hat_p_{t-1}
    compute rolling stats mu, sigma over N
    compute delta = p^{(f)}_t - p^{(f)}_{t-T}
    compute slope m via linear regression over last N samples

    determine candidate primary e_t = argmax_c hat_p_{c,t}
    if (hat_p_{e_t,t} > p_enter) and (hat_p_{e_t,t}-hat_p_{prev,t} > Delta_margin) and (sustained > T_sustain) and (m_{e_t} > m_thresh) and (cooldown passed):
        pause background daemon
        trigger_play(song_map[e_t])
        schedule_meme(T_meme)

    else:
        continue monitoring
```

## 16. Implementation notes

- Keep timestamps and sample counts consistent across modalities (align to a common frame clock).
- Always prefer smoothed probabilities for decision logic (use EMA or windowed mean) rather than raw model outputs.
- Expose thresholds and weights in configuration to allow data-driven tuning.

---

If you want, I can:
- Add a concrete code snippet (JS/Python) implementing EMA, slope calculation, and the transition rule for direct paste into `useDaemon.js` or `App.jsx`.
- Run a small calibration routine (grid-search over thresholds) on example recordings and produce suggested tuned thresholds.

End of specification.
