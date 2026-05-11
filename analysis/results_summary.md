# LKA Evaluation Results Summary

**Project:** Vision-Based Lane Keeping Assist — KMUTT FIBO Machine Vision
**Methods:** YOLO (`yolo26l-seg.pt`) · SCNN (ResNet-18) · Pure Vision (HSV + Canny/Hough)
**Simulator:** CARLA Town01 · 4 weather conditions: Rain, Clear, Fog, Night

---

## Overall Verdict

| Rank | Method | Strength |
|------|--------|----------|
| 🥇 1st | **SCNN** | Most consistent CTE, smoothest steering, lowest drop rate |
| 🥈 2nd | **YOLO** | Best in clear weather, high confidence, no tuning needed |
| 🥉 3rd | **Pure Vision** | Competitive in fog, but unreliable at night, needs per-weather tuning |

> All 3 methods kept the vehicle **within the lane (CTE < 10 cm) 100% of the time** across all weather conditions and all 36 closed-loop trials.

---

---

## Phase 3 — Perception Evaluation (Stationary)

**Bag:** `bags/20260511_141516` · Vehicle stationary · 60 s per weather condition
**Setup:** `enable_hysteresis: true` · Image 1600×900 px · ROI polygon applied

### Detection Rate

| Weather | YOLO | SCNN | Pure Vision |
|---------|:----:|:----:|:-----------:|
| Rain    | 100% | 100% | 100%        |
| Clear   | 100% | 100% | 100%        |
| Fog     | 100% | 100% | 99.87% ⚠️  |
| Night   | 100% | 100% | 100%        |

Pure Vision missed 1 frame in fog — the only non-perfect result.

---

### Lateral Error `|center − true_center|` — lower = better

> ⚠️ Vehicle is stationary so this measures calibration offset, not dynamic accuracy.
> Pure Vision's apparent advantage disappears once the vehicle moves.

| Weather | YOLO   | SCNN   | Pure Vision |
|---------|:------:|:------:|:-----------:|
| Rain    | 0.0098 | 0.0118 | **0.0088**  |
| Clear   | 0.0103 | 0.0117 | **0.0068**  |
| Fog     | **0.0109** | 0.0127 | 0.0110  |
| Night   | 0.0100 | 0.0122 | **0.0069**  |

---

### Center Stability `center_diff_std` — lower = better

Frame-to-frame jitter. More important than lateral error for controller performance.

| Weather | YOLO       | SCNN       | Pure Vision |
|---------|:----------:|:----------:|:-----------:|
| Rain    | **0.0001** | 0.0002     | 0.0022      |
| Clear   | **0.0000** | **0.0000** | 0.0009      |
| Fog     | **0.0001** | **0.0000** | 0.0016      |
| Night   | **0.0001** | **0.0001** | 0.0060 ❌  |

Pure Vision night jitter is **60× larger** than YOLO. Right-lane stability (rx_std) is even worse: **68× at night** (13.66 vs 0.20 px).

---

### YOLO Confidence

| Rain   | Clear  | Fog    | Night  |
|:------:|:------:|:------:|:------:|
| 0.9625 | 0.9681 | 0.9495 | 0.9620 |

High and consistent — no per-weather tuning required.

---

### Phase 3 Summary

| Aspect | YOLO | SCNN | Pure Vision |
|--------|:----:|:----:|:-----------:|
| Detection rate | ✅ 100% | ✅ 100% | ⚠️ 99.87% |
| Center stability | ✅ Best | ✅ ≈ Best | ❌ Worst |
| Right-lane stability | ✅ Best | ✅ ≈ Best | ❌ 68× worse at night |
| Lateral error (stationary) | 2nd | 3rd | 1st *(ROI artefact)* |
| Weather tuning needed | ❌ None | ❌ None | ✅ Per-weather HSV |
| GPU required | Yes | Yes | No (CPU only) |

---

---

## Phase 4 — Closed-Loop Controller Evaluation

**Bags:** `bags/closed_loop/` · 36 trials (3 methods × 4 weathers × **3 repeats**)
**Controller:** Pure Pursuit · `wheel_base=3.0046 m` · `ld_k=2.4` · `throttle=0.3`
**Route:** Town01 · Spawn x=317 → Stop x=108 (~209 m)

**Fairness measures:**
- Live bias calibration at spawn before every trial (`bias = mean(center) − 0.5` subtracted from controller)
- 3 repeats → results reported as **mean ± std**
- Spawn verified identical: all 36 trials start at `cte_m = 0.00023 m`

---

### Bias Offset (Calibration)

Bias is a **model geometry artefact** — constant across weather for neural methods.

| Method | Rain | Clear | Fog | Night |
|--------|:----:|:-----:|:---:|:-----:|
| YOLO | 0.0104 | 0.0104 | 0.0105 | 0.0115 |
| SCNN | 0.0121 | 0.0118 | 0.0117 | 0.0127 |
| Pure Vision | 0.0066 | 0.0090 | 0.0085 | 0.0108 |

YOLO/SCNN std < 0.001 across repeats. Pure Vision varies more due to weather-adaptive HSV thresholds.

---

### CTE RMSE — mean ± std (m) — lower = better

| Method | Rain | Clear | Fog | Night |
|--------|:----:|:-----:|:---:|:-----:|
| YOLO | 0.022 ± 0.002 | **0.013 ± 0.009** | 0.019 ± 0.008 | 0.020 ± 0.014 |
| SCNN | **0.016 ± 0.007** | 0.024 ± 0.001 | **0.017 ± 0.002** | **0.014 ± 0.008** |
| Pure Vision | 0.030 ± 0.001 | 0.034 ± 0.011 | 0.017 ± 0.011 | 0.040 ± 0.005 |

- **YOLO** best in clear (0.013 m — lowest single result overall)
- **SCNN** best in rain, fog, night — lowest variance (most repeatable)
- **Pure Vision** worst in clear and night; fog competitive but inconsistent (±0.011)

---

### Steering Jitter — mean ± std — lower = smoother

| Method | Rain | Clear | Fog | Night |
|--------|:----:|:-----:|:---:|:-----:|
| YOLO | 0.011 ± 0.004 | 0.014 ± 0.009 | 0.008 ± 0.007 | 0.008 ± 0.003 |
| **SCNN** | **0.006 ± 0.001** | **0.005 ± 0.000** | **0.003 ± 0.001** | **0.006 ± 0.001** |
| Pure Vision | 0.021 ± 0.002 | 0.007 ± 0.000 | 0.015 ± 0.003 | 0.048 ± 0.012 |

Pure Vision night jitter **(0.048) = 8× SCNN** — directly caused by noisy right-lane detection.

---

### Perception Drop Rate (during driving) — mean % — lower = better

| Method | Rain | Clear | Fog | Night |
|--------|:----:|:-----:|:---:|:-----:|
| YOLO | 1.29 | 1.29 | 0.85 | 1.23 |
| **SCNN** | 0.96 | **0.26** | **0.15** | 1.08 |
| Pure Vision | 1.83 | 1.72 | 1.97 | **0.37** |

> ⚠️ Pure Vision drops least at night but has the **highest night CTE** (0.040 m) — it detects frequently but inaccurately.

---

### Off-Lane Rate `|CTE| > 0.10 m`

**0.00% — all methods, all weathers, all 36 trials.**

---

### Route Consistency (Fairness Check)

| Check | Result |
|-------|--------|
| Start CTE (all 36 trials) | 0.00023 m — identical ✅ |
| Mean speed | 4.59–4.72 m/s (throttle constant) ✅ |
| Path shape | X-Y paths overlap exactly across methods ✅ |

---

### Phase 4 Summary

| Aspect | YOLO | SCNN | Pure Vision |
|--------|:----:|:----:|:-----------:|
| CTE RMSE rank | 2nd | **1st** | 3rd |
| Best weather | Clear | Rain / Fog / Night | — |
| Steering smoothness | 2nd | **Best** | Worst |
| Perception drop rate | 2nd | **Best** | Worst |
| Result consistency | Medium | **Best** | Worst |
| Off-lane events | None ✅ | None ✅ | None ✅ |

---

### Final Conclusion

**SCNN wins closed-loop.** Lowest CTE in 3/4 weathers, smoothest steering, fewest detection drops. Bias is a fixed geometric offset that the calibration step fully corrects.

**YOLO is the best for clear-weather** and a strong all-rounder. Higher variance in night suggests occasional detection instability, but overall performance is reliable.

**Pure Vision is not recommended for closed-loop control.** Night steering jitter (8× SCNN), highest drop rate in most conditions, and weather-dependent bias make it the least reliable option despite requiring no GPU.
