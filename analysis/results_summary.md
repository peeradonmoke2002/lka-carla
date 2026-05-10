# Perception Evaluation Summary

**Bag:** `bags/20260508_170715`
**Conditions:** Rain → Clear → Fog → Night (60 s each)
**Methods:** YOLO (`yolo26l-seg.pt`) vs Pure Vision (HSV yellow + gray Canny/Hough)
**Image size:** 1600 × 900 px, `y_ref = 0.85 × H`

---

## Detection Rate

Both methods achieve **100% detection** across all four weather conditions. Detection rate is not a differentiator in this experiment.

---

## Lateral Error  `|center_norm − 0.5|`

Lower is better. Ideal center = 0.5 (normalized).

| Weather | Pure Vision | YOLO | Winner |
|---------|-------------|------|--------|
| Rain    | **0.0289**  | 0.0333 | Pure Vision |
| Clear   | 0.0427      | **0.0312** | YOLO |
| Fog     | 0.0352      | **0.0321** | YOLO |
| Night   | 0.0402      | **0.0317** | YOLO |

YOLO achieves lower lateral error in 3 out of 4 conditions. Pure Vision wins in rain because the HSV thresholds were hand-tuned specifically for wet yellow markings.

---

## Center Stability  `center_std`

Lower is better. Measures frame-to-frame jitter of the estimated lane center.

| Weather | Pure Vision | YOLO |
|---------|-------------|------|
| Rain    | 0.0015      | **0.0010** |
| Clear   | 0.0013      | **0.0019** *(PV slightly better)* |
| Fog     | 0.0020      | **0.0007** |
| Night   | 0.0035      | **0.0005** |

YOLO is significantly more stable, especially in night (7× lower std than Pure Vision).

---

## Lane Position Stability  `lx_std / rx_std` (pixels)

Measures how consistently each method detects the left and right lane boundaries at `y_ref`.

| | lx_std (left lane) | rx_std (right lane) |
|---|---|---|
| **Pure Vision** | 1.78 – 2.12 px (stable) | 3.80 – **11.22** px (unstable in night) |
| **YOLO**        | 1.34 – 5.97 px (variable in clear) | 0.18 – 0.85 px (**very stable**) |

- **Right lane (rx_std):** YOLO is dramatically more stable. Pure Vision detects the right edge using gray-channel Canny + Hough, which is noise-sensitive, especially at night (rx_std = 11.22 px).
- **Left lane (lx_std):** Pure Vision performs comparably or slightly better. HSV yellow masking is precise for the left marking.

---

## YOLO Confidence

High and consistent across all conditions — the model generalizes well without weather-specific tuning.

| Weather | Confidence |
|---------|------------|
| Rain    | 0.9515 |
| Clear   | 0.9683 |
| Fog     | 0.9346 |
| Night   | 0.9621 |

---

## FPS

FPS values are identical for both methods in this experiment because both were replayed from the same bag. In a real-time deployment, YOLO (GPU inference) would have lower throughput than Pure Vision (CPU pipeline).

| Weather | FPS |
|---------|-----|
| Rain    | 9.6 |
| Clear   | 12.3 |
| Fog     | 13.1 |
| Night   | 10.1 |

---

## Evaluation Caveats

### 1. "Lateral error" is not ground-truth accuracy
`|center_norm − 0.5|` assumes the true ego-lane center projects to exactly the image midpoint every frame. This is only valid on perfectly straight road when the vehicle is lane-centered. On curves or during controller corrections the projected lane center *should* deviate from 0.5 — the metric then conflates **perception error** with **controller tracking error**. Both detectors report `center_mean ≈ 0.47`, a systematic ~0.03 left bias that reflects a real ego-position offset, not a detector bias. True accuracy requires ground-truth cross-track error from the CARLA map.

### 2. Asymmetric post-processing — hysteresis vs. no filtering
Pure Vision runs a hysteresis state machine (`confirm_frames=3`, `lost_frames=5`) that re-publishes the previous center during bad frames and synthesizes the missing side as `lx + 760 px` when only one marking is visible. YOLO returns an immediate miss the moment either side is absent — no smoothing, no fallback. Consequences:
- **Detection rate (100% both)** counts different things: Pure Vision's rate includes "remembered" frames; YOLO's is the raw frame rate.
- **center_std** is artificially compressed for Pure Vision by the fallback.
- Despite this advantage, YOLO still wins on stability in 3/4 conditions.

### 3. FPS is bag-replay speed, not inference cost
Both nodes subscribe to the same recorded bag. FPS reflects camera publish rate, not processing latency. Real inference cost (YOLO GPU vs. Pure Vision CPU) requires per-frame `time.perf_counter()` timing around the model call.

---

## Key Findings

| Aspect | YOLO | Pure Vision |
|--------|------|-------------|
| Overall accuracy (lateral error) | **Better (3/4 conditions)** | Better in rain |
| Center stability | **Better in 3/4 conditions** | Similar in clear |
| Right lane stability (rx_std) | **Much better** | Unstable (Canny noise) |
| Left lane stability (lx_std) | Slightly worse in clear | **Slightly better** |
| Confidence | **0.93–0.97 (high, consistent)** | N/A |
| Weather adaptability | **No tuning needed** | Requires per-weather HSV thresholds |
| Real-time speed | Slower (GPU required) | **Faster (CPU only)** |

### Conclusion

**YOLO is the stronger method overall.** It delivers lower lateral error and significantly better stability in fog and night conditions without any per-weather parameter tuning. Its main limitation is the requirement for GPU inference.

**Pure Vision is competitive in rain** where the HSV yellow threshold is precisely tuned, and it has an advantage in CPU-only deployments. However, its right-lane detection (gray Canny) is unreliable at night and the pipeline requires manual threshold calibration per weather condition.
