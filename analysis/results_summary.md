# Perception Evaluation Summary

**Bag:** `bags/20260511_141516`
**Conditions:** Rain → Clear → Fog → Night (60 s each), stationary ego vehicle
**Methods:** YOLO (`yolo26l-seg.pt`) · Pure Vision (HSV yellow + gray Canny/Hough) · SCNN (ResNet-18)
**Image size:** 1600 × 900 px · `y_ref = 0.85 × H = 765 px`
**GT formula:** `true_center = 0.5 − cte_m × (760 / (4.0 × 1600))`
**Node mode:** `enable_hysteresis: true` (hysteresis + single-side synthesis enabled)

---

## Detection Rate

| Weather | YOLO | SCNN | Pure Vision |
|---------|------|------|-------------|
| Rain    | 100% | 100% | 100%        |
| Clear   | 100% | 100% | 100%        |
| Fog     | 100% | 100% | **99.87%** ← 1 miss |
| Night   | 100% | 100% | 100%        |

All methods achieve near-100% detection. Pure Vision missed 1 frame in fog — the only non-perfect result across all conditions.

---

## Lateral Error `|center_norm − true_center|`

Lower is better. Vehicle is stationary so CTE ≈ 0, making this equivalent to `|center_norm − 0.5|`.

| Weather | Pure Vision | YOLO   | SCNN   | Winner |
|---------|-------------|--------|--------|--------|
| Rain    | **0.0088**  | 0.0098 | 0.0118 | Pure Vision |
| Clear   | **0.0068**  | 0.0103 | 0.0117 | Pure Vision |
| Fog     | **0.0110**  | 0.0109 | 0.0127 | YOLO *(by 0.0001)* |
| Night   | **0.0069**  | 0.0100 | 0.0122 | Pure Vision |

> ⚠️ **Caveat:** Pure Vision's lower error reflects an ROI/calibration offset, not true accuracy. PV's center estimate consistently sits closer to 0.5 due to its ROI polygon geometry. This advantage disappears once the vehicle moves and CTE becomes non-zero.

---

## Center Stability `center_diff_std` (frame-to-frame jitter)

Lower is better.

| Weather | YOLO       | SCNN       | Pure Vision |
|---------|------------|------------|-------------|
| Rain    | **0.0001** | 0.0002     | 0.0022      |
| Clear   | **0.0000** | **0.0000** | 0.0009      |
| Fog     | **0.0001** | **0.0000** | 0.0016      |
| Night   | **0.0001** | **0.0001** | 0.0060      |

YOLO and SCNN are dramatically more stable. At night, PV jitter is **60× larger** than YOLO's.

---

## Right Lane Stability `rx_std` (pixels)

| Weather | YOLO       | SCNN | Pure Vision |
|---------|------------|------|-------------|
| Rain    | **0.12**   | 0.28 | 5.63        |
| Clear   | **0.06**   | 0.07 | 1.98        |
| Fog     | **0.13**   | 0.16 | 3.66        |
| Night   | **0.20**   | 0.23 | 13.66 ❌   |

PV uses gray-channel Canny for the right edge. At night, its right-lane jitter is **68× larger** than YOLO's.

---

## Left Lane Stability `lx_std` (pixels)

| Weather | YOLO       | SCNN       | Pure Vision |
|---------|------------|------------|-------------|
| Rain    | **0.22**   | 0.37       | 0.66        |
| Clear   | **0.17**   | 0.05       | 0.61        |
| Fog     | **0.15**   | 0.06       | 0.70        |
| Night   | **0.14**   | 0.08       | 0.84        |

All three detect the left (yellow) marking reasonably. YOLO and SCNN outperform PV at night.

---

## YOLO Confidence

| Rain   | Clear  | Fog    | Night  |
|--------|--------|--------|--------|
| 0.9625 | 0.9681 | 0.9495 | 0.9620 |

High and consistent across all conditions with no per-weather tuning.

---

## FPS

FPS reflects bag replay rate, not real inference cost.

| Rain | Clear | Fog  | Night |
|------|-------|------|-------|
| 11.7 | 12.0  | 12.4 | 9.0   |

---

## Evaluation Caveats

### 1. Lateral error is not ground-truth accuracy
With a stationary vehicle, CTE ≈ constant, so the metric reduces to `|center − 0.5|`. PV's lower error is a calibration artefact from its ROI polygon, not a sign of superior detection. True accuracy requires a moving vehicle with varying CTE.

### 2. enable_hysteresis = true (consistent baseline)
All three nodes ran with `enable_hysteresis: true` — same hysteresis state machine (confirm × 3, tolerate × 5) and same single-side synthesis (± 760 px) applied equally across all methods. This setting will also be used for the controller test, ensuring perception eval and controller eval share the same baseline with no bias between phases.

### 3. FPS = bag replay rate
Real inference latency requires live `time.perf_counter()` measurements around the model forward pass.

---

## Key Findings

| Aspect | YOLO | SCNN | Pure Vision |
|--------|------|------|-------------|
| Lateral error | Middle | Worst | **Best** *(ROI artefact)* |
| Center stability | **Best** | **≈ Best** | Worst (60× YOLO at night) |
| Right lane stability | **Best** | ≈ Best | Worst (68× YOLO at night) |
| Left lane stability | **≈ Best** | **≈ Best** | Worst |
| Confidence | **0.95–0.97** | N/A | N/A |
| Weather adaptability | **No tuning needed** | **No tuning needed** | Per-weather HSV required |
| GPU required | Yes | Yes | **No (CPU only)** |

### Conclusion

**YOLO is the most stable method overall.** It delivers the lowest frame-to-frame jitter in all conditions, the most stable right-lane detection (especially at night), and high consistent confidence without any per-weather tuning.

**SCNN is competitive with YOLO** on stability in clear and fog conditions but is slightly worse in rain and night. It requires a more involved training pipeline.

**Pure Vision appears best on lateral error but this is misleading** — the advantage is a calibration offset that disappears once the vehicle moves. Its right-lane detection (gray Canny) is unreliable at night (rx_std = 13.66 vs YOLO 0.20) and requires manual HSV threshold calibration per weather. High jitter will translate directly into noisy steering when connected to the Pure Pursuit controller.
