#!/usr/bin/python3
"""
Plot perception evaluation from raw_frames.csv and metrics.csv.
Each panel is saved as a separate PNG file.

Output per method (e.g. YOLO):
  eval_yolo_lane_center.png
  eval_yolo_lateral_error_time.png
  eval_yolo_lateral_error_dist.png
  eval_yolo_detection_rate.png
  eval_yolo_confidence.png        (YOLO only) / eval_yolo_fps.png (others)
  eval_yolo_lane_stability.png

Plus comparison plots:
  eval_compare_lane_center.png
  eval_compare_lateral_error_time.png
  eval_compare_lateral_error_dist.png
  eval_compare_detection_rate.png
  eval_compare_fps.png
  eval_compare_lane_stability.png

Usage:
    python3 analysis/plot_perception.py
    python3 analysis/plot_perception.py --data analysis/results/perception
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

WEATHER_ORDER = ['rain', 'clear', 'fog', 'night']
W_COLORS      = {'rain': '#4C9BE8', 'clear': '#F5A623', 'fog': '#9B9B9B', 'night': '#2C3E50'}
M_COLORS      = {'YOLO': '#E74C3C', 'Pure Vision': '#2ECC71', 'SCNN': '#9B59B6'}
BAR_WIDTH     = 0.25
W_IMAGE       = 1600
BG_COLORS     = {'rain': '#AED6F1', 'clear': '#FAD7A0', 'fog': '#D5DBDB', 'night': '#AEB6BF'}
W_TXT_CLR     = {'rain': '#1A5276', 'clear': '#784212', 'fog': '#424949', 'night': '#1C2833'}

TOPIC_MAP = {
    '/lka/yolo/lane_center':        'YOLO',
    '/lka/pure_vision/lane_center': 'Pure Vision',
    '/lka/scnn/lane_center':        'SCNN',
    '/lka/lane_center':             'shared',
}


def load(data_dir: Path):
    raw     = pd.read_csv(data_dir / 'raw_frames.csv')
    metrics = pd.read_csv(data_dir / 'metrics.csv')
    raw['method'] = raw['topic'].map(TOPIC_MAP).fillna(raw['topic'])
    return raw, metrics


def _shade_weather(ax, spans):
    for sp in spans:
        ax.axvspan(sp['start'], sp['end'], color=BG_COLORS[sp['weather']], alpha=0.25, zorder=0)
        mid = (sp['start'] + sp['end']) / 2
        ax.text(mid, 0.97, sp['weather'].upper(),
                ha='center', va='top', fontsize=10, fontweight='bold',
                color=W_TXT_CLR[sp['weather']], transform=ax.get_xaxis_transform(), zorder=5)


def _annotate_bar_values(ax, bars, vals, fmt, pad_ratio=0.03, inside_ratio=0.06):
    y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
    for bar, val in zip(bars, vals):
        if np.isnan(val):
            continue
        if val > 0.6 * y_top:
            y, va = val - inside_ratio * y_top, 'top'
        else:
            y, va = val + pad_ratio * y_top, 'bottom'
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                fmt.format(val), ha='center', va=va, fontsize=9)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ── Per-method panels ──────────────────────────────────────────────────────────

def _prep_method(raw, metrics, method):
    W = W_IMAGE
    r = raw[raw['method'] == method].copy().sort_values('timestamp_ns')
    r['t_sec'] = (r['timestamp_ns'] - r['timestamp_ns'].min()) / 1e9
    use_gt = 'err_gt' in r.columns and r['err_gt'].notna().any()
    r['err_px'] = r['err_gt'] * W if use_gt else (r['center'] - 0.5) * W
    m = metrics[metrics['method'] == method].set_index('weather').reindex(WEATHER_ORDER)
    spans = sorted(
        [{'weather': w,
          'start': r[r['weather'] == w]['t_sec'].min(),
          'end':   r[r['weather'] == w]['t_sec'].max()}
         for w in WEATHER_ORDER if not r[r['weather'] == w].empty],
        key=lambda s: s['start']
    )
    return r, m, spans, use_gt


def panel_lane_center(r, m, spans, method, out_path):
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle(f'{method} — Lane Center over Time', fontsize=12, fontweight='bold')
    det    = r[r['detected']]
    missed = r[~r['detected']]
    ax.scatter(det['t_sec'], det['center'], s=2, alpha=0.4, color='steelblue', zorder=2)
    if not missed.empty:
        ax.scatter(missed['t_sec'], [0.5] * len(missed), s=15, alpha=0.5,
                   color='red', marker='x', zorder=3, label='miss')
    ax.axhline(0.5, color='gray', lw=1.2, ls='--', label='ideal center = 0.5', zorder=1)
    ax.set_ylim(0.3, 0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Lane Center (normalized)')
    ax.legend(fontsize=8, loc='upper right')
    for sp in spans:
        w = sp['weather']
        if w in m.index and not np.isnan(m.loc[w, 'det_rate_%']):
            mid = (sp['start'] + sp['end']) / 2
            ax.text(mid, 0.31, f'det={m.loc[w,"det_rate_%"]:.0f}%',
                    ha='center', va='bottom', fontsize=8, color=W_TXT_CLR[w])
    _shade_weather(ax, spans)
    plt.tight_layout()
    _save(fig, out_path)


def panel_lateral_error_time(r, spans, use_gt, method, out_path):
    W = W_IMAGE
    fig, ax = plt.subplots(figsize=(14, 4))
    err_title = 'GT' if use_gt else 'Lane Center'
    fig.suptitle(f'{method} — Lateral Error over Time (ref={err_title})',
                 fontsize=12, fontweight='bold')
    det = r[r['detected']]
    legend_handles = []
    for w in WEATHER_ORDER:
        wd = det[det['weather'] == w]
        if wd.empty:
            continue
        ax.scatter(wd['t_sec'], wd['err_px'], s=2, alpha=0.35, color=W_COLORS[w], zorder=2)
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=W_COLORS[w],
                       markersize=6, label=f'{w}: μ={wd["err_px"].mean():.0f}px')
        )
    ax.axhline(0, color='gray', lw=1.2, ls='--', zorder=1)
    legend_handles.append(plt.Line2D([0], [0], color='gray', ls='--', label='ideal error = 0'))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Lateral Error (px,  W={W})')
    ax.legend(handles=legend_handles, fontsize=8, loc='lower right')
    _shade_weather(ax, spans)
    plt.tight_layout()
    _save(fig, out_path)


def panel_lateral_error_dist(r, use_gt, method, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    err_title = 'GT' if use_gt else 'Lane Center'
    fig.suptitle(f'{method} — Lateral Error Distribution per Weather\n(ref={err_title})',
                 fontsize=12, fontweight='bold')
    det = r[r['detected']]
    box_data, box_labels, box_colors = [], [], []
    for w in WEATHER_ORDER:
        wd = det[det['weather'] == w]
        if wd.empty:
            continue
        box_data.append(wd['err_px'].values)
        box_labels.append(w.capitalize())
        box_colors.append(W_COLORS[w])
    bp = ax.boxplot(box_data, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='.', markersize=2, alpha=0.3))
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.axhline(0, color='gray', lw=1.2, ls='--', label='ideal = 0 px', zorder=1)
    for i, (w, data) in enumerate(zip(WEATHER_ORDER, box_data)):
        total   = len(r[r['weather'] == w])
        det_pct = len(det[det['weather'] == w]) / total * 100 if total else 0
        med = float(np.median(data))
        std = float(np.std(data))
        ymin, _ = ax.get_ylim()
        ax.text(i + 1, ymin + abs(ymin) * 0.04,
                f'med={med:.0f}px\nstd={std:.1f}\ndet={det_pct:.0f}%',
                ha='center', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75))
    ax.set_xticklabels(box_labels)
    ax.set_ylabel('Lateral Error (px)')
    ax.set_title('box = IQR, line = median', fontsize=9, style='italic')
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    _save(fig, out_path)


def panel_detection_rate(m, method, out_path):
    weathers = WEATHER_ORDER
    colors   = [W_COLORS[w] for w in weathers]
    fig, ax  = plt.subplots(figsize=(7, 5))
    fig.suptitle(f'{method} — Detection Rate', fontsize=12, fontweight='bold')
    vals = m['det_rate_%'].values.astype(float)
    bars = ax.bar(weathers, vals, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylim(0, 115)
    ax.axhline(100, color='green', lw=1, ls='--', alpha=0.4)
    ax.set_ylabel('Detection Rate (%)')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    _annotate_bar_values(ax, bars, vals, '{:.0f}%')
    plt.tight_layout()
    _save(fig, out_path)


def panel_confidence_or_fps(r, m, method, out_path):
    weathers = WEATHER_ORDER
    colors   = [W_COLORS[w] for w in weathers]
    fig, ax  = plt.subplots(figsize=(7, 5))
    det = r[r['detected']]
    if method == 'YOLO':
        conf_vals = []
        for w in weathers:
            d = det[(det['weather'] == w) & (det['confidence'] > 0)]['confidence'].values
            conf_vals.append(float(np.mean(d)) if len(d) else np.nan)
        vals = np.array(conf_vals, dtype=float)
        bars = ax.bar(weathers, vals, color=colors, edgecolor='black', linewidth=0.8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Mean Confidence')
        fig.suptitle(f'{method} — Detection Confidence', fontsize=12, fontweight='bold')
        _annotate_bar_values(ax, bars, vals, '{:.3f}', pad_ratio=0.01, inside_ratio=0.03)
    else:
        vals = m['fps'].values.astype(float)
        bars = ax.bar(weathers, vals, color=colors, edgecolor='black', linewidth=0.8)
        ax.set_ylabel('FPS')
        fig.suptitle(f'{method} — Frames per Second', fontsize=12, fontweight='bold')
        _annotate_bar_values(ax, bars, vals, '{:.1f}', pad_ratio=0.02, inside_ratio=0.04)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, out_path)


def panel_lane_stability(m, method, out_path):
    weathers = WEATHER_ORDER
    x        = np.arange(len(weathers))
    w6       = 0.35
    fig, ax  = plt.subplots(figsize=(7, 5))
    fig.suptitle(f'{method} — Lane Stability (lx_std, rx_std)\nlower = more stable',
                 fontsize=12, fontweight='bold')
    lx_stds = m['lx_std'].values.astype(float)
    rx_stds = m['rx_std'].values.astype(float)
    bars_lx = ax.bar(x - w6 / 2, lx_stds, w6, label='lx_std',
                     color='#3498DB', edgecolor='black', linewidth=0.8)
    bars_rx = ax.bar(x + w6 / 2, rx_stds, w6, label='rx_std',
                     color='#E67E22', edgecolor='black', linewidth=0.8)
    _annotate_bar_values(ax, list(bars_lx) + list(bars_rx),
                         np.concatenate([lx_stds, rx_stds]), '{:.1f}')
    ax.set_xticks(x)
    ax.set_xticklabels(weathers)
    ax.set_ylabel('Std (pixels)')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out_path)


def plot_method_figure(raw, metrics, method, out_dir: Path):
    slug = method.lower().replace(' ', '_')
    r, m, spans, use_gt = _prep_method(raw, metrics, method)

    panel_lateral_error_time(r, spans, use_gt, method,
                             out_dir / f'eval_{slug}_lateral_error_time.png')
    panel_detection_rate(m, method,
                         out_dir / f'eval_{slug}_detection_rate.png')
    panel_confidence_or_fps(r, m, method,
                            out_dir / f'eval_{slug}_confidence.png' if method == 'YOLO'
                            else out_dir / f'eval_{slug}_fps.png')
    panel_lane_stability(m, method,
                         out_dir / f'eval_{slug}_lane_stability.png')


# ── Comparison panels ──────────────────────────────────────────────────────────

def _prep_compare(raw):
    W = W_IMAGE
    t0  = raw['timestamp_ns'].min()
    raw = raw.copy()
    raw['t_sec'] = (raw['timestamp_ns'] - t0) / 1e9
    use_gt = 'err_gt' in raw.columns and raw['err_gt'].notna().any()
    raw['err_px'] = raw['err_gt'] * W if use_gt else (raw['center'] - 0.5) * W
    ref = raw[raw['method'] == 'YOLO'].copy()
    spans = sorted(
        [{'weather': w,
          'start': ref[ref['weather'] == w]['t_sec'].min(),
          'end':   ref[ref['weather'] == w]['t_sec'].max()}
         for w in WEATHER_ORDER if not ref[ref['weather'] == w].empty],
        key=lambda s: s['start']
    )
    _order  = ['YOLO', 'Pure Vision', 'SCNN']
    methods = [m for m in _order if m in raw['method'].unique()]
    return raw, spans, use_gt, methods


def compare_lane_center(raw, spans, methods, out_path):
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle('Comparison — Lane Center over Time', fontsize=12, fontweight='bold')
    for method in methods:
        det = raw[(raw['method'] == method) & raw['detected']]
        ax.scatter(det['t_sec'], det['center'], s=2, alpha=0.35,
                   color=M_COLORS[method], label=method, zorder=2)
    ax.axhline(0.5, color='gray', lw=1.2, ls='--', label='ideal = 0.5', zorder=1)
    ax.set_ylim(0.3, 0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Lane Center (normalized)')
    ax.legend(fontsize=8, loc='upper right', markerscale=4)
    _shade_weather(ax, spans)
    plt.tight_layout()
    _save(fig, out_path)


def compare_lateral_error_time(raw, spans, use_gt, methods, out_path):
    W = W_IMAGE
    err_title = 'GT' if use_gt else 'Lane Center'
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle(f'Comparison — Lateral Error over Time (ref={err_title})',
                 fontsize=12, fontweight='bold')
    for method in methods:
        det = raw[(raw['method'] == method) & raw['detected']]
        ax.scatter(det['t_sec'], det['err_px'], s=2, alpha=0.30,
                   color=M_COLORS[method], zorder=2)
    ax.axhline(0, color='gray', lw=1.2, ls='--', zorder=1)
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=M_COLORS[m],
                   markersize=7, label=m)
        for m in methods
    ] + [plt.Line2D([0], [0], color='gray', ls='--', label='ideal = 0')]
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Lateral Error (px,  W={W})')
    ax.set_ylim(0, 28)
    ax.legend(handles=legend_handles, fontsize=8, loc='upper right')
    _shade_weather(ax, spans)
    plt.tight_layout()
    _save(fig, out_path)


def compare_lateral_error_dist(raw, use_gt, methods, out_path):
    err_title = 'GT' if use_gt else 'Lane Center'
    fig, ax   = plt.subplots(figsize=(10, 5))
    fig.suptitle(f'Comparison — Lateral Error Distribution (ref={err_title})',
                 fontsize=12, fontweight='bold')
    n_methods = len(methods)
    positions_all, colors_all, box_data_all = [], [], []
    tick_pos, tick_lbl = [], []
    for wi, w in enumerate(WEATHER_ORDER):
        group_center = wi * (n_methods + 1)
        tick_pos.append(group_center + (n_methods - 1) / 2)
        tick_lbl.append(w.capitalize())
        for mi, method in enumerate(methods):
            det = raw[(raw['method'] == method) & raw['detected'] & (raw['weather'] == w)]
            positions_all.append(group_center + mi)
            colors_all.append(M_COLORS[method])
            box_data_all.append(det['err_px'].values if not det.empty else np.array([]))
    bp = ax.boxplot(box_data_all, positions=positions_all, patch_artist=True, widths=0.7,
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='.', markersize=2, alpha=0.3))
    for patch, c in zip(bp['boxes'], colors_all):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.axhline(0, color='gray', lw=1.2, ls='--', label='ideal = 0 px', zorder=1)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl)
    ax.set_ylabel('Lateral Error (px)')
    ax.set_title('box = IQR, line = median', fontsize=9, style='italic')
    legend_handles = [plt.matplotlib.patches.Patch(facecolor=M_COLORS[m], label=m)
                      for m in methods]
    ax.legend(handles=legend_handles, fontsize=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    _save(fig, out_path)


def compare_detection_rate(metrics, methods, out_path):
    x   = np.arange(len(WEATHER_ORDER))
    n_m = len(methods)
    offsets = [((i - (n_m - 1) / 2) * BAR_WIDTH) for i in range(n_m)]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle('Comparison — Detection Rate', fontsize=12, fontweight='bold')
    for i, method in enumerate(methods):
        m    = metrics[metrics['method'] == method].set_index('weather').reindex(WEATHER_ORDER)
        vals = m['det_rate_%'].values.astype(float)
        bars = ax.bar(x + offsets[i], vals, BAR_WIDTH,
                      label=method, color=M_COLORS.get(method, '#888'),
                      edgecolor='black', linewidth=0.7)
        _annotate_bar_values(ax, bars, vals, '{:.0f}%', pad_ratio=0.02, inside_ratio=0.05)
    ax.set_xticks(x)
    ax.set_xticklabels(WEATHER_ORDER)
    ax.set_ylim(0, 115)
    ax.axhline(100, color='green', lw=1, ls='--', alpha=0.4)
    ax.set_ylabel('Detection Rate (%)')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out_path)


def compare_fps(metrics, methods, out_path):
    x   = np.arange(len(WEATHER_ORDER))
    n_m = len(methods)
    offsets = [((i - (n_m - 1) / 2) * BAR_WIDTH) for i in range(n_m)]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle('Comparison — Frames per Second', fontsize=12, fontweight='bold')
    for i, method in enumerate(methods):
        m    = metrics[metrics['method'] == method].set_index('weather').reindex(WEATHER_ORDER)
        vals = m['fps'].values.astype(float)
        bars = ax.bar(x + offsets[i], vals, BAR_WIDTH,
                      label=method, color=M_COLORS.get(method, '#888'),
                      edgecolor='black', linewidth=0.7)
        _annotate_bar_values(ax, bars, vals, '{:.1f}', pad_ratio=0.02, inside_ratio=0.04)
    ax.set_xticks(x)
    ax.set_xticklabels(WEATHER_ORDER)
    ax.set_ylabel('FPS')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out_path)


def compare_lane_stability(raw, methods, out_path):
    x     = np.arange(len(WEATHER_ORDER))
    n_m   = len(methods)
    bar_w = 0.7 / n_m
    base  = -(n_m - 1) / 2 * bar_w
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle('Comparison — Max Lane Center Jitter per Weather\nlower = more stable',
                 fontsize=12, fontweight='bold')
    for i, method in enumerate(methods):
        vals = np.array([
            raw[(raw['method'] == method) & raw['detected'] & (raw['weather'] == w)]
            .sort_values('timestamp_ns')['center']
            .diff().abs().max() * W_IMAGE
            if not raw[(raw['method'] == method) & raw['detected'] & (raw['weather'] == w)].empty
            else np.nan
            for w in WEATHER_ORDER
        ])
        bars = ax.bar(x + base + i * bar_w, vals, bar_w,
                      label=method, color=M_COLORS.get(method, '#888'),
                      edgecolor='black', linewidth=0.7)
        _annotate_bar_values(ax, bars, vals, '{:.1f}', pad_ratio=0.02, inside_ratio=0.05)
    ax.set_xticks(x)
    ax.set_xticklabels([w.capitalize() for w in WEATHER_ORDER])
    ax.set_ylabel('Max Center Jitter (px, frame-to-frame)')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out_path)


def compare_lateral_error_rmse(metrics, methods, out_path):
    x     = np.arange(len(WEATHER_ORDER))
    n_m   = len(methods)
    bar_w = 0.7 / n_m
    base  = -(n_m - 1) / 2 * bar_w
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle('Comparison — Lateral Error RMSE per Weather\nlower = more accurate',
                 fontsize=12, fontweight='bold')
    for i, method in enumerate(methods):
        m        = metrics[metrics['method'] == method].set_index('weather').reindex(WEATHER_ORDER)
        err_mean = m['err_mean'].values.astype(float)
        err_std  = m['err_std'].values.astype(float)
        vals     = np.sqrt(err_mean**2 + err_std**2) * W_IMAGE
        bars = ax.bar(x + base + i * bar_w, vals, bar_w,
                      label=method, color=M_COLORS.get(method, '#888'),
                      edgecolor='black', linewidth=0.7)
        _annotate_bar_values(ax, bars, vals, '{:.1f}', pad_ratio=0.02, inside_ratio=0.05)
    ax.set_xticks(x)
    ax.set_xticklabels([w.capitalize() for w in WEATHER_ORDER])
    ax.set_ylabel('Lateral Error RMSE (px)')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out_path)


def compare_lateral_error_max(raw, methods, out_path):
    x     = np.arange(len(WEATHER_ORDER))
    n_m   = len(methods)
    bar_w = 0.7 / n_m
    base  = -(n_m - 1) / 2 * bar_w
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle('Comparison — Max Lateral Error per Weather\nlower = better worst-case',
                 fontsize=12, fontweight='bold')
    for i, method in enumerate(methods):
        vals = np.array([
            raw[(raw['method'] == method) & raw['detected'] & (raw['weather'] == w)]['err_px'].abs().max()
            if not raw[(raw['method'] == method) & raw['detected'] & (raw['weather'] == w)].empty
            else np.nan
            for w in WEATHER_ORDER
        ])
        bars = ax.bar(x + base + i * bar_w, vals, bar_w,
                      label=method, color=M_COLORS.get(method, '#888'),
                      edgecolor='black', linewidth=0.7)
        _annotate_bar_values(ax, bars, vals, '{:.1f}', pad_ratio=0.02, inside_ratio=0.05)
    ax.set_xticks(x)
    ax.set_xticklabels([w.capitalize() for w in WEATHER_ORDER])
    ax.set_ylabel('Max Lateral Error (px)')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out_path)


def plot_compare_figures(raw, metrics, out_dir: Path):
    raw, spans, use_gt, methods = _prep_compare(raw)

    compare_lateral_error_time(raw, spans, use_gt, methods,
                               out_dir / 'eval_compare_lateral_error_time.png')
    compare_lateral_error_rmse(metrics, methods,
                               out_dir / 'eval_compare_lateral_error_rmse.png')
    compare_lateral_error_max(raw, methods,
                              out_dir / 'eval_compare_lateral_error_max.png')
    compare_detection_rate(metrics, methods,
                           out_dir / 'eval_compare_detection_rate.png')
    compare_fps(metrics, methods,
                out_dir / 'eval_compare_fps.png')
    compare_lane_stability(raw, methods,
                           out_dir / 'eval_compare_lane_stability.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/peeradon/lka-carla-yolo/analysis/results/perception')
    args = parser.parse_args()

    data_dir = Path(args.data)
    raw, metrics = load(data_dir)
    print(f'Loaded {len(raw)} frames, {len(metrics)} metric rows')
    print(metrics[['method', 'weather', 'det_rate_%', 'err_mean', 'fps']].to_string(index=False))

    slug_map = {'YOLO': 'yolo', 'Pure Vision': 'pure_vision', 'SCNN': 'scnn'}
    for method in ['YOLO', 'Pure Vision', 'SCNN']:
        if method in raw['method'].unique():
            sub = data_dir / slug_map[method]
            sub.mkdir(exist_ok=True)
            plot_method_figure(raw, metrics, method, sub)

    compare_dir = data_dir / 'compare'
    compare_dir.mkdir(exist_ok=True)
    plot_compare_figures(raw, metrics, compare_dir)


if __name__ == '__main__':
    main()
