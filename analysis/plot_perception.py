#!/usr/bin/python3
"""
Plot perception evaluation from raw_frames.csv and metrics.csv.
Generates 3 output files:
  - eval_yolo.png
  - eval_pure_vision.png
  - eval_compare.png  (side-by-side comparison)

Usage:
    python3 analysis/plot_perception.py
    python3 analysis/plot_perception.py --data analysis/results
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
BAR_WIDTH     = 0.35
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


# ── Per-method figure (time-series + summary bars) ─────────────

def plot_method_figure(raw, metrics, method, out_path):
    W = W_IMAGE
    r = raw[raw['method'] == method].copy().sort_values('timestamp_ns')
    r['t_sec']  = (r['timestamp_ns'] - r['timestamp_ns'].min()) / 1e9
    r['err_px'] = (r['center'] - 0.5) * W

    m = metrics[metrics['method'] == method].set_index('weather').reindex(WEATHER_ORDER)

    spans = sorted(
        [{'weather': w,
          'start': r[r['weather'] == w]['t_sec'].min(),
          'end':   r[r['weather'] == w]['t_sec'].max()}
         for w in WEATHER_ORDER if not r[r['weather'] == w].empty],
        key=lambda s: s['start']
    )

    det    = r[r['detected']]
    missed = r[~r['detected']]
    weathers = WEATHER_ORDER
    colors   = [W_COLORS[w] for w in weathers]

    fig = plt.figure(figsize=(14, 17))
    gs  = fig.add_gridspec(4, 3, height_ratios=[2, 2, 2, 1.8], hspace=0.50, wspace=0.35)
    ax_ts1  = fig.add_subplot(gs[0, :])
    ax_ts2  = fig.add_subplot(gs[1, :])
    ax_box  = fig.add_subplot(gs[2, :])
    ax_det  = fig.add_subplot(gs[3, 0])
    ax_c4   = fig.add_subplot(gs[3, 1])
    ax_stab = fig.add_subplot(gs[3, 2])

    fig.suptitle(f'{method} Lane Detection — Bag Analysis\n(4 weather conditions, 60 s each)',
                 fontsize=13, fontweight='bold')

    # ── Panel 1: center_norm over time ─────────────────────────
    ax = ax_ts1
    ax.scatter(det['t_sec'], det['center'], s=2, alpha=0.4, color='steelblue', zorder=2)
    if not missed.empty:
        ax.scatter(missed['t_sec'], [0.5] * len(missed), s=15, alpha=0.5,
                   color='red', marker='x', zorder=3, label='miss')
    ax.axhline(0.5, color='gray', lw=1.2, ls='--', label='ideal center = 0.5', zorder=1)
    ax.set_ylim(0.3, 0.7)
    ax.set_ylabel('Lane Center (normalized)')
    ax.set_title('Lane Center over Time  (x = miss)')
    ax.legend(fontsize=8, loc='upper right')
    for sp in spans:
        w = sp['weather']
        if w in m.index and not np.isnan(m.loc[w, 'det_rate_%']):
            mid = (sp['start'] + sp['end']) / 2
            ax.text(mid, 0.31, f'det={m.loc[w,"det_rate_%"]:.0f}%',
                    ha='center', va='bottom', fontsize=8, color=W_TXT_CLR[w])
    _shade_weather(ax, spans)

    # ── Panel 2: lateral error over time ───────────────────────
    ax = ax_ts2
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
    ax.set_ylabel(f'Lateral Error (px,  W={W})')
    ax.set_title(f'Lateral Error from Lane Center  (pixels, W={W})')
    ax.legend(handles=legend_handles, fontsize=8, loc='lower right')
    _shade_weather(ax, spans)

    # ── Panel 3: boxplot per weather ───────────────────────────
    ax = ax_box
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
        total    = len(r[r['weather'] == w])
        det_pct  = len(det[det['weather'] == w]) / total * 100 if total else 0
        med      = float(np.median(data))
        std      = float(np.std(data))
        ymin, _  = ax.get_ylim()
        ax.text(i + 1, ymin + abs(ymin) * 0.04,
                f'med={med:.0f}px\nstd={std:.1f}\ndet={det_pct:.0f}%',
                ha='center', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75))
    ax.set_xticklabels(box_labels)
    ax.set_ylabel('Lateral Error (px)')
    ax.set_title('Lateral Error Distribution per Weather  (box = IQR, line = median)')
    ax.legend(fontsize=8)

    # ── Panel 4: Detection rate ─────────────────────────────────
    ax = ax_det
    vals = m['det_rate_%'].values.astype(float)
    bars = ax.bar(weathers, vals, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylim(0, 115)
    ax.axhline(100, color='green', lw=1, ls='--', alpha=0.4)
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Detection Rate')
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
    ax.tick_params(axis='x', labelsize=8)

    # ── Panel 5: Confidence (YOLO only) or FPS (Pure Vision / SCNN) ────────
    ax = ax_c4
    if method == 'YOLO':
        conf_vals = []
        for w in weathers:
            d = det[(det['weather'] == w) & (det['confidence'] > 0)]['confidence'].values
            conf_vals.append(float(np.mean(d)) if len(d) else np.nan)
        bars = ax.bar(weathers, conf_vals, color=colors, edgecolor='black', linewidth=0.8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Mean Confidence')
        ax.set_title('Detection Confidence')
        for bar, val in zip(bars, conf_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        fps_vals = m['fps'].values.astype(float)
        bars = ax.bar(weathers, fps_vals, color=colors, edgecolor='black', linewidth=0.8)
        ax.set_ylabel('FPS')
        ax.set_title('Frames per Second')
        for bar, val in zip(bars, fps_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    ax.tick_params(axis='x', labelsize=8)

    # ── Panel 6: Lane position stability (lx_std, rx_std) ──────
    ax = ax_stab
    x6      = np.arange(len(weathers))
    lx_stds = m['lx_std'].values.astype(float)
    rx_stds = m['rx_std'].values.astype(float)
    w6      = 0.35
    bars_lx = ax.bar(x6 - w6 / 2, lx_stds, w6, label='lx_std',
                     color='#3498DB', edgecolor='black', linewidth=0.8)
    bars_rx = ax.bar(x6 + w6 / 2, rx_stds, w6, label='rx_std',
                     color='#E67E22', edgecolor='black', linewidth=0.8)
    for bar, val in zip(list(bars_lx) + list(bars_rx),
                        list(lx_stds) + list(rx_stds)):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x6)
    ax.set_xticklabels(weathers, fontsize=8)
    ax.set_ylabel('Std (pixels)')
    ax.set_title('Lane Stability  (lx_std, rx_std)\nlower = more stable')
    ax.legend(fontsize=8)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


# ── Comparison figure (time-series style) ─────────────────────

def plot_compare_figure(raw, metrics, out_path):
    W       = W_IMAGE
    _order  = ['YOLO', 'Pure Vision', 'SCNN']
    methods = [m for m in _order if m in raw['method'].unique()]

    # Align both methods to a common time axis (YOLO as reference)
    t0  = raw['timestamp_ns'].min()
    raw = raw.copy()
    raw['t_sec']  = (raw['timestamp_ns'] - t0) / 1e9
    raw['err_px'] = (raw['center'] - 0.5) * W

    # Weather spans from YOLO (both share same bag)
    ref = raw[raw['method'] == 'YOLO'].copy()
    spans = sorted(
        [{'weather': w,
          'start': ref[ref['weather'] == w]['t_sec'].min(),
          'end':   ref[ref['weather'] == w]['t_sec'].max()}
         for w in WEATHER_ORDER if not ref[ref['weather'] == w].empty],
        key=lambda s: s['start']
    )

    x = np.arange(len(WEATHER_ORDER))

    fig = plt.figure(figsize=(14, 17))
    gs  = fig.add_gridspec(4, 3, height_ratios=[2, 2, 2, 1.8], hspace=0.50, wspace=0.35)
    ax_ts1  = fig.add_subplot(gs[0, :])
    ax_ts2  = fig.add_subplot(gs[1, :])
    ax_box  = fig.add_subplot(gs[2, :])
    ax_det  = fig.add_subplot(gs[3, 0])
    ax_fps  = fig.add_subplot(gs[3, 1])
    ax_stab = fig.add_subplot(gs[3, 2])

    fig.suptitle('Perception Comparison — ' + ' vs '.join(methods) + '\n(4 weather conditions, 60 s each)',
                 fontsize=13, fontweight='bold')

    # ── Panel 1: center_norm over time (both methods) ──────────
    ax = ax_ts1
    for method in methods:
        det = raw[(raw['method'] == method) & raw['detected']]
        ax.scatter(det['t_sec'], det['center'], s=2, alpha=0.35,
                   color=M_COLORS[method], label=method, zorder=2)
    ax.axhline(0.5, color='gray', lw=1.2, ls='--', label='ideal = 0.5', zorder=1)
    ax.set_ylim(0.3, 0.7)
    ax.set_ylabel('Lane Center (normalized)')
    ax.set_title('Lane Center over Time')
    ax.legend(fontsize=8, loc='upper right', markerscale=4)
    _shade_weather(ax, spans)

    # ── Panel 2: lateral error over time (both methods) ────────
    ax = ax_ts2
    for method in methods:
        det = raw[(raw['method'] == method) & raw['detected']]
        for w in WEATHER_ORDER:
            wd = det[det['weather'] == w]
            if wd.empty:
                continue
            ax.scatter(wd['t_sec'], wd['err_px'], s=2, alpha=0.30,
                       color=M_COLORS[method], zorder=2)
    ax.axhline(0, color='gray', lw=1.2, ls='--', zorder=1)
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=M_COLORS[m],
                   markersize=7, label=m)
        for m in methods
    ] + [plt.Line2D([0], [0], color='gray', ls='--', label='ideal = 0')]
    ax.set_ylabel(f'Lateral Error (px,  W={W})')
    ax.set_title(f'Lateral Error over Time  (pixels, W={W})')
    ax.legend(handles=legend_handles, fontsize=8, loc='lower right')
    _shade_weather(ax, spans)

    # ── Panel 3: boxplot per weather — both methods side by side
    ax = ax_box
    n_methods = len(methods)
    positions_all = []
    labels_all    = []
    colors_all    = []
    box_data_all  = []
    tick_pos      = []
    tick_lbl      = []

    for wi, w in enumerate(WEATHER_ORDER):
        group_center = wi * (n_methods + 1)
        tick_pos.append(group_center + (n_methods - 1) / 2)
        tick_lbl.append(w.capitalize())
        for mi, method in enumerate(methods):
            det  = raw[(raw['method'] == method) & raw['detected'] & (raw['weather'] == w)]
            pos  = group_center + mi
            positions_all.append(pos)
            colors_all.append(M_COLORS[method])
            box_data_all.append(det['err_px'].values if not det.empty else np.array([]))
            labels_all.append(method)

    bp = ax.boxplot(box_data_all, positions=positions_all, patch_artist=True,
                    widths=0.7,
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='.', markersize=2, alpha=0.3))
    for patch, c in zip(bp['boxes'], colors_all):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.axhline(0, color='gray', lw=1.2, ls='--', label='ideal = 0 px', zorder=1)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl)
    ax.set_ylabel('Lateral Error (px)')
    ax.set_title('Lateral Error Distribution per Weather  (box = IQR, line = median)')
    legend_handles = [plt.matplotlib.patches.Patch(facecolor=M_COLORS[m], label=m)
                      for m in methods]
    ax.legend(handles=legend_handles, fontsize=8)

    n_m  = len(methods)
    bw   = BAR_WIDTH
    offsets_m = [((i - (n_m - 1) / 2) * bw) for i in range(n_m)]

    # ── Panel 4: Detection rate comparison ─────────────────────
    ax = ax_det
    for i, method in enumerate(methods):
        m    = metrics[metrics['method'] == method].set_index('weather').reindex(WEATHER_ORDER)
        vals = m['det_rate_%'].values.astype(float)
        bars = ax.bar(x + offsets_m[i], vals, bw,
                      label=method, color=M_COLORS.get(method, '#888'), edgecolor='black', linewidth=0.7)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(WEATHER_ORDER, fontsize=8)
    ax.set_ylim(0, 115)
    ax.axhline(100, color='green', lw=1, ls='--', alpha=0.4)
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Detection Rate')
    ax.legend(fontsize=7)

    # ── Panel 5: FPS comparison ─────────────────────────────────
    ax = ax_fps
    for i, method in enumerate(methods):
        m    = metrics[metrics['method'] == method].set_index('weather').reindex(WEATHER_ORDER)
        vals = m['fps'].values.astype(float)
        bars = ax.bar(x + offsets_m[i], vals, bw,
                      label=method, color=M_COLORS.get(method, '#888'), edgecolor='black', linewidth=0.7)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(WEATHER_ORDER, fontsize=8)
    ax.set_ylabel('FPS')
    ax.set_title('Frames per Second')
    ax.legend(fontsize=7)

    # ── Panel 6: lx_std / rx_std comparison (2 bars per method per weather)
    ax   = ax_stab
    w6   = bw / 2
    _lx_shades = {'YOLO': '#C0392B', 'Pure Vision': '#1E8449'}
    _rx_shades = {'YOLO': '#E74C3C', 'Pure Vision': '#2ECC71'}
    n_bars  = 2 * n_m
    bar_w6  = 0.8 / n_bars
    base_off = -(n_bars - 1) / 2 * bar_w6
    for i, method in enumerate(methods):
        off_lx = base_off + (2 * i) * bar_w6
        off_rx = base_off + (2 * i + 1) * bar_w6
        m = metrics[metrics['method'] == method].set_index('weather').reindex(WEATHER_ORDER)
        lc = _lx_shades.get(method, '#555')
        rc = _rx_shades.get(method, '#888')
        ax.bar(x + off_lx, m['lx_std'].values.astype(float), bar_w6,
               label=f'{method} lx', color=lc, edgecolor='black', linewidth=0.6)
        ax.bar(x + off_rx, m['rx_std'].values.astype(float), bar_w6,
               label=f'{method} rx', color=rc, edgecolor='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(WEATHER_ORDER, fontsize=8)
    ax.set_ylabel('Std (pixels)')
    ax.set_title('Lane Stability  (lx_std, rx_std)\nlower = more stable')
    ax.legend(fontsize=6, ncol=2)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/peeradon/lka-carla-yolo/analysis/results')
    args = parser.parse_args()

    data_dir = Path(args.data)
    raw, metrics = load(data_dir)
    print(f'Loaded {len(raw)} frames')
    print(metrics[['method', 'weather', 'det_rate_%', 'err_mean', 'fps']].to_string(index=False))

    _fname = {
        'YOLO':        'eval_yolo.png',
        'Pure Vision': 'eval_pure_vision.png',
        'SCNN':        'eval_scnn.png',
    }
    for method in raw['method'].unique():
        if method in _fname:
            plot_method_figure(raw, metrics, method, data_dir / _fname[method])
    plot_compare_figure(raw, metrics, data_dir / 'eval_compare.png')


if __name__ == '__main__':
    main()
