#!/usr/bin/python3
"""
Plot perception evaluation from raw_frames.csv and metrics.csv.
Generates 3 output files:
  - eval_yolo.png
  - eval_pure_vision.png
  - eval_compare.png  (side-by-side comparison, no confidence)

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
M_COLORS      = {'YOLO': '#E74C3C', 'Pure Vision': '#2ECC71'}
BAR_WIDTH     = 0.35

TOPIC_MAP = {
    '/lka/yolo/lane_center':        'YOLO',
    '/lka/pure_vision/lane_center': 'Pure Vision',
    '/lka/lane_center':             'shared',
}


def load(data_dir: Path):
    raw     = pd.read_csv(data_dir / 'raw_frames.csv')
    metrics = pd.read_csv(data_dir / 'metrics.csv')
    raw['method'] = raw['topic'].map(TOPIC_MAP).fillna(raw['topic'])
    return raw, metrics


# ── Per-method figure ──────────────────────────────────────────

def plot_method_figure(raw, metrics, method, out_path):
    color = M_COLORS[method]
    m     = metrics[metrics['method'] == method].set_index('weather').reindex(WEATHER_ORDER)
    r     = raw[(raw['method'] == method)]

    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    fig.suptitle(f'Perception Evaluation — {method}', fontsize=14, fontweight='bold')

    weathers = WEATHER_ORDER
    colors   = [W_COLORS[w] for w in weathers]

    # Panel 1 — Detection rate
    ax = axes[0]
    vals = m['det_rate_%'].values.astype(float)
    bars = ax.bar(weathers, vals, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylim(0, 115)
    ax.axhline(100, color='green', lw=1, ls='--', alpha=0.4)
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Detection Rate')
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=10)

    # Panel 2 — Lateral error
    ax = axes[1]
    means = m['err_mean'].values.astype(float)
    stds  = m['err_std'].values.astype(float)
    ax.bar(weathers, means, yerr=stds, color=colors,
           edgecolor='black', linewidth=0.8, capsize=5)
    for i, (mn, sd) in enumerate(zip(means, stds)):
        ax.text(i, mn + sd + 0.001, f'{mn:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('|center_norm − 0.5|')
    ax.set_title('Lateral Error  (mean ± std)\nlower = better')

    # Panel 3 — center_norm boxplot
    ax = axes[2]
    data = [r[(r['weather'] == w) & r['detected']]['center'].values for w in weathers]
    bp   = ax.boxplot(data, patch_artist=True,
                      medianprops=dict(color='black', linewidth=2))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
    ax.axhline(0.5, color='green', lw=1.5, ls='--', label='ideal (0.5)')
    ax.set_xticklabels(weathers)
    ax.set_ylabel('center_norm')
    ax.set_title('center_norm Distribution')
    ax.legend(fontsize=8)

    # Panel 4 — confidence (YOLO only) or FPS
    ax = axes[3]
    if method == 'YOLO':
        conf_means = []
        for w in weathers:
            d = r[(r['weather'] == w) & r['detected'] & (r['confidence'] > 0)]['confidence'].values
            conf_means.append(float(np.mean(d)) if len(d) else np.nan)
        bars = ax.bar(weathers, conf_means, color=colors, edgecolor='black', linewidth=0.8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Mean Confidence')
        ax.set_title('Detection Confidence')
        for bar, val in zip(bars, conf_means):
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

    # Panel 5 — Lane position stability (lx_std, rx_std)
    ax = axes[4]
    x5 = np.arange(len(weathers))
    lx_stds = m['lx_std'].values.astype(float)
    rx_stds = m['rx_std'].values.astype(float)
    w5 = 0.35
    bars_lx = ax.bar(x5 - w5 / 2, lx_stds, w5, label='lx_std',
                     color='#3498DB', edgecolor='black', linewidth=0.8)
    bars_rx = ax.bar(x5 + w5 / 2, rx_stds, w5, label='rx_std',
                     color='#E67E22', edgecolor='black', linewidth=0.8)
    for bar, val in zip(list(bars_lx) + list(bars_rx),
                        list(lx_stds) + list(rx_stds)):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x5)
    ax.set_xticklabels(weathers)
    ax.set_ylabel('Std (pixels)')
    ax.set_title('Lane Position Stability\n(lx_std, rx_std — lower = more stable)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


# ── Comparison figure (no confidence) ─────────────────────────

def plot_compare_figure(raw, metrics, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle('Perception Comparison — YOLO vs Pure Vision', fontsize=14, fontweight='bold')

    x = np.arange(len(WEATHER_ORDER))

    # Panel 1 — Detection rate
    ax = axes[0]
    for i, method in enumerate(['YOLO', 'Pure Vision']):
        m    = metrics[metrics['method'] == method].set_index('weather').reindex(WEATHER_ORDER)
        vals = m['det_rate_%'].values.astype(float)
        ax.bar(x + i * BAR_WIDTH, vals, BAR_WIDTH,
               label=method, color=M_COLORS[method], edgecolor='black', linewidth=0.7)
    ax.set_xticks(x + BAR_WIDTH / 2)
    ax.set_xticklabels(WEATHER_ORDER)
    ax.set_ylim(0, 115)
    ax.axhline(100, color='green', lw=1, ls='--', alpha=0.4)
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Detection Rate')
    ax.legend(fontsize=9)

    # Panel 2 — Lateral error
    ax = axes[1]
    for i, method in enumerate(['YOLO', 'Pure Vision']):
        m     = metrics[metrics['method'] == method].set_index('weather').reindex(WEATHER_ORDER)
        means = m['err_mean'].values.astype(float)
        stds  = m['err_std'].values.astype(float)
        ax.bar(x + i * BAR_WIDTH, means, BAR_WIDTH, yerr=stds,
               label=method, color=M_COLORS[method],
               edgecolor='black', linewidth=0.7, capsize=4)
    ax.set_xticks(x + BAR_WIDTH / 2)
    ax.set_xticklabels(WEATHER_ORDER)
    ax.set_ylabel('|center_norm − 0.5|')
    ax.set_title('Lateral Error  (mean ± std)\nlower = better')
    ax.legend(fontsize=9)

    # Panel 3 — FPS
    ax = axes[2]
    for i, method in enumerate(['YOLO', 'Pure Vision']):
        m    = metrics[metrics['method'] == method].set_index('weather').reindex(WEATHER_ORDER)
        vals = m['fps'].values.astype(float)
        ax.bar(x + i * BAR_WIDTH, vals, BAR_WIDTH,
               label=method, color=M_COLORS[method], edgecolor='black', linewidth=0.7)
    ax.set_xticks(x + BAR_WIDTH / 2)
    ax.set_xticklabels(WEATHER_ORDER)
    ax.set_ylabel('FPS')
    ax.set_title('Frames per Second')
    ax.legend(fontsize=9)

    # Panel 4 — Lane width mean
    ax = axes[3]
    for i, method in enumerate(['YOLO', 'Pure Vision']):
        m    = metrics[metrics['method'] == method].set_index('weather').reindex(WEATHER_ORDER)
        vals = m['lane_width_mean'].values.astype(float)
        bars = ax.bar(x + i * BAR_WIDTH, vals, BAR_WIDTH,
                      label=method, color=M_COLORS[method], edgecolor='black', linewidth=0.7)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x + BAR_WIDTH / 2)
    ax.set_xticklabels(WEATHER_ORDER)
    ax.set_ylabel('Lane Width (pixels)')
    ax.set_title('Lane Width (rx − lx at y_ref)')
    ax.legend(fontsize=9)

    plt.tight_layout()
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

    plot_method_figure(raw, metrics, 'YOLO',
                       data_dir / 'eval_yolo.png')
    plot_method_figure(raw, metrics, 'Pure Vision',
                       data_dir / 'eval_pure_vision.png')
    plot_compare_figure(raw, metrics,
                        data_dir / 'eval_compare.png')


if __name__ == '__main__':
    main()
