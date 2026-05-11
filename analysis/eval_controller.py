#!/usr/bin/python3
"""
Controller evaluation script.
Reads closed-loop bags and computes per-(method, weather) driving metrics.

Single bag:
    python3 analysis/eval_controller.py bags/closed_loop/yolo_rain

All bags in a directory (12-trial sweep):
    python3 analysis/eval_controller.py bags/closed_loop/
    python3 analysis/eval_controller.py bags/closed_loop/ --out analysis/results/controller
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

VALID_METHODS  = ['yolo', 'pure_vision', 'scnn']
VALID_WEATHERS = ['rain', 'clear', 'fog', 'night']
COLORS = {
    'yolo': '#2196F3', 'pure_vision': '#FF9800', 'scnn': '#4CAF50',
}
LABELS = {'yolo': 'YOLO', 'pure_vision': 'Pure Vision', 'scnn': 'SCNN'}
OFF_LANE_THRESH_M = 0.10  # |cte_m| beyond which we count as off-lane
MAX_DT_SEC        = 0.15  # max gap for nearest-neighbour merge


def parse_method_weather(bag_dir: Path):
    """Infer method, weather, and rep from directory name.

    Handles:
      'yolo_rain'
      'yolo_rain_20260511_155234'          (old timestamped, rep=1 fallback)
      'yolo_rain_rep2_20260511_155234'     (new format with explicit rep)
    Returns (method, weather, rep) or (None, None, None).
    """
    name = bag_dir.name
    for m in VALID_METHODS:
        if name.startswith(m):
            suffix = name[len(m):].lstrip('_')
            for w in VALID_WEATHERS:
                if suffix == w:
                    return m, w, 1
                if suffix.startswith(w + '_'):
                    rest = suffix[len(w):].lstrip('_')
                    rep = 1
                    if rest.startswith('rep'):
                        try:
                            rep = int(rest[3:].split('_')[0])
                        except ValueError:
                            pass
                    return m, w, rep
    return None, None, None


def read_bag(bag_path: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''),
    )

    LaneCenterMsg = get_message('lka_msgs/msg/LaneCenter')
    CteMsg        = get_message('std_msgs/msg/Float64')
    CtrlMsg       = get_message('carla_msgs/msg/CarlaEgoVehicleControl')
    OdomMsg       = get_message('nav_msgs/msg/Odometry')

    lane_rows = []
    cte_rows  = []
    ctrl_rows = []
    odom_rows = []

    while reader.has_next():
        topic, data, ts = reader.read_next()

        if topic == '/lka/lane_center':
            msg = deserialize_message(data, LaneCenterMsg)
            lane_rows.append({'ts': ts, 'detected': bool(msg.detected), 'center': float(msg.center)})

        elif topic == '/lka/gt/cross_track_m':
            msg = deserialize_message(data, CteMsg)
            cte_rows.append({'ts': ts, 'cte_m': float(msg.data)})

        elif topic == '/carla/ego_vehicle/vehicle_control_cmd':
            msg = deserialize_message(data, CtrlMsg)
            ctrl_rows.append({'ts': ts, 'steer': float(msg.steer)})

        elif topic == '/carla/ego_vehicle/odometry':
            msg = deserialize_message(data, OdomMsg)
            v = msg.twist.twist.linear
            speed = float(np.sqrt(v.x**2 + v.y**2 + v.z**2))
            odom_rows.append({
                'ts':        ts,
                'pos_x':     float(msg.pose.pose.position.x),
                'pos_y':     float(msg.pose.pose.position.y),
                'speed_mps': speed,
            })

    return (
        pd.DataFrame(lane_rows),
        pd.DataFrame(cte_rows),
        pd.DataFrame(ctrl_rows),
        pd.DataFrame(odom_rows),
    )


def _calc_hz(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return float('nan')
    dur = (df['ts'].max() - df['ts'].min()) / 1e9
    return round((len(df) - 1) / dur, 2) if dur > 0 else float('nan')


def compute_metrics(lane_df, cte_df, ctrl_df, odom_df):
    """Return a dict of scalar metrics for one bag."""
    metrics = {}

    # ── CTE metrics ────────────────────────────────────────────────
    if not cte_df.empty:
        cte = cte_df['cte_m'].values
        metrics['cte_rmse']     = round(float(np.sqrt(np.mean(cte**2))), 4)
        metrics['cte_max']      = round(float(np.max(np.abs(cte))), 4)
        metrics['cte_p95']      = round(float(np.percentile(np.abs(cte), 95)), 4)
        metrics['cte_mean']     = round(float(np.mean(cte)), 4)
        metrics['off_lane_pct'] = round(float(np.mean(np.abs(cte) > OFF_LANE_THRESH_M) * 100), 2)
    else:
        for k in ('cte_rmse', 'cte_max', 'cte_p95', 'cte_mean', 'off_lane_pct'):
            metrics[k] = float('nan')

    # ── Steer jitter ───────────────────────────────────────────────
    if len(ctrl_df) > 1:
        dt = np.diff(ctrl_df['ts'].values) / 1e9          # seconds
        ds = np.diff(ctrl_df['steer'].values)
        # filter out large time gaps (pauses between bag segments)
        mask = dt < 1.0
        steer_rate = ds[mask] / dt[mask]
        metrics['steer_jitter'] = round(float(np.std(steer_rate)), 4) if len(steer_rate) else float('nan')
        metrics['steer_std']    = round(float(np.std(ctrl_df['steer'])), 4)
    else:
        metrics['steer_jitter'] = float('nan')
        metrics['steer_std']    = float('nan')

    # ── Perception drop rate ───────────────────────────────────────
    if not lane_df.empty:
        metrics['drop_rate_pct'] = round(float((~lane_df['detected']).mean() * 100), 2)
        metrics['total_frames']  = len(lane_df)
    else:
        metrics['drop_rate_pct'] = float('nan')
        metrics['total_frames']  = 0

    # ── Lane stability (center jitter while driving) ───────────────
    if not lane_df.empty:
        detected_centers = lane_df[lane_df['detected']]['center'].values
        metrics['center_diff_std'] = round(float(np.std(np.diff(detected_centers))), 4) \
            if len(detected_centers) > 1 else float('nan')
    else:
        metrics['center_diff_std'] = float('nan')

    # ── Speed ──────────────────────────────────────────────────────
    if not odom_df.empty:
        metrics['mean_speed_mps'] = round(float(odom_df['speed_mps'].mean()), 3)
        metrics['max_speed_mps']  = round(float(odom_df['speed_mps'].max()), 3)
    else:
        metrics['mean_speed_mps'] = float('nan')
        metrics['max_speed_mps']  = float('nan')

    # ── Duration ───────────────────────────────────────────────────
    if not cte_df.empty:
        metrics['duration_s'] = round((cte_df['ts'].max() - cte_df['ts'].min()) / 1e9, 1)
    elif not lane_df.empty:
        metrics['duration_s'] = round((lane_df['ts'].max() - lane_df['ts'].min()) / 1e9, 1)
    else:
        metrics['duration_s'] = float('nan')

    # ── Hz per topic ───────────────────────────────────────────────
    metrics['lane_hz'] = _calc_hz(lane_df)
    metrics['cte_hz']  = _calc_hz(cte_df)
    metrics['ctrl_hz'] = _calc_hz(ctrl_df)
    metrics['odom_hz'] = _calc_hz(odom_df)

    return metrics


def process_bag(bag_dir: Path):
    method, weather, rep = parse_method_weather(bag_dir)
    if method is None:
        print(f'  [skip] cannot parse method/weather from: {bag_dir.name}')
        return None, None, None, None, None

    print(f'  {bag_dir.name}  ({method} / {weather} / rep={rep})')
    lane_df, cte_df, ctrl_df, odom_df = read_bag(str(bag_dir))
    m = compute_metrics(lane_df, cte_df, ctrl_df, odom_df)
    m['method']  = method
    m['weather'] = weather
    m['rep']     = rep
    m['bag']     = bag_dir.name
    return m, lane_df, cte_df, ctrl_df, odom_df


def save_raw_csv(raw_data: list, out_dir: Path):
    """Merge per-frame data from all bags into one CSV aligned to CTE timestamps."""
    frames = []
    for entry in raw_data:
        cte_df  = entry['cte_df'].copy().sort_values('ts')
        lane_df = entry['lane_df'].copy().sort_values('ts')
        ctrl_df = entry['ctrl_df'].copy().sort_values('ts')
        odom_df = entry['odom_df'].copy().sort_values('ts')

        if cte_df.empty:
            continue

        base = cte_df.rename(columns={'ts': 'ts'})

        tol = int(MAX_DT_SEC * 1e9)
        if not lane_df.empty:
            base = pd.merge_asof(base, lane_df, on='ts', tolerance=tol, direction='nearest')
        if not ctrl_df.empty:
            base = pd.merge_asof(base, ctrl_df, on='ts', tolerance=tol, direction='nearest')
        if not odom_df.empty:
            base = pd.merge_asof(base, odom_df, on='ts', tolerance=tol, direction='nearest')

        base['time_s']  = (base['ts'] - base['ts'].iloc[0]) / 1e9
        base['method']  = entry['method']
        base['weather'] = entry['weather']
        base['rep']     = entry.get('rep', 1)
        base['bag']     = entry['bag']
        frames.append(base)

    if not frames:
        return

    col_order = ['method', 'weather', 'rep', 'bag', 'time_s',
                 'cte_m', 'center', 'detected', 'steer', 'speed_mps', 'pos_x', 'pos_y']
    df = pd.concat(frames, ignore_index=True)
    df = df[[c for c in col_order if c in df.columns]]
    df = df.round({'time_s': 3, 'cte_m': 5, 'center': 5, 'steer': 5, 'speed_mps': 4})

    out_path = out_dir / 'raw_frames.csv'
    df.to_csv(out_path, index=False)
    print(f'Raw frames saved: {out_path}  ({len(df)} rows)')


def plot_trajectories(raw_data: list, out_dir: Path):
    """3-row plot: CTE / Speed / Position-X — mean ± 1 std band per method per weather."""
    weathers = VALID_WEATHERS
    methods  = VALID_METHODS
    n        = len(weathers)

    fig, axes = plt.subplots(3, n, figsize=(5 * n, 10), sharex=True)
    fig.suptitle('Trajectory Profiles — Mean ± 1 std over Repeats',
                 fontsize=13, fontweight='bold')

    # Global minimum duration across ALL weathers × methods × reps
    t_end = min(
        (entry['cte_df']['ts'].values[-1] - entry['cte_df']['ts'].values[0]) / 1e9
        for entry in raw_data if not entry['cte_df'].empty
    )
    t_grid = np.linspace(0, t_end, 500)
    kernel = np.ones(20) / 20   # ~1.6s rolling average

    for col, weather in enumerate(weathers):
        ax_cte   = axes[0, col]
        ax_spd   = axes[1, col]
        ax_posx  = axes[2, col]

        ax_cte.set_title(weather, fontsize=11)
        ax_cte.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

        for method in methods:
            color = COLORS.get(method, 'gray')

            cte_series  = []
            spd_series  = []
            posx_series = []

            for entry in raw_data:
                if entry['weather'] != weather or entry['method'] != method:
                    continue
                cte_df  = entry['cte_df']
                odom_df = entry['odom_df']
                if cte_df.empty:
                    continue
                t0 = cte_df['ts'].values[0]
                t_cte = (cte_df['ts'].values - t0) / 1e9
                cte_series.append((t_cte, cte_df['cte_m'].values))

                if not odom_df.empty and 'pos_x' in odom_df.columns:
                    t_odom = (odom_df['ts'].values - t0) / 1e9
                    spd_series.append((t_odom, odom_df['speed_mps'].values))
                    posx_series.append((t_odom, odom_df['pos_x'].values))

            def plot_band(ax, series, refline=None):
                if not series:
                    return
                mat  = np.stack([np.interp(t_grid, t, v) for t, v in series])
                mean = np.convolve(mat.mean(axis=0), kernel, mode='same')
                std  = np.convolve(mat.std(axis=0),  kernel, mode='same')
                ax.plot(t_grid, mean, color=color, linewidth=1.5, label=method)
                ax.fill_between(t_grid, mean - std, mean + std, color=color, alpha=0.2)
                if refline is not None:
                    ax.axhline(refline, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

            plot_band(ax_cte,  cte_series)
            plot_band(ax_spd,  spd_series)
            plot_band(ax_posx, posx_series)

        ax_cte.legend(fontsize=7)
        ax_posx.set_xlabel(f'Time (s)  [0 – {t_end:.0f}s]')

    axes[0, 0].set_ylabel('CTE (m)')
    axes[1, 0].set_ylabel('Speed (m/s)')
    axes[2, 0].set_ylabel('Position X (m)')

    plt.tight_layout()
    out_path = out_dir / 'trajectories.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Trajectory plot saved: {out_path}')


def plot_cte_trajectory(raw_data: list, out_dir: Path):
    """Single-row CTE plot — mean ± 1 std band per method per weather."""
    weathers = VALID_WEATHERS
    methods  = VALID_METHODS
    n        = len(weathers)

    t_end = min(
        (entry['cte_df']['ts'].values[-1] - entry['cte_df']['ts'].values[0]) / 1e9
        for entry in raw_data if not entry['cte_df'].empty
    )
    t_grid = np.linspace(0, t_end, 500)
    kernel = np.ones(20) / 20

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    fig.suptitle('CTE over Time — Mean ± 1 std over Repeats',
                 fontsize=13, fontweight='bold')

    for col, weather in enumerate(weathers):
        ax = axes[col]
        ax.set_title(weather.capitalize(), fontsize=12)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_xlabel(f'Time (s)  [0 – {t_end:.0f}s]', fontsize=10)

        for method in methods:
            color      = COLORS.get(method, 'gray')
            cte_series = []
            for entry in raw_data:
                if entry['weather'] != weather or entry['method'] != method:
                    continue
                cte_df = entry['cte_df']
                if cte_df.empty:
                    continue
                t0 = cte_df['ts'].values[0]
                cte_series.append(
                    ((cte_df['ts'].values - t0) / 1e9, cte_df['cte_m'].values)
                )
            if not cte_series:
                continue
            mat  = np.stack([np.interp(t_grid, t, v) for t, v in cte_series])
            mean = np.convolve(mat.mean(axis=0), kernel, mode='same')
            std  = np.convolve(mat.std(axis=0),  kernel, mode='same')
            ax.plot(t_grid, mean, color=color, linewidth=1.5,
                    label=LABELS.get(method, method))
            ax.fill_between(t_grid, mean - std, mean + std, color=color, alpha=0.2)

        ax.legend(fontsize=9)

    axes[0].set_ylabel('CTE (m)', fontsize=11)

    plt.tight_layout()
    out_path = out_dir / 'cte_trajectory.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'CTE trajectory plot saved: {out_path}')


def plot_results(df: pd.DataFrame, out_dir: Path):
    weathers = VALID_WEATHERS
    methods  = [m for m in VALID_METHODS if m in df['method'].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Closed-Loop Controller Evaluation', fontsize=16, fontweight='bold', y=1.01)

    w      = 0.26
    x      = np.arange(len(weathers))

    def grouped_bars(ax, metric, ylabel, title):
        for i, method in enumerate(methods):
            vals, errs = [], []
            for wx in weathers:
                sub = df[(df['method'] == method) & (df['weather'] == wx)][metric].dropna()
                vals.append(float(sub.mean()) if len(sub) else float('nan'))
                errs.append(float(sub.std())  if len(sub) > 1 else 0.0)

            ax.bar(x + i * w, vals, width=w, yerr=errs, capsize=4,
                   label=LABELS.get(method, method),
                   color=COLORS.get(method, 'gray'),
                   edgecolor='white', linewidth=0.5,
                   error_kw={'linewidth': 1.2, 'ecolor': 'black'})

        ax.set_xticks(x + w)
        ax.set_xticklabels([wx.capitalize() for wx in weathers], fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xlim(-0.2, len(weathers) - 0.2)

    grouped_bars(axes[0, 0], 'cte_rmse',     'RMSE (m)',       'CTE RMSE')
    grouped_bars(axes[0, 1], 'cte_max',       'Max |CTE| (m)',  'Max |CTE|')
    grouped_bars(axes[1, 0], 'steer_jitter',  'Steer rate std', 'Steering Jitter')
    grouped_bars(axes[1, 1], 'drop_rate_pct', '% frames',       'Perception Drop Rate')

    # Single shared legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(methods),
               fontsize=12, frameon=True, bbox_to_anchor=(0.5, 1.0))

    plt.tight_layout()
    out_path = out_dir / 'eval_controller.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved: {out_path}')


def plot_path_xy(raw_data: list, out_dir: Path):
    """Bird's eye view (pos_x vs pos_y) of vehicle paths per weather."""
    weathers = VALID_WEATHERS
    methods  = VALID_METHODS
    n        = len(weathers)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle("Vehicle Path — Bird's Eye View", fontsize=14, fontweight='bold')

    for col, weather in enumerate(weathers):
        ax = axes[col]
        ax.set_title(weather.capitalize(), fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', adjustable='datalim')

        for method in methods:
            color  = COLORS.get(method, 'gray')
            labeled = False
            for entry in raw_data:
                if entry['weather'] != weather or entry['method'] != method:
                    continue
                odom_df = entry['odom_df']
                if odom_df.empty or 'pos_x' not in odom_df.columns:
                    continue
                ax.plot(odom_df['pos_x'].values, odom_df['pos_y'].values,
                        color=color, linewidth=1.2, alpha=0.6,
                        label=LABELS.get(method, method) if not labeled else None)
                labeled = True

        ax.legend(fontsize=9)
        ax.set_xlabel('Position X (m)')
        if col == 0:
            ax.set_ylabel('Position Y (m)')

    plt.tight_layout()
    out_path = out_dir / 'path_xy.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Path plot saved: {out_path}')


def plot_cte_distribution(raw_data: list, out_dir: Path):
    """Boxplot of CTE values combining all repeats per (method × weather)."""
    weathers = VALID_WEATHERS
    methods  = VALID_METHODS
    n        = len(weathers)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    fig.suptitle('CTE Distribution — All Frames Combined (3 Repeats)',
                 fontsize=14, fontweight='bold')

    for col, weather in enumerate(weathers):
        ax = axes[col]
        ax.set_title(weather.capitalize(), fontsize=12)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        data, labels, colors_list = [], [], []
        for method in methods:
            ctes = []
            for entry in raw_data:
                if entry['weather'] != weather or entry['method'] != method:
                    continue
                cte_df = entry['cte_df']
                if cte_df.empty:
                    continue
                ctes.extend(cte_df['cte_m'].values.tolist())
            if ctes:
                data.append(ctes)
                labels.append(LABELS.get(method, method))
                colors_list.append(COLORS.get(method, 'gray'))

        if not data:
            continue

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                        medianprops=dict(color='black', linewidth=1.5),
                        flierprops=dict(marker='.', markersize=3, alpha=0.3),
                        showfliers=True)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        if col == 0:
            ax.set_ylabel('CTE (m)', fontsize=12)

    plt.tight_layout()
    out_path = out_dir / 'cte_distribution.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'CTE distribution plot saved: {out_path}')


def plot_bias_comparison(df: pd.DataFrame, out_dir: Path):
    """Bar chart of calibrated bias_offset per (method × weather)."""
    if 'bias_offset' not in df.columns:
        print('  [warn] bias_offset column missing — skipping bias plot')
        return

    weathers = VALID_WEATHERS
    methods  = [m for m in VALID_METHODS if m in df['method'].unique()]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle('Calibrated Bias Offset per Method × Weather',
                 fontsize=14, fontweight='bold', y=0.98)
    ax.set_title('bias = mean(center) − 0.5  (measured at spawn, before driving)',
                 fontsize=10, style='italic', pad=8)

    w = 0.26
    x = np.arange(len(weathers))

    for i, method in enumerate(methods):
        vals, errs = [], []
        for wx in weathers:
            sub = df[(df['method'] == method) & (df['weather'] == wx)]['bias_offset'].dropna()
            vals.append(float(sub.mean()) if len(sub) else float('nan'))
            errs.append(float(sub.std())  if len(sub) > 1 else 0.0)

        ax.bar(x + i * w, vals, width=w, yerr=errs, capsize=4,
               label=LABELS.get(method, method),
               color=COLORS.get(method, 'gray'),
               edgecolor='white', linewidth=0.5,
               error_kw={'linewidth': 1.2, 'ecolor': 'black'})

    ax.set_xticks(x + w)
    ax.set_xticklabels([wx.capitalize() for wx in weathers], fontsize=12)
    ax.set_ylabel('Bias offset', fontsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, loc='upper left')
    ax.tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    out_path = out_dir / 'bias_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Bias comparison plot saved: {out_path}')


def plot_hz_consistency(df: pd.DataFrame, out_dir: Path):
    """Boxplot of Hz per topic per method — verifies no missing data or rate issues."""
    topics = {
        'lane_hz':  '/lka/lane_center',
        'cte_hz':   '/lka/gt/cross_track_m',
        'ctrl_hz':  'vehicle_control_cmd',
        'odom_hz':  'odometry',
    }
    methods = [m for m in VALID_METHODS if m in df['method'].unique()]

    fig, axes = plt.subplots(1, len(topics), figsize=(14, 5), sharey=False)
    fig.suptitle('Topic Hz Consistency — All 36 Trials', fontsize=14, fontweight='bold')

    for ax, (col, topic_name) in zip(axes, topics.items()):
        data, labels, colors_list = [], [], []
        for method in methods:
            vals = df[df['method'] == method][col].dropna().values
            if len(vals):
                data.append(vals)
                labels.append(LABELS.get(method, method))
                colors_list.append(COLORS.get(method, 'gray'))

        if not data:
            continue

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', linewidth=2),
                        flierprops=dict(marker='o', markersize=5, alpha=0.6))
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Expected Hz reference line
        all_vals = np.concatenate(data)
        expected = round(float(np.median(all_vals)))
        ax.axhline(expected, color='red', linewidth=1.2, linestyle='--', alpha=0.7,
                   label=f'median {expected} Hz')

        ax.set_title(topic_name, fontsize=9)
        ax.set_ylabel('Hz')
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)

        # Annotate min/max
        ax.text(0.97, 0.04, f'min={all_vals.min():.1f}  max={all_vals.max():.1f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                color='gray')

    plt.tight_layout()
    out_path = out_dir / 'hz_consistency.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Hz consistency plot saved: {out_path}')


def plot_lane_stability(df: pd.DataFrame, out_dir: Path):
    """Grouped bar chart of center_diff_std (lane jitter while driving) per method × weather."""
    weathers = VALID_WEATHERS
    methods  = [m for m in VALID_METHODS if m in df['method'].unique()]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Lane Stability During Driving — lower = better',
                 fontsize=14, fontweight='bold')
    ax.set_title('center_diff_std: frame-to-frame jitter of lane center while vehicle moves',
                 fontsize=9, style='italic', pad=6)

    w = 0.26
    x = np.arange(len(weathers))

    for i, method in enumerate(methods):
        vals, errs = [], []
        for wx in weathers:
            sub = df[(df['method'] == method) & (df['weather'] == wx)]['center_diff_std'].dropna()
            vals.append(float(sub.mean()) if len(sub) else float('nan'))
            errs.append(float(sub.std())  if len(sub) > 1 else 0.0)
        ax.bar(x + i * w, vals, width=w, yerr=errs, capsize=4,
               label=LABELS.get(method, method),
               color=COLORS.get(method, 'gray'),
               edgecolor='white', linewidth=0.5,
               error_kw={'linewidth': 1.2, 'ecolor': 'black'})

    ax.set_xticks(x + w)
    ax.set_xticklabels([wx.capitalize() for wx in weathers], fontsize=12)
    ax.set_ylabel('center_diff_std (norm)', fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.2, len(weathers) - 0.2)

    plt.tight_layout()
    out_path = out_dir / 'lane_stability.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Lane stability plot saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Controller bag evaluation')
    parser.add_argument('bag_path',
                        help='Path to one closed-loop bag directory, or a parent directory of bags')
    parser.add_argument('--out', default='/home/peeradon/lka-carla-yolo/analysis/results/controller',
                        help='Output directory for CSV and plots')
    args = parser.parse_args()

    bag_path = Path(args.bag_path)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect single bag vs directory of bags
    has_db3 = any(bag_path.glob('*.db3'))
    if has_db3:
        bags = [bag_path]
    else:
        all_bags = sorted([d for d in bag_path.iterdir() if d.is_dir() and any(d.glob('*.db3'))])
        bags = [d for d in all_bags if parse_method_weather(d)[0] is not None]
        skipped = len(all_bags) - len(bags)
        if skipped:
            print(f'  (skipped {skipped} bag(s) with unrecognised names)')

    # Load calibration log for bias_offset lookup
    bias_lookup: dict = {}
    calib_log_dir = bag_path if not has_db3 else bag_path.parent
    calib_log_path = calib_log_dir / 'calibration_log.csv'
    if calib_log_path.exists():
        try:
            calib_df = pd.read_csv(calib_log_path)
            for _, row in calib_df.iterrows():
                key = (row['method'], row['weather'], int(row['rep']))
                bias_lookup[key] = float(row['bias_offset'])
            print(f'  Loaded {len(bias_lookup)} calibration entries from {calib_log_path.name}')
        except Exception as e:
            print(f'  [warn] could not load calibration log: {e}')

    if not bags:
        print(f'No bags found in {bag_path}')
        return

    print(f'Processing {len(bags)} bag(s) ...')
    rows     = []
    raw_data = []
    for b in bags:
        r, lane_df, cte_df, ctrl_df, odom_df = process_bag(b)
        if r:
            key = (r['method'], r['weather'], r.get('rep', 1))
            r['bias_offset'] = bias_lookup.get(key, 0.0)
            rows.append(r)
            raw_data.append({
                'method':  r['method'],
                'weather': r['weather'],
                'rep':     r.get('rep', 1),
                'bag':     r['bag'],
                'lane_df': lane_df,
                'cte_df':  cte_df,
                'ctrl_df': ctrl_df,
                'odom_df': odom_df,
            })

    if not rows:
        print('No valid bags processed.')
        return

    col_order = [
        'method', 'weather', 'rep', 'bag',
        'bias_offset',
        'duration_s', 'total_frames',
        'lane_hz', 'cte_hz', 'ctrl_hz', 'odom_hz',
        'cte_rmse', 'cte_max', 'cte_p95', 'cte_mean',
        'off_lane_pct', 'drop_rate_pct',
        'steer_jitter', 'steer_std',
        'center_diff_std',
        'mean_speed_mps', 'max_speed_mps',
    ]
    df = pd.DataFrame(rows)
    df = df[[c for c in col_order if c in df.columns]]

    csv_path = out_dir / 'controller_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nResults saved: {csv_path}  ({len(df)} bag(s))')

    print('\n── Controller Metrics ────────────────────────────────────')
    print(df.to_string(index=False))

    # Per-condition summary (mean ± std over repeats)
    if 'rep' in df.columns and df['rep'].nunique() > 0:
        summary_df = df.groupby(['method', 'weather']).agg(
            n_repeats=('rep', 'count'),
            cte_rmse_mean=('cte_rmse', 'mean'),
            cte_rmse_std=('cte_rmse', 'std'),
            off_lane_pct_mean=('off_lane_pct', 'mean'),
            off_lane_pct_std=('off_lane_pct', 'std'),
            steer_jitter_mean=('steer_jitter', 'mean'),
            steer_jitter_std=('steer_jitter', 'std'),
            center_diff_std_mean=('center_diff_std', 'mean'),
            center_diff_std_std=('center_diff_std', 'std'),
        ).reset_index().round(4)
        summary_path = out_dir / 'controller_metrics_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f'\nSummary saved: {summary_path}  ({len(summary_df)} conditions)')
        print('\n── Summary (mean ± std) ──────────────────────────────────')
        print(summary_df.to_string(index=False))

    save_raw_csv(raw_data, out_dir)
    if len(df) > 1:
        for sub in ('summary', 'trajectories', 'cte', 'stability', 'hz'):
            (out_dir / sub).mkdir(exist_ok=True)
        plot_results(df,           out_dir / 'summary')
        plot_trajectories(raw_data,     out_dir / 'trajectories')
        plot_cte_trajectory(raw_data,   out_dir / 'trajectories')
        plot_path_xy(raw_data,          out_dir / 'trajectories')
        plot_cte_distribution(raw_data, out_dir / 'cte')
        plot_bias_comparison(df,   out_dir / 'stability')
        plot_lane_stability(df,    out_dir / 'stability')
        plot_hz_consistency(df,    out_dir / 'hz')


if __name__ == '__main__':
    main()
