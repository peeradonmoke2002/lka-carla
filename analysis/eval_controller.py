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
OFF_LANE_THRESH_M = 0.10  # |cte_m| beyond which we count as off-lane
MAX_DT_SEC        = 0.15  # max gap for nearest-neighbour merge


def parse_method_weather(bag_dir: Path):
    """Infer method and weather from directory name.
    Handles both 'yolo_rain' and 'yolo_rain_20260511_155234' (timestamped)."""
    name = bag_dir.name
    for m in VALID_METHODS:
        if name.startswith(m):
            suffix = name[len(m):].lstrip('_')
            # strip optional _YYYYMMDD_HHMMSS suffix
            for w in VALID_WEATHERS:
                if suffix == w or suffix.startswith(w + '_'):
                    return m, w
    return None, None


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
            odom_rows.append({'ts': ts, 'speed_mps': speed})

    return (
        pd.DataFrame(lane_rows),
        pd.DataFrame(cte_rows),
        pd.DataFrame(ctrl_rows),
        pd.DataFrame(odom_rows),
    )


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

    return metrics


def process_bag(bag_dir: Path):
    method, weather = parse_method_weather(bag_dir)
    if method is None:
        print(f'  [skip] cannot parse method/weather from: {bag_dir.name}')
        return None, None, None, None, None

    print(f'  {bag_dir.name}  ({method} / {weather})')
    lane_df, cte_df, ctrl_df, odom_df = read_bag(str(bag_dir))
    m = compute_metrics(lane_df, cte_df, ctrl_df, odom_df)
    m['method']  = method
    m['weather'] = weather
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
        base['bag']     = entry['bag']
        frames.append(base)

    if not frames:
        return

    col_order = ['method', 'weather', 'bag', 'time_s',
                 'cte_m', 'center', 'detected', 'steer', 'speed_mps']
    df = pd.concat(frames, ignore_index=True)
    df = df[[c for c in col_order if c in df.columns]]
    df = df.round({'time_s': 3, 'cte_m': 5, 'center': 5, 'steer': 5, 'speed_mps': 4})

    out_path = out_dir / 'raw_frames.csv'
    df.to_csv(out_path, index=False)
    print(f'Raw frames saved: {out_path}  ({len(df)} rows)')


def plot_trajectories(raw_data: list, out_dir: Path):
    """Time-series CTE and lane center for each weather, one line per method."""
    weathers = VALID_WEATHERS
    n = len(weathers)

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), sharex='col')
    fig.suptitle('Controller Trajectories — CTE and Lane Center Over Time',
                 fontsize=13, fontweight='bold')

    for col, weather in enumerate(weathers):
        ax_cte  = axes[0, col]
        ax_lane = axes[1, col]

        ax_cte.set_title(weather, fontsize=11)
        ax_cte.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax_cte.axhline( OFF_LANE_THRESH_M, color='red', linewidth=0.7, linestyle=':')
        ax_cte.axhline(-OFF_LANE_THRESH_M, color='red', linewidth=0.7, linestyle=':',
                       label=f'±{OFF_LANE_THRESH_M}m threshold')
        ax_lane.axhline(0.5, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

        for entry in raw_data:
            if entry['weather'] != weather:
                continue
            method = entry['method']
            color  = COLORS.get(method, 'gray')
            cte_df  = entry['cte_df']
            lane_df = entry['lane_df']

            if not cte_df.empty:
                t = (cte_df['ts'].values - cte_df['ts'].values[0]) / 1e9
                ax_cte.plot(t, cte_df['cte_m'].values, color=color,
                            linewidth=0.9, label=method, alpha=0.85)

            if not lane_df.empty:
                t2 = (lane_df['ts'].values - lane_df['ts'].values[0]) / 1e9
                centers = lane_df['center'].values.copy().astype(float)
                centers[~lane_df['detected'].values] = float('nan')
                ax_lane.plot(t2, centers, color=color,
                             linewidth=0.9, label=method, alpha=0.85)

        ax_cte.legend(fontsize=7)
        ax_lane.legend(fontsize=7)
        ax_lane.set_xlabel('Time (s)')

    axes[0, 0].set_ylabel('CTE (m)')
    axes[1, 0].set_ylabel('Lane center (normalised)')

    plt.tight_layout()
    out_path = out_dir / 'trajectories.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Trajectory plot saved: {out_path}')


def plot_results(df: pd.DataFrame, out_dir: Path):
    weathers = VALID_WEATHERS
    methods  = [m for m in VALID_METHODS if m in df['method'].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Closed-Loop Controller Evaluation', fontsize=14, fontweight='bold')

    def grouped_bars(ax, metric, ylabel, title):
        x    = np.arange(len(weathers))
        w    = 0.25
        all_bars = []
        for i, method in enumerate(methods):
            vals = [
                df[(df['method'] == method) & (df['weather'] == wx)][metric].values[0]
                if len(df[(df['method'] == method) & (df['weather'] == wx)]) > 0
                else float('nan')
                for wx in weathers
            ]
            bars = ax.bar(x + i * w, vals, width=w, label=method,
                          color=COLORS.get(method, 'gray'), edgecolor='black', linewidth=0.7)
            all_bars.append((bars, vals))
        ax.set_xticks(x + w)
        ax.set_xticklabels(weathers)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        # label bars after ylim is finalised
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        for bars, vals in all_bars:
            for bar, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                label_text = '0' if v == 0 else f'{v:.2f}'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + y_range * 0.01,
                        label_text, ha='center', va='bottom', fontsize=6, rotation=90)

    grouped_bars(axes[0, 0], 'cte_rmse',     'RMSE (m)',        'CTE RMSE — lower = better')
    grouped_bars(axes[0, 1], 'cte_max',       'Max |CTE| (m)',   'Max |CTE| — lower = better')
    grouped_bars(axes[1, 0], 'steer_jitter',  'Steer rate std',  'Steering Jitter — lower = smoother')
    grouped_bars(axes[1, 1], 'off_lane_pct',  '% time',          f'Off-Lane Time (|CTE|>{OFF_LANE_THRESH_M}m)')

    plt.tight_layout()
    out_path = out_dir / 'eval_controller.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved: {out_path}')


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
        # Keep only the latest bag per (method, weather) — bags are named
        # <method>_<weather>_<timestamp> so sorting by name gives chronological order.
        latest: dict = {}
        for d in all_bags:
            m, w = parse_method_weather(d)
            if m is not None:
                latest[(m, w)] = d   # later sort order wins → newest timestamp kept
        bags = sorted(latest.values())
        skipped = len(all_bags) - len(bags)
        if skipped:
            print(f'  (skipped {skipped} older duplicate bag(s) — keeping latest per method/weather)')

    if not bags:
        print(f'No bags found in {bag_path}')
        return

    print(f'Processing {len(bags)} bag(s) ...')
    rows     = []
    raw_data = []
    for b in bags:
        r, lane_df, cte_df, ctrl_df, odom_df = process_bag(b)
        if r:
            rows.append(r)
            raw_data.append({
                'method':  r['method'],
                'weather': r['weather'],
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
        'method', 'weather', 'bag',
        'duration_s', 'total_frames',
        'cte_rmse', 'cte_max', 'cte_p95', 'cte_mean',
        'off_lane_pct', 'drop_rate_pct',
        'steer_jitter', 'steer_std',
        'mean_speed_mps', 'max_speed_mps',
    ]
    df = pd.DataFrame(rows)
    df = df[[c for c in col_order if c in df.columns]]

    csv_path = out_dir / 'controller_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nResults saved: {csv_path}  ({len(df)} bag(s))')

    print('\n── Controller Metrics ────────────────────────────────────')
    print(df.to_string(index=False))

    save_raw_csv(raw_data, out_dir)
    if len(df) > 1:
        plot_results(df, out_dir)
        plot_trajectories(raw_data, out_dir)


if __name__ == '__main__':
    main()
