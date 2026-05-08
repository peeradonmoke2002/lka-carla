#!/usr/bin/python3
"""
Perception evaluation script.
Reads a ROS2 bag and computes detection metrics per weather condition.

Usage:
    python3 analysis/eval_perception.py <bag_path>
    python3 analysis/eval_perception.py bags/20260508_163903
    python3 analysis/eval_perception.py bags/20260508_163903 --out analysis/results
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

WEATHER_ORDER = ['rain', 'clear', 'fog', 'night']
COLORS = {'rain': '#4C9BE8', 'clear': '#F5A623', 'fog': '#9B9B9B', 'night': '#2C3E50'}


def classify_weather(msg):
    if msg.fog_density > 40:
        return 'fog'
    if msg.precipitation > 30:
        return 'rain'
    if msg.sun_altitude_angle < 0:
        return 'night'
    return 'clear'


def read_bag(bag_path: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''),
    )

    WeatherMsg    = get_message('carla_msgs/msg/CarlaWeatherParameters')
    LaneCenterMsg = get_message('lka_msgs/msg/LaneCenter')

    weather_events = []
    lane_records   = []

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if topic == '/carla/weather_control':
            msg = deserialize_message(data, WeatherMsg)
            weather_events.append({'timestamp_ns': timestamp,
                                   'weather': classify_weather(msg)})

        elif topic in ('/lka/lane_center',
                       '/lka/yolo/lane_center',
                       '/lka/pure_vision/lane_center'):
            msg = deserialize_message(data, LaneCenterMsg)
            lane_records.append({
                'timestamp_ns': timestamp,
                'topic':        topic,
                'center':       float(msg.center),
                'confidence':   float(msg.confidence),
                'detected':     bool(msg.detected),
                'lx':           float(msg.lx),
                'rx':           float(msg.rx),
            })

    return pd.DataFrame(weather_events), pd.DataFrame(lane_records)


def assign_weather(lane_df, weather_df, window_sec=60.0):
    """Label each frame with weather. Cut each window to window_sec seconds."""
    lane_df = lane_df.copy()
    lane_df['weather']   = 'unknown'
    lane_df['t_in_window'] = np.nan   # seconds since weather change
    if weather_df.empty:
        return lane_df

    ts = weather_df['timestamp_ns'].values
    ws = weather_df['weather'].values
    for i in range(len(ts)):
        end  = ts[i] + int(window_sec * 1e9)   # cut at 60 s
        mask = (lane_df['timestamp_ns'] >= ts[i]) & (lane_df['timestamp_ns'] < end)
        lane_df.loc[mask, 'weather']      = ws[i]
        lane_df.loc[mask, 't_in_window']  = (lane_df.loc[mask, 'timestamp_ns'] - ts[i]) / 1e9

    return lane_df[lane_df['weather'] != 'unknown'].reset_index(drop=True)


TOPIC_LABELS = {
    '/lka/lane_center':             'shared',
    '/lka/yolo/lane_center':        'YOLO',
    '/lka/pure_vision/lane_center': 'Pure Vision',
}


def compute_metrics(lane_df):
    lane_df = lane_df.copy()
    lane_df['method'] = lane_df['topic'].map(TOPIC_LABELS).fillna(lane_df['topic'])

    rows = []
    for weather in WEATHER_ORDER:
        for method in lane_df['method'].unique():
            w = lane_df[(lane_df['weather'] == weather) & (lane_df['method'] == method)]
            if w.empty:
                continue
            total      = len(w)
            n_det      = int(w['detected'].sum())
            valid      = w[w['detected']]
            centers    = valid['center'].values
            lat_err    = np.abs(centers - 0.5)
            confs      = valid['confidence'].values
            lx_vals    = valid[valid['lx'] > 0]['lx'].values
            rx_vals    = valid[valid['rx'] > 0]['rx'].values
            lane_widths = valid[(valid['lx'] > 0) & (valid['rx'] > 0)].eval('rx - lx').values
            duration_s = (w['timestamp_ns'].max() - w['timestamp_ns'].min()) / 1e9
            rows.append({
                'method':        method,
                'weather':       weather,
                'total_frames':  total,
                'detected':      n_det,
                'det_rate_%':    round(n_det / total * 100, 2),
                'center_mean':   round(float(np.mean(centers)),     4) if len(centers)     else np.nan,
                'center_std':    round(float(np.std(centers)),      4) if len(centers)     else np.nan,
                'err_mean':      round(float(np.mean(lat_err)),     4) if len(lat_err)     else np.nan,
                'err_std':       round(float(np.std(lat_err)),      4) if len(lat_err)     else np.nan,
                'lx_mean':       round(float(np.mean(lx_vals)),     2) if len(lx_vals)     else np.nan,
                'lx_std':        round(float(np.std(lx_vals)),      2) if len(lx_vals)     else np.nan,
                'rx_mean':       round(float(np.mean(rx_vals)),     2) if len(rx_vals)     else np.nan,
                'rx_std':        round(float(np.std(rx_vals)),      2) if len(rx_vals)     else np.nan,
                'lane_width_mean': round(float(np.mean(lane_widths)), 2) if len(lane_widths) else np.nan,
                'conf_mean':     round(float(np.mean(confs)),       4) if len(confs)       else np.nan,
                'fps':           round(total / duration_s,          1) if duration_s > 0   else np.nan,
            })
    return pd.DataFrame(rows)


def plot_results(metrics, lane_df, out_dir: Path):
    weathers = metrics['weather'].tolist()
    colors   = [COLORS.get(w, 'gray') for w in weathers]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Perception Evaluation', fontsize=14, fontweight='bold')

    # Panel 1 — Detection rate
    ax = axes[0]
    bars = ax.bar(weathers, metrics['det_rate_%'], color=colors,
                  edgecolor='black', linewidth=0.8)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Detection Rate per Weather')
    ax.axhline(100, color='green', lw=1, ls='--', alpha=0.4)
    for bar, val in zip(bars, metrics['det_rate_%']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Panel 2 — Lateral error mean ± std
    ax = axes[1]
    ax.bar(weathers, metrics['err_mean'], yerr=metrics['err_std'],
           color=colors, edgecolor='black', linewidth=0.8, capsize=5)
    ax.set_ylabel('|center_norm − 0.5|')
    ax.set_title('Lateral Error (mean ± std)\nlower = better')
    ax.axhline(0, color='green', lw=1, ls='--', alpha=0.4)

    # Panel 3 — center_norm time series
    ax = axes[2]
    t0 = lane_df['timestamp_ns'].min()
    lane_df = lane_df.copy()
    lane_df['t_sec'] = (lane_df['timestamp_ns'] - t0) / 1e9
    valid = lane_df[lane_df['detected']]
    ax.scatter(valid['t_sec'], valid['center'], s=1, alpha=0.3, color='steelblue')
    ax.axhline(0.5, color='green', lw=1.5, ls='--', label='ideal (0.5)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('center_norm')
    ax.set_title('center_norm over Time')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = out_dir / 'eval_perception.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Perception bag evaluation')
    parser.add_argument('bag_path', help='Path to ROS2 bag directory')
    parser.add_argument('--out', default='/home/peeradon/lka-carla-yolo/analysis/results',
                        help='Output directory for plots and CSV')
    args = parser.parse_args()

    print(f'Reading: {args.bag_path}')
    weather_df, lane_df = read_bag(args.bag_path)
    print(f'Weather events : {len(weather_df)}')
    print(f'Lane messages  : {len(lane_df)}')
    if not weather_df.empty:
        print(f'Weather sequence: {list(weather_df["weather"])}')

    lane_df = assign_weather(lane_df, weather_df, window_sec=60.0)
    metrics = compute_metrics(lane_df)

    print('\n── Metrics ──────────────────────────────────────────────')
    print(metrics.to_string(index=False))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # raw data CSV — every frame with weather label
    raw_cols = ['timestamp_ns', 't_in_window', 'weather', 'topic',
                'center', 'confidence', 'detected', 'lx', 'rx']
    lane_df[raw_cols].to_csv(out_dir / 'raw_frames.csv', index=False)
    print(f'Raw data saved: {out_dir}/raw_frames.csv  ({len(lane_df)} rows)')

    metrics.to_csv(out_dir / 'metrics.csv', index=False)
    print(f'Results saved to {out_dir}')


if __name__ == '__main__':
    main()
