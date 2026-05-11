#!/usr/bin/python3
"""
Automated closed-loop evaluation — 12 trials (3 methods × 4 weathers).

Prerequisites:
  - CARLA is running
  - Bridge + ego spawned (no teleop window):
      ros2 launch lka_bringup bring_up_carla.launch.py
  - Workspace is built: cd lka_ws && colcon build --symlink-install

Usage:
  python3 run_trials.py                       # all 12 trials
  python3 run_trials.py --methods yolo scnn   # only these methods
  python3 run_trials.py --weathers clear rain # only these weathers
  python3 run_trials.py --dry-run             # print plan, do nothing

Per-trial sequence:
  1. respawn     — teleport ego to spawn via set_transform, wait to settle
  2. perception  — launch selected node, wait for detected=True via rclpy subscriber
  3. weather     — publish weather preset once
  4. record      — start bag recorder (ros2 bag record)
  5. controller  — launch Pure Pursuit; ego starts driving
  6. drive       — monitor odometry x via rclpy subscriber until x < STOP_X or timeout
  7. stop        — kill controller → recorder → perception (in order)
  8. verify      — confirm /lka/lane_center goes quiet (no messages for 3 s)
"""

import argparse
import collections
import csv
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from lka_msgs.msg import LaneCenter

# ── Project paths ──────────────────────────────────────────────────────────────
ROOT       = Path('/home/peeradon/lka-carla-yolo')
SETUP_BASH = ROOT / 'lka_ws' / 'install' / 'setup.bash'

# ── Spawn point ────────────────────────────────────────────────────────────────
# ros_pose_to_carla_transform does carla.Location(x, -y, z).
# Publish y=-195.158 → CARLA y=+195.158 (correct lane position).
# z=2.2 drops the vehicle onto the road via CARLA physics.
SPAWN_POSE = (
    '{position: {x: 317.099, y: -195.158, z: 2.2},'
    ' orientation: {x: 0.0, y: 0.0, z: 1.0, w: 0.0}}'
)

# ── Weather presets ────────────────────────────────────────────────────────────
WEATHER_PRESETS = {
    'rain':  ('{cloudiness: 60.0, precipitation: 40.0, precipitation_deposits: 40.0,'
              ' wind_intensity: 30.0, sun_azimuth_angle: 275.0, sun_altitude_angle: 20.0,'
              ' fog_density: 5.0, fog_distance: 0.75, wetness: 80.0}'),
    'clear': ('{cloudiness: 0.0, precipitation: 0.0, precipitation_deposits: 0.0,'
              ' wind_intensity: 0.0, sun_azimuth_angle: 0.0, sun_altitude_angle: 75.0,'
              ' fog_density: 0.0, fog_distance: 0.0, wetness: 0.0}'),
    'fog':   ('{cloudiness: 80.0, precipitation: 0.0, precipitation_deposits: 0.0,'
              ' wind_intensity: 0.0, sun_azimuth_angle: 0.0, sun_altitude_angle: 45.0,'
              ' fog_density: 80.0, fog_distance: 10.0, wetness: 0.0}'),
    'night': ('{cloudiness: 0.0, precipitation: 0.0, precipitation_deposits: 0.0,'
              ' wind_intensity: 0.0, sun_azimuth_angle: 0.0, sun_altitude_angle: -90.0,'
              ' fog_density: 0.0, fog_distance: 0.0, wetness: 0.0}'),
}

# ── Bag topics ─────────────────────────────────────────────────────────────────
RECORD_TOPICS = [
    '/lka/lane_center',
    '/lka/gt/cross_track_m',
    '/carla/ego_vehicle/odometry',
    '/carla/ego_vehicle/vehicle_control_cmd',
    '/carla/weather_control',
]

# ── Timing / thresholds ────────────────────────────────────────────────────────
TRIAL_TIMEOUT_S          = 120     # hard timeout per trial
PERCEPTION_READY_TIMEOUT = 45      # max wait for first detected=True (s)
QUIET_TIMEOUT_S          = 3.0     # seconds of silence to call perception "stopped"

BRAKE_WAIT_S    = 2    # hold brake before teleport
SETTLE_WAIT_S   = 7    # wait after teleport for physics to settle
RECORDER_INIT_S = 1    # pause after starting recorder
KILL_WAIT_S     = 8    # seconds to wait for process to die

CALIB_COLLECT_S   = 5.0   # seconds to collect centers for bias estimation
CALIB_MIN_SAMPLES = 50    # minimum detected frames required for valid bias

CALIB_LOG = ROOT / 'bags' / 'closed_loop' / 'calibration_log.csv'

METHODS  = ['yolo', 'pure_vision', 'scnn']
WEATHERS = ['rain', 'clear', 'fog', 'night']


# ── ROS2 monitor node ──────────────────────────────────────────────────────────

class TrialMonitor(Node):
    """Subscribes to lane_center and controller/state for the trial runner."""

    def __init__(self):
        super().__init__('trial_monitor')
        self._lock = threading.Lock()

        self._detected:         bool  = False
        self._ctrl_state:       str   = ''
        self._last_lane_ts:     float = 0.0   # wall-clock of last /lka/lane_center msg
        self._recent_centers: collections.deque = collections.deque(maxlen=200)

        self.create_subscription(LaneCenter, '/lka/lane_center',      self._lane_cb,  10)
        self.create_subscription(String,     '/lka/controller/state', self._state_cb, 10)

    def _lane_cb(self, msg: LaneCenter):
        with self._lock:
            self._detected     = bool(msg.detected)
            self._last_lane_ts = time.monotonic()
            if msg.detected:
                self._recent_centers.append(float(msg.center))

    def _state_cb(self, msg: String):
        with self._lock:
            self._ctrl_state = msg.data

    def detected(self) -> bool:
        with self._lock:
            return self._detected

    def ctrl_state(self) -> str:
        with self._lock:
            return self._ctrl_state

    def reset(self):
        """Clear stale state from the previous trial before starting a new one."""
        with self._lock:
            self._ctrl_state   = ''
            self._detected     = False
            self._last_lane_ts = 0.0
            self._recent_centers.clear()

    def clear_centers(self):
        with self._lock:
            self._recent_centers.clear()

    def calibrate_bias(self, min_samples: int = CALIB_MIN_SAMPLES) -> tuple:
        """Return (bias, n) where bias = mean(centers) - 0.5, or (None, n) if too few samples."""
        with self._lock:
            centers = list(self._recent_centers)
        n = len(centers)
        if n < min_samples:
            return None, n
        return sum(centers) / n - 0.5, n

    def center_count(self) -> int:
        with self._lock:
            return len(self._recent_centers)

    def seconds_since_last_lane(self) -> float:
        with self._lock:
            if self._last_lane_ts == 0.0:
                return float('inf')
            return time.monotonic() - self._last_lane_ts


def spin_for(executor, seconds: float):
    """Spin the executor for `seconds` seconds."""
    t0 = time.time()
    while time.time() - t0 < seconds:
        executor.spin_once(timeout_sec=0.1)


# ── Process helpers ────────────────────────────────────────────────────────────

def _popen(cmd: str) -> subprocess.Popen:
    """Launch a bash command in its own process group; stdout/stderr discarded."""
    return subprocess.Popen(
        ['bash', '-c', f'source {SETUP_BASH} && {cmd}'],
        preexec_fn=os.setsid,      # own process group — kills all child nodes cleanly
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_proc(proc: subprocess.Popen, name: str):
    if proc is None or proc.poll() is not None:
        return
    print(f'  [stop] {name} ...', flush=True)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=KILL_WAIT_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()


def ros2_once(topic: str, msg_type: str, payload: str) -> int:
    cmd = f'source {SETUP_BASH} && ros2 topic pub --once {topic} {msg_type} "{payload}"'
    return subprocess.call(['bash', '-c', cmd],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── Step helpers ───────────────────────────────────────────────────────────────

def reset_vehicle(executor):
    """Step 1 — brake then teleport ego to spawn, wait for physics to settle."""
    print('  [1-respawn] braking ...', flush=True)
    ros2_once('/carla/ego_vehicle/vehicle_control_cmd',
              'carla_msgs/msg/CarlaEgoVehicleControl',
              '{throttle: 0.0, steer: 0.0, brake: 1.0, hand_brake: true}')
    spin_for(executor, BRAKE_WAIT_S)

    print('  [1-respawn] teleporting via set_transform ...', flush=True)
    ret = ros2_once('/carla/ego_vehicle/control/set_transform',
                    'geometry_msgs/msg/Pose', SPAWN_POSE)
    if ret != 0:
        print(f'  [1-respawn] WARNING: set_transform failed (exit {ret}) — is bridge running?')

    print(f'  [1-respawn] settling {SETTLE_WAIT_S} s ...', flush=True)
    spin_for(executor, SETTLE_WAIT_S)


def start_perception(method: str) -> subprocess.Popen:
    """Step 2a — launch perception node."""
    cmd = f'ros2 launch lka_bringup perception_only.launch.py method:={method}'
    print(f'  [2-perception] starting {method} ...', flush=True)
    return _popen(cmd)


def wait_for_perception(monitor: TrialMonitor, executor,
                        timeout_s: int = PERCEPTION_READY_TIMEOUT) -> bool:
    """Step 2b — spin until /lka/lane_center publishes detected=True."""
    print(f'  [2-perception] waiting for detected=True (timeout {timeout_s} s) ...', flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        executor.spin_once(timeout_sec=0.2)
        if monitor.detected():
            print('  [2-perception] OK — detected=True', flush=True)
            return True
    return False


def set_weather(weather: str):
    """Step 3 — publish weather preset."""
    print(f'  [3-weather] → {weather.upper()} ...', flush=True)
    ros2_once('/carla/weather_control',
              'carla_msgs/msg/CarlaWeatherParameters',
              WEATHER_PRESETS[weather])


def start_recorder(method: str, weather: str, rep: int) -> subprocess.Popen:
    """Step 4 — start bag recorder, wait briefly for it to initialise."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_path = str(ROOT / 'bags' / 'closed_loop' / f'{method}_{weather}_rep{rep}_{ts}')
    cmd = ('ros2 bag record --output ' + bag_path
           + ' --storage sqlite3 ' + ' '.join(RECORD_TOPICS))
    print(f'  [4-record] bag → {bag_path}', flush=True)
    return _popen(cmd)


def start_controller(bias_offset: float = 0.0) -> subprocess.Popen:
    """Step 5 — launch Pure Pursuit controller with calibrated bias."""
    cmd = (f'ros2 launch lka_control lka_controller.launch.py '
           f'wheel_base:=3.0046 ld_velocity_ratio:=2.4 '
           f'max_steer_rad:=1.2217 throttle:=0.3 '
           f'bias_offset:={bias_offset:.6f}')
    print(f'  [5-controller] starting — bias_offset={bias_offset:+.4f} ...', flush=True)
    return _popen(cmd)


def calibrate_perception_bias(monitor: TrialMonitor, executor) -> float:
    """Step 2.5 — collect CALIB_COLLECT_S seconds of center samples while stationary, return bias."""
    print(f'  [2.5-calibrate] collecting {CALIB_COLLECT_S:.0f} s of center samples ...', flush=True)
    monitor.clear_centers()
    spin_for(executor, CALIB_COLLECT_S)
    bias, n = monitor.calibrate_bias()
    if bias is None:
        print(f'  [2.5-calibrate] WARNING: only {n} samples (need {CALIB_MIN_SAMPLES}) — using bias=0.0', flush=True)
        return 0.0
    print(f'  [2.5-calibrate] bias={bias:+.4f}  (n={n})', flush=True)
    return bias


def _write_calib_log(method: str, weather: str, rep: int, bias: float, n_samples: int):
    CALIB_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CALIB_LOG.exists() or CALIB_LOG.stat().st_size == 0
    with open(CALIB_LOG, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['timestamp', 'method', 'weather', 'rep', 'bias_offset', 'samples_used'])
        w.writerow([datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                    method, weather, rep, f'{bias:.6f}', n_samples])


def wait_for_stop(monitor: TrialMonitor, executor) -> str:
    """Step 6 — spin until controller reports goal_reached or timeout."""
    deadline    = time.monotonic() + TRIAL_TIMEOUT_S
    last_print  = -5.0
    stopped_by  = 'timeout'
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.2)
        state   = monitor.ctrl_state()
        elapsed = int(time.monotonic() - (deadline - TRIAL_TIMEOUT_S))
        if elapsed - last_print >= 3:
            print(f'  t+{elapsed:3d}s  controller_state={state!r}', flush=True)
            last_print = elapsed
        if state == 'goal_reached':
            stopped_by = 'goal_reached'
            break
    return stopped_by


def verify_quiet(monitor: TrialMonitor, executor) -> bool:
    """Step 8 — return True if /lka/lane_center is silent for QUIET_TIMEOUT_S."""
    print(f'  [8-verify] checking /lka/lane_center quiet ({QUIET_TIMEOUT_S:.0f} s) ...', flush=True)
    spin_for(executor, QUIET_TIMEOUT_S)
    silence = monitor.seconds_since_last_lane()
    if silence >= QUIET_TIMEOUT_S:
        print('  [8-verify] OK — no data')
        return True
    print(f'  [8-verify] WARNING: data still arriving ({silence:.1f} s ago)')
    return False


# ── Trial runner ───────────────────────────────────────────────────────────────

def run_trial(method: str, weather: str, rep: int, idx: int, total: int,
              monitor: TrialMonitor, executor, dry_run: bool) -> bool:
    tag = f'[{idx}/{total}] {method} × {weather}  rep={rep}'
    print(f'\n{"─" * 60}')
    print(f'{tag}')

    if dry_run:
        print(f'  [2-perception] start, wait detected=True')
        print(f'  [2.5-calibrate] collect {CALIB_COLLECT_S:.0f}s, compute bias')
        print(f'  [3-weather] set weather')
        print(f'  [4-record] start recorder  → <method>_<weather>_rep{rep}_<ts>')
        print(f'  [5-controller] launch Pure Pursuit with bias')
        print(f'  [6-drive] wait goal_reached or timeout')
        print(f'  [7-stop] kill controller → recorder → perception')
        print(f'  [8-verify] check quiet')
        return True

    monitor.reset()   # clear goal_reached / detected from previous trial

    perception_proc = None
    recorder_proc   = None
    controller_proc = None
    bias_offset     = 0.0
    try:
        # 2. Perception
        perception_proc = start_perception(method)
        if not wait_for_perception(monitor, executor):
            print(f'  [2-perception] WARNING: no detected=True after {PERCEPTION_READY_TIMEOUT} s — aborting')
            return False

        # 2.5. Calibrate bias — vehicle is stationary at spawn, perception already running
        bias_offset = calibrate_perception_bias(monitor, executor)
        _write_calib_log(method, weather, rep, bias_offset, monitor.center_count())

        # 3. Weather
        set_weather(weather)

        # 4. Recorder
        recorder_proc = start_recorder(method, weather, rep)
        spin_for(executor, RECORDER_INIT_S)

        # 5. Controller
        controller_proc = start_controller(bias_offset)

        # 6. Drive
        stopped_by = wait_for_stop(monitor, executor)
        print(f'  [6-drive] stopped — {stopped_by}')

    except KeyboardInterrupt:
        print(f'\n{tag}  interrupted — aborting all trials')
        _stop_proc(controller_proc, 'controller')
        _stop_proc(recorder_proc,   'recorder')
        _stop_proc(perception_proc, 'perception')
        sys.exit(1)

    finally:
        # 7. Stop: controller → recorder → perception
        _stop_proc(controller_proc, 'controller')
        _stop_proc(recorder_proc,   'recorder')
        _stop_proc(perception_proc, 'perception')

    # 8. Verify quiet
    verify_quiet(monitor, executor)

    print(f'  {tag} done')
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Automated 12-trial closed-loop evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--methods',  nargs='+', default=METHODS,  choices=METHODS,
                        metavar='METHOD',  help='methods to run (default: all 3)')
    parser.add_argument('--weathers', nargs='+', default=WEATHERS, choices=WEATHERS,
                        metavar='WEATHER', help='weathers to run (default: all 4)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per (method × weather) condition (default: 3)')
    parser.add_argument('--skip-respawn', action='store_true',
                        help='do not reset vehicle between trials')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the trial plan without running anything')
    parser.add_argument('--test-respawn', action='store_true',
                        help='run the respawn sequence once and exit')
    args = parser.parse_args()

    if not args.dry_run and not SETUP_BASH.exists():
        print(f'\nERROR: workspace not built — {SETUP_BASH} not found')
        print('  Run: cd lka_ws && colcon build --symlink-install && source install/setup.bash')
        sys.exit(1)

    rclpy.init()
    monitor  = TrialMonitor()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(monitor)

    if args.test_respawn:
        print('[test-respawn] running ...')
        reset_vehicle(executor)
        print('[test-respawn] done')
        monitor.destroy_node()
        rclpy.shutdown()
        return

    # Outer loop = repeat round so all (method×weather) get rep=1 before any gets rep=2
    trials   = [(m, w, r + 1) for r in range(args.repeats)
                               for m in args.methods
                               for w in args.weathers]
    total    = len(trials)
    secs_per = (TRIAL_TIMEOUT_S + CALIB_COLLECT_S
                + (0 if args.skip_respawn else BRAKE_WAIT_S + SETTLE_WAIT_S + KILL_WAIT_S))
    eta      = timedelta(seconds=total * secs_per)

    print('═' * 60)
    print('  Closed-loop trial runner')
    print(f'  methods   : {args.methods}')
    print(f'  weathers  : {args.weathers}')
    print(f'  repeats   : {args.repeats}')
    print(f'  trials    : {total}  ({args.repeats} × {len(args.methods)} methods × {len(args.weathers)} weathers)')
    print(f'  stop point: controller state=goal_reached  (timeout {TRIAL_TIMEOUT_S} s)')
    print(f'  respawn   : {"OFF" if args.skip_respawn else "ON"}')
    print(f'  dry-run   : {args.dry_run}')
    print(f'  est. total: ~{int(eta.total_seconds() // 60)} min')
    print(f'  started   : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('═' * 60)

    if not args.dry_run:
        first_weather = args.weathers[0]
        print(f'[pre-run] setting initial weather → {first_weather.upper()} ...')
        set_weather(first_weather)

    try:
        for i, (method, weather, rep) in enumerate(trials, start=1):
            # 1. Respawn
            if not args.skip_respawn:
                if args.dry_run:
                    print(f'[{i}/{total}] [1-respawn] brake {BRAKE_WAIT_S}s → set_transform → settle {SETTLE_WAIT_S}s')
                else:
                    print(f'[{i}/{total}] [1-respawn] resetting vehicle ...')
                    reset_vehicle(executor)

            run_trial(method, weather, rep, i, total, monitor, executor, args.dry_run)

    except KeyboardInterrupt:
        print('\nInterrupted.')
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

    print(f'\n{"═" * 60}')
    print(f'  All {total} trial(s) complete')
    print(f'  bags → {ROOT / "bags" / "closed_loop"}/  ')
    print(f'  calib log → {CALIB_LOG}')
    print(f'  eval → python3 analysis/eval_controller.py bags/closed_loop/')
    print(f'  done : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('═' * 60)


if __name__ == '__main__':
    main()
