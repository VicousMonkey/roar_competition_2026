"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np
import time
import sys
import atexit


def normalize_rad(rad: float):
    return (rad + np.pi) % (2 * np.pi) - np.pi


def filter_waypoints(location: np.ndarray, current_idx: int, waypoints: List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint: roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(location[:2] - waypoint.location[:2])
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i % len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx


class RoarCompetitionSolution:

    # Vehicle limits. Real tire_friction is 3.50; 3.3 is what the inward racing
    # line below can actually hold. Single-lap deltas under ~1s are noise.
    MU_LATERAL = 3.3
    A_BRAKE = 11.25
    A_ACCEL = 6.0
    V_MAX = 120.0
    V_MIN = 33.75

    # Corner measurement: 3 points this far apart, then smoothed.
    CURVATURE_STRIDE_M = 6.0
    CURVATURE_SMOOTH_M = 12.0

    # Pure pursuit lookahead, growing with speed.
    LOOKAHEAD_BASE_M = 8.0
    LOOKAHEAD_GAIN_S = 0.675
    LOOKAHEAD_MAX_M = 40.0

    STEER_GAIN = 6.25
    STEER_SPEED_EXP = 0.2

    THROTTLE_KP = 0.22
    BRAKE_KP = 0.28
    SPEED_DEADBAND = 0.5

    STEER_THROTTLE_CUT = 0

    LOG_INTERVAL_SEC = 1.0
    LOG_ENABLED = False

    # Every step() is one world tick of this length regardless of how long the
    # machine takes to compute it, so all timing here uses sim time, not wall
    # clock. Wall clock inflates whenever rendering lags.
    SIM_SECONDS_PER_TICK = 0.05

    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle: roar_py_interface.RoarPyActor,
        camera_sensor: roar_py_interface.RoarPyCameraSensor = None,
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor: roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor: roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor: roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor

        self._summary_printed = False
        self._logging_ready = False

    # ------------------------------------------------------------------
    # Track pre-processing - runs once, before the race
    # ------------------------------------------------------------------

    # Apex cuts, (waypoint, inset in metres). The offset only ever moves toward
    # the inside of a corner. Outward offsets fail here - the drivable surface
    # outside is narrower than the 12m lane_width metadata claims, and every
    # outward line tested DNFed. The two 18m corners take a gentler cut.
    CORNER_INSETS = [
        (511, 1.0), (711, 1.0), (874, 1.0), (1468, 1.0), (1961, 1.0), (2073, 1.0),
        (1372, 0.5), (2724, 0.5),
    ]
    CORNER_HALF_SPAN = 60
    OFFSET_SMOOTH_M = 25.0

    def _build_corner_offset(self, pts, stride, spacing):
        n = len(pts)
        off = np.zeros(n)

        prev = np.roll(pts, stride, axis=0)
        nxt = np.roll(pts, -stride, axis=0)
        turn = np.sign((pts[:, 0] - prev[:, 0]) * (nxt[:, 1] - prev[:, 1])
                       - (pts[:, 1] - prev[:, 1]) * (nxt[:, 0] - prev[:, 0]))

        # Raised cosine: 1 at the apex, 0 at the ends, never negative.
        half = self.CORNER_HALF_SPAN
        k = np.arange(-half, half + 1)
        f = k / float(half)
        shape = 0.5 * (1.0 + np.cos(np.pi * f))

        for apex_i, inset in self.CORNER_INSETS:
            d = 1.0 if turn[apex_i % n] >= 0 else -1.0
            off[(apex_i + k) % n] += d * inset * shape

        win = max(1, int(round(self.OFFSET_SMOOTH_M / max(spacing, 1e-6))))
        if win > 1:
            kern = np.ones(win) / win
            pad = np.concatenate([off[-win:], off, off[:win]])
            off = np.convolve(pad, kern, mode="same")[win:win + n]

        return np.clip(off, -2.5, 2.5)

    def _build_track_geometry(self) -> None:
        centre = np.array([wp.location[:2] for wp in self.maneuverable_waypoints], dtype=float)
        n = len(centre)
        self.n_wp = n

        seg0 = np.linalg.norm(np.roll(centre, -1, axis=0) - centre, axis=1)
        seg0[seg0 < 1e-6] = 1e-6
        spacing0 = float(seg0.sum() / n)
        stride0 = max(1, int(round(self.CURVATURE_STRIDE_M / spacing0)))

        # Shift the centerline sideways by the apex offsets to get the driven line.
        offset = self._build_corner_offset(centre, stride0, spacing0)
        tangent = np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0)
        tlen = np.linalg.norm(tangent, axis=1, keepdims=True)
        tlen[tlen < 1e-9] = 1e-9
        tangent = tangent / tlen
        normals = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

        pts = centre + offset[:, None] * normals
        self.offset = offset
        z = np.array([wp.location[2] for wp in self.maneuverable_waypoints])
        self.path_pts = np.column_stack([pts, z])

        seg = np.roll(pts, -1, axis=0) - pts
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.seg_len[self.seg_len < 1e-6] = 1e-6
        self.track_length = float(self.seg_len.sum())
        self.mean_spacing = self.track_length / n

        stride = max(1, int(round(self.CURVATURE_STRIDE_M / self.mean_spacing)))
        self.stride = stride

        # Menger curvature: 4 * triangle area / product of the three side lengths.
        prev = np.roll(pts, stride, axis=0)
        nxt = np.roll(pts, -stride, axis=0)

        a = np.linalg.norm(pts - prev, axis=1)
        b = np.linalg.norm(nxt - pts, axis=1)
        c = np.linalg.norm(nxt - prev, axis=1)

        cross = ((pts[:, 0] - prev[:, 0]) * (nxt[:, 1] - prev[:, 1])
                 - (pts[:, 1] - prev[:, 1]) * (nxt[:, 0] - prev[:, 0]))
        area = 0.5 * np.abs(cross)

        denom = a * b * c
        denom[denom < 1e-9] = 1e-9
        curvature = 4.0 * area / denom

        win = max(1, int(round(self.CURVATURE_SMOOTH_M / self.mean_spacing)))
        if win > 1:
            kernel = np.ones(win) / win
            padded = np.concatenate([curvature[-win:], curvature, curvature[:win]])
            smoothed = np.convolve(padded, kernel, mode="same")
            curvature = smoothed[win:win + n]

        self.curvature = curvature

    # A single global MU gets dragged down to whatever the worst corner
    # tolerates. Section 19 (a 31m corner, waypoints ~1465-1542) takes more grip
    # than the rest and is the only one of nine tested that improved.
    NUM_SECTIONS = 36
    SECTION_MU = {19: 3.45}

    def _build_speed_profile(self) -> None:
        k = np.maximum(self.curvature, 1e-6)
        radius = 1.0 / k

        mu = np.full(self.n_wp, float(self.MU_LATERAL))
        if self.SECTION_MU:
            sec = (np.arange(self.n_wp) * self.NUM_SECTIONS // self.n_wp)
            sec = np.clip(sec, 0, self.NUM_SECTIONS - 1)
            for s, val in self.SECTION_MU.items():
                mu[sec == int(s)] = float(val)

        v_corner = np.sqrt(mu * 9.81 * radius)
        v_corner = np.clip(v_corner, self.V_MIN, self.V_MAX)

        v = v_corner.copy()
        n = self.n_wp

        # Backward pass: every point must be able to brake down to the next one.
        for _ in range(12):
            max_change = 0.0
            for i in range(n - 1, -1, -1):
                nxt = (i + 1) % n
                limit = np.sqrt(v[nxt] ** 2 + 2.0 * self.A_BRAKE * self.seg_len[i])
                if limit < v[i]:
                    max_change = max(max_change, v[i] - limit)
                    v[i] = limit
            if max_change < 0.01:
                break

        # Forward pass: and must be reachable by accelerating from the previous one.
        for _ in range(12):
            max_change = 0.0
            for i in range(n):
                prv = (i - 1) % n
                limit = np.sqrt(v[prv] ** 2 + 2.0 * self.A_ACCEL * self.seg_len[prv])
                if limit < v[i]:
                    max_change = max(max_change, v[i] - limit)
                    v[i] = limit
            if max_change < 0.01:
                break

        self.target_speeds = v

    def _lookahead_index(self, idx: int, speed: float) -> int:
        dist = self.LOOKAHEAD_BASE_M + self.LOOKAHEAD_GAIN_S * speed
        dist = min(dist, self.LOOKAHEAD_MAX_M)

        travelled = 0.0
        i = idx
        steps = 0
        max_steps = self.n_wp - 1
        while travelled < dist and steps < max_steps:
            travelled += self.seg_len[i]
            i = (i + 1) % self.n_wp
            steps += 1
        return i

    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

        t0 = time.perf_counter()
        self._build_track_geometry()
        self._build_speed_profile()
        precompute_ms = (time.perf_counter() - t0) * 1000.0

        now = time.perf_counter()
        self.last_log_time = now
        self.wall_start_time = now

        self.step_count = 0
        self.lap_start_step = 0
        self.last_waypoint_change_step = 0
        self.lap_count = 0
        self.lap_times = []
        self.max_speed = 0.0
        self.collision_count = 0

        self._prev_waypoint_idx = self.current_waypoint_idx
        self._in_collision = False
        self._logging_ready = True

        atexit.register(self._print_summary)

        self._log("track: {} waypoints, {:.0f} m, {:.2f} m spacing".format(
            self.n_wp, self.track_length, self.mean_spacing))
        self._log("speed profile: min {:.1f} / mean {:.1f} / max {:.1f} m/s  (built in {:.0f} ms)".format(
            self.target_speeds.min(), self.target_speeds.mean(),
            self.target_speeds.max(), precompute_ms))
        self._log("starting at wp {} | logging to stderr every {:.1f}s".format(
            self.current_waypoint_idx, self.LOG_INTERVAL_SEC))

    async def step(self) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        speed = float(np.linalg.norm(vehicle_velocity))

        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

        # Aim at the racing line, not the raw centerline waypoint.
        target_idx = self._lookahead_index(self.current_waypoint_idx, speed)
        target_point = self.path_pts[target_idx]

        vector_to_waypoint = (target_point - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1], vector_to_waypoint[0])
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        # Pure pursuit, softened as speed rises.
        if speed > 1e-2:
            gain = self.STEER_GAIN / (speed ** self.STEER_SPEED_EXP)
            steer_control = -gain * delta_heading / np.pi
        else:
            steer_control = -float(np.sign(delta_heading))
        steer_control = float(np.clip(steer_control, -1.0, 1.0))

        v_target = float(self.target_speeds[self.current_waypoint_idx])
        speed_error = v_target - speed

        throttle_control = 0.0
        brake_control = 0.0
        if speed_error > self.SPEED_DEADBAND:
            throttle_control = float(np.clip(self.THROTTLE_KP * speed_error, 0.0, 1.0))
            throttle_control *= (1.0 - self.STEER_THROTTLE_CUT * abs(steer_control))
        elif speed_error < -self.SPEED_DEADBAND:
            brake_control = float(np.clip(self.BRAKE_KP * (-speed_error), 0.0, 1.0))

        control = {
            "throttle": float(np.clip(throttle_control, 0.0, 1.0)),
            "steer": steer_control,
            "brake": float(np.clip(brake_control, 0.0, 1.0)),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)

        self._update_telemetry(control, speed, v_target, delta_heading)

        return control

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if not self.LOG_ENABLED:
            return
        sim_elapsed = self.step_count * self.SIM_SECONDS_PER_TICK if hasattr(self, "step_count") else 0.0
        print("[sim {:8.2f}s] {}".format(sim_elapsed, msg), file=sys.stderr, flush=True)

    def _read_collision_impulse(self) -> float:
        if self.collision_sensor is None:
            return 0.0
        try:
            obs = self.collision_sensor.get_last_observation()
            if obs is None:
                return 0.0
            return float(np.linalg.norm(np.asarray(obs.impulse_normal, dtype=float)))
        except Exception:
            return 0.0

    def _update_telemetry(self, control, speed: float, v_target: float, delta_heading: float) -> None:
        if not self._logging_ready:
            return

        now = time.perf_counter()
        self.step_count += 1

        if speed > self.max_speed:
            self.max_speed = speed

        if self.current_waypoint_idx != self._prev_waypoint_idx:
            self.last_waypoint_change_step = self.step_count

        # Waypoint index jumping from the end of the lap back to the start.
        wrapped = (self._prev_waypoint_idx > self.n_wp * 0.75
                   and self.current_waypoint_idx < self.n_wp * 0.25)
        if wrapped:
            lap_ticks = self.step_count - self.lap_start_step
            lap_time = lap_ticks * self.SIM_SECONDS_PER_TICK
            self.lap_count += 1
            self.lap_times.append(lap_time)
            self.lap_start_step = self.step_count
            self._log("*** LAP {} COMPLETE - {:.2f}s (sim time) ***".format(self.lap_count, lap_time))

        self._prev_waypoint_idx = self.current_waypoint_idx

        # Count each contact once, not once per tick of sustained contact.
        impulse = self._read_collision_impulse()
        if impulse > 0.0:
            if not self._in_collision:
                self.collision_count += 1
                self._in_collision = True
                self._log("!!! COLLISION #{} at wp {} | impulse {:.1f} N*s | speed {:.1f} m/s".format(
                    self.collision_count, self.current_waypoint_idx, impulse, speed))
        else:
            self._in_collision = False

        if now - self.last_log_time >= self.LOG_INTERVAL_SEC:
            self.last_log_time = now
            pct = 100.0 * self.current_waypoint_idx / self.n_wp
            self._log(
                "lap {} | wp {:>5}/{} ({:5.1f}%) | {:5.1f} -> {:5.1f} m/s | "
                "err {:+6.1f}deg | steer {:+.2f} thr {:.2f} brk {:.2f}".format(
                    self.lap_count + 1, self.current_waypoint_idx, self.n_wp, pct,
                    speed, v_target, np.degrees(delta_heading),
                    float(control["steer"]), float(control["throttle"]), float(control["brake"]),
                )
            )

    def _print_summary(self) -> None:
        if self._summary_printed or not self._logging_ready:
            return
        self._summary_printed = True

        total_elapsed_sim = self.step_count * self.SIM_SECONDS_PER_TICK
        wp_elapsed_sim = self.last_waypoint_change_step * self.SIM_SECONDS_PER_TICK
        total_elapsed_wall = time.perf_counter() - self.wall_start_time
        lag_ratio = total_elapsed_wall / total_elapsed_sim if total_elapsed_sim > 0 else 1.0

        lines = ["", "=" * 60, "  RUN SUMMARY", "=" * 60]
        lines.append("  Last waypoint reached : {} / {}".format(self.current_waypoint_idx, self.n_wp))
        lines.append("  Reached at            : {:.2f}s (sim time)".format(wp_elapsed_sim))
        lines.append("  Total run time (sim)  : {:.2f}s".format(total_elapsed_sim))
        lines.append("  Total run time (wall) : {:.2f}s".format(total_elapsed_wall))
        if lag_ratio > 1.15:
            lines.append("  ^ wall time is {:.1f}x sim time -> your machine is lagging behind".format(lag_ratio))
            lines.append("    real-time. Sim time above is the number that matters for lap times.")
        lines.append("  Simulation steps      : {}  (fixed {:.0f}ms/step by definition)".format(
            self.step_count, self.SIM_SECONDS_PER_TICK * 1000))
        lines.append("  Laps completed        : {}".format(self.lap_count))
        for i, lt in enumerate(self.lap_times, start=1):
            lines.append("      lap {} : {:.2f}s (sim time)".format(i, lt))
        if self.lap_count > 0:
            lines.append("  Best lap              : {:.2f}s".format(min(self.lap_times)))
            lines.append("  Total of laps         : {:.2f}s".format(sum(self.lap_times)))
        else:
            partial = (self.step_count - self.lap_start_step) * self.SIM_SECONDS_PER_TICK
            pct = 100.0 * self.current_waypoint_idx / self.n_wp
            lines.append("  Current lap progress  : {:.1f}%  ({:.2f}s sim time elapsed)".format(pct, partial))
        lines.append("  Max speed             : {:.1f} m/s  ({:.0f} km/h)".format(
            self.max_speed, self.max_speed * 3.6))
        lines.append("  Collision events      : {}".format(self.collision_count))
        lines.append("-" * 60)
        lines.append("  Tuning: MU={} A_BRAKE={} LOOKAHEAD_GAIN={} STEER_GAIN={}".format(
            self.MU_LATERAL, self.A_BRAKE, self.LOOKAHEAD_GAIN_S, self.STEER_GAIN))
        lines.append("=" * 60 + "\n")

        print("\n".join(lines), file=sys.stderr, flush=True)
