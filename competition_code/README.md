# ROAR Competition — Monza

Autonomous racing controller for the [Berkeley ROAR](https://roar.gitbook.io/roar-competition-documentation/)
competition, Monza map, Summer 2026 season.

**Result: 357.40s** over the graded three laps (~119.1s/lap), no collisions,
repeated across multiple runs.

The controller is entirely deterministic classical control — geometry and
physics, no learned policy and no neural network at race time.

---

## How it works

Everything runs once in `initialize()`; `step()` stays cheap enough to hold the
simulator's 20 Hz tick.

**1. Racing line.** The harness hands you `maneuverable_waypoints` — the track
centerline. Each of eight corners gets a lateral offset toward its inside apex,
shaped by a raised cosine so it eases in and out instead of stepping (a step in
lateral offset is a kink, i.e. curvature sharper than the corner it was meant to
help). The result is smoothed over 25 m and clamped to ±2.5 m.

The offsets only ever move the car *toward* the inside of a corner, never wide.
This is the opposite of textbook out-in-out, and it is deliberate: the drivable
surface here is narrower than the uniform 12 m `lane_width` metadata claims, and
every outward line tested went into a wall. A tracker that follows the
centerline accurately crashes at waypoint ~501 repeatedly; pure pursuit survives
that corner precisely *because* its lookahead cuts the apex.

**2. Curvature.** Menger curvature (`4 · area / (a·b·c)`) over three points 6 m
apart, smoothed with a 12 m moving average. Both windows matter — too tight and
waypoint quantisation shows up as phantom corners, too wide and real corners get
averaged away.

**3. Speed profile.** Start from the grip limit `v = √(μ·g·R)`, then two passes:

- *backward* — every point must be able to brake down to the next one
- *forward* — and must be reachable by accelerating from the previous one

These are sequential on purpose. They look vectorisable and are not: the
constraint propagates point to point, and a `np.roll`-based version moves it
only one step per sweep, silently inventing speed the car cannot reach.

Cornering grip is per-section rather than global, because one global value gets
dragged down to whatever the worst corner tolerates. Of nine grip-limited
sections tested, exactly one took more (section 19, a 31 m corner) — worth 0.45s
over three laps. Neighbouring corners of similar radius got *worse* with the
same increase, so this is corner-specific, not a rule about radius.

**4. Control.** Pure pursuit steering with a speed-dependent lookahead, aimed at
the racing line rather than the raw centerline waypoint, with gain softened as
speed rises. Throttle and brake are proportional to speed error with a deadband
between them.

## Tuning

Every constant lives at the top of `RoarCompetitionSolution`. The ones that
actually move lap time:

| constant | effect |
|---|---|
| `MU_LATERAL` | cornering grip. Higher = faster corners, more spin risk |
| `A_BRAKE` | assumed braking. Higher = brakes later |
| `LOOKAHEAD_GAIN_S` | lookahead metres per m/s. Bounded below by one fast corner, above by the last one |
| `STEER_GAIN` | steering aggressiveness. Too high and the car loses traction on turn-in |
| `CORNER_INSETS` | `(apex waypoint, inset in metres)` per corner |

**On measurement:** single-lap differences under about 1 second are noise.
`A_BRAKE = 10.0` looked 0.65s *faster* on one lap and was 0.50s *slower* over
the graded three. Confirm anything promising on a full three-lap run, twice,
before believing it.

Set `LOG_ENABLED = True` for per-second telemetry and an end-of-run summary on
stderr. Note that the summary reports both simulated and wall-clock time — only
**sim time** is graded. Wall clock inflates whenever rendering or CPU lags, and
on a loaded machine it can read 2× the real result.

---

## Setup

Requires **Python 3.8**, the CARLA 0.9.12 Monza build, and
[`roar_py`](https://github.com/augcog/roar_py). The simulator binary is several
gigabytes and is not in this repository.

```bash
conda create -n roar_competition python=3.8
conda activate roar_competition

git clone https://github.com/augcog/roar_py.git
pip install -e roar_py            # the trailing path matters
pip install -r roar_py/requirements.txt
```

Verify the install:

```bash
python -c "import roar_py_carla, roar_py_interface; print('ok')"
```

## Running

Start the Monza simulator first and let it fully load — a real load settles
around 1–3 GB of memory. Then:

```bash
cd competition_code
python competition_runner.py
```

It prints `Solution finished in <N> seconds` — that is the graded number.

If the client times out on port 2000, check for more than one `CarlaUE4.exe`
running. If a spawn fails, a vehicle left behind by a killed run is occupying
the spawn point; destroy the leftover actors or restart the simulator.

## Layout

```
competition_code/
  submission.py           the controller — the only file that is mine
  competition_runner.py   grading harness, unmodified
  infrastructure.py       grading harness, unmodified
```

`competition_runner.py` and `infrastructure.py` are the competition's own files
and are left exactly as provided.
