# amr_lab: learning the active hitch

Isaac Lab tasks for the towing AMR. Not a colcon package (`COLCON_IGNORE`): this runs on
Isaac Sim's Python in the `isaaclab` conda environment, with no ROS in the loop.

## Why this exists

The campaign in `amr_experiments/` measured the robot as it is built today: the hitch yaw
servo is limp and the cart trails behind. Those numbers are the baseline this task has to
beat, and they are blunt about the limits of a passive hitch:

| | passive hitch, E5 (`~/amr_results/report/results.md`) |
|---|---|
| towing runs finished | 9 / 10 (one jackknifed) |
| time over a 14.2 m route | 68.7 ± 23.0 s |
| off-tracking, rms | 219.6 ± 27.6 mm |
| hitch angle, max | 90.0 ± 0.0° - the joint limit, on **every** run |
| backing the cart up | not possible at any speed |

The hitch yaw servo is the one actuator on this robot that nothing has used yet. The
question is what a policy can do with it, and in particular whether it can reverse -
which is also what the fleet work needs, because a towing robot that cannot back up
cannot use any standard deadlock recovery.

## The task

`towing_env.py` couples the cart for the whole episode and commands a pose for the
**cart**. Reaching it may need forward towing, a turn, or backing the cart up; goals
behind the cart are sampled on purpose and nothing tells the policy which kind it got.

- **Actions** (3): linear velocity ±0.5 m/s, yaw rate ±1.5 rad/s, hitch angle ±1.4 rad.
  With `--passive-hitch` the third is dropped and the servo goes limp: same task, same
  reward, same randomisation, the system the campaign measured.
- **Actor observations**: only what the real robot can measure - forward speed and yaw
  rate from wheel odometry, the hitch servo's position, velocity and current - plus the
  goal against a cart pose *predicted* from the localised robot pose and the measured
  hitch angle. Four control frames of history, 52 numbers, every one of them carrying the
  device's noise, its quantisation and a calibration error that lasts the episode.
- **Critic observations**: the same plus the true cart pose and velocity, the randomised
  parameters and this episode's sensor calibration, 75 numbers. The asymmetry is what
  lets the actor learn to infer the payload from the hitch rather than be told it.
- **Rate**: 20 Hz, because that is what `amr_base` runs the MD driver's velocity loop and
  its odometry at. 200 Hz physics under it.
- **Termination**: |hitch angle| > 1.45 rad (jackknife), cart on its side, goal reached,
  or 20 s.

The asset (`asset.py`) is robot and cart as **one** articulation, so the hitch yaw is a
real joint PhysX reports an effort for and accepts a target on. The hook holds the bar in
a bearing that turns about Y only, so the hitch yaw joint *is* the hitch angle β. Masses,
inertias and the coupling geometry come from the same files the ROS simulator reads.

## Why the payload sweep is the headline, not a detail

`authority.py` drives the machine open loop over a grid of payloads, hitch limits and
floor frictions. The full tables are in `~/amr_results/amr_lab/`; measured 2026-10-17 on
the four-caster cart with the fitted 10.6 Nm hitch, against a 0.5 m/s command:

| floor mu | 0 kg | 20 kg | 60 kg | 100 kg |
|---|---|---|---|---|
| 0.4 | 0.215 +- 0.159 | 0.206 +- 0.153 | 0.177 +- 0.167 | 0.172 +- 0.168 |
| 0.7 | 0.448 +- 0.045 | 0.465 +- 0.030 | 0.417 +- 0.132 | 0.444 +- 0.067 |
| 1.0 | 0.466 +- 0.020 | 0.484 +- 0.007 | 0.459 +- 0.033 | 0.483 +- 0.007 |

- **The floor binds, and it binds independent of payload.** Below about mu = 0.7 the
  robot cannot hold speed with an *empty* cart, and 100 kg barely changes it. Payload
  adds drag, but it also presses down on the hook and so adds normal load to the drive
  wheels; the two roughly cancel. The real robot having pulled 100 kg is consistent with
  this.
- **The fitted hitch servo is enough for the four-caster cart.** It reaches 71-78 deg of
  the 69 deg commanded at every payload up to 122.6 kg total, using 1.1-3.3 Nm of its
  10.6. Raising the limit to 20 or 30 Nm changes nothing.
- **The trailer configuration is what wants more hitch torque.** At 10.6 Nm it averages
  58-64 deg with a standard deviation of 23-28 deg: it often cannot hold the angle. At
  20-30 Nm the spread falls to 5-10 deg. A fixed rear axle resists cart yaw; four swivel
  casters do not.
- **The caster scrub coefficient is unknown for the real cart** and is randomised
  (`swivel_scrub_range`) rather than assumed. It has to be measured on hardware.

Wheel motor torque is deliberately not reported: a velocity loop with damping 1000
saturates on a rounding error, so `applied_torque` says nothing about demand. Measuring
the torque a heavy cart demands needs a torque-controlled run, which this script is not.

`--payload MIN MAX` sets the mass added on the deck (on top of the 22.6 kg cart),
`--hitch-torque MIN MAX` the servo limit, `--pitch-target` the angle the hook is held at
while towing. Mass, centre of mass and inertia all move together: setting the mass alone
gives a cart that weighs 100 kg and turns like an empty one.

## Running it

```bash
conda activate isaaclab
cd ~/Repositories/amr-ros2

# does the asset build, does the task step, is the geometry right
python amr_lab/smoke.py --headless --num-envs 16 --cart-wheels trailer

# train
python amr_lab/train.py --headless --num-envs 2048 --max-iterations 1500
python amr_lab/train.py --headless --passive-hitch --run-name baseline   # the baseline
python amr_lab/train.py --headless --cart-wheels caster4 --payload 0 60

# the table, against a checkpoint
python amr_lab/play.py --headless --checkpoint logs/amr_lab/trailer/<run>/model_1500.pt
python amr_lab/play.py --num-envs 32 --checkpoint ...                    # watch it

# which actuator runs out first as the cart is loaded - no policy involved
python amr_lab/authority.py --headless
python amr_lab/authority.py --headless --payloads 0 20 40 60 80 \
    --hitch-torques 10.6 20 30 --cart-wheels caster4
```

`authority.py` is the measurement in the table above and it is worth running before any
training: it drives each (payload, servo limit, scrub) cell open loop through full speed
ahead, a hitch command while rolling, and the same command standing still. No policy is
involved, so what it reports is a property of the machine.

The training log carries `Episode/success_rate` and `Episode/jackknife_rate` alongside
the return, so a run that is collecting reward without doing the task shows up.

Logs and checkpoints land in `logs/amr_lab/<cart>[_passive]/<timestamp>_<run name>/`.

## Ablations the flags give you

| flag | question |
|---|---|
| `--passive-hitch` | what does the active hitch actually buy? |
| `--cart-wheels caster4` | does it still work when nothing holds the cart's heading? |
| `--true-cart-obs` | how much is lost by not knowing where the cart really is? |
| `--payload 0 60` | where does the servo run out of authority? |
| `--hitch-torque 6 30` | would a larger servo move that point? |
| `--legacy-obs` | what does being honest about the sensors cost? |
| `--no-sensor-noise` | how much of the final number is the noise model? |
| `--control-hz 50` | does the policy need to run faster than the driver can? |

`--true-cart-obs` is the interesting one on the four-caster cart. There the kinematic
prediction is wrong by construction - nothing holds the cart's heading, which is what the
two-cart comparison found - so the gap between the two runs measures what the missing
observability costs.

## Sim to real

The observation is the hardware's sensor set, not the simulator's state, and the sizes
come from the drivers rather than a guess:

| channel | source | noise | per-episode error | step |
|---|---|---|---|---|
| forward speed | reported motor rpm, 20 Hz | 0.01 m/s | scale 0.97-1.03 (tyre radius under load) | 2.4 mm/s |
| yaw rate | the same, across the track | 0.02 rad/s | scale 0.98-1.02 (track width) | 0.0129 rad/s |
| hitch angle | XM540 present position | 0.003 rad | zero offset ±2° (`zero_ticks` is a hand calibration) | 2π/4096 |
| hitch rate | XM540 present velocity | 0.02 rad/s | - | as above / dt |
| hitch torque | XM540 present current | 5% of the limit | - | - |
| robot pose | the localiser | 20 mm, 0.017 rad | bias 50 mm, 0.035 rad | - |
| command | cmd_vel → 19200 baud → driver | - | 0-2 control steps of delay | - |

Two channels of the first version were dropped because the robot has no sensor for them:
the body-frame **lateral velocity**, which differential-drive odometry defines as zero
(in simulation it is the real slip the hitch side-force causes, which is exactly why a
policy would lean on it), and the **tow bar angle**, which turns in an unsensored bearing
inside the hook. `--legacy-obs` puts them back, as an ablation.

The hitch zero offset is the one to watch: the cart pose is *predicted* from that angle,
so a couple of degrees of calibration error walks straight into where the policy thinks
the cart is - as it will on the robot.

## Known limits

- Flat ground, no obstacles, no narrow passage. The corridor work needs the swept volume
  and belongs with the planning paper.
- No sensors: no lidar, no cameras, no SLAM. The policy is proprioceptive on purpose.
- The trailer cart's front support is still an idealised frictionless sphere and the
  four-caster cart's is a mechanism, exactly as in the ROS simulator, so cross-cart
  numbers are indicative rather than a controlled comparison.
- The hook slot and the tow bar are an ideal revolute: there is no mechanical play. In
  reverse the cart does not respond until the play is taken up, which matters most for
  the one manoeuvre this task exists for. Measure it on the robot.
- The MD driver's internal acceleration ramp is not modelled, only its command rate and
  its clamp (0.5 m/s, 1.5 rad/s, from `base_node.py`).
- Nothing here has run on hardware. The hitch effort is the servo drive torque
  (Kp·error − Kd·rate, clipped), computed the same way as in `amr_isaac/amr_sim.py`, so
  it means what the Dynamixel current means - but the constant still has to be measured.
