# amr_experiments

The measured experiments behind the system paper, plus the tools that run them, analyse
them and record the videos. Results go to `$AMR_RESULTS` (default `~/amr_results`), which
is outside the repository: the CSVs are data, not source.

## The experiments

| id | runner | what it measures |
|---|---|---|
| E1 | `e1_charger_docking` | forward docking onto the wireless charger, and whether charging starts |
| E2 | `e2_cart_docking` | reverse docking onto the cart tow bar, and the release that follows |
| E3 | `e3_engagement` | whether the hitch effort separates a hooked cart from a missed one |
| E4 | `e4_corridor` | narrow passages with towing mode on and off, at several widths |
| E5 | `e5_towing` | off-tracking, hitch angle and effort along a route, against the no-cart baseline |
| E6 | `e6_endurance` | many unattended mission cycles with no reset between them |
| E7 | `e7_scenario` | all of the above once, in order, in one simulation run |

E7's route is deliberate. The cart parks west of the narrow passage and the charger
stands east of it, so the passage is driven twice: towed on the way out, empty on the
way back. The empty return goes by way of the passage's west mouth, because sent
straight to the home pose the robot goes round the north end of the wing wall instead,
which is shorter from the charger and leaves the empty pass never driven.

E1-E6 repeat one capability until its numbers are trustworthy. E7 is the opposite: one
pass of the whole mission, for showing that the parts work together and for the video.
It is not a substitute for the campaigns and its single-run numbers are not statistics.

## Running them

```bash
AMR_CART_WHEELS=trailer AMR_RESULTS=~/amr_results ROS_DOMAIN_ID=100 \
  ./amr_experiments/tools/run_experiment.sh e4 --width 1.00 --repeats 10
```

`run_experiment.sh` starts a world of its own per experiment; `run_session.sh` shares one
world between E1, E2, E5 and E6. `AMR_CART_WHEELS` picks the cart construction
(`trailer` or `caster4`) and is recorded in `campaign.txt` beside the results, because a
results directory that does not say which cart it used is not evidence of anything.

## Analysing them

```bash
ros2 run amr_experiments analyze --results ~/amr_results --out ~/amr_results/report
ros2 run amr_experiments compare --a ~/amr_results --b ~/amr_results/caster4 \
    --out ~/amr_results/report/comparison.md
```

`analyze` writes `results.md` and `figures/`; `compare` puts two campaigns in one table.
The two cart constructions are not interchangeable --- one front support is an idealised
frictionless sphere, the other a mechanism with eight joints --- so a cross-comparison is
a reference figure and not a controlled one. `campaign.txt` records which was used.

E4's clearance is **not** taken from the trial CSV. `/indoor_gps/cart` carries the 20 mm
position and 1 deg yaw noise of a real beacon system, and the logged clearance is the
minimum over about 85 noisy updates, which sits roughly 50 mm below the truth.
`corridor_clearance.py` recomputes it from the recorded trajectory with that noise
averaged over 1 s and the real deck width, and only counts a contact that lasts.

## Videos

```bash
AMR_VIDEOS=~/amr_results/videos ROS_DOMAIN_ID=100 HL_SECONDS=200 \
  ./amr_experiments/tools/record_highlight.sh e5 --repeats 1     # one experiment
  ./amr_experiments/tools/record_compare.sh   e5 --repeats 1     # both carts, for a split screen
  ./amr_experiments/tools/record_highlight.sh e7                 # the whole mission
```

`record_highlight.sh` brings up a world with the cameras on, runs a few trials and
composes the panels with `compose_highlight.py`; `compose_compare.py` joins two captures
into one split screen, with a time offset per side because the two runs reach the
interesting part at slightly different moments.

`HL_SECONDS` sets the capture length and `HL_OVERVIEW` overrides the overview camera as
six numbers, position then target. Both are worth setting for a comparison: off-tracking
only reads from overhead (`HL_OVERVIEW="4.8 0.5 8.0 4.8 -4.0 0.0"`), and the warehouse
has a roof, so a camera much above 8 m sees nothing and the panel comes out black.
