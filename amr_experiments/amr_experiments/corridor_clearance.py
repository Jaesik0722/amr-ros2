"""Clearance to the corridor walls, recomputed from the recorded trajectories.

E4 logs a clearance taken straight from `/indoor_gps/cart`, and that number is biased.
Two reasons, both of which only became visible once the four-caster campaign produced
negative clearances where the simulator shows no collision:

1. **The cart pose is noisy.** The simulated indoor GPS carries sigma = 20 mm in position
   and 1 deg in yaw (`amr_isaac/world.py`) at 5 Hz, which is what the real Marvelmind-class
   beacons do. One pass through the passage is about 85 independent updates, and the
   logged clearance is the *minimum* over all of them: the minimum of 85 draws from a
   20 mm normal sits about 2.6 sigma, roughly 50 mm, below the truth. The logged number
   is therefore an extreme value of the noise, not the closest the cart came.
2. **The deck was given 20 mm of margin.** The clearance used half-extents of
   0.42 x 0.30 m for a deck that is 0.84 x 0.56 m, so every reading is a further 20 mm
   pessimistic on each side.

This module recomputes both from the trajectory CSVs: the cart pose averaged over five
GPS updates (1.0 s, which cuts the noise to about 9 mm without smearing the slow lateral
motion that matters here), the true deck half-width, and a contact that has to last
0.5 s before it counts - a single sample below zero is still noise, a sustained one is
the cart riding the wall.

The robot pose comes from `/odom`, which is planar ground truth in simulation, so only
the cart term is filtered.
"""

import csv
import glob
import math
import os
import statistics

DECK_HALF_LENGTH = 0.42
DECK_HALF_WIDTH = 0.28           # deck 0.84 x 0.56 m
GPS_UPDATES = 5                  # averaged, 1.0 s at 5 Hz
CONTACT_SECONDS = 0.5


def corners(pose, half_length=DECK_HALF_LENGTH, half_width=DECK_HALF_WIDTH):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * px - s * py, y + s * px + c * py)
            for px, py in ((half_length, half_width), (-half_length, half_width),
                           (-half_length, -half_width), (half_length, -half_width))]


def read_trajectory(path):
    """[(t, robot x, robot y, measured cart pose)], samples with no cart pose dropped."""
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                cart = (float(row['cart_x']), float(row['cart_y']), float(row['cart_yaw']))
                sample = (float(row['t']), float(row['x']), float(row['y']), cart)
            except (KeyError, TypeError, ValueError):
                continue
            if any(math.isnan(v) for v in cart):
                continue
            out.append(sample)
    return out


def average_cart(samples, updates=GPS_UPDATES):
    """Average the cart pose over `updates` indoor-GPS updates, centred.

    The trajectory is logged at the odometry rate and the cart pose at 5 Hz, so the same
    measurement repeats six to ten times in a row. Averaging over samples would average
    those copies and remove nothing; the distinct updates have to be found first.
    """
    if updates <= 1 or not samples:
        return samples
    distinct, previous = [], None
    for i, sample in enumerate(samples):
        if sample[3] != previous:
            distinct.append((i, sample[3]))
            previous = sample[3]
    half = updates // 2
    smoothed = []
    for j, (index, _) in enumerate(distinct):
        block = [p for _, p in distinct[max(0, j - half):j + half + 1]]
        smoothed.append((index, (statistics.fmean(p[0] for p in block),
                                 statistics.fmean(p[1] for p in block),
                                 statistics.fmean(p[2] for p in block))))
    out, j = [], 0
    for i, sample in enumerate(samples):
        while j + 1 < len(smoothed) and smoothed[j + 1][0] <= i:
            j += 1
        out.append((sample[0], sample[1], sample[2], smoothed[j][1]))
    return out


def measure(samples, width, corridor=(3.0, 5.5, -4.5)):
    """(clearance [m], seconds in contact, longest unbroken contact [s]) for one pass."""
    x0, x1, y = corridor
    half = width / 2.0
    worst, run, longest, touching = float('inf'), 0, 0, 0
    if len(samples) < 2:
        return float('nan'), 0.0, 0.0
    dt = (samples[-1][0] - samples[0][0]) / (len(samples) - 1)
    for _, rx, ry, cart in samples:
        if not (x0 <= rx <= x1 and abs(ry - y) < half):
            continue
        gap = float('inf')
        for px, py in [(rx, ry)] + corners(cart):
            if x0 <= px <= x1:
                gap = min(gap, half - abs(py - y))
        if gap == float('inf'):
            continue
        worst = min(worst, gap)
        if gap <= 0.0:
            touching += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return (worst if worst < float('inf') else float('nan')), touching * dt, longest * dt


def recompute(csv_path, corridor=(3.0, 5.5, -4.5)):
    """{trial: (clearance, contact seconds, longest contact)} for one width's CSV.

    Returns {} when the trajectories are not beside the CSV, so a caller can fall back
    to the logged column rather than report nothing.
    """
    directory = os.path.join(os.path.dirname(csv_path), 'trajectories')
    base = os.path.basename(csv_path)[:-4]
    with open(csv_path) as f:
        widths = {row['trial']: row['width'] for row in csv.DictReader(f)}
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, f'{base}_*.csv'))):
        tail = os.path.basename(path)[:-4][len(base) + 1:]
        if not tail.isdigit():
            continue
        trial = str(int(tail))
        if trial not in widths or not widths[trial]:
            continue
        samples = average_cart(read_trajectory(path))
        if samples:
            out[trial] = measure(samples, float(widths[trial]), corridor)
    return out
