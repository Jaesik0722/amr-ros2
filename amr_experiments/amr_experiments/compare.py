"""Put two campaigns side by side: the same experiments, two cart constructions.

  ros2 run amr_experiments compare \
      --a ~/amr_results        --a-label 'single-axle trailer' \
      --b ~/amr_results/caster4 --b-label 'four-caster trolley' \
      --out ~/amr_results/report/comparison.md

Each directory is a results tree written by run_experiment.sh: <dir>/e2, <dir>/e4, ...
Only the experiments present in both are compared, and the cart construction each
directory used is read from its campaign.txt rather than assumed from the path, so a
report cannot quietly claim a comparison that was never run.

The two constructions are not modelled at the same level (CLAUDE.md trap 23): the
trailer's front support is an idealised frictionless sphere, the four-caster cart has
four real swivel mechanisms. Every table here therefore carries that caveat, because a
difference between the columns can come from the carts or from the modelling.
"""

import argparse
import glob
import math
import os
import sys

from .analyze import cell, fisher_exact, read_csv, successes
from .corridor_clearance import CONTACT_SECONDS, recompute
from .trial_log import describe, numbers

CAVEAT = (
    'The two carts are not modelled at the same level of detail: the single-axle '
    'trailer rolls on an idealised frictionless sphere at the front and a free axle at '
    'the back, while the four-caster cart has four swivel mechanisms with a 35 mm trail, '
    'each with its own swivel and wheel joint. Both have the same total mass (23.0 kg). '
    'A difference between the columns is therefore a difference between these two '
    'models, which is a reference point, not a controlled comparison of two real carts.')


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--a', required=True, help='first results directory')
    ap.add_argument('--b', required=True, help='second results directory')
    ap.add_argument('--a-label', default=None)
    ap.add_argument('--b-label', default=None)
    ap.add_argument('--out', default=None, help='markdown file to write (default: stdout)')
    return ap.parse_args(argv if argv is not None else sys.argv[1:])


def campaign(directory, fallback):
    """Read the cart construction recorded beside the CSVs, or return the fallback."""
    path = os.path.join(directory, 'campaign.txt')
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                key, _, value = line.partition(':')
                if key.strip() == 'cart_wheels' and value.strip():
                    return value.strip()
    return fallback


def load(directory, experiment):
    """Every CSV of one experiment in one campaign, concatenated, trajectories excluded.

    E4 rows gain the clearance recomputed from their trajectory: the logged column is an
    extreme value of the indoor GPS's own noise (see corridor_clearance).
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, experiment, 'e*_*.csv'))):
        if os.sep + 'trajectories' + os.sep in path:
            continue
        block = read_csv(path)
        if experiment == 'e4':
            fixed = recompute(path)
            for row in block:
                clearance, seconds, longest = fixed.get(row['trial'], (None, None, None))
                row['clearance'] = '' if clearance is None else f'{clearance:.4f}'
                row['contact_longest'] = '' if longest is None else f'{longest:.2f}'
        rows += block
    return rows


def pair_table(title, note, labels, metrics, left, right):
    """One table of 'quantity | A | B' rows; metrics are (label, fn) over a row set."""
    lines = [f'### {title}', '']
    if note:
        lines += [note, '']
    lines += [f'| quantity | {labels[0]} | {labels[1]} |', '|---|---|---|']
    for label, fn in metrics:
        lines.append(f'| {label} | {fn(left)} | {fn(right)} |')
    return lines + ['']


def rate(rows, key):
    if not rows:
        return '-'
    hits = sum(1 for r in rows if str(r.get(key, '')).strip() in ('1', 'True', 'true'))
    return f'{hits}/{len(rows)}'


def docking(title, left, right, labels):
    metrics = [('trials', lambda r: f'{len(successes(r))}/{len(r)}'),
               ('docking time [s]', lambda r: cell(successes(r), 'dock_time', 1.0, 2)),
               ('longitudinal error [mm]',
                lambda r: cell(successes(r), 'err_long', 1000.0, 1)),
               ('lateral error [mm]', lambda r: cell(successes(r), 'err_lat', 1000.0, 1)),
               ('yaw error [deg]',
                lambda r: cell(successes(r), 'err_yaw', 180.0 / math.pi, 2)),
               ('position error [mm]', lambda r: cell(successes(r), 'err_xy', 1000.0, 1)),
               ('hitch effort, hooked [Nm]',
                lambda r: cell(successes(r), 'effort_loaded', 1.0, 3)),
               ('hitch effort, free [Nm]',
                lambda r: cell(successes(r), 'effort_free', 1.0, 3)),
               ('released cleanly', lambda r: rate(r, 'release_ok')),
               ('still latched after the release', lambda r: rate(r, 'latched_after'))]
    return pair_table(title, 'Ground truth is the pose the cart geometry implies, so '
                      'these are docking errors, not estimation errors.',
                      labels, metrics, left, right)


def engagement(left, right, labels):
    def hooked(rows):
        return [r for r in rows if r.get('latched') == '1']

    def missed(rows):
        return [r for r in rows if r.get('latched') == '0']

    def extreme(rows, which):
        value = describe(numbers(rows, 'effort_loaded')).get(which, float('nan'))
        return f'{value:.3f}'

    def separation(rows):
        low = describe(numbers(hooked(rows), 'effort_loaded')).get('min', float('nan'))
        high = describe(numbers(missed(rows), 'effort_loaded')).get('max', float('nan'))
        return f'{low - high:.3f}'

    def accuracy(rows):
        if not rows:
            return '-'
        correct = sum(1 for r in rows if r.get('correct') == '1')
        return f'{100.0 * correct / len(rows):.1f} % ({correct}/{len(rows)})'

    def tolerance(rows):
        """Find the largest lateral offset at which every trial still hooked."""
        key = 'offset_lat'
        if not rows or key not in rows[0]:
            return '-'
        offsets = sorted({round(float(r[key]), 4) for r in rows if r[key]})
        best = None
        for offset in offsets:
            trials = [r for r in rows if abs(float(r[key]) - offset) < 1e-6]
            if trials and all(r.get('latched') == '1' for r in trials):
                best = offset
        return f'{best * 1000:.0f} mm' if best is not None else '-'

    metrics = [('trials', lambda r: str(len(r))),
               ('classification accuracy', accuracy),
               ('effort when hooked [Nm]',
                lambda r: cell(hooked(r), 'effort_loaded', 1.0, 3)),
               ('effort when missed [Nm]',
                lambda r: cell(missed(r), 'effort_loaded', 1.0, 3)),
               ('smallest hooked [Nm]', lambda r: extreme(hooked(r), 'min')),
               ('largest missed [Nm]', lambda r: extreme(missed(r), 'max')),
               ('separation [Nm]', separation),
               ('largest lateral offset that always hooked', tolerance)]
    return pair_table('E3 engagement detection',
                      'A static geometry question: whether the tow bar lands in the hook.',
                      labels, metrics, left, right)


def corridor(left, right, labels):
    """One row per corridor width and towing mode, both campaigns side by side."""
    def blocks(rows):
        out = {}
        for row in rows:
            key = (float(row['width']), row['towing_mode'])
            out.setdefault(key, []).append(row)
        return out

    a, b = blocks(left), blocks(right)
    widths = sorted({w for w, _ in list(a) + list(b)}, reverse=True)
    lines = ['### E4 narrow passage', '',
             'Passes and contacts are over every trial; the clearance column is the worst '
             'single trial of the block, which is the number a passage has to survive. '
             'Clearance is recomputed from the trajectories with the indoor GPS noise '
             'averaged out and the true deck width, and a contact has to last '
             f'{CONTACT_SECONDS} s to count (see corridor_clearance.py).', '',
             f'| width [m] | towing mode | passes {labels[0]} | passes {labels[1]} '
             f'| worst clearance {labels[0]} [mm] | worst clearance {labels[1]} [mm] '
             f'| contacts {labels[0]} | contacts {labels[1]} |',
             '|---|---|---|---|---|---|---|---|'[:len('|---' * 8) + 1]]

    def passes(rows):
        return f'{len(successes(rows))}/{len(rows)}' if rows else '-'

    def worst(rows):
        values = numbers(rows, 'clearance')
        return f'{min(values) * 1000:+.1f}' if values else '-'

    def contacts(rows):
        hits = [r for r in rows if r.get('contact_longest')
                and float(r['contact_longest']) >= CONTACT_SECONDS]
        return f'{len(hits)}/{len(rows)}' if rows else '-'

    for width in widths:
        for mode, name in (('1', 'on'), ('0', 'off')):
            ra, rb = a.get((width, mode), []), b.get((width, mode), [])
            if not ra and not rb:
                continue
            lines.append(f'| {width:.2f} | {name} | {passes(ra)} | {passes(rb)} '
                         f'| {worst(ra)} | {worst(rb)} | {contacts(ra)} | {contacts(rb)} |')
    lines.append('')

    for mode, name in (('1', 'towing mode on'), ('0', 'fixed hitch')):
        ra = [r for r in left if r['towing_mode'] == mode]
        rb = [r for r in right if r['towing_mode'] == mode]
        if not ra or not rb:
            continue
        pa, pb = len(successes(ra)), len(successes(rb))
        p = fisher_exact(pa, len(ra) - pa, pb, len(rb) - pb)
        lines.append(f'- pooled passes, {name}: {labels[0]} {pa}/{len(ra)}, '
                     f'{labels[1]} {pb}/{len(rb)} (Fisher exact, two-sided p = {p:.3f})')
    return lines + ['']


def towing(left, right, labels):
    def towed(rows):
        return successes([r for r in rows if r.get('towing') == '1'])

    def solo(rows):
        return successes([r for r in rows if r.get('towing') == '0'])

    def completed(rows):
        all_towed = [r for r in rows if r.get('towing') == '1']
        return f'{len(successes(all_towed))}/{len(all_towed)}'

    def at_limit(rows):
        """Routes whose hitch angle reached the 90 deg joint limit."""
        values = numbers(towed(rows), 'beta_max')
        hits = sum(1 for v in values if math.degrees(v) >= 89.0)
        return f'{hits}/{len(values)}' if values else '-'

    metrics = [('routes completed, towing', completed),
               ('route time, towing [s]', lambda r: cell(towed(r), 'route_time', 1.0, 1)),
               ('route time, no cart [s]', lambda r: cell(solo(r), 'route_time', 1.0, 1)),
               ('path length [m]', lambda r: cell(towed(r), 'path_length', 1.0, 2)),
               ('goal error [mm]', lambda r: cell(towed(r), 'goal_err_xy', 1000.0, 1)),
               ('max off-tracking [mm]',
                lambda r: cell(towed(r), 'offtrack_max', 1000.0, 1)),
               ('RMS off-tracking [mm]',
                lambda r: cell(towed(r), 'offtrack_rms', 1000.0, 1)),
               ('max hitch angle [deg]',
                lambda r: cell(towed(r), 'beta_max', 180.0 / math.pi, 1)),
               ('RMS hitch angle [deg]',
                lambda r: cell(towed(r), 'beta_rms', 180.0 / math.pi, 1)),
               ('routes that reached the 90 deg limit', at_limit),
               ('mean hitch effort [Nm]',
                lambda r: cell(towed(r), 'effort_mean', 1.0, 3))]
    return pair_table('E5 towing',
                      'Off-tracking is how far the cart centre cuts inside the path the '
                      'robot drove; the hitch angle says how much a passive hitch swings.',
                      labels, metrics, left, right)


def endurance(left, right, labels):
    stages = ('dock_charger', 'undock', 'dock_cart', 'tow', 'release', 'return_ok')
    metrics = [('cycles', lambda r: f'{len(successes(r))}/{len(r)}'),
               # Over the complete cycles: a cycle that failed spent its time in the
               # recovery, which is not what a cycle costs.
               ('cycle time [s]', lambda r: cell(successes(r), 'cycle_time', 1.0, 1)),
               ('interventions',
                lambda r: str(sum(int(x['intervention'] or 0) for x in r))),
               ('total time [h]',
                lambda r: f"{sum(float(x['cycle_time'] or 0) for x in r) / 3600.0:.2f}")]
    metrics += [(stage, (lambda s: lambda r: rate(r, s))(stage)) for stage in stages]
    return pair_table('E6 endurance', 'The world is never reset between cycles.',
                      labels, metrics, left, right)


def main(argv=None):
    args = parse_args(argv)
    labels = (args.a_label or campaign(args.a, os.path.basename(args.a.rstrip('/'))),
              args.b_label or campaign(args.b, os.path.basename(args.b.rstrip('/'))))
    lines = ['# Two cart constructions, the same experiments', '',
             f'Left column: {labels[0]} (`{campaign(args.a, "unrecorded")}`, `{args.a}`). ',
             f'Right column: {labels[1]} (`{campaign(args.b, "unrecorded")}`, `{args.b}`).', '',
             CAVEAT, '']
    builders = [('e2', lambda a, b: docking('E2 cart reverse docking', a, b, labels)),
                ('e3', lambda a, b: engagement(a, b, labels)),
                ('e4', lambda a, b: corridor(a, b, labels)),
                ('e5', lambda a, b: towing(a, b, labels)),
                ('e6', lambda a, b: endurance(a, b, labels))]
    for experiment, builder in builders:
        left, right = load(args.a, experiment), load(args.b, experiment)
        if not left or not right:
            lines += [f'### {experiment.upper()}', '',
                      'Not run in both campaigns, so there is nothing to compare.', '']
            continue
        lines += builder(left, right)
    text = '\n'.join(lines) + '\n'
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(text)
        print(f'wrote {args.out}')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    main()
