"""Turn the experiment CSVs into the tables and figures a paper needs.

Reads whatever `e*_*.csv` files are in the results directory and writes

  results.md     - one table per experiment, numbers as mean +- std (n)
  figures/*.pdf  - the plots that carry the result: error scatter, effort histogram,
                   off-tracking against the hitch angle, clearance against width

  ros2 run amr_experiments analyze --results results --out results/report
"""

import argparse
import csv
import glob
import math
import os
import sys

from .corridor_clearance import CONTACT_SECONDS, recompute
from .trial_log import describe, numbers

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:            # tables still work without matplotlib
    plt = None


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', default='results', help='directory with the experiment CSVs')
    ap.add_argument('--out', default=None, help='where to write (default: --results)')
    return ap.parse_args(argv if argv is not None else sys.argv[1:])


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def cell(rows, key, scale=1.0, digits=3):
    stats = describe([v * scale for v in numbers(rows, key)])
    if not stats:
        return '-'
    return f"{stats['mean']:.{digits}f} ± {stats['std']:.{digits}f} ({stats['n']})"


def successes(rows):
    return [r for r in rows if str(r.get('success', '')).lower() in ('1', 'true', 'yes')]


def table_docking(name, rows):
    ok = successes(rows)
    rate = 100.0 * len(ok) / len(rows) if rows else 0.0
    lines = [f'### {name}', '',
             f'Trials: {len(rows)}, successful: {len(ok)} ({rate:.1f} %)', '',
             '| quantity | value |', '|---|---|',
             f'| docking time [s] | {cell(ok, "dock_time", 1.0, 1)} |',
             f'| longitudinal error [mm] | {cell(ok, "err_long", 1000.0, 1)} |',
             f'| lateral error [mm] | {cell(ok, "err_lat", 1000.0, 1)} |',
             f'| yaw error [deg] | {cell(ok, "err_yaw", 180.0 / math.pi, 2)} |',
             f'| position error [mm] | {cell(ok, "err_xy", 1000.0, 1)} |']
    if rows and 'charging' in rows[0]:
        charging = sum(int(r['charging'] or 0) for r in rows)
        lines.append(f'| charging confirmed | {charging}/{len(rows)} |')
    if rows and 'effort_loaded' in rows[0]:
        lines.append(f'| hitch effort, hooked [Nm] | {cell(ok, "effort_loaded", 1.0, 3)} |')
        lines.append(f'| hitch effort, free [Nm] | {cell(ok, "effort_free", 1.0, 3)} |')
    return lines + ['']


def table_engagement(rows):
    hooked = [r for r in rows if r['latched'] == '1']
    missed = [r for r in rows if r['latched'] == '0']
    correct = sum(1 for r in rows if r['correct'] == '1')
    hooked_stats, missed_stats = (describe(numbers(hooked, 'effort_loaded')),
                                  describe(numbers(missed, 'effort_loaded')))
    gap = (hooked_stats.get('min', float('nan')) - missed_stats.get('max', float('nan'))
           if hooked and missed else float('nan'))
    return ['### E3 engagement detection', '',
            f'Trials: {len(rows)}, hooked: {len(hooked)}, missed: {len(missed)}, '
            f'classification accuracy: {100.0 * correct / len(rows):.1f} %', '',
            '| quantity | value |', '|---|---|',
            f'| effort when hooked [Nm] | {cell(hooked, "effort_loaded", 1.0, 3)} |',
            f'| effort when missed [Nm] | {cell(missed, "effort_loaded", 1.0, 3)} |',
            f'| effort with the hook down [Nm] | {cell(rows, "effort_free", 1.0, 3)} |',
            f'| smallest hooked / largest missed [Nm] | '
            f'{hooked_stats.get("min", float("nan")):.3f} / '
            f'{missed_stats.get("max", float("nan")):.3f} |',
            f'| separation [Nm] | {gap:.3f} |', '']


def table_towing(rows):
    towed, solo = ([r for r in rows if r['towing'] == '1'],
                   [r for r in rows if r['towing'] == '0'])
    lines = ['### E5 towing', '',
             f'Routes completed: {len(successes(towed))}/{len(towed)} towing, '
             f'{len(successes(solo))}/{len(solo)} with no cart. The table is over the '
             'completed routes; a route that was aborted never reached the goal, so its '
             'time and goal error measure the abort, not the towing.', '',
             '| quantity | towing | no cart |', '|---|---|---|']
    towed, solo = successes(towed), successes(solo)
    for label, key, scale, digits in (
            ('route time [s]', 'route_time', 1.0, 1),
            ('path length [m]', 'path_length', 1.0, 2),
            ('goal error [mm]', 'goal_err_xy', 1000.0, 1),
            ('max off-tracking [mm]', 'offtrack_max', 1000.0, 1),
            ('RMS off-tracking [mm]', 'offtrack_rms', 1000.0, 1),
            ('max hitch angle [deg]', 'beta_max', 180.0 / math.pi, 2),
            ('mean hitch effort [Nm]', 'effort_mean', 1.0, 3)):
        lines.append(f'| {label} | {cell(towed, key, scale, digits)} '
                     f'| {cell(solo, key, scale, digits)} |')
    return lines + ['']


def corridor_blocks(files):
    """(width, mode, all trials, successful trials) per width, widest corridor first.

    Each row gains `clearance` and `contact_seconds`, recomputed from its trajectory with
    the indoor-GPS noise averaged out and the true deck width (see corridor_clearance).
    The logged `min_clearance` column is left alone so the two can be compared.
    """
    blocks = []
    for path in files:
        rows = read_csv(path)
        fixed = recompute(path)
        for row in rows:
            clearance, seconds, longest = fixed.get(row['trial'], (None, None, None))
            row['clearance'] = '' if clearance is None else f'{clearance:.4f}'
            row['contact_seconds'] = '' if seconds is None else f'{seconds:.2f}'
            row['contact_longest'] = '' if longest is None else f'{longest:.2f}'
        for mode in ('1', '0'):
            subset = [r for r in rows if r['towing_mode'] == mode]
            if subset:
                blocks.append((float(subset[0]['width']), mode, subset, successes(subset)))
    return sorted(blocks, key=lambda b: (-b[0], -int(b[1])))


def touched(rows):
    """Trials where the cart rode the wall for at least CONTACT_SECONDS.

    A single sample below zero is the indoor GPS's own noise; a contact that lasts is
    the cart actually against the wall. Falls back to the logged contact count when the
    trajectory is missing.
    """
    out = []
    for row in rows:
        if row.get('contact_longest'):
            if float(row['contact_longest']) >= CONTACT_SECONDS:
                out.append(row)
        elif float(row.get('contacts') or 0) > 0:
            out.append(row)
    return out


def fisher_exact(a, b, c, d):
    """Two-sided p of the 2x2 table [[a, b], [c, d]].

    Sums the probability of every table with the same margins that is no more likely
    than the observed one.
    """
    rows, columns, total = a + b, a + c, a + b + c + d
    if not rows or not columns or rows == total or columns == total:
        return float('nan')

    def likelihood(x):
        return (math.comb(rows, x) * math.comb(total - rows, columns - x)
                / math.comb(total, columns))

    observed = likelihood(a)
    low, high = max(0, columns - (total - rows)), min(rows, columns)
    return sum(likelihood(x) for x in range(low, high + 1)
               if likelihood(x) <= observed * (1 + 1e-9))


def table_corridor(files):
    lines = ['### E4 narrow passage', '',
             'Time and the hitch angle are over the successful trials. Clearance and '
             'contacts are over every trial: the trials that scrape the wall are the '
             'ones that fail, so dropping them would hide exactly the result this '
             'experiment is about.', '',
             'Clearance is recomputed from the recorded trajectories with the 20 mm '
             'noise of the indoor GPS averaged over 1 s and the true 0.84 x 0.56 m '
             'deck. The '
             'number logged during the trials is the minimum over about 85 noisy '
             'updates and sits roughly 50 mm below the truth, so it is not used here. A '
             f'contact counts when the cart stays against the wall for {CONTACT_SECONDS} '
             's or more.', '',
             '| width [m] | towing mode | passes | clearance [mm] | worst clearance '
             '[mm] | time [s] | max hitch angle [deg] | trials with contact |',
             '|---|---|---|---|---|---|---|---|']
    passes = {'1': [0, 0], '0': [0, 0]}
    touches = {'1': [0, 0], '0': [0, 0]}
    for width, mode, subset, ok in corridor_blocks(files):
        hits = len(touched(subset))
        passes[mode][0] += len(ok)
        passes[mode][1] += len(subset)
        touches[mode][0] += hits
        touches[mode][1] += len(subset)
        clearances = [1000 * v for v in numbers(subset, 'clearance')]
        lines.append(f'| {width:.2f} | {"on" if mode == "1" else "off"} '
                     f'| {len(ok)}/{len(subset)} '
                     f'| {cell(subset, "clearance", 1000.0, 1)} '
                     f'| {min(clearances):.1f} '
                     f'| {cell(ok, "pass_time", 1.0, 1)} '
                     f'| {cell(ok, "beta_max", 180.0 / math.pi, 1)} '
                     f'| {hits}/{len(subset)} |')
    lines += ['']
    for key, label in ((passes, 'passes'), (touches, 'trials with contact')):
        on, off = key['1'], key['0']
        p = fisher_exact(on[0], on[1] - on[0], off[0], off[1] - off[0])
        lines.append(f'- pooled {label}: towing mode {on[0]}/{on[1]}, fixed hitch '
                     f'{off[0]}/{off[1]} (Fisher exact, two-sided p = {p:.3f})')
    return lines + ['']


def windowed_effort(path, seconds=2.0):
    """Hitch effort averaged over fixed windows along one recorded route."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    out, window, start = [], [], float(rows[0]['t'])
    for row in rows:
        t = float(row['t'])
        if t - start >= seconds and window:
            out.append(sum(window) / len(window))
            window, start = [], t
        window.append(float(row['pitch_effort']))
    if window:
        out.append(sum(window) / len(window))
    return out


def towing_effort_states(e5_path, rows, seconds=2.0):
    """Windowed hitch effort grouped by what the hitch was carrying on each route.

    Towing leaves the hook in one of two steady states - the tow bar either rests on it
    or the cart's own wheels keep carrying the load - and the state is set when the hook
    is lifted and then holds for the whole route. Both tow the cart, so the groups are
    found from the data rather than assumed.
    """
    directory = os.path.join(os.path.dirname(e5_path), 'trajectories')
    base = os.path.basename(e5_path)[:-4]
    groups = {'no cart': [], 'towing, hook unloaded': [], 'towing, hook loaded': []}
    for row in rows:
        path = os.path.join(directory, f"{base}_{int(row['trial']):03d}.csv")
        if not os.path.exists(path):
            continue
        values = windowed_effort(path, seconds)
        if not values:
            continue
        if row['towing'] != '1':
            groups['no cart'] += values
        else:
            mean = sum(values) / len(values)
            key = 'towing, hook loaded' if mean > 1.0 else 'towing, hook unloaded'
            groups[key] += values
    return {k: v for k, v in groups.items() if v}


def table_towing_effort(e5_path, rows, seconds=2.0):
    groups = towing_effort_states(e5_path, rows, seconds)
    if len(groups) < 2:
        return []
    lines = ['#### Hitch effort during towing', '',
             f'Hitch pitch effort averaged over {seconds:.0f} s windows along every '
             'recorded route. The steady state the hook settles in after the lift does '
             'not change the outcome of the route, but it changes the effort by more '
             'than an order of magnitude, so a coupling check that runs during towing '
             'has to hold against the lower of the two.', '',
             '| state | windows | mean [Nm] | min [Nm] | max [Nm] |',
             '|---|---|---|---|---|']
    for name, values in groups.items():
        lines.append(f'| {name} | {len(values)} | {sum(values) / len(values):.3f} '
                     f'| {min(values):.3f} | {max(values):.3f} |')
    free, loose = groups.get('no cart'), groups.get('towing, hook unloaded')
    if free and loose:
        lines += ['', f'The windows separate without overlap: the largest window with no '
                      f'cart is {max(free):.4f} Nm and the smallest towing window is '
                      f'{min(loose):.4f} Nm, a margin of {min(loose) - max(free):.4f} Nm.']
    return lines + ['']


def table_endurance(rows):
    ok = successes(rows)
    interventions = sum(int(r['intervention'] or 0) for r in rows)
    total = sum(float(r['cycle_time']) for r in rows if r['cycle_time'])
    # No intervention means the MTBI is censored by the length of the campaign, so state
    # the bound the data supports rather than an infinity.
    mtbi = (f'mean time between interventions: {total / 60.0 / interventions:.1f} min'
            if interventions else
            f'no intervention in {total / 60.0:.0f} min of operation (MTBI censored)')
    lines = ['### E6 endurance', '',
             f'Cycles: {len(rows)}, complete: {len(ok)} '
             f'({100.0 * len(ok) / len(rows):.1f} %), '
             f'total {total / 3600.0:.2f} h, interventions: {interventions}, '
             f'{mtbi}', '',
             '| stage | successes |', '|---|---|']
    for key in ('dock_charger', 'undock', 'dock_cart', 'tow', 'release', 'return_ok'):
        attempted = [r for r in rows if r.get(key, '') != '']
        lines.append(f'| {key} | {sum(int(r[key]) for r in attempted)}/{len(attempted)} |')
    failures = {}
    for row in rows:
        if row['failure_stage']:
            failures[row['failure_stage']] = failures.get(row['failure_stage'], 0) + 1
    if failures:
        lines += ['', '| failure | count |', '|---|---|']
        lines += [f'| {k} | {v} |' for k, v in sorted(failures.items())]
    return lines + ['']


def figures(data, directory):
    if plt is None:
        return []
    os.makedirs(directory, exist_ok=True)
    written = []
    for name, rows in data.items():
        if name.startswith('e1') or name.startswith('e2'):
            ok = successes(rows)
            x = [1000 * v for v in numbers(ok, 'err_lat')]
            y = [1000 * v for v in numbers(ok, 'err_long')]
            if not x:
                continue
            fig, ax = plt.subplots(figsize=(3.4, 3.2))
            ax.axhline(0, lw=0.5, color='0.7')
            ax.axvline(0, lw=0.5, color='0.7')
            ax.scatter(x, y, s=18, alpha=0.8, edgecolor='none')
            ax.set_xlabel('lateral error [mm]')
            ax.set_ylabel('longitudinal error [mm]')
            ax.set_title(name.replace('_', ' '), fontsize=9)
            ax.set_aspect('equal', adjustable='datalim')
            fig.tight_layout()
            written.append(save(fig, directory, f'{name}_scatter'))
        if name.startswith('e3'):
            hooked = numbers([r for r in rows if r['latched'] == '1'], 'effort_loaded')
            missed = numbers([r for r in rows if r['latched'] == '0'], 'effort_loaded')
            fig, ax = plt.subplots(figsize=(3.6, 2.6))
            # Log bins: the two distributions are two orders of magnitude apart, so on a
            # linear axis the missed ones collapse into the y axis and show nothing.
            bins = [10 ** (-2.5 + 0.1 * i) for i in range(0, 31)]
            ax.hist(missed, bins=bins, alpha=0.8, label='missed')
            ax.hist(hooked, bins=bins, alpha=0.8, label='hooked')
            ax.set_xscale('log')
            ax.set_xlabel('hitch pitch effort after lifting [Nm]')
            ax.set_ylabel('trials')
            ax.legend(fontsize=8)
            fig.tight_layout()
            written.append(save(fig, directory, 'e3_effort_histogram'))
        if name.startswith('e5'):
            towed = [r for r in rows if r['towing'] == '1']
            x = [math.degrees(v) for v in numbers(towed, 'beta_max')]
            y = [1000 * v for v in numbers(towed, 'offtrack_max')]
            if not x:
                continue
            fig, ax = plt.subplots(figsize=(3.4, 2.6))
            ax.scatter(x, y, s=20)
            ax.set_xlabel('max hitch angle [deg]')
            ax.set_ylabel('max off-tracking [mm]')
            fig.tight_layout()
            written.append(save(fig, directory, 'e5_offtracking'))
    return written


def figure_towing_effort(e5_path, rows, directory, seconds=2.0):
    groups = towing_effort_states(e5_path, rows, seconds)
    if plt is None or len(groups) < 2:
        return None
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    edges = [10 ** (-2.0 + 0.1 * i) for i in range(0, 26)]
    for name, values in groups.items():
        ax.hist(values, bins=edges, alpha=0.75, label=name)
    ax.set_xscale('log')
    ax.set_xlabel(f'hitch pitch effort, {seconds:.0f} s mean [Nm]')
    ax.set_ylabel('windows')
    ax.legend(fontsize=7)
    fig.tight_layout()
    return save(fig, directory, 'e5_effort_states')


def figure_corridor(files, directory):
    """Pass rate and clearance against corridor width for both hitch modes."""
    if plt is None or not files:
        return None
    blocks = corridor_blocks(files)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.6, 4.0), sharex=True)
    styles = {'1': ('towing mode', 'o-'), '0': ('fixed hitch', 's--')}
    for mode, (label, style) in styles.items():
        series = [b for b in blocks if b[1] == mode]
        if not series:
            continue
        widths = [b[0] for b in series]
        top.plot(widths, [100.0 * len(b[3]) / len(b[2]) for b in series], style, label=label)
        # The worst trial, over every trial rather than only the successes. A mean
        # would hide what this experiment is about: one wall contact is a failure, and
        # averaging it against nine clean passes leaves a comfortable-looking number.
        clear = [[1000 * v for v in numbers(b[2], 'clearance')] for b in series]
        bottom.plot(widths, [min(c) if c else float('nan') for c in clear], style,
                    label=label)
    top.set_ylabel('passes [%]')
    top.set_ylim(0, 105)
    top.legend(fontsize=8)
    bottom.axhline(0, lw=0.8, color='0.4')
    bottom.set_ylabel('worst clearance [mm]')
    bottom.set_xlabel('corridor width [m]')
    fig.tight_layout()
    return save(fig, directory, 'e4_corridor_width')


def save(fig, directory, name):
    path = os.path.join(directory, f'{name}.pdf')
    fig.savefig(path)
    fig.savefig(path[:-4] + '.png', dpi=200)
    plt.close(fig)
    return path


def main(argv=None):
    args = parse_args(argv)
    out = args.out or args.results
    os.makedirs(out, exist_ok=True)
    # Only <results>/e1 ... <results>/e6, never a recursive walk: runs that were thrown
    # away are kept beside them (contaminated_20261015/e1, e4_attempts, e5_jackknife) and
    # a report that silently mixes them in is worse than no report.
    paths = sorted(glob.glob(os.path.join(args.results, 'e[1-6]', 'e*_*.csv')))
    paths = [p for p in paths if os.sep + 'trajectories' + os.sep not in p]
    data = {os.path.basename(p)[:-4]: read_csv(p) for p in paths}
    by_path = {os.path.basename(p)[:-4]: p for p in paths}
    lines = ['# Simulation results', '',
             'Every number is mean ± standard deviation over the successful trials, '
             'with the trial count in brackets.', '']
    corridor_files = [p for p in paths if os.path.basename(p).startswith('e4')]
    for name, rows in data.items():
        if not rows:
            continue
        if name.startswith('e1'):
            lines += table_docking('E1 charger docking', rows)
        elif name.startswith('e2'):
            lines += table_docking('E2 cart reverse docking', rows)
        elif name.startswith('e3'):
            lines += table_engagement(rows)
        elif name.startswith('e4'):
            if corridor_files and name == os.path.basename(corridor_files[0])[:-4]:
                lines += table_corridor(corridor_files)   # all widths in one table
        elif name.startswith('e5'):
            lines += table_towing(rows)
            lines += table_towing_effort(by_path[name], rows)
        elif name.startswith('e6'):
            lines += table_endurance(rows)
    report = os.path.join(out, 'results.md')
    with open(report, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    figure_dir = os.path.join(out, 'figures')
    written = figures(data, figure_dir)
    extra = [figure_corridor(corridor_files, figure_dir) if corridor_files else None]
    for name, rows in data.items():
        if name.startswith('e5') and rows:
            extra.append(figure_towing_effort(by_path[name], rows, figure_dir))
    written += [p for p in extra if p]
    print(f'wrote {report}')
    for path in written:
        print(f'wrote {path}')
    return 0
