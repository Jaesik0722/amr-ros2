"""Per-trial CSV logs and the summary statistics a results table needs."""

import csv
import json
import math
import os
import statistics


class TrialLog:
    """Write one CSV per experiment, one row per trial, as the trials run."""

    def __init__(self, directory, name, fields):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f'{name}.csv')
        self.fields = ['trial'] + list(fields)
        self.rows = []
        self._file = open(self.path, 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=self.fields)
        self._writer.writeheader()

    def add(self, **values):
        row = {k: values.get(k, '') for k in self.fields}
        row['trial'] = len(self.rows) + 1
        self.rows.append(row)
        self._writer.writerow(row)
        self._file.flush()
        return row

    def save_trajectory(self, trial, samples):
        """Write one trajectory: t, robot pose, hitch angle, effort, measured cart pose."""
        if not samples:
            return
        directory = os.path.join(os.path.dirname(self.path), 'trajectories')
        os.makedirs(directory, exist_ok=True)
        base = os.path.basename(self.path)[:-4]
        with open(os.path.join(directory, f'{base}_{trial:03d}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'x', 'y', 'yaw', 'beta', 'pitch_effort',
                        'cart_x', 'cart_y', 'cart_yaw'])
            w.writerows(samples)

    def close(self, meta=None):
        self._file.close()
        if meta:
            with open(self.path[:-4] + '_meta.json', 'w') as f:
                json.dump(meta, f, indent=2)


def numbers(rows, key):
    out = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isnan(value):
            out.append(value)
    return out


def describe(values):
    """mean, standard deviation, median, min, max, 95th percentile of |value|."""
    if not values:
        return {}
    absolute = sorted(abs(v) for v in values)
    return {
        'n': len(values),
        'mean': statistics.fmean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
        'median': statistics.median(values),
        'min': min(values),
        'max': max(values),
        'rms': math.sqrt(statistics.fmean([v * v for v in values])),
        'p95_abs': absolute[min(len(absolute) - 1, int(0.95 * len(absolute)))],
    }


def summarize(rows, keys, success_key='success'):
    """Summary of a set of trials: success rate plus statistics over the successful ones."""
    total = len(rows)
    ok = [r for r in rows if str(r.get(success_key, '')).lower() in ('1', 'true', 'yes')]
    out = {'trials': total, 'successes': len(ok),
           'success_rate': len(ok) / total if total else 0.0}
    for key in keys:
        stats = describe(numbers(ok, key))
        if stats:
            out[key] = stats
    return out


def format_summary(title, summary):
    lines = [f'--- {title} ---',
             f"trials {summary['trials']}  success {summary['successes']}"
             f"  ({100 * summary['success_rate']:.1f} %)"]
    for key, stats in summary.items():
        # Summaries also carry dictionaries that are not statistics (stage counts,
        # failure tallies); only the ones describe() built have a mean.
        if isinstance(stats, dict) and 'mean' in stats:
            lines.append(f"  {key:<22} mean {stats['mean']:+.4f}  std {stats['std']:.4f}"
                         f"  |p95| {stats['p95_abs']:.4f}  max {stats['max']:+.4f}")
    return '\n'.join(lines)
