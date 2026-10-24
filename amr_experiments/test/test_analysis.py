"""Unit tests for the pure functions behind the experiment numbers."""

import math

from amr_experiments.analyze import cell, fisher_exact, successes
from amr_experiments.corridor_clearance import average_cart, measure
from amr_experiments.e5_towing import analyse, point_to_path, transform
from amr_experiments.harness import pose_error
from amr_experiments.trial_log import describe, format_summary, summarize


def test_pose_error_is_expressed_in_the_target_frame():
    # 0.1 m ahead and 0.05 m to the left of a target that points along +y.
    actual = (-0.05, 0.1, math.pi / 2)
    long, lat, yaw = pose_error(actual, (0.0, 0.0, math.pi / 2))
    assert math.isclose(long, 0.1) and math.isclose(lat, 0.05) and math.isclose(yaw, 0.0,
                                                                                abs_tol=1e-12)


def test_pose_error_wraps_the_yaw():
    long, lat, yaw = pose_error((0.0, 0.0, -math.pi + 0.1), (0.0, 0.0, math.pi - 0.1))
    assert (long, lat) == (0.0, 0.0)
    assert math.isclose(yaw, 0.2, abs_tol=1e-9)


def test_point_to_path_uses_the_nearest_segment():
    path = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert math.isclose(point_to_path((0.5, 0.25), path), 0.25)
    assert math.isclose(point_to_path((1.5, 1.0), path), 0.5)
    assert math.isclose(point_to_path((-1.0, 0.0), path), 1.0)


def test_transform_rotates_into_the_world_frame():
    x, y = transform((1.0, 2.0, math.pi / 2), (1.0, 0.0))
    assert math.isclose(x, 1.0, abs_tol=1e-9) and math.isclose(y, 3.0, abs_tol=1e-9)


def test_describe_reports_spread_and_rms():
    stats = describe([-1.0, 1.0, -1.0, 1.0])
    assert stats['n'] == 4 and stats['mean'] == 0.0
    assert math.isclose(stats['rms'], 1.0)
    assert stats['p95_abs'] == 1.0


def test_describe_of_nothing_is_empty():
    assert describe([]) == {}


def test_summarize_counts_only_successful_trials():
    rows = [{'success': 'True', 'err': '0.10'}, {'success': 'False', 'err': '9.9'},
            {'success': 'True', 'err': '0.20'}]
    summary = summarize(rows, ['err'])
    assert summary['trials'] == 3 and summary['successes'] == 2
    assert math.isclose(summary['success_rate'], 2 / 3)
    assert math.isclose(summary['err']['mean'], 0.15)


def test_successes_and_cell_ignore_failed_rows():
    rows = [{'success': '1', 'v': '1.0'}, {'success': '0', 'v': '100.0'}]
    assert len(successes(rows)) == 1
    assert cell(successes(rows), 'v', 1000.0, 1).startswith('1000.0')


def test_cell_without_data_is_a_dash():
    assert cell([], 'missing') == '-'


def test_analyse_straight_line_has_no_offtracking():
    # Robot driving along +x with the hitch straight: the cart stays on the path.
    # Samples carry the measured cart pose too; analyse() uses the hitch geometry.
    samples = [(0.1 * i, 0.1 * i, 0.0, 0.0, 0.0, 0.02, 0.1 * i - 0.85, 0.0, 0.0)
               for i in range(40)]
    stats = analyse(samples, 0.58, towing=True)
    assert stats['offtrack_max'] < 1e-6      # the cart runs exactly on the driven path
    assert math.isclose(stats['path_length'], 3.9, abs_tol=1e-6)
    assert stats['beta_max'] == 0.0


def test_analyse_needs_enough_samples():
    assert analyse([(0.0,) * 9], 0.58, towing=True) == {}


def test_clearance_ignores_samples_outside_the_passage():
    from types import SimpleNamespace

    from amr_experiments.e4_corridor import clearance
    args = SimpleNamespace(width=1.2, corridor=[3.0, 6.5, -4.5])
    # Robot still short of the corridor mouth: nothing here is a wall contact.
    samples = [(0.0, 2.0, -6.0, 0.0, 1.5, 0.1, 1.2, -6.2, 1.5)] * 5
    gaps, contacts = clearance(samples, args, 0.58, 0.42, 0.30)
    assert math.isnan(gaps['measured']) and contacts == 0


def test_clearance_prefers_the_measured_cart_pose():
    # The hitch angle says the cart is straight behind; the measured pose says it is
    # swung out to the side. The measured pose has to win.
    from types import SimpleNamespace

    from amr_experiments.e4_corridor import clearance
    args = SimpleNamespace(width=1.2, corridor=[3.0, 6.5, -4.5])
    samples = [(0.0, 4.0, -4.5, 0.0, 0.0, 0.1, 3.2, -5.0, 0.0)] * 5   # robot inside
    gaps, contacts = clearance(samples, args, 0.58, 0.42, 0.30)
    # Cart corner at y = -5.3, wall at -5.1: 0.2 m through it, counted as a contact.
    # Only the corners inside the corridor span count, so one per sample.
    assert math.isclose(gaps['measured'], -0.2, abs_tol=1e-9)
    assert gaps['geometric'] > 0.0
    assert contacts == len(samples)


def test_format_summary_ignores_entries_that_are_not_statistics():
    # E6 puts stage counts and a failure tally in the same summary; formatting used to
    # crash on them and threw away the whole endurance report at the end of a 30-cycle
    # run.
    summary = summarize([{'t': '1.0', 'success': 'True'}], ['t'])
    summary['stages'] = {'undock': (2, 30)}
    summary['failures'] = {'undock': 26}
    text = format_summary('E6 endurance', summary)
    assert 'trials 1' in text and 'stages' not in text


def test_fisher_exact_matches_a_known_table():
    # Fisher's own tea-tasting table: 3/4 against 1/4 is not significant.
    assert math.isclose(fisher_exact(3, 1, 1, 3), 0.4857142857, rel_tol=1e-6)
    # A clean separation is.
    assert math.isclose(fisher_exact(4, 0, 0, 4), 0.0285714286, rel_tol=1e-6)


def test_fisher_exact_of_an_empty_margin_is_not_a_number():
    # E4 pools one mode at a time; a width with no trials in a mode must not raise.
    assert math.isnan(fisher_exact(0, 0, 3, 2))
    # Nor must a table where every trial succeeded in both modes.
    assert math.isnan(fisher_exact(5, 0, 5, 0))


def test_average_cart_averages_distinct_gps_updates_not_repeated_samples():
    """The trajectory repeats each 5 Hz cart pose at the odometry rate.

    Averaging over samples would average those copies and remove no noise at all, which
    is what made the first corrected clearance identical to the raw one.
    """
    poses = [(0.0, 0.10, 0.0), (0.0, -0.10, 0.0), (0.0, 0.10, 0.0)]
    samples = [(0.1 * i, 0.0, 0.0, poses[i // 4]) for i in range(12)]
    smoothed = average_cart(samples, updates=3)
    # The middle update sees all three: (0.10 - 0.10 + 0.10) / 3. The edges see two,
    # which is what a centred window can do there.
    assert all(math.isclose(s[3][1], 0.10 / 3, abs_tol=1e-12) for s in smoothed[4:8])
    assert all(math.isclose(s[3][1], 0.0, abs_tol=1e-12) for s in smoothed[:4])
    # Every sample of one update gets that update's average, and nothing is dropped.
    assert len(smoothed) == len(samples)
    # One update averaged with itself is the identity.
    assert average_cart(samples, updates=1) == samples


def test_measure_reports_the_closest_corner_and_how_long_it_touched():
    # Corridor 1.0 m wide centred on y = -4.5, so each wall face is 0.5 m out.
    # A cart centred on the axis has its corners 0.28 m to each side: 0.22 m of room.
    samples = [(0.1 * i, 4.0, -4.5, (4.0, -4.5, 0.0)) for i in range(20)]
    clearance, seconds, longest = measure(samples, 1.0)
    assert math.isclose(clearance, 0.22, abs_tol=1e-9)
    assert seconds == 0.0 and longest == 0.0
    # Push the cart 0.25 m sideways and the far corner is 0.03 m inside the wall.
    touching = [(0.1 * i, 4.0, -4.5, (4.0, -4.25, 0.0)) for i in range(20)]
    clearance, seconds, longest = measure(touching, 1.0)
    assert math.isclose(clearance, -0.03, abs_tol=1e-9)
    assert math.isclose(longest, 2.0, abs_tol=0.11)


def test_measure_ignores_samples_outside_the_passage():
    outside = [(0.1 * i, 1.0, -4.5, (1.0, -4.25, 0.0)) for i in range(20)]
    clearance, seconds, _ = measure(outside, 1.0)
    assert math.isnan(clearance) and seconds == 0.0
