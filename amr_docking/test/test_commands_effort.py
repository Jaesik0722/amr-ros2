"""The hook effort reading must not turn a missing measurement into zero load."""

from amr_docking.commands import Commander


class FakeLogger:

    def __init__(self):
        self.warnings = []

    def warn(self, message):
        self.warnings.append(message)


class FakeCommander:
    """Just enough of Commander to exercise hook_effort without a ROS graph."""

    hook_effort = Commander.hook_effort

    def __init__(self, batches):
        self.batches = list(batches)      # effort samples delivered per attempt
        self.logger = FakeLogger()
        self.callback = None

    def create_subscription(self, _type, _topic, callback, _depth):
        self.callback = callback
        return object()

    def destroy_subscription(self, _sub):
        self.callback = None

    def get_logger(self):
        return self.logger

    def wait(self, _seconds):
        for value in (self.batches.pop(0) if self.batches else []):
            self.callback(Joints(value))


class Joints:

    def __init__(self, effort):
        self.name = ['hitch_yaw_joint', 'hitch_pitch_joint']
        self.effort = [0.0, effort]


def test_hook_effort_averages_the_samples_it_received():
    node = FakeCommander([[1.8, 2.0, 2.2]])
    assert abs(node.hook_effort() - 2.0) < 1e-9


def test_hook_effort_is_the_magnitude():
    node = FakeCommander([[-1.9, -2.1]])
    assert abs(node.hook_effort() - 2.0) < 1e-9


def test_hook_effort_returns_none_when_nothing_arrived():
    """None, never 0.0: a release that reads 0.0 drives away from a hooked cart."""
    node = FakeCommander([[], [], []])
    assert node.hook_effort() is None
    assert len(node.logger.warnings) == 3


def test_hook_effort_retries_until_a_sample_arrives():
    node = FakeCommander([[], [1.5, 1.7]])
    assert abs(node.hook_effort() - 1.6) < 1e-9
    assert len(node.logger.warnings) == 1
