"""The gait clock contract, which this deploy has now got wrong twice.

First: the wrapper held a walking gait_frequency at the goal, because a comment
asserted E0 does not gate the clock. Training samples gait_frequency as 0.0 for
stand goals, and the robot marched in place at (0, 0, 0).

Second: with the gate in place, the phase was still computed as
fmod(time_now * gait_frequency, 1). That agrees with training only while the
frequency is constant -- which it no longer is.

Both are checked here against envs/K1/goal_pose.py:621.

Runs on the robot, where torch is installed; skips elsewhere.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.policy_goal_pose import GoalPosePolicy
except ImportError as exc:  # torch absent off-robot
    GoalPosePolicy = None
    _IMPORT_ERROR = exc

POLICY_INTERVAL = 0.02
WALK_HZ = 2.0


def _clock(policy_interval=POLICY_INTERVAL):
    """A GoalPosePolicy with only the clock fields, bypassing the .pt load."""
    p = object.__new__(GoalPosePolicy)
    p.gait_frequency = WALK_HZ
    p.gait_process = 0.0
    p._last_time = None
    p.policy_interval = policy_interval
    return p


def _obs_step(a, b):
    """Distance the (cos, sin) observation pair moves between two phases."""
    return math.hypot(
        math.cos(2 * math.pi * b) - math.cos(2 * math.pi * a),
        math.sin(2 * math.pi * b) - math.sin(2 * math.pi * a),
    )


ONE_WALKING_STEP = _obs_step(0.0, WALK_HZ * POLICY_INTERVAL)  # 0.251


@unittest.skipIf(GoalPosePolicy is None, "torch not installed")
class GaitClockTest(unittest.TestCase):
    def test_advances_at_the_commanded_frequency(self):
        p = _clock()
        t = 0.0
        for _ in range(25):  # 0.5 s == exactly one cycle at 2 Hz
            t += POLICY_INTERVAL
            p.advance_gait_clock(t)
        self.assertAlmostEqual(p.gait_process, 0.0, places=6)

    def test_phase_freezes_while_stopped(self):
        """goal_pose.py:621 integrates, so zero frequency holds the phase."""
        p = _clock()
        t = 0.0
        for _ in range(27):  # deliberately not a whole number of cycles
            t += POLICY_INTERVAL
            p.advance_gait_clock(t)
        at_arrival = p.gait_process
        self.assertNotAlmostEqual(at_arrival, 0.0, places=3)

        p.gait_frequency = 0.0
        for _ in range(50):
            t += POLICY_INTERVAL
            self.assertAlmostEqual(p.advance_gait_clock(t), at_arrival, places=9)

    def test_no_observation_jump_at_arrival_or_departure(self):
        """The failure the closed form produced: a teleport in the clock pair."""
        p = _clock()
        t = 0.0
        for _ in range(27):
            t += POLICY_INTERVAL
            p.advance_gait_clock(t)

        before = p.gait_process
        p.gait_frequency = 0.0
        t += POLICY_INTERVAL
        self.assertAlmostEqual(_obs_step(before, p.advance_gait_clock(t)), 0.0, places=9)

        for _ in range(50):
            t += POLICY_INTERVAL
            p.advance_gait_clock(t)

        before = p.gait_process
        p.gait_frequency = WALK_HZ
        t += POLICY_INTERVAL
        self.assertLessEqual(
            _obs_step(before, p.advance_gait_clock(t)), ONE_WALKING_STEP + 1e-9
        )

    def test_closed_form_would_fail_this(self):
        """Guards the test itself: the old formula must not pass the above."""
        stop_after, stopped_for = 27, 50
        t = (stop_after + stopped_for) * POLICY_INTERVAL
        frozen = math.fmod(stop_after * POLICY_INTERVAL * WALK_HZ, 1.0)
        resumed = math.fmod((t + POLICY_INTERVAL) * WALK_HZ, 1.0)
        self.assertGreater(_obs_step(frozen, math.fmod(t * 0.0, 1.0)), ONE_WALKING_STEP)
        self.assertGreater(_obs_step(0.0, resumed), ONE_WALKING_STEP)

    def test_long_stall_does_not_wind_the_phase_forward(self):
        """A fall-recovery pause must not advance the clock by the whole gap."""
        p = _clock()
        p.advance_gait_clock(0.02)
        before = p.gait_process
        p.advance_gait_clock(8.02)  # GetUp takes about 8 s
        advanced = math.fmod(p.gait_process - before + 1.0, 1.0)
        self.assertLessEqual(advanced, 4 * POLICY_INTERVAL * WALK_HZ + 1e-9)

    def test_wall_clock_jitter_is_tracked_not_assumed(self):
        """A slow loop must yield 2 Hz in wall clock, not 2 Hz per step.

        The first call has no previous timestamp to measure against and so is
        seeded with the nominal interval; every call after it uses the elapsed
        time. Counting steps instead would run the gait 20% slow here.
        """
        slow_dt, steps = 0.025, 20
        p = _clock()
        t = 0.0
        for _ in range(steps):
            t += slow_dt
            p.advance_gait_clock(t)

        elapsed_driven = (POLICY_INTERVAL + (steps - 1) * slow_dt) * WALK_HZ
        step_counting = steps * POLICY_INTERVAL * WALK_HZ
        self.assertAlmostEqual(p.gait_process, math.fmod(elapsed_driven, 1.0), places=6)
        self.assertGreater(elapsed_driven, step_counting)


if __name__ == "__main__":
    unittest.main()
