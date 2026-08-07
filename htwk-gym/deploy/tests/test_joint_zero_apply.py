"""The correction must be a round trip, and a half-applied correction must be
visibly worse than none.  That second property is the one worth testing: it is
the failure mode that would otherwise ship silently.

    python3 -m pytest deploy/tests/test_joint_zero_apply.py -q
    python3 deploy/tests/test_joint_zero_apply.py          # no pytest needed
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.joint_zero import JointZero, load        # noqa: E402

N = 22
LS = 10


def _delta(scale=0.03):
    rng = np.random.default_rng(0)
    d = np.zeros(N, dtype=np.float32)
    d[LS:LS + 12] = rng.normal(0, scale, 12)
    return d


def _servo(q_cmd, delta):
    """What the robot physically does: its onboard PD closes on q_meas, so the
    TRUE angle settles at q_cmd - delta."""
    return q_cmd - delta


def test_disabled_is_exactly_noop():
    jz = JointZero(joint_cnt=N, enabled=False)
    q = np.linspace(-1, 1, N).astype(np.float32)
    assert np.array_equal(jz.correct_measurement(q), q)
    assert np.array_equal(jz.correct_command(q), q)


def test_round_trip_puts_the_robot_where_the_policy_asked():
    d = _delta()
    jz = JointZero(d, joint_cnt=N, leg_dof_start=LS, enabled=True)
    q_want = np.linspace(-0.5, 0.5, N).astype(np.float32)

    q_true = _servo(jz.correct_command(q_want), d)
    assert np.allclose(q_true, q_want, atol=1e-6), \
        "with both halves applied the robot must stand where asked"

    q_meas = q_true + d
    assert np.allclose(jz.correct_measurement(q_meas), q_want, atol=1e-6), \
        "and the policy must observe that same posture"


def test_half_applied_is_worse_than_nothing():
    """Correcting only the read, or only the write, leaves a bias of the same
    size somewhere else -- and the read-only case additionally lies to the
    policy about where the robot is."""
    d = _delta()
    jz = JointZero(d, joint_cnt=N, leg_dof_start=LS, enabled=True)
    q_want = np.zeros(N, dtype=np.float32)

    # no correction at all
    q_true_none = _servo(q_want, d)
    err_none = np.abs(q_true_none - q_want).max()

    # write-only: robot stands right, policy is misinformed by delta
    q_true_write = _servo(jz.correct_command(q_want), d)
    obs_err_write = np.abs((q_true_write + d) - q_true_write).max()
    assert np.allclose(q_true_write, q_want, atol=1e-6)
    assert obs_err_write > 0.9 * err_none

    # read-only: policy sees truth, robot stands at q_want - delta
    q_true_read = _servo(q_want, d)
    assert np.abs(q_true_read - q_want).max() > 0.9 * err_none


def test_load_rejects_an_implausible_delta(tmpdir=None):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write("joint_zero:\n  enabled: true\n  leg_dof_start: 10\n"
                 "  delta_rad: [%s]\n" % ", ".join(["0.9"] * 12))
        p = fh.name
    try:
        load(p, joint_cnt=N)
    except ValueError as e:
        assert "sanity limit" in str(e)
    else:
        raise AssertionError("a 51 deg delta must be refused")
    finally:
        os.unlink(p)


def test_missing_file_is_a_disabled_noop():
    jz = load("/nonexistent/joint_zero.yaml", joint_cnt=N)
    assert not jz.enabled
    q = np.ones(N, dtype=np.float32)
    assert np.array_equal(jz.correct_command(q), q)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  %s" % fn.__name__)
    print("%d passed" % len(fns))
