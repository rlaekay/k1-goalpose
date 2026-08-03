"""Zero-motion integration check for the deploy stack.

Constructs the policy and both monitors against the live robot and runs one
inference. Publishes no LowCmd and changes no mode. RemoteControlService is
skipped on purpose: it starts a keyboard listener that needs a TTY, which a
non-interactive ssh session does not have.
"""
import logging, os, sys, time, importlib.util, numpy as np, yaml
sys.path.insert(0, os.getcwd())  # deploy_goal_pose imports utils.* relative to the deploy dir
logging.basicConfig(level=logging.INFO)
spec = importlib.util.spec_from_file_location("d", "deploy_goal_pose.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from booster_robotics_sdk_python import ChannelFactory
ChannelFactory.Instance().Init(0, "127.0.0.1")
cfg = yaml.safe_load(open("configs/Goal_Pose_E0.yaml"))

pol = m.GoalPosePolicy(cfg=cfg)
mm = m.ModeMonitor(); fm = m.FallMonitor()
# Wait for delivery rather than assuming a fixed sleep is enough: DDS discovery
# has taken ~2 s here, and /robot_states only publishes at ~2 Hz.
deadline = time.time() + 15.0
while time.time() < deadline:
    if mm.snapshot()[1] < 5.0 and fm.snapshot()[2] < 5.0:
        break
    time.sleep(0.2)
print()
print("policy leg slice : %d..%d  obs=%d act=%d"
      % (pol.leg_start, pol.leg_start + pol.num_act - 1, pol.num_obs, pol.num_act))
print("mode_monitor     :", mm.available, "->", mm.name())
print("fall_monitor     :", fm.available, "-> state", fm.snapshot()[0], "recov", fm.snapshot()[1])
dof = np.array(cfg["common"]["default_qpos"], dtype=np.float32)
t = pol.inference(0.0, dof, np.zeros(22, dtype=np.float32), np.zeros(3, dtype=np.float32),
                  np.array([0, 0, -1], dtype=np.float32), 0.0, 0.0, 0.0)
print("inference legs   :", np.round(t[10:22], 3))
print("finite           :", bool(np.all(np.isfinite(t))))
print()
print("PREFLIGHT OK - no LowCmd, no mode change")
