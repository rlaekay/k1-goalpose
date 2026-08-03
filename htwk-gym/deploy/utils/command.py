from booster_robotics_sdk_python import LowCmd, LowCmdType, MotorCmd, B1JointCnt


def joint_count(cfg):
    """How many joints this robot actually has, per the config.

    NOT B1JointCnt. That constant is 23 and the SDK's B1JointIndex map places a
    waist at index 10, but this K1 has no waist: low_state carries 22 joints
    with the legs at 10..21. Indexing a 22-entry config array with range(23)
    raises IndexError mid-run -- which is exactly how the first hardware start
    died, right after the operator pressed 'r', taking LowCmd publication with
    it and dropping the robot into DAMPING.
    """
    common = cfg["common"]
    return int(common.get("joint_cnt", len(common["default_qpos"])))


def init_Cmd_T1(low_cmd: LowCmd, joint_cnt: int = B1JointCnt):
    low_cmd.cmd_type = LowCmdType.SERIAL
    motorCmds = [MotorCmd() for _ in range(joint_cnt)]
    low_cmd.motor_cmd = motorCmds

    for i in range(joint_cnt):
        low_cmd.motor_cmd[i].q = 0.0
        low_cmd.motor_cmd[i].dq = 0.0
        low_cmd.motor_cmd[i].tau = 0.0
        low_cmd.motor_cmd[i].kp = 0.0
        low_cmd.motor_cmd[i].kd = 0.0
        # weight is not effective in custom mode
        low_cmd.motor_cmd[i].weight = 0.0


def create_prepare_cmd(low_cmd: LowCmd, cfg):
    n = joint_count(cfg)
    init_Cmd_T1(low_cmd, n)
    for i in range(n):
        low_cmd.motor_cmd[i].kp = cfg["prepare"]["stiffness"][i]
        low_cmd.motor_cmd[i].kd = cfg["prepare"]["damping"][i]
        low_cmd.motor_cmd[i].q = cfg["prepare"]["default_qpos"][i]
    return low_cmd


def create_first_frame_rl_cmd(low_cmd: LowCmd, cfg):
    n = joint_count(cfg)
    init_Cmd_T1(low_cmd, n)
    for i in range(n):
        low_cmd.motor_cmd[i].kp = cfg["common"]["stiffness"][i]
        low_cmd.motor_cmd[i].kd = cfg["common"]["damping"][i]
        low_cmd.motor_cmd[i].q = cfg["common"]["default_qpos"][i]

    return low_cmd
