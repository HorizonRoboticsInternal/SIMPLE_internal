"""Replay-time robot / reset configuration (import before gym.make in every replay tool).

The per-episode scene fits and data/replay_datasets_phys were made with the SIMPLE working tree of 2026-09-25..29. Eval-side
defaults changed on this branch afterwards (2026-10-06 upstream merge: raised-arm controller init pose, stabilise threshold 1e-3;
2026-10-08: the three real-scene tasks reset arms + hands to a fixed level-3 pose). With those, the validated placements do not
reproduce (2026-10-19, bottle bottlevid2: 4 of 22 outcomes flip, base paths off by up to 14 cm). The replays therefore pin the
configuration they were validated with; evaluation is not affected (it does not import this module).
"""
import simple.robots.g1_sonic as _g1

_g1.STABILIZE_VEL_THRESHOLD = 1e-4                                    # the settle latch the fits ran with
for _arm in ("left_arm_cfg", "right_arm_cfg"):
    getattr(_g1.G1Sonic.controller_cfg, _arm).init_qpos = [0.0] * 7      # arms start at zero, not the raised eval pose


def pin_task(module) -> None:
    """Disable the eval-side fixed start pose (task.initial_joint_positions) on every task class of a kit module: the env then
    takes its first physics step from the robot's own reset, as the replays were fitted."""
    for v in list(vars(module).values()):
        if isinstance(v, type) and getattr(v, "initial_joint_positions", None):
            v.initial_joint_positions = {}
