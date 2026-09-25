"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from typing import Any

from simple.tasks.g1_wholebody_locomotion_pick_between_tables_teleop import (
    G1WholebodyLocomotionPickBetweenTablesTaskTeleop,
)
from simple.tasks.registry import TaskRegistry


@TaskRegistry.register("g1_wholebody_locomotion_pick_between_tables_holomotion_teleop")
class G1WholebodyLocomotionPickBetweenTablesHoloMotionTeleop(
    G1WholebodyLocomotionPickBetweenTablesTaskTeleop
):
    """Pick-between-tables teleop task driven by the HoloMotion motion-tracking
    policy (whole-body reference from PICO via HoloRetarget) instead of the
    decoupled WBC. Scene, objects, DR and success criteria are identical to
    ``g1_wholebody_locomotion_pick_between_tables_teleop``; only the controller
    (see ``simple.agents.holomotion_pico_agent``) differs.
    """

    uid: str = "g1_wholebody_locomotion_pick_between_tables_holomotion_teleop"
    label: str = "G1 HoloMotion TELEOP Pick Between Tables"
    description: str = (
        "A task where the G1 robot, controlled by the HoloMotion whole-body "
        "motion-tracking policy, must pick up an object from table1, walk to "
        "table2 and place it there."
    )

    metadata: dict[str, Any] = {
        **G1WholebodyLocomotionPickBetweenTablesTaskTeleop.metadata,
        "controller": "holomotion",
    }
