import math
import time

from robot_sim.client import SimulatorClient
from robot_sim.models import Gait, GaitPhase


client = SimulatorClient()
client.resume()
gait = Gait(
    name="walk_in_place",
    cycle_s=1.0,
    phases=(
        GaitPhase(
            duration=0.25,
            joints={
                "right_hip": -0.35,
                "right_knee": 1.35,
                "left_hip": 0.05,
                "left_knee": 0.95,
            },
        ),
        GaitPhase(
            duration=0.25,
            joints={
                "right_hip": 0.05,
                "right_knee": 0.95,
                "left_hip": -0.35,
                "left_knee": 1.35,
            },
        ),
        GaitPhase(
            duration=0.25,
            joints={
                "right_hip": 0.15,
                "right_knee": 1.05,
                "left_hip": -0.15,
                "left_knee": 1.15,
            },
        ),
        GaitPhase(
            duration=0.25,
            joints={
                "right_hip": -0.15,
                "right_knee": 1.15,
                "left_hip": 0.15,
                "left_knee": 1.05,
            },
        ),
    ),
)
client.send_gait(gait)
print("walk-in-place gait sent to local desktop server")
