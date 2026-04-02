from robot_sim.client import SimulatorClient
from robot_sim.models import Pose


client = SimulatorClient()
client.resume()
client.set_pose(
    Pose(
        joints={
            "right_hip": -0.12,
            "right_knee": 1.08,
            "left_hip": -0.12,
            "left_knee": 1.08,
        }
    )
)
print("stand pose sent to local desktop server")
