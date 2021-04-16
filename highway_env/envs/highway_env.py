import numpy as np
from typing import Tuple
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    RIGHT_LANE_REWARD: float = 0.1
    """The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""

    HIGH_SPEED_REWARD: float = 0.4
    """The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"]."""

    LANE_CHANGE_REWARD: float = 0
    """The reward received at each lane change action."""
    

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 3,
            "vehicles_count": 3,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 20,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self.alarm=False

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        t=np.random.randint(11)
        for others in other_per_controlled:
            # controlled_vehicle = self.action_type.vehicle_class.create_random(
            #     self.road,
            #     amir=False,
            #     speed=np.random.randint(23,29),

            #     lane_id=1,
            #     spacing=self.config["ego_spacing"]
            # )
            flags=[False,False]
            # array=[t,None,np.random.randint(23,25)]
            for i in range(others+1):
                if i==t:
                    controlled_vehicle=self.action_type.vehicle_class.create_random(
                    self.road,
                    amir=False,
                    speed=np.random.randint(23,29),

                    lane_id=1,
                    spacing=self.config["ego_spacing"])
                    self.controlled_vehicles.append(controlled_vehicle)
                    self.road.vehicles.append(controlled_vehicle)
                else:
                  back=False
                  speed=np.random.randint(23,29)
                  self.road.vehicles.append(
                      other_vehicles_type.create_random(self.road,amir=False,speed=speed,back=back, spacing=1 / self.config["vehicles_density"])
                  )

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        reward= utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        if self.vehicle.crashed:
          reward=self.config["collision_reward"] * reward
        
        return reward


    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            self.vehicle.speed<=0 or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)
