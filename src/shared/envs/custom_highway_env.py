from highway_env import utils
from highway_env.utils import near_split
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs import HighwayEnv
from .custom_behavior import CustomIDMVehicle

class CustomHighwayEnv(HighwayEnv):
    """
    A highway driving environment.
    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

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
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "initial_speed": 25,#25,
            "duration": 40,  # [s]
            "ego_spacing": 2,#2
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False,
            "policy_frequency": 1,

            "other_vehicles_initial_speed": None,
            "other_vehicles_delta": None,
        })
        return config

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=self.config["initial_speed"],
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, speed=self.config["other_vehicles_initial_speed"], spacing=1 / self.config["vehicles_density"])
                if self.config["other_vehicles_delta"]:
                    vehicle.DELTA = self.config["other_vehicles_delta"]
                else:
                    vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
    
    def spawn_vehicle(self):
        vehicle = CustomIDMVehicle.create_random_front_vehicle(self.road, self.controlled_vehicles[0], speed=self.config["other_vehicles_initial_speed"], spacing=1 / self.config["vehicles_density"])
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
