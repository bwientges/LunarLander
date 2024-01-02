from configparser import ConfigParser
import gym
from pathlib import Path

class Environment():

    def get_config(self):
        config = ConfigParser()
        path = Path(Path.cwd() / "config" / "config.ini")
        config.read(path)
        return config
    
    def create_environment(self):
        if str(self.config["Environment"]["environment_randomization"]) == "True":
            return gym.make(str(self.config["Environment"]["environment_name"]),enable_wind = True, wind_power = 10.0, turbulence_power = 1.0)
        elif str(self.config["Environment"]["environment_randomization"]) == "False":
            return gym.make(str(self.config["Environment"]["environment_name"]))
        else:
            print("Environment Randomization not specified! Please specify in config using True or False!")

    def get_randomization_status(self, rnd_status_config):
        if rnd_status_config == "True": 
            return "randomized"
        else:
            return "determinisitic"

    def __init__(self):
        self.config = self.get_config()
        self.gym = self.create_environment()
        self.terminated = False
        self.action_space = [integer for integer in range(int(self.config["Environment"]["action_space_dimensions"]))]
        self.randomization_status = self.get_randomization_status(str(self.config["Environment"]["environment_randomization"]))

    def reset_termination_flag(self) -> bool:
        self.terminated = False

if __name__ == "__main__":
    env = Environment()