from configparser import ConfigParser
import numpy as np
from pathlib import Path

class Memory():

    def get_config(self):
            config = ConfigParser()
            path = Path(Path.cwd() / "config" / "config.ini")
            config.read(path)
            return config
       
    def __init__(self):
        #read parameters for memory arrays
        self.memory_entry_counter = 0
        self.memory_config = self.get_config()
        self.memory_size = int(self.memory_config["Memory"]["maximal_memory_size"])
        self.memory_type = np.dtype(self.memory_config["Memory"]["data_type"])
        self.memory_input_dimensions = int(self.memory_config["Environment"]["observation_space_dimensions"])

        #create memory arrays for all relevant dimensions within the lunar lander environment
        self.actions_memory = np.zeros(self.memory_size, self.memory_type)
        self.rewards_memory = np.zeros(self.memory_size, self.memory_type)
        self.states_memory = np.zeros((self.memory_size, self.memory_input_dimensions), self.memory_type)                              
        self.following_states_memory = np.zeros((self.memory_size, self.memory_input_dimensions), self.memory_type)
        self.termination_flags_memory = np.zeros(self.memory_size, self.memory_type)

    def store_in_memory(self, action, reward, state, new_state, termination_flag):
        self.actions_memory[self.memory_entry_counter] = action
        self.rewards_memory[self.memory_entry_counter] = reward
        self.states_memory[self.memory_entry_counter] = state
        self.following_states_memory[self.memory_entry_counter] = new_state   
        self.termination_flags_memory[self.memory_entry_counter] = int(termination_flag)
        self.memory_entry_counter = self.memory_entry_counter + 1

if __name__ == "__main__":
    memory = Memory()