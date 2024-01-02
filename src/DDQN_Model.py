from configparser import ConfigParser
import numpy as np
import tensorflow as tf
from pathlib import Path
import datetime

class Model():

    def get_config(self):
            config = ConfigParser()
            path = Path(Path.cwd() / "config" / "config.ini")
            config.read(path)
            return config
        
    def __init__(self):
        self.model_config = self.get_config()
        self.nn_input_dimensions = int(self.model_config["Environment"]["observation_space_dimensions"])
        self.nn_output_dimensions = int(self.model_config["Environment"]["action_space_dimensions"])
        self.nn_fully_conn_layer_dim = int(self.model_config["DDQN"]["fully_connected_layer_dimensions"])
        self.nn_hidden_layer_dim = int(self.model_config["DDQN"]["hidden_layer_dimensions"])
        self.nn_fully_conn_layer_act_funct = str(self.model_config["DDQN"]["fully_connected_layer_activation"])
        self.nn_hidden_layer_act_funct = str(self.model_config["DDQN"]["hidden_layer_activation"])
        self.nn_output_layer_activation_function = str(self.model_config["DDQN"]["output_layer_activation"])
        self.nn_optimizer = str(self.model_config["DDQN"]["optimizer"])
        self.nn_loss_function = str(self.model_config["DDQN"]["loss_function"])
        self.nn_model_filename = str(self.model_config["DDQN"]["model_file_name"])
        self.model_name = str(self.model_config["Model"]["model_name"])
        self.nn_training_batch_size = int(self.model_config["Model"]["training_batch_size"]) 
        self.no_episodes = int(self.model_config["Model"]["number_of_episodes"])
        self.current_epsilon = float(self.model_config["Model"]["epsilon_initial"])
        self.minimal_epsilon = float(self.model_config["Model"]["epsilon_minimal"])
        self.epsilon_decay = float(self.model_config["Model"]["epsilon_decay"])
        self.discount_rate = float(self.model_config["Model"]["discount_rate"])
        self.weight_replacement_thresh = float(self.model_config["DDQN"]["weight_replacement_thresh"])
    
    def create_neural_network (self):
        tf.compat.v1.disable_eager_execution()
        neural_network = tf.keras.Sequential([
                tf.keras.layers.Dense(self.nn_fully_conn_layer_dim, activation=self.nn_fully_conn_layer_act_funct),
                tf.keras.layers.Dense(self.nn_hidden_layer_dim, activation=self.nn_hidden_layer_act_funct),
                tf.keras.layers.Dense(self.nn_hidden_layer_dim, activation=self.nn_hidden_layer_act_funct),
                tf.keras.layers.Dense(self.nn_output_dimensions)])
        neural_network.compile(optimizer=self.nn_optimizer, loss=self.nn_loss_function)
        return neural_network
    
    def check_minimal_availiable_samples(self, memory) -> bool:
        if memory.memory_entry_counter > self.nn_training_batch_size:
            return True
        else:
            return False

    def restrict_to_availiable_samples(self, memory) -> int:
        if memory.memory_entry_counter <= memory.memory_size:        
            return memory.memory_entry_counter
        
    def pick_random_training_batch(self, availiable_samples) -> np.array:
        random_batch = np.random.choice(availiable_samples, self.nn_training_batch_size, replace=False)
        return random_batch

    def sample_from_memory(self, memory):
        availiable_samples = self.restrict_to_availiable_samples(memory)
        random_batch = self.pick_random_training_batch(availiable_samples)        
        batch_actions = memory.actions_memory[random_batch]
        batch_rewards = memory.rewards_memory[random_batch]
        batch_states = memory.states_memory[random_batch]
        batch_following_states = memory.following_states_memory[random_batch]
        batch_termination_flags = memory.termination_flags_memory[random_batch]
        return batch_actions, batch_rewards, batch_states, batch_following_states, batch_termination_flags

    def decay_epsilon(self) -> float:
        if self.current_epsilon > self.minimal_epsilon:
            self.current_epsilon -= self.epsilon_decay     
            return self.current_epsilon
        
    def calculuate_q_target_values(self, q_values_state, q_values_following_state, q_predicted_values, batch_actions, batch_rewards, batch_termination_flags) -> np.array:
        q_target_values = q_values_state
        batch_index = np.arange(self.nn_training_batch_size, dtype=np.int32)
        q_target_values[batch_index, np.int32(batch_actions)] = batch_rewards + self.discount_rate \
            * q_values_following_state[batch_index, np.argmax(q_values_state, axis=1)]*(1-batch_termination_flags) #the real update pt2
        return q_target_values
    
    def check_for_weight_replacement(self, memory) -> bool:
        if memory.memory_entry_counter % self.weight_replacement_thresh == 0:
            return True
        else:
            return False

    def conditionally_update_target_network_weights(self, memory, target_network, eval_network):
        if self.check_for_weight_replacement(memory) == True:
            target_network.set_weights(eval_network.get_weights())
        return target_network
    
    def update(self, eval_network, target_network, memory):
        
        if self.check_minimal_availiable_samples(memory) == True:
            
            batch_actions, batch_rewards, batch_states, batch_following_states, batch_termination_flags = self.sample_from_memory(memory)
            
            q_values_state = eval_network.predict(batch_states)
            q_values_following_state = target_network.predict(batch_following_states)
            q_predicted_values = eval_network.predict(batch_following_states)
            
            q_target_values = self.calculuate_q_target_values(q_values_state, q_values_following_state, q_predicted_values, batch_actions, batch_rewards, batch_termination_flags) # the real update!

            eval_network.train_on_batch(batch_states, q_target_values)
        
            self.decay_epsilon()

            target_network = self.conditionally_update_target_network_weights(memory, target_network, eval_network)

            return eval_network, target_network, memory

    def save_trained_nn(self, neural_network, environment) -> None:
        time = str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))
        path = Path(Path.cwd() / "models" / str(self.model_name+"_model_"+str(self.no_episodes) +"_eps_"+str(environment.randomization_status)+"_env_"+ time + self.nn_model_filename)) 
        neural_network.save(path, save_format = "h5")


if __name__ == "__main__":
    model = Model()
